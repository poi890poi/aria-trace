"""Calibrate HIK Bayer contrast/color against synchronized Android game frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import cv2
import numpy as np
import yaml

from aria_trace.adapters.filesystem.profile_registry import (
    AdapterRequest,
    ProfileContext,
    ProfileRegistry,
    context_from_rig_calibration,
)
from aria_trace.adapters.filesystem.session import SessionReader
from aria_trace.services.calibration.rig.hik.color_match import (
    optimize_mvs_bayer_conversion,
    synchronized_frame_pairs,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_indices(
    video: Path,
    records: Sequence[Mapping[str, object]],
    *,
    content_size_px: Optional[Sequence[int]] = None,
) -> np.ndarray:
    requested = {int(record["frame_index"]) for record in records}
    if not requested:
        raise ValueError("No video frame indices were selected")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError("Cannot open synchronized video: {}".format(video))
    decoded: Dict[int, np.ndarray] = {}
    try:
        index = 0
        last = max(requested)
        while index <= last:
            ok, frame = capture.read()
            if not ok:
                break
            if index in requested:
                if content_size_px is not None:
                    width, height = map(int, content_size_px)
                    if width > frame.shape[1] or height > frame.shape[0]:
                        raise ValueError(
                            "Declared content {}x{} exceeds decoded {}x{}".format(
                                width, height, frame.shape[1], frame.shape[0]
                            )
                        )
                    frame = frame[:height, :width]
                decoded[index] = frame.copy()
            index += 1
    finally:
        capture.release()
    missing = sorted(requested.difference(decoded))
    if missing:
        raise RuntimeError(
            "Video ended before selected frame indices: {}".format(missing)
        )
    return np.stack([decoded[int(record["frame_index"])] for record in records])


def _session_game_context(
    reader: SessionReader,
    rig_document: Mapping[str, object],
    *,
    game_id: Optional[str] = None,
) -> ProfileContext:
    capture = dict(reader.manifest.get("context") or {})
    launch = dict(capture.get("game_launch") or {})
    surface = dict(capture.get("phone_surface_orientation") or {})
    selected_game = game_id or capture.get("game_id") or launch.get("game_id")
    if not selected_game:
        raise ValueError(
            "Game color calibration requires --game-id when the session has no game identity"
        )
    rig_context = context_from_rig_calibration(rig_document)
    logical = surface.get("logical_size_px")
    natural = surface.get("natural_size_px")
    if not logical or not natural:
        raise ValueError("Session does not declare Android logical and natural display sizes")
    return ProfileContext(
        game_id=str(selected_game),
        platform="android",
        package=launch.get("package"),
        camera_adapter=rig_context.camera_adapter,
        camera_id=rig_context.camera_id,
        phone_id=rig_context.phone_id,
        phone_model=rig_context.phone_model,
        panel_display=rig_context.panel_display,
        game_display={
            "natural_panel_px": natural,
            "logical_frame_px": logical,
            "game_viewport_xywh": [0, 0, int(logical[0]), int(logical[1])],
            "rotation_quarter_turns": int(
                surface.get("quarter_turns_clockwise_from_natural", 0)
            ),
            "ui_layout_id": str(capture.get("ui_layout_id") or "default"),
        },
    )


def calibrate_game_color_session(
    session: Path,
    output: Path,
    *,
    profile_root: Optional[Path] = None,
    game_id: Optional[str] = None,
    maximum_pairs: int = 16,
    activate: bool = True,
) -> Mapping[str, object]:
    """Fit and publish one immutable rig-game color profile."""

    session = Path(session).resolve()
    output = Path(output).resolve()
    reader = SessionReader(session)
    if reader.manifest.get("status") != "complete":
        raise ValueError("Game color calibration requires a complete capture session")
    for stream_id in ("android_phone", "hik_phone"):
        if not reader.frames_by_stream.get(stream_id):
            raise ValueError("Session has no {} frames".format(stream_id))

    coordinate_file = session / "coordinate_spaces.yaml"
    if not coordinate_file.is_file():
        raise FileNotFoundError(
            "Synchronized session has no coordinate_spaces.yaml: {}".format(session)
        )
    spaces = yaml.safe_load(coordinate_file.read_text(encoding="utf-8"))
    matrix = (spaces.get("conversions") or {}).get(
        "adb_to_hik_phone_video_3x3"
    )
    if matrix is None:
        raise ValueError("Session does not define ADB-to-HIK video conversion")
    hik_stream = (spaces.get("streams") or {}).get("hik_phone") or {}
    content_size = hik_stream.get("content_size_px")

    calibration_value = ((reader.manifest.get("context") or {}).get("hik_capture") or {}).get(
        "rig_calibration"
    )
    if not calibration_value:
        raise ValueError("Session does not identify the rig calibration used for HIK capture")
    rig_calibration = Path(str(calibration_value)).resolve()
    if not rig_calibration.is_file():
        raise FileNotFoundError(
            "Session rig calibration is unavailable: {}".format(rig_calibration)
        )
    rig_document = json.loads(rig_calibration.read_text(encoding="utf-8"))
    context = _session_game_context(reader, rig_document, game_id=game_id)
    registry = ProfileRegistry(profile_root)
    active_rig = registry.resolve(
        "rig",
        ProfileContext(
            camera_id=context.camera_id,
            phone_id=context.phone_id,
            phone_model=context.phone_model,
            panel_display=context.panel_display,
        ),
    )
    active_rig_file = registry.runtime_file(
        active_rig, "hik_camera_calibration"
    ).resolve()
    if _sha256(active_rig_file) != _sha256(rig_calibration):
        raise ValueError(
            "Session HIK frames were not captured with the active rig revision; "
            "capture fresh synchronized frames before publishing game color"
        )

    android_records = list(reader.frames_by_stream["android_phone"])
    hik_records = list(reader.frames_by_stream["hik_phone"])
    android_times = np.asarray(
        [int(record["host_capture_time_ns"]) for record in android_records],
        dtype=np.int64,
    )
    hik_times = np.asarray(
        [int(record["host_capture_time_ns"]) for record in hik_records],
        dtype=np.int64,
    )
    selected = synchronized_frame_pairs(
        android_times, hik_times, maximum_pairs=maximum_pairs
    )
    if len(selected) < 4:
        raise ValueError("Fewer than four synchronized ADB/HIK frame pairs")
    selected_android = [android_records[a] for a, _h, _delta in selected]
    selected_hik = [hik_records[h] for _a, h, _delta in selected]
    android_frames = _decode_indices(
        reader.video_path("android_phone"), selected_android
    )
    hik_frames = _decode_indices(
        reader.video_path("hik_phone"),
        selected_hik,
        content_size_px=content_size,
    )
    selected_android_times = np.asarray(
        [int(record["host_capture_time_ns"]) for record in selected_android],
        dtype=np.int64,
    )
    selected_hik_times = np.asarray(
        [int(record["host_capture_time_ns"]) for record in selected_hik],
        dtype=np.int64,
    )

    mask_path = session / "cross_source_check" / "valid_mask.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.is_file() else None
    if mask is None or mask.shape != hik_frames.shape[1:3]:
        mask = np.full(hik_frames.shape[1:3], 255, np.uint8)
    conversion, evidence = optimize_mvs_bayer_conversion(
        android_frames,
        selected_android_times,
        hik_frames,
        selected_hik_times,
        matrix,
        mask,
        maximum_pairs=maximum_pairs,
    )

    output.mkdir(parents=True, exist_ok=False)
    for filename, image in evidence.items():
        if not cv2.imwrite(str(output / filename), image):
            raise RuntimeError("Cannot write color evidence {}".format(filename))
    summary = {
        "schema_version": "1.0",
        "status": "calibrated_pending_publication",
        "calibration_kind": "hik_game_color",
        "session": str(session),
        "profile_context": context.as_dict(),
        "rig_revision": active_rig["revision_id"],
        "hik_bayer_conversion": conversion,
        "synchronized_source_frames": {
            "android_phone": [int(row["frame_index"]) for row in selected_android],
            "hik_phone": [int(row["frame_index"]) for row in selected_hik],
            "host_delta_ms": [float(row[2]) for row in selected],
            "coordinate_conversion": "coordinate_spaces.yaml#conversions.adb_to_hik_phone_video_3x3",
        },
        "evidence": list(evidence),
    }
    summary_path = output / "game_color_calibration.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    profile = registry.publish(
        "rig_game_color",
        context,
        {
            "profile_kind": "rig_game_color",
            "hik_bayer_conversion": conversion,
            "capabilities": {
                "game_matched_color": conversion.get("status") == "selected",
                "runtime_frame_passes": 0,
            },
        },
        dependencies={"rig": active_rig["revision_id"]},
        provenance={
            "game_color_calibration": str(summary_path),
            "session": str(session),
            "evidence": list(evidence),
        },
        review_state="accepted" if activate else "review_required",
        activate=activate,
    )
    summary["profile_revision"] = profile["revision_id"]
    summary["profile_publication"] = profile["publication"]
    summary["status"] = "accepted" if activate else "review_required"
    if activate:
        from aria_trace.workflows.adapter_export import export_resolved_adapter

        summary["standalone_camera_adapter"] = export_resolved_adapter(
            output / "hikcam_adapter.py",
            registry=registry,
            context=context,
            request=AdapterRequest(
                mode="full", color_order="BGR", color_policy="game_matched"
            ),
        )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Fit HIK MVS gamma/color from one synchronized ADB + rig-normalized "
            "HIK game capture and publish the active rig-game color profile"
        )
    )
    value.add_argument("session", type=Path)
    value.add_argument("output", type=Path)
    value.add_argument("--profile-root", type=Path)
    value.add_argument("--game-id")
    value.add_argument("--maximum-pairs", type=int, default=16)
    value.add_argument(
        "--candidate",
        action="store_true",
        help="publish for review without activating the profile",
    )
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.game_id is None:
        from aria_trace.adapters.filesystem.system_configuration import (
            load_system_configuration,
        )

        arguments.game_id = load_system_configuration(arguments.profile_root)[
            "game"
        ].get("game_id")
    result = calibrate_game_color_session(
        arguments.session,
        arguments.output,
        profile_root=arguments.profile_root,
        game_id=arguments.game_id,
        maximum_pairs=arguments.maximum_pairs,
        activate=not arguments.candidate,
    )
    conversion = result["hik_bayer_conversion"]
    print("Game color calibration: {}".format(Path(arguments.output).resolve()))
    print("Profile: {}".format(result["profile_revision"]))
    print(
        "Validation RGB MAE: {:.3f} -> {:.3f} DN".format(
            float(conversion["fit"]["baseline_validation"]["rgb_mae_dn"]),
            float(conversion["fit"]["selected_validation"]["rgb_mae_dn"]),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
