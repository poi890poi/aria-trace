"""Localize a mini-map in one completed dual-source zigzag session.

Android discovery and fitting are delegated directly to minimap_calibration.
HIK geometry is only projected through the same session's declared transform.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from .commented_yaml import write_commented_yaml
from .hik_bayer_color_match import optimize_mvs_bayer_conversion
from .minimap_calibration import (
    _color_heatmap,
    _stacked_difference_heatmap,
    android_minimap_discovery_config,
    calibrate_minimap_boundary_frames,
)
from .session import SessionReader


HEADER = """# AriaTrace current-session mini-map localization.
#
# Android geometry is fitted in the complete logical display. HIK geometry is
# only a projection through this capture session's declared registration."""

COMMENTS = {
    "scope": "The single responsibility and explicit exclusions of this stage.",
    "provenance": "The completed session and exact streams used as input.",
    "android_discovery": "Broad bounds used by the verified boundary backend.",
    "android": "Fitted full-display Android geometry and review evidence.",
    "cross_source_registration": "Current-session matrix and timestamp pairing.",
    "hik_session_observation": "Projected, session-local HIK geometry only.",
    "hik_bayer_conversion": (
        "Offline ADB/HIK color fit applied once inside MVS Bayer conversion."
    ),
    "evidence": "Review images tied to this fresh session.",
}


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


def _fractions(value: object, count: int, label: str) -> list:
    values = (
        [float(item.strip()) for item in value.split(",")]
        if isinstance(value, str)
        else [float(item) for item in value]
    )
    if len(values) != count:
        raise ValueError("{} must contain {} fractions".format(label, count))
    return values


def read_representative_frames(
    session: SessionReader, stream_id: str, maximum_frames: int = 48
) -> Tuple[np.ndarray, np.ndarray]:
    records = list(session.frames_by_stream.get(stream_id) or [])
    if len(records) < 12:
        raise ValueError("{} contains fewer than 12 frames".format(stream_id))
    selected = set(
        np.linspace(
            0, len(records) - 1, min(max(12, maximum_frames), len(records))
        )
        .round()
        .astype(int)
        .tolist()
    )
    capture = cv2.VideoCapture(str(session.video_path(stream_id)))
    if not capture.isOpened():
        raise RuntimeError("Cannot open {}".format(session.video_path(stream_id)))
    frames, times = [], []
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index in selected and index < len(records):
                frames.append(frame)
                times.append(int(records[index]["host_capture_time_ns"]))
            index += 1
    finally:
        capture.release()
    if len(frames) < 12:
        raise ValueError("{} yielded fewer than 12 frames".format(stream_id))
    return np.stack(frames), np.asarray(times, dtype=np.int64)


def load_session_registration(
    session_path: Path,
    android_size_px: Sequence[int],
    hik_size_px: Sequence[int],
) -> dict:
    path = Path(session_path) / "coordinate_spaces.yaml"
    if not path.is_file():
        raise RuntimeError(
            "Current session has no coordinate_spaces.yaml; HIK geometry will not be guessed"
        )
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    streams = (document or {}).get("streams") or {}
    if "android_phone" not in streams or "hik_phone" not in streams:
        raise RuntimeError("Session registration lacks the required two streams")
    for stream_id, actual in (
        ("android_phone", android_size_px),
        ("hik_phone", hik_size_px),
    ):
        declared = list(map(int, streams[stream_id].get("stored_size_px") or []))
        if declared and declared != list(map(int, actual)):
            raise RuntimeError(
                "{} size {} disagrees with registration {}".format(
                    stream_id, list(actual), declared
                )
            )
    matrix = np.asarray(
        ((document or {}).get("conversions") or {}).get(
            "adb_to_hik_phone_video_3x3"
        ),
        np.float64,
    )
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or abs(float(np.linalg.det(matrix))) < 1.0e-12
    ):
        raise RuntimeError("Current-session ADB-to-HIK matrix is invalid")
    return {"path": path.resolve(), "document": document, "matrix": matrix}


def nearest_synchronized_pair(
    android_times_ns: np.ndarray, hik_times_ns: np.ndarray
) -> dict:
    pairs = []
    for hik_index, hik_time in enumerate(hik_times_ns.tolist()):
        android_index = int(np.argmin(np.abs(android_times_ns - int(hik_time))))
        signed = int(android_times_ns[android_index]) - int(hik_time)
        pairs.append((abs(signed), signed, android_index, hik_index))
    pairs.sort(key=lambda item: item[0])
    absolute_ms = np.asarray([item[0] / 1.0e6 for item in pairs])
    _, signed, android_index, hik_index = pairs[0]
    return {
        "method": "nearest_host_capture_time_ns",
        "android_frame_array_index": android_index,
        "hik_frame_array_index": hik_index,
        "best_signed_delta_ms": signed / 1.0e6,
        "pair_count_considered": len(pairs),
        "absolute_delta_p50_ms": float(np.percentile(absolute_ms, 50)),
        "absolute_delta_p95_ms": float(np.percentile(absolute_ms, 95)),
        "absolute_delta_max_ms": float(np.max(absolute_ms)),
    }


def project_android_boundary(
    boundary: Mapping[str, object],
    android_to_hik_3x3: Sequence[Sequence[float]],
    hik_size_px: Sequence[int],
) -> dict:
    center = np.asarray([boundary["center_x"], boundary["center_y"]], np.float64)
    radius = float(boundary["radius"])
    angles = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    android_points = np.column_stack(
        (center[0] + np.cos(angles) * radius, center[1] + np.sin(angles) * radius)
    )
    matrix = np.asarray(android_to_hik_3x3, np.float64)
    points = cv2.perspectiveTransform(
        android_points.reshape((-1, 1, 2)), matrix
    ).reshape((-1, 2))
    mapped_center = cv2.perspectiveTransform(
        center.reshape((1, 1, 2)), matrix
    ).reshape(2)
    width, height = map(int, hik_size_px)
    visible = (
        (points[:, 0] >= 0)
        & (points[:, 0] < width)
        & (points[:, 1] >= 0)
        & (points[:, 1] < height)
    )
    polygon = np.round(points).astype(np.int32)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [polygon], 255, cv2.LINE_AA)
    nonzero = cv2.findNonZero(mask)
    if nonzero is None:
        raise RuntimeError("Mapped mini-map does not intersect the HIK frame")
    ellipse = cv2.fitEllipse(points.astype(np.float32))
    return {
        "center_xy": mapped_center.tolist(),
        "polygon": polygon,
        "polygon_xy": points[::10].tolist(),
        "bounding_xywh": list(map(int, cv2.boundingRect(nonzero))),
        "visible_circumference_fraction": float(np.mean(visible)),
        "projected_ellipse": {
            "center_xy": list(map(float, ellipse[0])),
            "diameter_xy": list(map(float, ellipse[1])),
            "angle_deg": float(ellipse[2]),
        },
        "mask": mask,
    }


def localize_session_minimap(
    session_path: Path,
    output_path: Path,
    *,
    android_discovery: Optional[dict] = None,
    maximum_frames: int = 48,
) -> dict:
    session_path = Path(session_path).resolve()
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("Output already exists: {}".format(output_path))
    session = SessionReader(session_path)
    context = session.manifest.get("context") or {}
    if session.manifest.get("status") != "complete":
        raise ValueError("Localization requires a complete session")
    if context.get("capture_kind") != "zigzag_minimap_source_data":
        raise ValueError("Session is not zigzag mini-map source data")
    if not {"android_phone", "hik_phone"}.issubset(session.frames_by_stream):
        raise ValueError("Session requires android_phone and registered hik_phone")

    output_path.mkdir(parents=True)
    android_output = output_path / "android"
    hik_output = output_path / "hik_session"
    android_output.mkdir()
    hik_output.mkdir()
    android_frames, android_times = read_representative_frames(
        session, "android_phone", maximum_frames
    )
    hik_frames, hik_times = read_representative_frames(
        session, "hik_phone", maximum_frames
    )
    discovery = android_minimap_discovery_config(android_discovery)
    try:
        fitted = calibrate_minimap_boundary_frames(
            android_frames, android_output, config={"discovery": discovery}
        )
    except Exception as exc:
        cv2.imwrite(
            str(android_output / "failure_source_average.png"),
            android_frames.mean(axis=0).astype(np.uint8),
        )
        cv2.imwrite(
            str(android_output / "failure_stacked_difference_heatmap.png"),
            _color_heatmap(_stacked_difference_heatmap(android_frames)),
        )
        _atomic_json(
            output_path / "failure.json",
            {
                "stage": "verified_android_boundary_backend",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "session_path": str(session_path),
            },
        )
        raise

    android_height, android_width = android_frames.shape[1:3]
    hik_height, hik_width = hik_frames.shape[1:3]
    registration = load_session_registration(
        session_path, [android_width, android_height], [hik_width, hik_height]
    )
    synchronization = nearest_synchronized_pair(android_times, hik_times)
    projection = project_android_boundary(
        fitted["outer_boundary"], registration["matrix"], [hik_width, hik_height]
    )

    boundary = fitted["outer_boundary"]
    android_mask = np.zeros((android_height, android_width), np.uint8)
    cv2.circle(
        android_mask,
        (round(boundary["center_x"]), round(boundary["center_y"])),
        round(boundary["radius"]),
        255,
        -1,
    )
    cv2.imwrite(str(android_output / "actual_shift_estimation_mask.png"), android_mask)
    cv2.imwrite(str(hik_output / "projected_shift_estimation_mask.png"), projection["mask"])

    ai = synchronization["android_frame_array_index"]
    hi = synchronization["hik_frame_array_index"]
    android_image, hik_image = android_frames[ai], hik_frames[hi]
    warped = cv2.warpPerspective(
        android_image, registration["matrix"], (hik_width, hik_height)
    )
    cv2.imwrite(
        str(hik_output / "synchronized_registration_triptych.png"),
        np.hstack((warped, hik_image, cv2.absdiff(warped, hik_image))),
    )
    overlay = hik_image.copy()
    cv2.polylines(overlay, [projection["polygon"]], True, (255, 0, 255), 3, cv2.LINE_AA)
    cv2.imwrite(str(hik_output / "mapped_boundary_overlay.png"), overlay)

    try:
        bayer_conversion, color_evidence = optimize_mvs_bayer_conversion(
            android_frames,
            android_times,
            hik_frames,
            hik_times,
            registration["matrix"],
            projection["mask"],
        )
        for name, image in color_evidence.items():
            cv2.imwrite(str(hik_output / name), image)
        bayer_conversion["evidence"] = [
            "hik_session/{}".format(name) for name in color_evidence
        ]
    except Exception as exc:
        # Boundary calibration remains useful even when synchronized game
        # content is too sparse or delayed for a trustworthy color fit.
        bayer_conversion = {
            "schema_version": "1.0",
            "status": "unavailable",
            "reason": str(exc),
            "non_gating": True,
            "runtime_application": {
                "additional_frame_passes": 0,
                "additional_frame_copies": 0,
            },
        }

    check_path = session_path / "cross_source_check" / "summary.json"
    check = (
        json.loads(check_path.read_text(encoding="utf-8"))
        if check_path.is_file()
        else None
    )
    discovered = fitted["model"].get("discovery") or {}
    result = {
        "schema_version": "1.0",
        "status": "review_required",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "includes": [
                "Android mini-map discovery",
                "verified boundary fit",
                "current-session HIK projection",
            ],
            "excludes": [
                "rig calibration",
                "acquisition",
                "game control",
                "camera-space discovery",
                "runtime profiles",
                "cursor",
                "pose",
                "tracking",
            ],
        },
        "provenance": {
            "session_path": str(session_path),
            "session_id": session.manifest.get("session_id"),
            "android_stream_id": "android_phone",
            "hik_stream_id": "hik_phone",
            "representative_frame_count": {
                "android_phone": len(android_frames),
                "hik_phone": len(hik_frames),
            },
        },
        "android_discovery": {
            "config": discovery,
            "method": discovered.get("method"),
            "selected": discovered.get("selected"),
            "ranked_candidates": discovered.get("candidates", []),
            "precise_prior_used": False,
        },
        "android": {
            "coordinate_space": "android_logical_display_pixels",
            "frame_size_px": [android_width, android_height],
            "outer_boundary": boundary,
            "shift_estimation_mask": "android/actual_shift_estimation_mask.png",
            "verified_backend_evidence": [
                "android/{}".format(item["name"]) for item in fitted["evidence"]
            ],
        },
        "cross_source_registration": {
            "method": "current_session_coordinate_spaces_yaml",
            "file": str(registration["path"]),
            "android_to_hik_3x3": registration["matrix"].tolist(),
            "synchronization": synchronization,
            "game_frame_evidence": {
                "summary": str(check_path) if check is not None else None,
                "status": check.get("status") if check is not None else "unavailable",
                "metrics": check.get("metrics") if check is not None else None,
                "non_gating": True,
            },
        },
        "hik_session_observation": {
            "coordinate_space": "hik_session_aligned_visible_phone_pixels",
            "frame_size_px": [hik_width, hik_height],
            "center_xy": projection["center_xy"],
            "boundary_polygon_xy": projection["polygon_xy"],
            "bounding_xywh": projection["bounding_xywh"],
            "visible_circumference_fraction": projection["visible_circumference_fraction"],
            "projected_ellipse": projection["projected_ellipse"],
            "session_local": True,
            "reusable_camera_prior": False,
        },
        "hik_bayer_conversion": bayer_conversion,
        "evidence": {
            "android_shift_mask": "android/actual_shift_estimation_mask.png",
            "hik_projected_shift_mask": "hik_session/projected_shift_estimation_mask.png",
            "hik_mapped_boundary_overlay": "hik_session/mapped_boundary_overlay.png",
            "synchronized_registration_triptych": "hik_session/synchronized_registration_triptych.png",
        },
    }
    np.savez_compressed(
        str(output_path / "minimap_geometry.npz"),
        android_boundary=np.asarray(
            [boundary["center_x"], boundary["center_y"], boundary["radius"]]
        ),
        android_mask=android_mask,
        hik_boundary_polygon=projection["polygon"],
        hik_mask=projection["mask"],
        android_to_hik_3x3=registration["matrix"],
    )
    _atomic_json(output_path / "localization_summary.json", result)
    write_commented_yaml(
        output_path / "localization_summary.yaml",
        result,
        header=HEADER,
        section_comments=COMMENTS,
    )
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Localize a mini-map in one completed dual-source session"
    )
    value.add_argument("session", type=Path)
    value.add_argument("--output", type=Path)
    value.add_argument("--android-center-region", default="0,0,0.35,0.35")
    value.add_argument("--android-radius-fraction", default="0.07,0.22")
    value.add_argument("--android-min-visible", type=float, default=0.85)
    value.add_argument("--maximum-frames", type=int, default=48)
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    output = arguments.output or Path("artifacts") / (
        "session-minimap-localization-"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    result = localize_session_minimap(
        arguments.session,
        output,
        android_discovery={
            "center_region_xyxy_fraction": _fractions(
                arguments.android_center_region, 4, "Android center region"
            ),
            "radius_fraction_range": _fractions(
                arguments.android_radius_fraction, 2, "Android radius range"
            ),
            "minimum_circle_visible_fraction": arguments.android_min_visible,
        },
        maximum_frames=arguments.maximum_frames,
    )
    boundary = result["android"]["outer_boundary"]
    print("Mini-map localization: {}".format(Path(output).resolve()))
    print(
        "Android boundary: ({:.2f}, {:.2f}), radius {:.2f}px".format(
            boundary["center_x"], boundary["center_y"], boundary["radius"]
        )
    )
    print("Status: review_required; inspect localization_summary.yaml and evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
