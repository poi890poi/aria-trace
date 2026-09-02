"""Calibrate HIK Bayer contrast/color against synchronized Android game frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
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
from aria_trace.adapters.android.spaces import natural_to_logical_matrix
from aria_trace.domain.spatial import (
    normalize_legacy_geometry,
    raster_space,
    require_spatial_geometry,
    transform_circle_similarity,
)
from aria_trace.evidence.rig_alignment import (
    DEFAULT_MAXIMUM_RESIDUAL_TRANSLATION_PX,
    cross_source_alignment_evidence,
    cross_source_alignment_warning,
)
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


def _adb_color_statistics(
    frames: np.ndarray, sampling_mask: np.ndarray
) -> Mapping[str, object]:
    """Compact camera-independent appearance reference in decoded BGR space."""

    stride = 8
    sampled_frames = frames[:, ::stride, ::stride, :]
    sampled_mask = sampling_mask[::stride, ::stride] > 0
    sampled = sampled_frames[:, sampled_mask, :].reshape(-1, 3).astype(np.float32)
    if sampled.shape[0] < 32:
        stride = 1
        sampled = frames[:, sampling_mask > 0, :].reshape(-1, 3).astype(
            np.float32
        )
    if sampled.shape[0] < 32:
        raise ValueError("Mini-map mask contains insufficient color-reference pixels")
    channels = {}
    for index, name in enumerate(("blue", "green", "red")):
        values = sampled[:, index]
        channels[name] = {
            "mean_dn": float(np.mean(values)),
            "stddev_dn": float(np.std(values)),
            "percentiles_dn": {
                str(percentile): float(np.percentile(values, percentile))
                for percentile in (1, 5, 25, 50, 75, 95, 99)
            },
        }
    return {
        "color_order": "BGR",
        "source_frame_count": int(frames.shape[0]),
        "sample_stride_px": stride,
        "sample_count": int(sampled.shape[0]),
        "sampling_region": "inset_minimap_circle",
        "channels": channels,
    }


def _logical_minimap_sampling_mask(
    registry: ProfileRegistry,
    context: ProfileContext,
    frame_size_px: Sequence[int],
    *,
    phone_game_revision: Optional[str] = None,
    radius_fraction: float = 0.96,
) -> tuple[np.ndarray, Mapping[str, object]]:
    """Resolve portable mini-map geometry into this session's ADB raster."""

    if phone_game_revision:
        profile = registry.resolve_revision(
            phone_game_revision,
            context,
            expected_kind="phone_game",
        )
        selection = "explicit_current_game_calibration_revision"
    else:
        profile = registry.resolve("phone_game", context)
        selection = "active_compatible_phone_game_revision"
    payload = dict(profile.get("payload") or {})
    boundary_value = payload.get("outer_boundary")
    if not isinstance(boundary_value, Mapping):
        raise ValueError("Phone-game profile has no mini-map boundary geometry")
    logical_size = list(map(int, context.game_display.get("logical_frame_px") or []))
    natural_size = list(map(int, context.panel_display.get("natural_panel_px") or []))
    actual_size = list(map(int, frame_size_px))
    if len(logical_size) != 2 or len(natural_size) != 2:
        raise ValueError("Profile context lacks logical or natural panel dimensions")
    if actual_size != logical_size:
        raise ValueError(
            "ADB color frames {} do not match game logical raster {}".format(
                actual_size, logical_size
            )
        )
    natural_space = raster_space(
        "android_phone_natural_display_pixels", natural_size
    )
    logical_space = raster_space("android_logical_display_pixels", logical_size)
    boundary = dict(boundary_value)
    if "space" not in boundary:
        boundary = normalize_legacy_geometry(boundary, "circle", natural_space)
    else:
        boundary = require_spatial_geometry(boundary, "circle")
    source_space = boundary["space"]
    if source_space["space_id"] == natural_space["space_id"]:
        if list(map(int, source_space["size_px"])) != natural_size:
            raise ValueError("Phone-game natural boundary raster is incompatible")
        logical_boundary = transform_circle_similarity(
            boundary,
            natural_to_logical_matrix(
                natural_size,
                int(context.game_display.get("rotation_quarter_turns", 0)),
            ),
            logical_space,
        )
    elif source_space["space_id"] == logical_space["space_id"]:
        if list(map(int, source_space["size_px"])) != logical_size:
            raise ValueError("Phone-game logical boundary raster is incompatible")
        logical_boundary = boundary
    else:
        raise ValueError(
            "Phone-game boundary uses unsupported space {!r}".format(
                source_space["space_id"]
            )
        )
    radius = float(logical_boundary["radius"]) * float(radius_fraction)
    if radius <= 1.0:
        raise ValueError("Phone-game mini-map radius is too small for color fitting")
    mask = np.zeros((logical_size[1], logical_size[0]), np.uint8)
    cv2.circle(
        mask,
        (
            int(round(float(logical_boundary["center_x"]))),
            int(round(float(logical_boundary["center_y"]))),
        ),
        int(round(radius)),
        255,
        -1,
        cv2.LINE_AA,
    )
    return mask, {
        "phone_game_revision": profile["revision_id"],
        "selection": selection,
        "radius_fraction": float(radius_fraction),
        "logical_boundary": logical_boundary,
        "sample_pixel_count": int(np.count_nonzero(mask)),
    }


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


def _decode_session_records(
    reader: SessionReader,
    stream_id: str,
    records: Sequence[Mapping[str, object]],
    *,
    content_size_px: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Decode one traceable stream from either images or its video container."""

    storage_kinds = {
        str((record.get("storage") or {}).get("kind") or "video")
        for record in records
    }
    if storage_kinds == {"image_series"}:
        frames = reader.read_image_frames(records)
        if content_size_px is not None:
            width, height = map(int, content_size_px)
            if width > frames.shape[2] or height > frames.shape[1]:
                raise ValueError(
                    "Declared content {}x{} exceeds decoded {}x{}".format(
                        width, height, frames.shape[2], frames.shape[1]
                    )
                )
            frames = frames[:, :height, :width]
        return frames
    if storage_kinds != {"video"}:
        raise ValueError(
            "Stream {} mixes incompatible frame storage: {}".format(
                stream_id, sorted(storage_kinds)
            )
        )
    return _decode_indices(
        reader.video_path(stream_id),
        records,
        content_size_px=content_size_px,
    )


def _check_color_spatial_alignment(
    android_frames: np.ndarray,
    hik_frames: np.ndarray,
    adb_to_hik_3x3: np.ndarray,
    hik_sampling_mask: np.ndarray,
    output: Path,
) -> Mapping[str, object]:
    """Verify declared geometry before fitting HIK appearance parameters."""

    pair_results = []
    representative = None
    for index, (adb_frame, hik_frame) in enumerate(
        zip(android_frames, hik_frames)
    ):
        adb_in_hik = cv2.warpPerspective(
            adb_frame,
            adb_to_hik_3x3,
            (hik_frame.shape[1], hik_frame.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        try:
            metrics, images = cross_source_alignment_evidence(
                adb_in_hik,
                hik_frame,
                hik_sampling_mask,
            )
        except ValueError as exc:
            pair_results.append(
                {
                    "pair_index": index,
                    "confidence": 0.0,
                    "information_eligible": False,
                    "residual_translation": {
                        "status": "ineligible",
                        "reason": str(exc),
                    },
                    "warning": str(exc),
                }
            )
            continue
        residual = dict(metrics.get("residual_translation") or {})
        pair_results.append(
            {
                "pair_index": index,
                "confidence": float(metrics["confidence"]),
                "information_eligible": bool(metrics["information_eligible"]),
                "residual_translation": residual,
                "warning": cross_source_alignment_warning(metrics),
            }
        )
        rank = (
            residual.get("status") == "measured",
            float(residual.get("median_phase_response", 0.0)),
            float(metrics["information_quality"]),
        )
        if representative is None or rank > representative[0]:
            representative = (rank, index, images)

    measured = [
        item
        for item in pair_results
        if item["information_eligible"]
        and item["residual_translation"].get("status") == "measured"
    ]
    status = "inconclusive"
    warning = (
        "Cross-source mini-map alignment could not be verified from the "
        "selected synchronized frames; color fitting will continue with a "
        "review-required spatial warning"
    )
    aggregate = None
    if len(measured) >= 2:
        offsets = np.asarray(
            [
                item["residual_translation"]["hik_offset_xy_px_from_adb"]
                for item in measured
            ],
            dtype=np.float64,
        )
        median_offset = np.median(offsets, axis=0)
        spread = float(
            np.median(np.linalg.norm(offsets - median_offset, axis=1))
        )
        magnitude = float(np.linalg.norm(median_offset))
        aggregate = {
            "hik_offset_xy_px_from_adb": median_offset.tolist(),
            "hik_correction_xy_px_to_adb": (-median_offset).tolist(),
            "magnitude_px": magnitude,
            "pair_consensus_spread_px": spread,
            "measured_pair_count": len(measured),
            "maximum_allowed_residual_translation_px": (
                DEFAULT_MAXIMUM_RESIDUAL_TRANSLATION_PX
            ),
        }
        if spread <= DEFAULT_MAXIMUM_RESIDUAL_TRANSLATION_PX:
            if magnitude <= DEFAULT_MAXIMUM_RESIDUAL_TRANSLATION_PX:
                status = "aligned"
                warning = ""
            else:
                status = "displaced"
                warning = (
                    "Declared ADB-to-HIK mini-map conversion is consistently "
                    "displaced by {:.2f}px (HIK relative to ADB: dx={:.2f}, "
                    "dy={:.2f}). Color fitting was stopped because it would "
                    "compare different pixels; repair/recompose the owning "
                    "space conversion and capture fresh synchronized data."
                ).format(magnitude, median_offset[0], median_offset[1])
        else:
            warning = (
                "Cross-source residual translations disagree by {:.2f}px "
                "across synchronized frames; color fitting will continue with "
                "a review-required spatial warning"
            ).format(spread)

    output.mkdir(parents=True, exist_ok=False)
    evidence_files = []
    if representative is not None:
        _rank, pair_index, images = representative
        for filename, image in images.items():
            evidence_name = "spatial_alignment_{}".format(filename)
            if not cv2.imwrite(str(output / evidence_name), image):
                raise RuntimeError(
                    "Cannot write spatial-alignment evidence {}".format(
                        evidence_name
                    )
                )
            evidence_files.append(evidence_name)
    summary = {
        "schema_version": "1.0",
        "status": status,
        "method": "declared_space_transform_then_multilevel_threshold_features",
        "non_correcting": True,
        "space_contract": {
            "comparison_space": "hik_phone_video",
            "adb_source_space": "android_logical_display_pixels",
            "hik_source_space": "hik_phone_video",
            "adb_to_comparison_3x3": adb_to_hik_3x3.tolist(),
            "mask_space": "hik_phone_video",
            "coordinates": "pixel_center_xy",
        },
        "representative_pair_index": (
            representative[1] if representative is not None else None
        ),
        "aggregate": aggregate,
        "pairs": pair_results,
        "evidence_files": evidence_files,
        "warning": warning or None,
    }
    (output / "spatial_alignment_check.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if status == "displaced":
        raise ValueError(
            "{} Evidence: {}".format(
                warning,
                output / "spatial_alignment_check.json",
            )
        )
    if warning:
        print("Warning: {} Evidence: {}".format(
            warning,
            output / "spatial_alignment_check.json",
        ))
    else:
        print(
            "Cross-source mini-map alignment verified: residual {:.2f}px."
            .format(float(aggregate["magnitude_px"]))
        )
    return summary


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


# Shared, public session contracts used by the task-level game calibration
# orchestrator. Private aliases above remain for compatibility with existing
# callers and tests.
decode_session_records = _decode_session_records
session_game_context = _session_game_context
sha256_file = _sha256


def calibrate_game_color_session(
    session: Path,
    output: Path,
    *,
    profile_root: Optional[Path] = None,
    game_id: Optional[str] = None,
    maximum_pairs: int = 16,
    activate: bool = True,
    phone_game_revision: Optional[str] = None,
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
    android_frames = _decode_session_records(
        reader, "android_phone", selected_android
    )
    hik_frames = _decode_session_records(
        reader,
        "hik_phone",
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

    adb_mask, sampling_geometry = _logical_minimap_sampling_mask(
        registry,
        context,
        [android_frames.shape[2], android_frames.shape[1]],
        phone_game_revision=phone_game_revision,
    )

    mask_path = session / "cross_source_check" / "valid_mask.png"
    valid_hik_mask = (
        cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_path.is_file()
        else None
    )
    if valid_hik_mask is None or valid_hik_mask.shape != hik_frames.shape[1:3]:
        valid_hik_mask = np.full(hik_frames.shape[1:3], 255, np.uint8)
    hik_map_mask = cv2.warpPerspective(
        adb_mask,
        np.asarray(matrix, np.float64),
        (hik_frames.shape[2], hik_frames.shape[1]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask = cv2.bitwise_and(valid_hik_mask, hik_map_mask)
    if np.count_nonzero(mask) == 0:
        raise ValueError(
            "Projected mini-map has no valid HIK pixels for color fitting"
        )
    spatial_alignment = _check_color_spatial_alignment(
        android_frames,
        hik_frames,
        np.asarray(matrix, np.float64),
        mask,
        output,
    )
    conversion, evidence = optimize_mvs_bayer_conversion(
        android_frames,
        selected_android_times,
        hik_frames,
        selected_hik_times,
        matrix,
        mask,
        maximum_pairs=maximum_pairs,
    )

    evidence = dict(evidence)
    evidence["adb_minimap_color_sampling_mask.png"] = adb_mask
    evidence["hik_minimap_color_sampling_mask.png"] = mask
    for filename, image in evidence.items():
        if not cv2.imwrite(str(output / filename), image):
            raise RuntimeError("Cannot write color evidence {}".format(filename))
    adb_reference_path = output / "adb_game_color_reference.png"
    adb_reference_mask_path = output / "adb_game_color_reference_mask.png"
    reference_index = int(len(android_frames) // 2)
    masked_reference = cv2.bitwise_and(
        android_frames[reference_index],
        android_frames[reference_index],
        mask=adb_mask,
    )
    if not cv2.imwrite(str(adb_reference_path), masked_reference):
        raise RuntimeError("Cannot write portable ADB game-color reference")
    if not cv2.imwrite(str(adb_reference_mask_path), adb_mask):
        raise RuntimeError("Cannot write portable ADB game-color reference mask")
    adb_color_reference = _adb_color_statistics(android_frames, adb_mask)
    summary = {
        "schema_version": "1.0",
        "status": "calibrated_pending_publication",
        "calibration_kind": "hik_game_color",
        "session": str(session),
        "profile_context": context.as_dict(),
        "rig_revision": active_rig["revision_id"],
        "hik_bayer_conversion": conversion,
        "adb_game_color_reference": adb_color_reference,
        "sampling_geometry": sampling_geometry,
        "spatial_alignment": spatial_alignment,
        "synchronized_source_frames": {
            "android_phone": [int(row["frame_index"]) for row in selected_android],
            "hik_phone": [int(row["frame_index"]) for row in selected_hik],
            "host_delta_ms": [float(row[2]) for row in selected],
            "coordinate_conversion": "coordinate_spaces.yaml#conversions.adb_to_hik_phone_video_3x3",
        },
        "evidence": list(evidence) + list(spatial_alignment["evidence_files"]),
    }
    summary_path = output / "game_color_calibration.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    phone_color_profile = registry.publish(
        "phone_game_color",
        context,
        {
            "profile_kind": "phone_game_color",
            "coordinate_space": "android_logical_display_pixels",
            "adb_game_color_reference": adb_color_reference,
            "sampling_geometry": sampling_geometry,
            "capabilities": {
                "camera_independent": True,
                "portable": True,
                "local_hik_fit_required": True,
            },
        },
        runtime_files={
            "adb_game_color_reference": adb_reference_path,
            "adb_game_color_reference_mask": adb_reference_mask_path,
        },
        provenance={
            "session": str(session),
            "selected_android_frame_index": int(
                selected_android[reference_index]["frame_index"]
            ),
        },
        review_state="accepted" if activate else "review_required",
        activate=activate,
    )
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
            "spatial_alignment": {
                "status": spatial_alignment["status"],
                "aggregate": spatial_alignment["aggregate"],
                "method": spatial_alignment["method"],
            },
        },
        dependencies={
            "rig": active_rig["revision_id"],
            "phone_game_color": phone_color_profile["revision_id"],
        },
        provenance={
            "game_color_calibration": str(summary_path),
            "session": str(session),
            "evidence": list(evidence)
            + list(spatial_alignment["evidence_files"]),
        },
        review_state="accepted" if activate else "review_required",
        activate=activate,
    )
    summary["profile_revision"] = profile["revision_id"]
    summary["portable_phone_game_color_revision"] = phone_color_profile[
        "revision_id"
    ]
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
    value.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="diagnostic evidence override; default is under IRIS_PROFILE_ROOT",
    )
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
    from aria_trace.adapters.filesystem.system_configuration import (
        load_system_configuration,
    )

    configuration = load_system_configuration(arguments.profile_root)
    if arguments.game_id is None:
        arguments.game_id = configuration["game"].get("game_id")
    if arguments.output is None:
        registry = ProfileRegistry(arguments.profile_root)
        game_label = re.sub(
            r"[^A-Za-z0-9_.-]+", "-", str(arguments.game_id or "game")
        ).strip("-.") or "game"
        arguments.output = (
            registry.root
            / "calibrations"
            / "game-color"
            / "{}-{}".format(
                game_label, datetime.now().strftime("%Y%m%d-%H%M%S")
            )
        )
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
