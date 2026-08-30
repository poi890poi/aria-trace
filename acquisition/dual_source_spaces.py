"""Traceable coordinate-space manifest for synchronized ADB/HIK sessions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Mapping

import numpy as np

from .commented_yaml import write_commented_yaml
from aria_trace.evidence.media_trace import image_size_px, raster_record, validate_media_registry
from .rig_calibration.hik.spaces import RigCalibratedSpaceConverter


DUAL_SOURCE_SPACES_HEADER = """# AriaTrace dual-source coordinate spaces.
#
# This file is the human-readable authority for converting coordinates among
# the saved ADB video, the saved rig-normalized HIK video, and the reusable
# camera adapter output. Coordinates are pixel-center XY with top-left [0, 0],
# +X right, and +Y down. Apply 3x3 matrices to column [x, y, 1]. Encoder padding
# is storage only and is outside the valid image coordinate space."""

DUAL_SOURCE_SPACES_COMMENTS = {
    "rig": "Immutable calibration and device identities that own this mapping.",
    "orientation_selection": "Session quarter-turn selected from the first game ADB/HIK image pair; Android's report is retained only for comparison.",
    "spaces": "Named image rasters. Size is [width, height] in pixels.",
    "conversions": "Forward and inverse matrices; no image matching or fitted session transform is used.",
    "streams": "Video filenames, timing authority, content sizes, and storage-only padding.",
    "media": "Every persisted image/video in this session, with its raster space, crop/ROI, orientation, and provenance.",
    "usage": "Short operational rules for downstream developers.",
}

ANDROID_SOURCE_SPACES_HEADER = """# AriaTrace Android-only coordinate space.
#
# This session intentionally contains no HIK stream. Coordinates are Android
# logical-display pixel centers with top-left [0, 0], +X right, and +Y down."""

ANDROID_SOURCE_SPACES_COMMENTS = {
    "spaces": "The single saved Android raster and its orientation.",
    "streams": "Video filename, size, and timestamp authority.",
    "conversions": "No cross-source conversion exists in an Android-only session.",
    "usage": "Rules for the one-time phone-game mini-map calibration input.",
}


def build_dual_source_media_registry(
    session_path: Path, document: Mapping[str, object]
) -> list[dict]:
    """Describe every video and evidence image in a dual-source session."""

    session_path = Path(session_path)
    spaces = document["spaces"]
    streams = document["streams"]
    conversions = document["conversions"]
    orientation = document.get("orientation_selection") or {}
    rig_reference = Path(str(document["rig"]["calibration"]))
    rig_config = json.loads(rig_reference.read_text(encoding="utf-8"))
    hardware_roi = list(map(int, rig_config["camera"]["hardware_roi_xywh"]))
    records = []
    android = streams["android_phone"]
    android_file = android.get("video")
    if android_file:
        android_size = list(map(int, android["stored_size_px"]))
        records.append(
            raster_record(
                android_file,
                media_type="video",
                stored_size_px=android_size,
                space_id="android_logical_display_pixels",
                operation="scrcpy_decode_and_encode_without_spatial_crop",
                source_space_id="android_logical_display_pixels",
                source_region={"kind": "full_frame", "xywh": [0, 0] + android_size},
                orientation={
                    "quarter_turns_clockwise_from_phone_natural": (
                        orientation.get(
                            "effective_quarter_turns_clockwise_from_phone_natural"
                        )
                    ),
                    "selection_source": orientation.get("source"),
                },
                metadata_reference="manifest.json#frame_sources",
                timing={"authority": android["timestamp_authority"]},
            )
        )
    hik = streams["hik_phone"]
    hik_file = hik.get("video")
    if hik_file:
        stored = list(map(int, hik["stored_size_px"]))
        content = list(map(int, hik["content_size_px"]))
        records.append(
            raster_record(
                hik_file,
                media_type="video",
                stored_size_px=stored,
                content_size_px=content,
                space_id="hik_session_aligned_visible_phone_pixels",
                operation="hardware_roi_then_rig_rectification_then_quarter_turn",
                source_space_id="hik_full_sensor_camera_pixels",
                source_region={
                    "kind": "hardware_roi",
                    "xywh": hardware_roi,
                    "xywh_reference": "rig calibration camera.hardware_roi_xywh",
                },
                orientation={
                    "quarter_turns_clockwise_from_calibration_display": (
                        conversions[
                            "camera_adapter_image_quarter_turns_clockwise_to_hik_phone_video"
                        ]
                    ),
                    "selection_source": orientation.get("source"),
                },
                transform={
                    "rig_rectification": "rig calibration normalization",
                    "to_adb_3x3_reference": (
                        "coordinate_spaces.yaml#conversions.hik_phone_video_to_adb_3x3"
                    ),
                },
                metadata_reference="manifest.json#frame_sources",
                timing={"authority": hik["timestamp_authority"]},
                notes="stored_size may include right/bottom encoder padding outside content_size",
            )
        )

    main_summary_path = session_path / "cross_source_check" / "summary.json"
    main_summary = (
        json.loads(main_summary_path.read_text(encoding="utf-8"))
        if main_summary_path.is_file()
        else {}
    )
    comparison_crop = main_summary.get("logical_adb_crop_xywh") or conversions.get(
        "hik_phone_video_bounds_in_adb_xywh"
    )
    for path in sorted(session_path.rglob("*.png")):
        relative = str(path.relative_to(session_path)).replace("\\", "/")
        name = path.name
        size = image_size_px(path)
        metadata_reference = "cross_source_check/summary.json"
        source_space = "cross_source_comparison_inputs"
        source_region = {
            "kind": "comparison",
            "adb_crop_xywh": comparison_crop,
        }
        space_id = "cross_source_comparison_local_pixels"
        operation = "cross_source_diagnostic_visualization"
        image_orientation = {
            "matches": "hik_session_aligned_visible_phone_pixels"
        }

        if "/orientation_match/" in "/" + relative:
            metadata_reference = "cross_source_check/orientation_match/summary.json"
            if name == "first_adb_game_image.png":
                space_id = "android_logical_display_pixels"
                source_space = space_id
                source_region = {"kind": "full_frame", "xywh": [0, 0] + size}
                operation = "first_game_adb_frame_copy"
                image_orientation = {
                    "as_captured": True,
                    "quarter_turn_selected_by_this_evidence": True,
                }
            elif name == "first_hik_rig_normalized_calibration_display.png":
                space_id = "hik_rig_rectified_visible_phone_pixels"
                source_space = "hik_camera_adapter_roi_pixels"
                source_region = {
                    "kind": "hardware_roi",
                    "xywh": hardware_roi,
                    "xywh_reference": "rig calibration camera.hardware_roi_xywh",
                }
                operation = "one_time_rig_rectification_for_orientation_evidence"
                image_orientation = {"rig_calibration_display": True}
            else:
                match = re.match(
                    r"candidate_surface_(\d)_adapter_([0-9]+)deg_(.+)", name
                )
                surface_turns = int(match.group(1)) if match else None
                image_degrees = int(match.group(2)) if match else None
                source_region = {
                    "kind": "orientation_candidate",
                    "candidate_adb_surface_quarter_turns_clockwise_from_natural": surface_turns,
                    "candidate_adapter_image_degrees_clockwise_from_calibration_display": image_degrees,
                    "candidate_metadata_reference": metadata_reference,
                }
                image_orientation = {
                    "candidate_adapter_image_degrees_clockwise_from_calibration_display": image_degrees
                }
                if name.endswith("_adb_crop.png"):
                    source_space = "android_logical_display_pixels"
                    operation = "candidate_adb_crop"
                elif name.endswith("_hik.png"):
                    source_space = "hik_rig_rectified_visible_phone_pixels"
                    operation = "candidate_hik_quarter_turn"
        elif name == "adb_visible_crop.png":
            source_space = "android_logical_display_pixels"
            source_region = {"kind": "crop", "xywh": comparison_crop}
            operation = "crop_without_resampling"
        elif name == "hik_rectified.png":
            source_space = "hik_session_aligned_visible_phone_pixels"
            source_region = {"kind": "full_frame", "xywh": [0, 0] + size}
            operation = "frame_copy"
        elif name == "valid_mask.png":
            source_space = "hik_rig_rectified_visible_phone_pixels"
            operation = "rotate_saved_rig_validity_mask"
        elif name.startswith("side_by_side"):
            space_id = "diagnostic_composite_pixels"
            operation = "horizontal_composite_adb_then_hik"

        records.append(
            raster_record(
                relative,
                media_type="image",
                stored_size_px=size,
                space_id=space_id,
                operation=operation,
                source_space_id=source_space,
                source_region=source_region,
                orientation=image_orientation,
                metadata_reference=metadata_reference,
            )
        )
    validate_media_registry(session_path, records)
    return records


def build_dual_source_space_document(
    rig_calibration: Path,
    phone_surface_orientation: Mapping[str, object],
    manifest: Mapping[str, object],
) -> dict:
    turns = int(
        phone_surface_orientation.get(
            "quarter_turns_clockwise_from_natural", 0
        )
    )
    converter = RigCalibratedSpaceConverter(rig_calibration, turns)
    description = converter.describe()
    crop_x, crop_y, crop_width, crop_height = converter.camera_adapter_bounds_in_adb_xywh()
    content_width, content_height = converter.output_image_size_px
    hik_video_to_adb = (
        converter.adapter_to_adb_3x3 @ converter.output_image_to_adapter_3x3
    )
    adb_to_hik_video = np.linalg.inv(hik_video_to_adb)
    first_frames = {}
    for frame in manifest.get("first_frames", []):
        first_frames[str(frame["stream_id"])] = frame
    android_frame = first_frames.get("android_phone", {})
    hik_frame = first_frames.get("hik_phone", {})
    hik_stored_size = [
        int(hik_frame.get("width", content_width)),
        int(hik_frame.get("height", content_height)),
    ]
    hik_padding = [
        max(0, hik_stored_size[0] - content_width),
        max(0, hik_stored_size[1] - content_height),
    ]
    if hik_stored_size[0] < content_width or hik_stored_size[1] < content_height:
        raise RuntimeError(
            "Stored HIK raster {}x{} is smaller than declared output space {}x{}"
            .format(*hik_stored_size, content_width, content_height)
        )
    corners = np.asarray(
        [[0, 0, 1], [content_width - 1, 0, 1],
         [content_width - 1, content_height - 1, 1], [0, content_height - 1, 1]],
        dtype=np.float64,
    ).T
    mapped = hik_video_to_adb @ corners
    mapped = (mapped[:2] / mapped[2:3]).T
    mapped_min = np.rint(np.min(mapped, axis=0)).astype(int)
    mapped_max = np.rint(np.max(mapped, axis=0)).astype(int)
    mapped_bounds = [
        int(mapped_min[0]), int(mapped_min[1]),
        int(mapped_max[0] - mapped_min[0] + 1),
        int(mapped_max[1] - mapped_min[1] + 1),
    ]
    expected_bounds = [crop_x, crop_y, crop_width, crop_height]
    if mapped_bounds != expected_bounds:
        raise RuntimeError(
            "HIK output-space matrix maps to {}, but rig geometry declares {}"
            .format(mapped_bounds, expected_bounds)
        )
    rig_config = converter.calibration
    return {
        "schema_version": "1.0",
        "scope": "one synchronized dual-source capture session",
        "rig": {
            "calibration": str(Path(rig_calibration).resolve()),
            "camera_id": str(rig_config["camera"]["device_id"]),
            "phone_serial": str(rig_config["phone"]["serial"]),
        },
        "orientation_selection": {
            "effective_quarter_turns_clockwise_from_phone_natural": turns,
            "effective_degrees_clockwise_from_phone_natural": turns * 90,
            "rig_calibration_display_quarter_turns_clockwise_from_natural": (
                converter.calibration_display_quarter_turns_clockwise_from_natural
            ),
            "camera_adapter_image_quarter_turns_clockwise_from_calibration_display": (
                converter.output_image_quarter_turns_clockwise_from_calibration_display
            ),
            "source": phone_surface_orientation.get("source"),
            "android_reported_quarter_turns_clockwise_from_natural": (
                phone_surface_orientation.get(
                    "android_reported_quarter_turns_clockwise_from_natural"
                )
            ),
            "image_evidence": phone_surface_orientation.get(
                "orientation_evidence"
            ),
        },
        "spaces": {
            **description["spaces"],
            "hik_phone_video": {
                "id": "hik_session_aligned_visible_phone_pixels",
                "content_size_px": [content_width, content_height],
                "stored_size_px": hik_stored_size,
                "valid_content_xywh": [0, 0, content_width, content_height],
                "relationship_to_camera_adapter": (
                    "camera adapter image rotated from rig-calibration display space by the declared quarter turns"
                ),
            },
        },
        "conversions": {
            "camera_adapter_to_adb_3x3": description["conversion"][
                "camera_adapter_to_adb_3x3"
            ],
            "adb_to_camera_adapter_3x3": description["conversion"][
                "adb_to_camera_adapter_3x3"
            ],
            "hik_phone_video_to_adb_3x3": hik_video_to_adb.tolist(),
            "adb_to_hik_phone_video_3x3": adb_to_hik_video.tolist(),
            "camera_adapter_image_quarter_turns_clockwise_to_hik_phone_video": (
                converter.output_image_quarter_turns_clockwise_from_calibration_display
            ),
            "hik_phone_video_bounds_in_adb_xywh": [
                crop_x,
                crop_y,
                crop_width,
                crop_height,
            ],
        },
        "streams": {
            "android_phone": {
                "video": (manifest.get("videos") or {}).get("android_phone"),
                "space": "adb",
                "stored_size_px": [
                    int(android_frame.get("width", converter.adb_size_px[0])),
                    int(android_frame.get("height", converter.adb_size_px[1])),
                ],
                "timestamp_authority": (
                    "frames.jsonl host_capture_time_ns mapped from Android CLOCK_MONOTONIC"
                ),
            },
            "hik_phone": {
                "video": (manifest.get("videos") or {}).get("hik_phone"),
                "space": "hik_phone_video",
                "stored_size_px": hik_stored_size,
                "content_size_px": [content_width, content_height],
                "encoder_padding_right_bottom_px": hik_padding,
                "timestamp_authority": (
                    "frames.jsonl host_capture_time_ns at HIK frame receive; raw device counter is metadata"
                ),
            },
        },
        "usage": {
            "camera_adapter_api": (
                "Use camera_adapter_to_adb_points or adb_to_camera_adapter_points "
                "with the same Android surface quarter-turn value."
            ),
            "saved_video_api": (
                "Use hik_phone_video_to_adb_3x3 for pixels decoded from streams.hik_phone.video."
            ),
            "visibility": (
                "ADB points outside hik_phone_video_bounds_in_adb_xywh are not visible to HIK."
            ),
            "quality_gate": (
                "Space conversion is rig-calibration-defined. The session "
                "quarter-turn is selected from image evidence; its confidence "
                "and every scored candidate are retained at image_evidence."
            ),
        },
    }


def write_dual_source_space_yaml(
    session_path: Path,
    rig_calibration: Path,
    phone_surface_orientation: Mapping[str, object],
    manifest: Mapping[str, object],
) -> Path:
    enriched_manifest = dict(manifest)
    first_frames = {}
    frames_file = Path(session_path) / "frames.jsonl"
    if frames_file.is_file():
        with frames_file.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                frame = json.loads(line)
                first_frames.setdefault(str(frame["stream_id"]), frame)
                if len(first_frames) >= 2:
                    break
    enriched_manifest["first_frames"] = list(first_frames.values())
    document = build_dual_source_space_document(
        rig_calibration, phone_surface_orientation, enriched_manifest
    )
    document["media"] = build_dual_source_media_registry(
        Path(session_path), document
    )
    return write_commented_yaml(
        Path(session_path) / "coordinate_spaces.yaml",
        document,
        header=DUAL_SOURCE_SPACES_HEADER,
        section_comments=DUAL_SOURCE_SPACES_COMMENTS,
    )


def write_android_source_space_yaml(
    session_path: Path,
    phone_surface_orientation: Mapping[str, object],
    manifest: Mapping[str, object],
) -> Path:
    """Write the explicit one-space authority for an ADB-only zigzag."""

    session_path = Path(session_path)
    first_android = None
    frames_file = session_path / "frames.jsonl"
    if frames_file.is_file():
        with frames_file.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                frame = json.loads(line)
                if str(frame.get("stream_id")) == "android_phone":
                    first_android = frame
                    break
    if first_android is None:
        raise RuntimeError("ADB-only session has no Android frame metadata")
    size = [int(first_android["width"]), int(first_android["height"])]
    video = (manifest.get("videos") or {}).get("android_phone")
    document = {
        "schema_version": "1.0",
        "capture_mode": "android_only",
        "spaces": {
            "android_logical_display_pixels": {
                "size_px": size,
                "origin": "top_left_pixel_center",
                "x_axis": "right",
                "y_axis": "down",
                "quarter_turns_clockwise_from_phone_natural": int(
                    phone_surface_orientation.get(
                        "quarter_turns_clockwise_from_natural", 0
                    )
                ),
            }
        },
        "streams": {
            "android_phone": {
                "video": video,
                "space": "android_logical_display_pixels",
                "stored_size_px": size,
                "timestamp_authority": (
                    "frames.jsonl host_capture_time_ns mapped from Android CLOCK_MONOTONIC"
                ),
            }
        },
        "conversions": {
            "cross_source": None,
            "reason": "No HIK stream was acquired for this session",
        },
        "usage": {
            "mini_map_calibration": (
                "Use android_phone directly for one-time phone-game mini-map discovery."
            ),
            "hik_projection": "Unavailable; acquire a synchronized HIK stream later if needed.",
        },
    }
    return write_commented_yaml(
        session_path / "coordinate_spaces.yaml",
        document,
        header=ANDROID_SOURCE_SPACES_HEADER,
        section_comments=ANDROID_SOURCE_SPACES_COMMENTS,
    )
