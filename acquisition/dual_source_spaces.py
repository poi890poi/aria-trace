"""Traceable coordinate-space manifest for synchronized ADB/HIK sessions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .commented_yaml import write_commented_yaml
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
    "spaces": "Named image rasters. Size is [width, height] in pixels.",
    "conversions": "Forward and inverse matrices; no image matching or fitted session transform is used.",
    "streams": "Video filenames, timing authority, content sizes, and storage-only padding.",
    "usage": "Short operational rules for downstream developers.",
}


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
    crop_x, crop_y, content_width, content_height = (
        converter.camera_adapter_bounds_in_adb_xywh()
    )
    hik_video_to_adb = np.asarray(
        [[1.0, 0.0, crop_x], [0.0, 1.0, crop_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
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
    rig_config = converter.calibration
    return {
        "schema_version": "1.0",
        "scope": "one synchronized dual-source capture session",
        "rig": {
            "calibration": str(Path(rig_calibration).resolve()),
            "camera_id": str(rig_config["camera"]["device_id"]),
            "phone_serial": str(rig_config["phone"]["serial"]),
        },
        "spaces": {
            **description["spaces"],
            "hik_phone_video": {
                "id": "hik_session_aligned_visible_phone_pixels",
                "content_size_px": [content_width, content_height],
                "stored_size_px": hik_stored_size,
                "valid_content_xywh": [0, 0, content_width, content_height],
                "relationship_to_camera_adapter": (
                    "camera adapter image rotated by the declared quarter turns"
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
                converter.output_image_quarter_turns_clockwise_from_phone_natural
            ),
            "hik_phone_video_bounds_in_adb_xywh": [
                crop_x,
                crop_y,
                content_width,
                content_height,
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
            "quality_gate": "Space conversion is calibration-defined; cross-source confidence is diagnostic only.",
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
    return write_commented_yaml(
        Path(session_path) / "coordinate_spaces.yaml",
        document,
        header=DUAL_SOURCE_SPACES_HEADER,
        section_comments=DUAL_SOURCE_SPACES_COMMENTS,
    )
