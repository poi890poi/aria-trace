"""Media provenance registries for HIK rig-calibration artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

from aria_trace.evidence.media_trace import (
    image_size_px,
    raster_record,
    validate_media_registry,
)


def _camera_source_region(
    size_px: Sequence[int],
    full_size_px: Sequence[int],
    hardware_roi_xywh: Optional[Sequence[int]],
) -> tuple[str, dict]:
    size = list(map(int, size_px))
    full = list(map(int, full_size_px))
    roi = list(map(int, hardware_roi_xywh or [0, 0] + full))
    if size == full:
        return "hik_full_sensor_camera_pixels", {
            "kind": "full_frame",
            "xywh": [0, 0] + full,
        }
    if size == roi[2:4]:
        return "hik_full_sensor_camera_pixels", {
            "kind": "hardware_roi",
            "xywh": roi,
            "local_origin_in_source_xy": roi[:2],
        }
    return "hik_current_camera_acquisition_pixels", {
        "kind": "explicit_unresolved_acquisition_raster",
        "size_px": size,
        "warning": "No saved full-sensor/ROI dimensions match this frame",
    }


def build_hik_calibration_media_registry(
    root: Path, config: Mapping[str, object]
) -> list[dict]:
    """Describe every persisted image in a completed HIK calibration bundle."""

    root = Path(root)
    camera = config["camera"]
    full_mode = camera["full_sensor_mode"]
    full_size = [int(full_mode["width_px"]), int(full_mode["height_px"])]
    hardware_roi = list(map(int, camera["hardware_roi_xywh"]))
    normalization = config["normalization"]
    output_size = list(map(int, normalization["output_size_px"]))
    records = []

    last_frame_path = root / "last_camera_frame.png"
    if last_frame_path.is_file():
        size = image_size_px(last_frame_path)
        source_space, source_region = _camera_source_region(
            size, full_size, hardware_roi
        )
        records.append(
            raster_record(
                "last_camera_frame.png",
                media_type="image",
                stored_size_px=size,
                space_id=(
                    "hik_camera_adapter_roi_image_pixels"
                    if source_region["kind"] == "hardware_roi"
                    else source_space
                ),
                operation="last_settled_hik_acquisition_frame_copy",
                source_space_id=source_space,
                source_region=source_region,
                orientation={"camera_native": True},
                metadata_reference="hik_camera_calibration.yaml#camera",
                notes="This is not the rectified visible-phone output.",
            )
        )

    mask_path = root / "valid_screen_mask.png"
    if mask_path.is_file():
        records.append(
            raster_record(
                "valid_screen_mask.png",
                media_type="image",
                stored_size_px=image_size_px(mask_path),
                space_id="hik_rig_rectified_visible_phone_pixels",
                operation="validity_mask_from_camera_to_phone_rectification",
                source_space_id="hik_full_sensor_camera_pixels",
                source_region={"kind": "full_frame", "xywh": [0, 0] + full_size},
                orientation={"phone_natural": True},
                transform={
                    "matrix_reference": (
                        "hik_camera_calibration.yaml#normalization."
                        "full_sensor_camera_to_output_3x3"
                    )
                },
                metadata_reference="hik_camera_calibration.yaml#normalization",
            )
        )

    cross = (config.get("results") or {}).get("cross_source_check") or {}
    adb_crop = cross.get("camera_visible_screen_region_xywh")
    camera_roi = cross.get("camera_hardware_roi_xywh") or hardware_roi
    evidence_root = root / "cross_source_check"
    for path in sorted(evidence_root.glob("*.png")) if evidence_root.is_dir() else []:
        relative = "cross_source_check/{}".format(path.name)
        size = image_size_px(path)
        space_id = "rig_cross_source_comparison_local_pixels"
        source_space = "rig_cross_source_comparison_inputs"
        source_region = {
            "kind": "comparison",
            "adb_crop_xywh": adb_crop,
            "camera_hardware_roi_xywh": camera_roi,
        }
        operation = "cross_source_diagnostic_visualization"
        orientation = {"phone_natural": True}
        if path.name == "adb_visible_crop.png":
            source_space = "android_phone_natural_display_pixels"
            source_region = {"kind": "crop", "xywh": adb_crop}
            operation = "crop_without_resampling"
        elif path.name == "hik_rectified.png":
            source_space = "hik_full_sensor_camera_pixels"
            source_region = {"kind": "hardware_roi", "xywh": camera_roi}
            operation = "rig_rectification"
        elif path.name == "valid_mask.png":
            source_space = "hik_rig_rectified_visible_phone_pixels"
            source_region = {"kind": "full_frame", "xywh": [0, 0] + output_size}
            operation = "validity_mask_copy"
        elif path.name.startswith("side_by_side"):
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
                orientation=orientation,
                metadata_reference="cross_source_check/cross_source_check.yaml",
            )
        )
    validate_media_registry(root, records)
    return records


def build_hik_failure_media_registry(
    root: Path,
    *,
    full_camera_size_px: Sequence[int],
    hardware_roi_xywh: Optional[Sequence[int]],
    phone_logical_size_px: Optional[Sequence[int]],
) -> list[dict]:
    """Describe every persisted review image in a failed HIK run."""

    root = Path(root)
    phone_size = list(map(int, phone_logical_size_px or [1, 1]))
    records = []
    for path in sorted(root.glob("*.png")):
        size = image_size_px(path)
        name = path.name
        if "hik-frame" in name:
            source_space, source_region = _camera_source_region(
                size, full_camera_size_px, hardware_roi_xywh
            )
            record = raster_record(
                name,
                media_type="image",
                stored_size_px=size,
                space_id=source_space,
                operation="failure_evidence_camera_frame_copy",
                source_space_id=source_space,
                source_region=source_region,
                orientation={"camera_native": True},
                metadata_reference="failure.json#camera",
            )
        else:
            is_screenshot = "screenshot" in name
            record = raster_record(
                name,
                media_type="image",
                stored_size_px=size,
                space_id="android_logical_display_pixels",
                operation=(
                    "adb_screenshot_copy"
                    if is_screenshot
                    else "generated_phone_display_target"
                ),
                source_space_id="android_logical_display_pixels",
                source_region={
                    "kind": "full_frame",
                    "xywh": [0, 0, size[0], size[1]],
                    "expected_logical_size_px": phone_size,
                },
                orientation={"as_displayed_by_android": True},
                metadata_reference="failure.json#phone",
            )
        records.append(record)
    validate_media_registry(root, records)
    return records


def build_data_matrix_media_registry(
    root: Path,
    failures: Sequence[Mapping[str, object]],
    *,
    full_camera_size_px: Sequence[int],
    phone_logical_size_px: Sequence[int],
) -> list[dict]:
    """Describe every Data Matrix target, camera frame, and decoder crop."""

    root = Path(root)
    full_size = list(map(int, full_camera_size_px))
    phone_size = list(map(int, phone_logical_size_px))
    file_metadata = {}
    for index, failure in enumerate(failures):
        for role, filename in (failure.get("files") or {}).items():
            if filename:
                file_metadata[str(filename)] = (role, failure, index)
    records = []
    for path in sorted(root.glob("*.png")):
        size = image_size_px(path)
        role, failure, index = file_metadata.get(
            path.name, ("diagnostic", {}, None)
        )
        source_space = "hik_full_sensor_camera_pixels"
        source_region = {"kind": "full_frame", "xywh": [0, 0] + full_size}
        space_id = source_space
        operation = "camera_frame_copy"
        orientation = {"camera_native": True}
        transform = None
        if role == "raw_camera_crop":
            source_region = {
                "kind": "crop",
                "xywh": failure.get("raw_camera_crop_xywh"),
            }
            space_id = "hik_camera_crop_local_pixels"
            operation = "crop_without_resampling"
        elif role == "rectified_decoder_crop":
            source_space = "android_logical_display_pixels"
            source_region = {
                "kind": "crop",
                "xywh": failure.get("decode_rect_screen_xywh"),
            }
            space_id = "data_matrix_decoder_crop_local_pixels"
            operation = "screen_rectification_then_crop"
            orientation = {"as_displayed_by_android": True}
            transform = {
                "matrix_reference": (
                    "index.json#space_conversion.camera_to_phone_3x3"
                )
            }
        elif role == "rectified_camera_frame":
            source_region = {"kind": "full_frame", "xywh": [0, 0] + full_size}
            space_id = "android_logical_display_pixels"
            operation = "camera_to_phone_perspective_rectification"
            orientation = {"as_displayed_by_android": True}
            transform = {
                "matrix_reference": (
                    "index.json#space_conversion.camera_to_phone_3x3"
                )
            }
        elif role == "display_target":
            source_space = "android_logical_display_pixels"
            source_region = {"kind": "full_frame", "xywh": [0, 0] + phone_size}
            space_id = source_space
            operation = "generated_data_matrix_display_target"
            orientation = {"as_displayed_by_android": True}
        records.append(
            raster_record(
                path.name,
                media_type="image",
                stored_size_px=size,
                space_id=space_id,
                operation=operation,
                source_space_id=source_space,
                source_region=source_region,
                orientation=orientation,
                transform=transform,
                metadata_reference=(
                    "index.json#failures[{}]".format(index)
                    if index is not None
                    else "index.json"
                ),
            )
        )
    validate_media_registry(root, records)
    return records
