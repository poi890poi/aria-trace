"""Media provenance registries for HIK rig-calibration artifacts."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Mapping, Optional, Sequence

from rig_runtime.evidence.media_trace import (
    image_size_px,
    raster_record,
    validate_media_registry,
)
from rig_runtime.evidence.rig_spatial import rig_sample_media_record
from rig_runtime.services.calibration.rig.contracts import FrameSample
from rig_runtime.domain.spaces import RigSpaceId


def build_hik_calibration_media_registry(
    root: Path,
    config: Mapping[str, object],
    *,
    last_camera_sample: Optional[FrameSample] = None,
    additional_records: Sequence[Mapping[str, object]] = (),
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
        if last_camera_sample is None:
            raise RuntimeError(
                "Completed HIK bundle has a camera frame without producer metadata"
            )
        if image_size_px(last_frame_path) != [
            int(last_camera_sample.image.shape[1]),
            int(last_camera_sample.image.shape[0]),
        ]:
            raise RuntimeError("Persisted last camera frame differs from its sample")
        records.append(
            rig_sample_media_record(
                "last_camera_frame.png",
                last_camera_sample,
                operation="last_settled_hik_acquisition_frame_copy",
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
                space_id=RigSpaceId.HIK_RIG_RECTIFIED_VISIBLE_PHONE,
                operation="validity_mask_from_camera_to_phone_rectification",
                source_space_id=RigSpaceId.HIK_FULL_SENSOR_CAMERA,
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

    evidence_root = root / "cross_source_check"
    if evidence_root.is_dir():
        result_path = evidence_root / "cross_source_check.json"
        cross_document = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.is_file()
            else {}
        )
        cross_records = list(cross_document.get("media") or [])
        persisted_cross_media = list(evidence_root.glob("*.png"))
        if persisted_cross_media and not cross_records:
            raise RuntimeError(
                "Cross-source rig evidence was persisted without producer records"
            )
        for record in cross_records:
            value = dict(record)
            value["file"] = "cross_source_check/{}".format(value["file"])
            records.append(value)
    records.extend(dict(value) for value in additional_records)
    validate_media_registry(root, records)
    return records


def build_hik_failure_media_registry(
    root: Path,
    *,
    camera_samples: Mapping[str, FrameSample],
    phone_media: Mapping[str, Mapping[str, object]],
    additional_records: Sequence[Mapping[str, object]] = (),
) -> list[dict]:
    """Describe every persisted review image in a failed HIK run."""

    root = Path(root)
    records = []
    for path in sorted(root.glob("*.png")):
        size = image_size_px(path)
        name = path.name
        if name in camera_samples:
            record = rig_sample_media_record(
                name,
                camera_samples[name],
                operation="failure_evidence_camera_frame_copy",
                metadata_reference="failure.json#camera",
            )
        elif name in phone_media:
            supplied = dict(phone_media[name])
            record = raster_record(
                name,
                media_type="image",
                stored_size_px=size,
                space_id=str(supplied["space_id"]),
                operation=str(supplied["operation"]),
                source_space_id=str(supplied["source_space_id"]),
                source_region=dict(supplied["source_region"]),
                orientation=dict(supplied["orientation"]),
                metadata_reference="failure.json#phone",
            )
        else:
            matches = [row for row in additional_records if row.get("file") == name]
            if len(matches) != 1:
                raise RuntimeError(
                    "Rig failure image {} lacks exactly one producer record"
                    .format(name)
                )
            record = dict(matches[0])
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
        source_space = RigSpaceId.HIK_FULL_SENSOR_CAMERA
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
            space_id = RigSpaceId.HIK_CAMERA_CROP_LOCAL
            operation = "crop_without_resampling"
        elif role == "rectified_decoder_crop":
            source_space = RigSpaceId.ANDROID_LOGICAL_DISPLAY
            source_region = {
                "kind": "crop",
                "xywh": failure.get("decode_rect_screen_xywh"),
            }
            space_id = RigSpaceId.DATA_MATRIX_DECODER_CROP_LOCAL
            operation = "screen_rectification_then_crop"
            orientation = {"as_displayed_by_android": True}
            transform = {
                "matrix_reference": (
                    "index.json#space_conversion.camera_to_phone_3x3"
                )
            }
        elif role == "rectified_camera_frame":
            source_region = {"kind": "full_frame", "xywh": [0, 0] + full_size}
            space_id = RigSpaceId.ANDROID_LOGICAL_DISPLAY
            operation = "camera_to_phone_perspective_rectification"
            orientation = {"as_displayed_by_android": True}
            transform = {
                "matrix_reference": (
                    "index.json#space_conversion.camera_to_phone_3x3"
                )
            }
        elif role == "display_target":
            source_space = RigSpaceId.ANDROID_LOGICAL_DISPLAY
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
