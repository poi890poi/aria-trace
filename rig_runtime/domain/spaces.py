"""Coordinate-space references shared by pixels and geometric results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple


class RigSpaceId:
    """Canonical identifiers for distinct IRIS rig raster spaces.

    Similar names are intentionally not aliases: raw calibration rasters,
    runtime ROI rasters, rectified phone rasters, and game-upright rasters are
    different spaces and must remain distinguishable.
    """

    ANDROID_PHONE_NATURAL = "android_phone_natural_display_pixels"
    ANDROID_LOGICAL_DISPLAY = "android_logical_display_pixels"
    ANDROID_CALIBRATION_LOGICAL_DISPLAY = (
        "android_calibration_logical_display_pixels"
    )
    ANDROID_GAME_CAPTURE = "android_game_capture_pixels"
    ANDROID_VISIBLE_PHONE_CROP = "android_visible_phone_crop_pixels"
    PHONE_NATURAL_DISPLAY = "phone_natural_display_pixels"
    HIK_FULL_SENSOR_CAMERA = "hik_full_sensor_camera_pixels"
    HIK_FULL_SENSOR_BGR = "hik_full_sensor_bgr_pixels"
    HIK_FULL_SENSOR_UNDISTORTED = "hik_full_sensor_undistorted_pixels"
    NATIVE_HIK_SENSOR_BGR = "native_hik_sensor_bgr_pixels"
    HIK_CAMERA_ACQUISITION = "hik_camera_acquisition_pixels"
    HIK_CAMERA_ADAPTER_HARDWARE_ROI_BGR = (
        "hik_camera_adapter_hardware_roi_bgr_pixels"
    )
    HIK_CAMERA_ADAPTER_ROI_IMAGE = "hik_camera_adapter_roi_image_pixels"
    HIK_RIG_RECTIFIED_VISIBLE_PHONE = "hik_rig_rectified_visible_phone_pixels"
    HIK_GAME_UPRIGHT_RECTIFIED_VISIBLE_PHONE = (
        "hik_game_upright_rectified_visible_phone_pixels"
    )
    HIK_GAME_UPRIGHT_CAMERA_ADAPTER_ROI = (
        "hik_game_upright_camera_adapter_roi_pixels"
    )
    HIK_NORMALIZED_MINIMAP = "hik_phone_game_normalized_minimap_pixels"
    HIK_SESSION_ALIGNED_VISIBLE_PHONE = (
        "hik_session_aligned_visible_phone_pixels"
    )
    HIK_SESSION_ROTATED_CAMERA_ADAPTER_ROI = (
        "hik_session_rotated_camera_adapter_roi_pixels"
    )
    HIK_CAMERA_CROP_LOCAL = "hik_camera_crop_local_pixels"
    DATA_MATRIX_DECODER_CROP_LOCAL = "data_matrix_decoder_crop_local_pixels"
    CURRENT_MINIMAP_CROP = "current_minimap_crop_pixels"
    RIG_EXPANDED_CAMERA_REVIEW = "rig_expanded_camera_review_pixels"
    RIG_STANDARDIZED_THREE_SPACE_COMPARISON = (
        "rig_standardized_three_space_comparison_pixels"
    )
    RIG_CROSS_SOURCE_COMPARISON_LOCAL = (
        "rig_cross_source_comparison_local_pixels"
    )
    CROSS_SOURCE_COMPARISON_LOCAL = "cross_source_comparison_local_pixels"
    DIAGNOSTIC_COMPOSITE = "diagnostic_composite_pixels"
    ALIGNED_ADB_AND_HIK_COMPARISON_INPUTS = (
        "aligned_adb_and_hik_comparison_inputs"
    )


class RigTransformOperation:
    """Canonical operation identities used by IRIS frame producers."""

    RIG_RECTIFICATION = "rig_rectification"
    GAME_UPRIGHT_QUARTER_TURN = "game_upright_quarter_turn"
    HARDWARE_ROI = "hardware_roi"
    ADAPTER_OUTPUT_VIEW = "adapter_output_view"
    SESSION_STREAM_VIEW = "session_stream_view"
    MINIMAP_CROP = "minimap_crop"
    MINIMAP_MASK = "minimap_mask"
    ENCODER_PADDING = "encoder_padding"


def compiled_transform_lineage(
    source_space_id: str,
    target_space_id: str,
    operation_ids: Sequence[str],
    *,
    inherited: Mapping[str, Any] = None,
) -> Dict[str, Any]:
    """Describe one already-applied transform and reject duplicate operations.

    This record travels with produced media. A producer that consumes another
    IRIS frame must pass its parent's lineage; applying an operation already in
    that lineage is an error rather than a second compensating transform.
    """

    source = str(source_space_id).strip()
    target = str(target_space_id).strip()
    if not source or not target:
        raise ValueError("Transform lineage requires source and target spaces")
    previous_lineage = dict(inherited or {})
    previous = list(previous_lineage.get("operation_ids") or [])
    if previous_lineage:
        previous_target = str(previous_lineage.get("target_space_id") or "")
        if previous_target != source:
            raise ValueError(
                "Transform chain discontinuity: parent ends in {!r}, child starts "
                "in {!r}".format(previous_target, source)
            )
    current = [str(value).strip() for value in operation_ids]
    if any(not value for value in current):
        raise ValueError("Transform operation IDs must be non-empty")
    combined = previous + current
    duplicates = sorted({value for value in combined if combined.count(value) > 1})
    if duplicates:
        raise ValueError(
            "Spatial operation already applied: {}".format(", ".join(duplicates))
        )
    root_source = str(previous_lineage.get("source_space_id") or source)
    plan_id = "{}|{}|{}".format(
        root_source, ",".join(combined) or "identity", target
    )
    return {
        "schema_version": "1.0",
        "plan_id": plan_id,
        "source_space_id": root_source,
        "target_space_id": target,
        "operation_ids": combined,
        "application_count": 1,
    }


@dataclass(frozen=True)
class SpaceRef:
    space_id: str
    kind: str = "unspecified"
    transform_refs: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.space_id.strip():
            raise ValueError("Space ID is required; use 'unknown' explicitly")
        if not self.kind.strip():
            raise ValueError("Space kind is required")
        if any(not value.strip() for value in self.transform_refs):
            raise ValueError("Transform references must be non-empty")
