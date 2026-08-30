"""High-level construction of a reusable rig-calibration bundle."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .artifact import CALIBRATION_SCHEMA_VERSION, write_calibration_yaml
from .contracts import GeometryEstimate
from .geometry import estimate_screen_geometry


def _normalization_matrix(
    camera_to_screen: np.ndarray,
    origin_screen_xy: Sequence[float],
    screen_units_per_output_pixel_xy: Sequence[float],
) -> np.ndarray:
    origin_x, origin_y = map(float, origin_screen_xy)
    scale_x, scale_y = map(float, screen_units_per_output_pixel_xy)
    if scale_x <= 0 or scale_y <= 0:
        raise ValueError("Screen units per output pixel must be positive")
    screen_to_output = np.asarray(
        [
            [1.0 / scale_x, 0.0, -origin_x / scale_x],
            [0.0, 1.0 / scale_y, -origin_y / scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    result = screen_to_output.dot(camera_to_screen)
    return result / result[2, 2]


def valid_output_mask(
    input_size_px: Sequence[int],
    output_size_px: Sequence[int],
    matrix_3x3: Sequence[Sequence[float]],
) -> np.ndarray:
    input_width, input_height = map(int, input_size_px)
    output_width, output_height = map(int, output_size_px)
    source = np.full((input_height, input_width), 255, dtype=np.uint8)
    return cv2.warpPerspective(
        source,
        np.asarray(matrix_3x3, dtype=np.float64),
        (output_width, output_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def build_calibration(
    calibration_id: str,
    camera_points_xy: Sequence[Sequence[float]],
    screen_points_xy: Sequence[Sequence[float]],
    camera_size_px: Sequence[int],
    screen_size_px: Sequence[int],
    input_frame_id: str,
    canonical_screen_frame_id: str,
    required_region_screen_xy: Optional[Sequence[Sequence[float]]] = None,
    output_origin_screen_xy: Sequence[float] = (0.0, 0.0),
    screen_units_per_output_pixel_xy: Sequence[float] = (1.0, 1.0),
    output_size_px: Optional[Sequence[int]] = None,
    output_frame_id: Optional[str] = None,
    rig: Optional[Mapping[str, Any]] = None,
    optics: Optional[Mapping[str, Any]] = None,
    required_roi: Optional[Mapping[str, Any]] = None,
    image_quality: Optional[Mapping[str, Any]] = None,
    feature_matching: Optional[Mapping[str, Any]] = None,
    timing: Optional[Mapping[str, Any]] = None,
    status: str = "warning",
    ransac_threshold_screen_px: float = 2.0,
    data_matrix_decode: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], GeometryEstimate, np.ndarray]:
    """Fit geometry and assemble the complete consumer-facing YAML value."""

    if not calibration_id:
        raise ValueError("calibration_id is required")
    camera_size = tuple(map(int, camera_size_px))
    screen_size = tuple(map(int, screen_size_px))
    origin = tuple(map(float, output_origin_screen_xy))
    scale = tuple(map(float, screen_units_per_output_pixel_xy))
    if output_size_px is None:
        output_size = (
            int(round(screen_size[0] / scale[0])),
            int(round(screen_size[1] / scale[1])),
        )
    else:
        output_size = tuple(map(int, output_size_px))
    if min(output_size) <= 0:
        raise ValueError("Output size must be positive")

    geometry = estimate_screen_geometry(
        camera_points_xy,
        screen_points_xy,
        camera_size,
        screen_size,
        required_region_screen_xy=required_region_screen_xy,
        ransac_threshold_screen_px=ransac_threshold_screen_px,
    )
    normalization_matrix = _normalization_matrix(
        geometry.matrix_3x3, origin, scale
    )
    mask = valid_output_mask(camera_size, output_size, normalization_matrix)
    geometry_value = geometry.to_dict()
    geometry_value.update(geometry.metrics)
    geometry_value["mode"] = (
        "intrinsics_and_homography"
        if (optics or {}).get("lens_model", {}).get("source")
        in ("measured", "reused")
        else "homography_only"
    )
    geometry_value["transform_p95_error_px_at_required_roi"] = geometry_value[
        "reprojection_p95_px"
    ]

    confidence = {
        "geometry": float(geometry.confidence),
        "image_quality": float((image_quality or {}).get("confidence", 0.0)),
        "feature_matching": float((feature_matching or {}).get("confidence", 0.0)),
        "timing": float((timing or {}).get("confidence", 0.0)),
        "overall": float(geometry.confidence),
        "assumptions": [],
        "warnings": list(geometry.warnings),
    }
    measured_confidences = [confidence["geometry"]]
    if image_quality:
        measured_confidences.append(confidence["image_quality"])
    if feature_matching:
        measured_confidences.append(confidence["feature_matching"])
    if timing:
        measured_confidences.append(confidence["timing"])
    confidence["overall"] = float(min(measured_confidences))

    phone = dict((rig or {}).get("phone", {}))
    phone.setdefault("screen_size_px", list(screen_size))
    rig_value = dict(rig or {})
    rig_value["phone"] = phone
    optics_value = dict(optics or {})
    optics_value.setdefault("lens_model", {"source": "unavailable"})
    normalization = {
        "input_frame_id": input_frame_id,
        "input_space": "camera_undistorted_px",
        "input_size_px": list(camera_size),
        "input_origin": "top_left_pixel_center",
        "input_axes": ["right", "down"],
        "transform_direction": "input_pixel_to_output_pixel",
        "matrix_3x3": normalization_matrix.tolist(),
        "output_size_px": list(output_size),
        "output_frame_id": output_frame_id
        or "aria://artifact/{}/normalized".format(calibration_id),
        "output_origin": "top_left_pixel_center",
        "output_axes": ["right", "down"],
        "canonical_screen_frame_id": canonical_screen_frame_id,
        "origin_screen_xy": list(origin),
        "screen_units_per_output_pixel_xy": list(scale),
        "valid_mask_file": "valid_screen_mask.png",
        "border_mode": "constant",
        "border_value_bgr": [0, 0, 0],
    }
    value: Dict[str, Any] = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "calibration_id": calibration_id,
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rig": rig_value,
        "optics": optics_value,
        "normalization": normalization,
        "geometry": geometry_value,
        "required_roi": dict(required_roi or {}),
        "image_quality": dict(image_quality or {}),
        "data_matrix_decode": dict(data_matrix_decode or {}),
        "feature_matching": dict(feature_matching or {}),
        "timing": dict(timing or {}),
        "confidence": confidence,
        "applicability": {
            "require_same_camera_hardware_id": True,
            "require_same_camera_mode": True,
            "require_same_phone_orientation": True,
            "require_same_phone_screen_size_px": True,
        },
        "evidence": {},
    }
    return value, geometry, mask


def write_calibration_bundle(
    output_directory: Path,
    calibration: Mapping[str, Any],
    valid_mask: np.ndarray,
    rectification_maps: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Path:
    """Write the YAML, valid mask, and optional dense lens maps."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    mask_path = output_directory / "valid_screen_mask.png"
    if valid_mask.dtype != np.uint8:
        valid_mask = np.clip(valid_mask, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(mask_path), valid_mask):
        raise RuntimeError("Cannot write calibration valid mask")
    if rectification_maps is not None:
        map_x, map_y = rectification_maps
        np.savez_compressed(
            str(output_directory / "rectification_maps.npz"),
            map_x=np.asarray(map_x, dtype=np.float32),
            map_y=np.asarray(map_y, dtype=np.float32),
        )
    return write_calibration_yaml(output_directory / "calibration.yaml", calibration)
