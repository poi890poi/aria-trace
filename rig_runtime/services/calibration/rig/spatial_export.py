"""Dependency-free spatial-fragment export for a rig calibration artifact."""

from typing import Any, Dict, List, Mapping

import numpy as np

from .artifact import validate_calibration
from .contracts import matrix_3x3


SPATIAL_SCHEMA_VERSION = "1.0"


def _frame(
    frame_id: str,
    revision: str,
    unit: str,
    size_px: Any,
) -> Dict[str, Any]:
    return {
        "frame_id": frame_id,
        "revision": revision,
        "kind": "raster_2d",
        "unit": unit,
        "size_px": list(map(int, size_px)),
        "origin": "top_left_pixel_center",
        "axes": {"x": "right", "y": "down"},
        "pixel_center_convention": "integer_is_pixel_center",
    }


def export_spatial_fragment(calibration: Mapping[str, Any]) -> Dict[str, Any]:
    """Export frames/transforms without importing or contacting a registry."""

    validate_calibration(calibration)
    normalization = calibration["normalization"]
    calibration_id = str(calibration["calibration_id"])
    canonical_size = (
        calibration.get("rig", {}).get("phone", {}).get("screen_size_px")
        or normalization["output_size_px"]
    )
    frames = [
        _frame(
            normalization["input_frame_id"],
            calibration_id,
            "pixel",
            normalization["input_size_px"],
        ),
        _frame(
            normalization["output_frame_id"],
            calibration_id,
            "pixel",
            normalization["output_size_px"],
        ),
        _frame(
            normalization["canonical_screen_frame_id"],
            str(calibration.get("rig", {}).get("phone", {}).get("orientation", calibration_id)),
            "screen_pixel",
            canonical_size,
        ),
    ]
    geometry = calibration.get("geometry", {})
    transforms = [
        {
            "transform_id": "{}:camera-to-normalized".format(calibration_id),
            "from_frame": normalization["input_frame_id"],
            "to_frame": normalization["output_frame_id"],
            "model": "projective_2d",
            "behavior": "static",
            "direction": "from_to",
            "matrix_3x3": matrix_3x3(normalization["matrix_3x3"]).tolist(),
            "valid_mask_file": normalization.get("valid_mask_file"),
            "uncertainty": {
                "model": "point_error",
                "p95_px_at_required_roi": geometry.get(
                    "transform_p95_error_px_at_required_roi"
                ),
            },
            "confidence": float(
                calibration.get("confidence", {}).get("geometry", 0.0)
            ),
            "status": calibration.get("status"),
            "estimator": {"name": "iris_rig_calibration", "version": "1.0"},
        }
    ]
    scale_x, scale_y = map(
        float, normalization["screen_units_per_output_pixel_xy"]
    )
    origin_x, origin_y = map(float, normalization["origin_screen_xy"])
    transforms.append(
        {
            "transform_id": "{}:normalized-to-screen".format(calibration_id),
            "from_frame": normalization["output_frame_id"],
            "to_frame": normalization["canonical_screen_frame_id"],
            "model": "affine_2d",
            "behavior": "static",
            "direction": "from_to",
            "matrix_3x3": [
                [scale_x, 0.0, origin_x],
                [0.0, scale_y, origin_y],
                [0.0, 0.0, 1.0],
            ],
            "confidence": 1.0,
            "status": calibration.get("status"),
            "estimator": {"name": "declared_raster_origin_scale", "version": "1.0"},
        }
    )

    timing = calibration.get("timing", {})
    clock_ids = set()
    clock_mapping = timing.get("clocks", {})
    if isinstance(clock_mapping, Mapping):
        clock_ids.update(str(value) for value in clock_mapping.values())
    clock_transform = timing.get("clock_transform")
    if isinstance(clock_transform, Mapping):
        clock_ids.add(str(clock_transform.get("from_clock")))
        clock_ids.add(str(clock_transform.get("to_clock")))
    clocks = [
        {"clock_id": clock_id, "unit": "nanosecond"}
        for clock_id in sorted(clock_ids)
        if clock_id and clock_id != "None"
    ]
    latencies = []
    for name, value in timing.items():
        if not isinstance(value, Mapping) or "median_ns" not in value:
            continue
        latency = dict(value)
        latency["latency_id"] = name
        latencies.append(latency)
    fragment = {
        "spatial_schema_version": SPATIAL_SCHEMA_VERSION,
        "artifact_id": calibration_id,
        "frames": frames,
        "transforms": transforms,
        "clocks": clocks,
        "latencies": latencies,
    }
    validate_spatial_fragment(fragment)
    return fragment


def validate_spatial_fragment(value: Mapping[str, Any]) -> None:
    if value.get("spatial_schema_version") != SPATIAL_SCHEMA_VERSION:
        raise ValueError("Unsupported spatial schema version")
    frames = value.get("frames")
    transforms = value.get("transforms")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Spatial fragment needs at least one frame")
    if not isinstance(transforms, list) or not transforms:
        raise ValueError("Spatial fragment needs at least one transform")
    frame_ids = [item.get("frame_id") for item in frames]
    if any(not frame_id for frame_id in frame_ids) or len(frame_ids) != len(set(frame_ids)):
        raise ValueError("Spatial frame IDs must be present and unique")
    known = set(frame_ids)
    transform_ids = set()
    for transform in transforms:
        transform_id = transform.get("transform_id")
        if not transform_id or transform_id in transform_ids:
            raise ValueError("Spatial transform IDs must be present and unique")
        transform_ids.add(transform_id)
        if transform.get("from_frame") not in known or transform.get("to_frame") not in known:
            raise ValueError("Spatial transform references an unknown frame")
        matrix_3x3(transform.get("matrix_3x3"))
        confidence = float(transform.get("confidence", 0.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Spatial transform confidence must be within [0, 1]")
