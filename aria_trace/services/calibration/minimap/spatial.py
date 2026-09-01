"""Spatial contracts for mini-map calibration geometry."""

from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional, Sequence

from aria_trace.domain.spatial import (
    bind_geometry,
    normalize_legacy_geometry,
    raster_space,
    require_same_space,
    require_spatial_geometry,
    validate_raster_space,
)


MINIMAP_CROP_SPACE_ID = "minimap_calibration_crop_pixels"


def minimap_crop_space(
    size_px: Sequence[int],
    *,
    space_id: str = MINIMAP_CROP_SPACE_ID,
    parent_space_id: Optional[str] = None,
    crop_xywh: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Describe the raster in which calibrated mini-map shapes are measured."""

    transform = None
    if parent_space_id is not None:
        if crop_xywh is None:
            raise ValueError("A parent mini-map space requires crop_xywh")
        x, y, width, height = [int(item) for item in crop_xywh]
        if [width, height] != [int(item) for item in size_px]:
            raise ValueError("Mini-map crop size does not match its raster space")
        transform = [[1.0, 0.0, float(x)], [0.0, 1.0, float(y)], [0.0, 0.0, 1.0]]
    return raster_space(
        space_id,
        size_px,
        parent_space_id=parent_space_id,
        local_to_parent_3x3=transform,
    )


def normalize_minimap_geometry(
    calibration: Mapping[str, Any],
    space: Mapping[str, Any],
    *,
    require_rotation_center: bool = False,
    allow_legacy: bool = True,
) -> Dict[str, Any]:
    """Return calibration geometry validated in one explicit raster space.

    Legacy values are accepted only when the caller supplies the containing
    raster space.  The returned document is always fully spatially bound.
    """

    result = copy.deepcopy(dict(calibration))
    space = validate_raster_space(space)
    boundary = result.get("outer_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("Mini-map calibration has no outer_boundary geometry")
    if allow_legacy:
        boundary = normalize_legacy_geometry(boundary, "circle", space)
    else:
        boundary = require_spatial_geometry(boundary, "circle")
    boundary = require_spatial_geometry(
        boundary, "circle", expected_space_id=str(space["space_id"])
    )
    if boundary["space"] != space:
        raise ValueError("Mini-map boundary does not match the containing raster space")
    result["outer_boundary"] = boundary

    center = result.get("rotation_center")
    if center is not None:
        if allow_legacy:
            center = normalize_legacy_geometry(center, "point", space)
        else:
            center = require_spatial_geometry(center, "point")
        center = require_spatial_geometry(
            center, "point", expected_space_id=str(space["space_id"])
        )
        require_same_space(boundary, center)
        result["rotation_center"] = center
    elif require_rotation_center:
        raise ValueError("Mini-map calibration has no rotation_center geometry")

    offset = result.get("center_offset")
    if offset is not None:
        if allow_legacy:
            offset = normalize_legacy_geometry(offset, "vector", space)
        else:
            offset = require_spatial_geometry(offset, "vector")
        offset = require_spatial_geometry(
            offset, "vector", expected_space_id=str(space["space_id"])
        )
        require_same_space(boundary, offset)
        result["center_offset"] = offset
    result["geometry_space"] = copy.deepcopy(boundary["space"])
    return result


def bind_minimap_boundary(
    boundary: Mapping[str, Any], space: Mapping[str, Any]
) -> Dict[str, Any]:
    return bind_geometry(boundary, "circle", space)


__all__ = [
    "MINIMAP_CROP_SPACE_ID",
    "bind_minimap_boundary",
    "minimap_crop_space",
    "normalize_minimap_geometry",
]
