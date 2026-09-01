"""Validated, JSON-compatible coordinate-space contracts for geometry."""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Mapping, Optional, Sequence

from .spaces import SpaceRef


SPATIAL_SCHEMA_VERSION = 1
PIXEL_SPACE_KIND = "raster_pixel_centers"


def raster_space(
    space_id: str,
    size_px: Sequence[int],
    *,
    parent_space_id: Optional[str] = None,
    local_to_parent_3x3: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, Any]:
    """Create and validate a serializable raster coordinate-space reference."""

    width, height = [int(value) for value in size_px]
    value: Dict[str, Any] = {
        "space_id": str(space_id),
        "kind": PIXEL_SPACE_KIND,
        "size_px": [width, height],
        "coordinates": "pixel_center_xy",
        "axis_directions": "x_right_y_down",
        "units": "px",
    }
    if parent_space_id is not None:
        value["parent_space_id"] = str(parent_space_id)
    if local_to_parent_3x3 is not None:
        value["local_to_parent_3x3"] = _finite_transform(local_to_parent_3x3)
    return validate_raster_space(value)


def validate_raster_space(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Reject missing, malformed, or ambiguous raster-space metadata."""

    if not isinstance(value, Mapping):
        raise ValueError("Spatial geometry requires a coordinate-space mapping")
    result = copy.deepcopy(dict(value))
    space = SpaceRef(
        str(result.get("space_id") or ""),
        str(result.get("kind") or ""),
    )
    if space.kind != PIXEL_SPACE_KIND:
        raise ValueError(
            "Spatial pixel geometry requires kind {!r}, got {!r}".format(
                PIXEL_SPACE_KIND, space.kind
            )
        )
    size = result.get("size_px")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("Spatial raster space requires size_px [width, height]")
    width, height = [int(item) for item in size]
    if min(width, height) <= 0:
        raise ValueError("Spatial raster dimensions must be positive")
    result["size_px"] = [width, height]
    if result.get("coordinates") != "pixel_center_xy":
        raise ValueError("Spatial raster coordinates must be pixel_center_xy")
    if result.get("axis_directions") != "x_right_y_down":
        raise ValueError("Spatial raster axes must be x_right_y_down")
    if result.get("units") != "px":
        raise ValueError("Spatial raster units must be px")
    parent = result.get("parent_space_id")
    matrix = result.get("local_to_parent_3x3")
    if (parent is None) != (matrix is None):
        raise ValueError(
            "parent_space_id and local_to_parent_3x3 must be declared together"
        )
    if parent is not None:
        if not str(parent).strip():
            raise ValueError("Parent coordinate-space ID cannot be empty")
        transform = _finite_transform(matrix)
        result["parent_space_id"] = str(parent)
        result["local_to_parent_3x3"] = transform
    return result


def bind_geometry(
    metrics: Mapping[str, Any], geometry_type: str, space: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return metrics carrying their own immutable spatial identity."""

    result = copy.deepcopy(dict(metrics))
    result["spatial_schema_version"] = SPATIAL_SCHEMA_VERSION
    result["geometry_type"] = str(geometry_type)
    result["space"] = validate_raster_space(space)
    return require_spatial_geometry(result, geometry_type)


def require_spatial_geometry(
    value: Mapping[str, Any],
    geometry_type: Optional[str] = None,
    *,
    expected_space_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate geometry before any consumer uses its numeric coordinates."""

    if not isinstance(value, Mapping):
        raise ValueError("Spatial geometry must be a mapping")
    result = copy.deepcopy(dict(value))
    if int(result.get("spatial_schema_version") or 0) != SPATIAL_SCHEMA_VERSION:
        raise ValueError("Geometry has no supported spatial schema")
    actual_type = str(result.get("geometry_type") or "")
    if not actual_type:
        raise ValueError("Spatial geometry requires geometry_type")
    if geometry_type is not None and actual_type != geometry_type:
        raise ValueError(
            "Expected {} geometry, got {}".format(geometry_type, actual_type)
        )
    result["space"] = validate_raster_space(result.get("space"))
    actual_space_id = result["space"]["space_id"]
    if expected_space_id is not None and actual_space_id != expected_space_id:
        raise ValueError(
            "Geometry space {!r} does not match required space {!r}".format(
                actual_space_id, expected_space_id
            )
        )
    if actual_type == "circle":
        numeric = [result.get("center_x"), result.get("center_y"), result.get("radius")]
        if not all(item is not None and math.isfinite(float(item)) for item in numeric):
            raise ValueError("Circle requires finite center_x, center_y, and radius")
        if float(result["radius"]) <= 0.0:
            raise ValueError("Circle radius must be positive")
    elif actual_type == "point":
        numeric = [result.get("x"), result.get("y")]
        if not all(item is not None and math.isfinite(float(item)) for item in numeric):
            raise ValueError("Point requires finite x and y")
    elif actual_type == "vector":
        numeric = [result.get("dx"), result.get("dy")]
        if not all(item is not None and math.isfinite(float(item)) for item in numeric):
            raise ValueError("Vector requires finite dx and dy")
    elif actual_type in ("polygon", "relative_polygon"):
        if result.get("points_xy") is None and str(result.get("array_name") or ""):
            pass
        else:
            raw_points = result.get("points_xy")
            if not isinstance(raw_points, (list, tuple)) or len(raw_points) < 3:
                raise ValueError("Polygon requires at least three XY points")
            points = []
            for point in raw_points:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ValueError("Polygon points must be XY pairs")
                x, y = [float(item) for item in point]
                if not math.isfinite(x) or not math.isfinite(y):
                    raise ValueError("Polygon points must be finite")
                points.append([x, y])
            result["points_xy"] = points
    elif actual_type == "mask":
        if not str(result.get("array_name") or result.get("file") or "").strip():
            raise ValueError("Mask geometry requires array_name or file")
    else:
        raise ValueError("Unsupported spatial geometry type {!r}".format(actual_type))
    return result


def require_same_space(*values: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the shared space, rejecting mixed-space geometry."""

    if not values:
        raise ValueError("At least one spatial geometry is required")
    geometries = [require_spatial_geometry(value) for value in values]
    first = geometries[0]["space"]
    for geometry in geometries[1:]:
        if geometry["space"] != first:
            raise ValueError(
                "Spatial geometry cannot be combined across coordinate spaces"
            )
    return first


def normalize_legacy_geometry(
    value: Mapping[str, Any], geometry_type: str, fallback_space: Mapping[str, Any]
) -> Dict[str, Any]:
    """Normalize one legacy shape at an explicit compatibility boundary.

    New producers must use :func:`bind_geometry`.  This function exists only
    for readers of older persisted profiles whose containing contract already
    proves the coordinate space.
    """

    if not isinstance(value, Mapping):
        raise ValueError("Geometry must be a mapping")
    if "space" in value or "geometry_type" in value or "spatial_schema_version" in value:
        return require_spatial_geometry(value, geometry_type)
    return bind_geometry(value, geometry_type, fallback_space)


def transform_point(
    value: Mapping[str, Any],
    matrix_3x3: Sequence[Sequence[float]],
    target_space: Mapping[str, Any],
) -> Dict[str, Any]:
    """Transform a point and bind the result to the target space."""

    point = require_spatial_geometry(value, "point")
    matrix = _finite_transform(matrix_3x3)
    x, y = float(point["x"]), float(point["y"])
    target = [
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2],
    ]
    if abs(target[2]) < 1.0e-12:
        raise ValueError("Point transform produced an invalid projective scale")
    metrics = {
        key: copy.deepcopy(item)
        for key, item in point.items()
        if key not in ("spatial_schema_version", "geometry_type", "space", "x", "y")
    }
    metrics.update(x=float(target[0] / target[2]), y=float(target[1] / target[2]))
    metrics["spatial_provenance"] = {
        "operation": "explicit_point_transform",
        "source_space": copy.deepcopy(point["space"]),
        "transform_3x3": copy.deepcopy(matrix),
        "previous": copy.deepcopy(point.get("spatial_provenance")),
    }
    return bind_geometry(metrics, "point", target_space)


def transform_circle_similarity(
    value: Mapping[str, Any],
    matrix_3x3: Sequence[Sequence[float]],
    target_space: Mapping[str, Any],
    *,
    tolerance: float = 1.0e-6,
) -> Dict[str, Any]:
    """Transform a circle only when the mapping preserves circular geometry."""

    circle = require_spatial_geometry(value, "circle")
    matrix = _finite_transform(matrix_3x3)
    if not all(
        math.isclose(matrix[2][index], expected, abs_tol=tolerance)
        for index, expected in enumerate((0.0, 0.0, 1.0))
    ):
        raise ValueError("A projective transform does not preserve a circle")
    a, b = matrix[0][0], matrix[0][1]
    c, d = matrix[1][0], matrix[1][1]
    gram_00 = a * a + c * c
    gram_01 = a * b + c * d
    gram_11 = b * b + d * d
    scale_squared = (gram_00 + gram_11) / 2.0
    close = lambda first, second: math.isclose(
        first, second, abs_tol=tolerance, rel_tol=tolerance
    )
    if (
        scale_squared <= 0.0
        or not close(gram_00, scale_squared)
        or not close(gram_11, scale_squared)
        or not close(gram_01, 0.0)
    ):
        raise ValueError("A non-similarity transform converts a circle to an ellipse")
    point = bind_geometry(
        {"x": float(circle["center_x"]), "y": float(circle["center_y"])},
        "point",
        circle["space"],
    )
    transformed = transform_point(point, matrix, target_space)
    metrics = {
        key: copy.deepcopy(item)
        for key, item in circle.items()
        if key
        not in (
            "spatial_schema_version",
            "geometry_type",
            "space",
            "center_x",
            "center_y",
            "radius",
        )
    }
    metrics.update(
        center_x=float(transformed["x"]),
        center_y=float(transformed["y"]),
        radius=float(circle["radius"]) * math.sqrt(scale_squared),
    )
    metrics["spatial_provenance"] = {
        "operation": "explicit_circle_similarity_transform",
        "source_space": copy.deepcopy(circle["space"]),
        "transform_3x3": copy.deepcopy(matrix),
        "previous": copy.deepcopy(circle.get("spatial_provenance")),
    }
    return bind_geometry(metrics, "circle", target_space)


def _finite_transform(value: Sequence[Sequence[float]]) -> list:
    try:
        rows = list(value)
    except (TypeError, ValueError):
        rows = []
    if len(rows) != 3:
        raise ValueError("Geometry transform must be a finite 3x3 matrix")
    matrix = []
    for row in rows:
        try:
            columns = list(row)
        except (TypeError, ValueError):
            columns = []
        if len(columns) != 3:
            raise ValueError("Geometry transform must be a finite 3x3 matrix")
        converted = [float(item) for item in columns]
        if not all(math.isfinite(item) for item in converted):
            raise ValueError("Geometry transform must be a finite 3x3 matrix")
        matrix.append(converted)
    determinant = (
        matrix[0][0]
        * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1]
        * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2]
        * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    if abs(determinant) < 1.0e-12:
        raise ValueError("Geometry transform must be invertible")
    return matrix


__all__ = [
    "PIXEL_SPACE_KIND",
    "SPATIAL_SCHEMA_VERSION",
    "bind_geometry",
    "raster_space",
    "normalize_legacy_geometry",
    "require_same_space",
    "require_spatial_geometry",
    "transform_circle_similarity",
    "transform_point",
    "validate_raster_space",
]
