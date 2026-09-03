"""Independent broad-edge measurement of residual phone-panel axis rotation."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


def _axial_degrees(value: float) -> float:
    """Resolve only the 180-degree ambiguity of an undirected fitted line."""

    result = float(value)
    while result < -90.0:
        result += 180.0
    while result >= 90.0:
        result -= 180.0
    return result


def _subpixel_offsets(responses: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Vectorized three-sample parabola offsets for independent scan lines."""

    indices = np.asarray(indices, dtype=np.int64)
    offsets = np.zeros(indices.shape, dtype=np.float32)
    valid = (indices > 0) & (indices < responses.shape[1] - 1)
    if not np.any(valid):
        return offsets
    rows = np.nonzero(valid)[0]
    centers = indices[valid]
    left = responses[rows, centers - 1].astype(np.float64)
    center = responses[rows, centers].astype(np.float64)
    right = responses[rows, centers + 1].astype(np.float64)
    denominator = left - 2.0 * center + right
    stable = np.abs(denominator) > 1.0e-9
    values = np.zeros(rows.shape, dtype=np.float64)
    values[stable] = 0.5 * (left[stable] - right[stable]) / denominator[stable]
    offsets[rows] = np.clip(values, -0.75, 0.75).astype(np.float32)
    return offsets


def _fit_line(points_xy: np.ndarray, axis: str) -> Dict[str, Any]:
    points = np.asarray(points_xy, dtype=np.float32).reshape((-1, 2))
    if len(points) < 12:
        raise RuntimeError("Panel edge has fewer than 12 detected samples")
    fitted = cv2.fitLine(
        points.reshape((-1, 1, 2)),
        cv2.DIST_WELSCH,
        0.0,
        0.01,
        0.01,
    ).reshape(-1)
    vx, vy, x0, y0 = map(float, fitted)
    if axis == "horizontal":
        if vx < 0.0:
            vx, vy = -vx, -vy
        residual = _axial_degrees(math.degrees(math.atan2(vy, vx)))
    else:
        if vy < 0.0:
            vx, vy = -vx, -vy
        residual = _axial_degrees(math.degrees(math.atan2(vy, vx)) - 90.0)
    normal = np.asarray([-vy, vx], dtype=np.float64)
    distances = np.abs((points - np.asarray([x0, y0])) @ normal)
    return {
        "point_xy": [x0, y0],
        "direction_unit_xy": [vx, vy],
        "residual_clockwise_degrees": float(residual),
        "orthogonal_rmse_px": float(np.sqrt(np.mean(np.square(distances)))),
        "orthogonal_p95_px": float(np.percentile(distances, 95)),
    }


def _edge_points(
    gradient: np.ndarray,
    edge: Mapping[str, Any],
    output_origin_screen_xy: Sequence[float],
) -> Tuple[np.ndarray, int]:
    origin_x, origin_y = map(float, output_origin_screen_xy)
    axis = str(edge["axis"])
    coordinate = float(edge["coordinate_screen_px"]) - (
        origin_x if axis == "vertical" else origin_y
    )
    segments = [list(map(float, value)) for value in edge["segments_screen_px"]]
    polarity = float(edge.get("polarity", 1.0))
    half_width = int(edge.get("search_half_width_px", 12))
    height, width = gradient.shape
    points: List[List[float]] = []
    possible = 0
    if axis == "vertical":
        low = max(1, int(math.floor(coordinate - half_width)))
        high = min(width - 2, int(math.ceil(coordinate + half_width)))
        if high <= low:
            return np.empty((0, 2), np.float32), 0
        for start, stop in segments:
            first = max(1, int(math.ceil(start - origin_y)))
            last = min(height - 1, int(math.floor(stop - origin_y)))
            sample_rows = np.arange(first, last, dtype=np.int32)
            possible += int(len(sample_rows))
            if not len(sample_rows):
                continue
            responses = polarity * gradient[first:last, low : high + 1]
            indices = np.argmax(responses, axis=1)
            peaks = responses[np.arange(len(indices)), indices]
            valid = peaks >= 12.0
            offsets = _subpixel_offsets(responses, indices)
            points.extend(
                np.column_stack(
                    [low + indices[valid] + offsets[valid], sample_rows[valid]]
                ).tolist()
            )
    elif axis == "horizontal":
        low = max(1, int(math.floor(coordinate - half_width)))
        high = min(height - 2, int(math.ceil(coordinate + half_width)))
        if high <= low:
            return np.empty((0, 2), np.float32), 0
        for start, stop in segments:
            first = max(1, int(math.ceil(start - origin_x)))
            last = min(width - 1, int(math.floor(stop - origin_x)))
            sample_columns = np.arange(first, last, dtype=np.int32)
            possible += int(len(sample_columns))
            if not len(sample_columns):
                continue
            responses = (
                polarity * gradient[low : high + 1, first:last]
            ).T
            indices = np.argmax(responses, axis=1)
            peaks = responses[np.arange(len(indices)), indices]
            valid = peaks >= 12.0
            offsets = _subpixel_offsets(responses, indices)
            points.extend(
                np.column_stack(
                    [sample_columns[valid], low + indices[valid] + offsets[valid]]
                ).tolist()
            )
    else:
        raise ValueError("Panel edge axis must be horizontal or vertical")
    return np.asarray(points, dtype=np.float32).reshape((-1, 2)), possible


def _sample_output_to_raw(
    points_xy: np.ndarray, output_to_raw_maps: Tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    map_x, map_y = output_to_raw_maps
    query_x = np.asarray(points_xy[:, 0], dtype=np.float32).reshape((1, -1))
    query_y = np.asarray(points_xy[:, 1], dtype=np.float32).reshape((1, -1))
    raw_x = cv2.remap(
        np.asarray(map_x, dtype=np.float32), query_x, query_y, cv2.INTER_LINEAR
    ).reshape(-1)
    raw_y = cv2.remap(
        np.asarray(map_y, dtype=np.float32), query_x, query_y, cv2.INTER_LINEAR
    ).reshape(-1)
    return np.column_stack([raw_x, raw_y]).astype(np.float32)


def _mean_direction(lines: Sequence[Mapping[str, Any]]) -> Optional[List[float]]:
    vectors = []
    for line in lines:
        vector = np.asarray(line["raw_direction_unit_xy"], dtype=np.float64)
        if vectors and float(np.dot(vector, vectors[0])) < 0.0:
            vector = -vector
        vectors.append(vector)
    if not vectors:
        return None
    value = np.mean(np.asarray(vectors), axis=0)
    norm = float(np.linalg.norm(value))
    return (value / norm).tolist() if norm > 1.0e-9 else None


def measure_panel_axis_edges(
    rectified_bgr: np.ndarray,
    edge_specs: Sequence[Mapping[str, Any]],
    output_origin_screen_xy: Sequence[float],
    *,
    output_to_raw_maps: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    maximum_correction_degrees: float = 30.0,
) -> Dict[str, Any]:
    """Measure residual clockwise panel rotation after coarse rectification."""

    if rectified_bgr is None or rectified_bgr.size == 0:
        raise ValueError("Panel-axis image is empty")
    gray = (
        cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2GRAY)
        if rectified_bgr.ndim == 3
        else np.asarray(rectified_bgr)
    )
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
    gradient_x = cv2.Scharr(blurred, cv2.CV_32F, 1, 0)
    gradient_y = cv2.Scharr(blurred, cv2.CV_32F, 0, 1)
    lines = []
    failures = []
    for edge in edge_specs:
        axis = str(edge["axis"])
        gradient = gradient_x if axis == "vertical" else gradient_y
        points, possible = _edge_points(
            gradient, edge, output_origin_screen_xy
        )
        minimum = max(12, int(math.ceil(float(possible) * 0.25)))
        if len(points) < minimum:
            failures.append(
                {
                    "id": str(edge["id"]),
                    "reason": "insufficient_edge_support",
                    "detected_samples": int(len(points)),
                    "required_samples": int(minimum),
                }
            )
            continue
        fitted = _fit_line(points, axis)
        fitted.update(
            id=str(edge["id"]),
            axis=axis,
            detected_samples=int(len(points)),
            possible_samples=int(possible),
            support_fraction=float(len(points)) / float(max(possible, 1)),
            detected_points_output_xy=points[:: max(1, len(points) // 80)].tolist(),
        )
        if output_to_raw_maps is not None:
            raw_points = _sample_output_to_raw(points, output_to_raw_maps)
            raw_fit = _fit_line(raw_points, axis)
            raw_direction = np.asarray(
                raw_fit["direction_unit_xy"], dtype=np.float64
            )
            ordered_direction = np.asarray(
                raw_points[-1] - raw_points[0], dtype=np.float64
            )
            if float(np.dot(raw_direction, ordered_direction)) < 0.0:
                raw_direction = -raw_direction
            fitted["raw_direction_unit_xy"] = raw_direction.tolist()
            fitted["raw_orthogonal_rmse_px"] = raw_fit["orthogonal_rmse_px"]
        lines.append(fitted)

    horizontal = [row for row in lines if row["axis"] == "horizontal"]
    vertical = [row for row in lines if row["axis"] == "vertical"]
    residuals = [float(row["residual_clockwise_degrees"]) for row in lines]
    result: Dict[str, Any] = {
        "schema_version": "1.0",
        "method": "broad_orthogonal_edges_subpixel_scharr_robust_fit",
        "maximum_correction_degrees": float(maximum_correction_degrees),
        "lines": lines,
        "failures": failures,
        "status": "unavailable",
        "applied": False,
    }
    if not horizontal or not vertical:
        result["reason"] = "both_horizontal_and_vertical_edges_are_required"
        return result
    residual = float(np.median(np.asarray(residuals, dtype=np.float64)))
    spread = float(
        np.percentile(np.abs(np.asarray(residuals) - residual), 95)
    )
    support = float(np.mean([row["support_fraction"] for row in lines]))
    confidence = float(
        np.clip(support * math.exp(-spread / 0.35), 0.0, 1.0)
    )
    result.update(
        residual_clockwise_degrees=residual,
        correction_counterclockwise_degrees=residual,
        line_angle_p95_deviation_degrees=spread,
        mean_support_fraction=support,
        confidence=confidence,
    )
    if abs(residual) >= float(maximum_correction_degrees):
        result.update(
            status="rejected_charuco_disagreement",
            reason="residual_rotation_is_not_smaller_than_safety_limit",
        )
        return result
    result.update(status="accepted", applied=True)
    if output_to_raw_maps is not None:
        right = _mean_direction(horizontal)
        down = _mean_direction(vertical)
        if right is not None:
            result["panel_right_unit_vector_full_sensor_camera_xy"] = right
        if down is not None:
            up = [-float(down[0]), -float(down[1])]
            result["panel_up_unit_vector_full_sensor_camera_xy"] = up
            result["camera_up_to_panel_up_clockwise_degrees"] = float(
                math.degrees(math.atan2(up[0], -up[1])) % 360.0
            )
    return result


def aggregate_panel_axis_measurements(
    measurements: Sequence[Mapping[str, Any]],
    *,
    maximum_correction_degrees: float = 30.0,
) -> Dict[str, Any]:
    """Aggregate independent frames without turning failure into a rig gate."""

    accepted = [row for row in measurements if row.get("status") == "accepted"]
    if not accepted:
        return {
            "schema_version": "1.0",
            "status": "unavailable",
            "applied": False,
            "non_gating": True,
            "reason": "no_frame_produced_a_usable_axis_measurement",
            "attempted_frames": len(measurements),
            "maximum_correction_degrees": float(maximum_correction_degrees),
        }
    angles = np.asarray(
        [float(row["residual_clockwise_degrees"]) for row in accepted],
        dtype=np.float64,
    )
    residual = float(np.median(angles))
    temporal_p95 = float(np.percentile(np.abs(angles - residual), 95))
    selected = min(
        accepted,
        key=lambda row: abs(float(row["residual_clockwise_degrees"]) - residual),
    )
    result = {
        key: value
        for key, value in dict(selected).items()
        if key not in ("lines", "failures")
    }
    result.update(
        schema_version="1.0",
        status="accepted",
        applied=abs(residual) < float(maximum_correction_degrees),
        non_gating=True,
        residual_clockwise_degrees=residual,
        correction_counterclockwise_degrees=residual,
        attempted_frames=len(measurements),
        accepted_frames=len(accepted),
        temporal_p95_deviation_degrees=temporal_p95,
        per_frame_residual_clockwise_degrees=angles.tolist(),
        representative_lines=list(selected.get("lines") or []),
        frame_failures=[
            dict(row) for row in measurements if row.get("status") != "accepted"
        ],
    )
    if not result["applied"]:
        result.update(
            status="rejected_charuco_disagreement",
            reason="aggregate_rotation_is_not_smaller_than_safety_limit",
        )
    return result


def panel_axis_correction_matrix(
    output_size_px: Sequence[int], correction_counterclockwise_degrees: float
) -> np.ndarray:
    """Return an output-space correction suitable for left composition."""

    width, height = map(int, output_size_px)
    affine = cv2.getRotationMatrix2D(
        ((width - 1) / 2.0, (height - 1) / 2.0),
        float(correction_counterclockwise_degrees),
        1.0,
    )
    return np.vstack([affine, [0.0, 0.0, 1.0]]).astype(np.float64)


def refine_camera_to_screen_matrix(
    camera_to_screen_3x3: Sequence[Sequence[float]],
    output_origin_screen_xy: Sequence[float],
    screen_units_per_output_pixel_xy: Sequence[float],
    correction_output_3x3: Sequence[Sequence[float]],
) -> np.ndarray:
    """Fold an output-space axis observation into camera-to-screen geometry.

    The broad-edge observation is measured after the ChArUco mapping has been
    normalized into the adapter output raster.  Conjugating that correction
    through the screen-to-output mapping makes it part of the single
    authoritative camera-to-phone transform.  Consumers therefore do not
    need a second panel-axis coordinate convention.
    """

    origin_x, origin_y = map(float, output_origin_screen_xy)
    scale_x, scale_y = map(float, screen_units_per_output_pixel_xy)
    if min(scale_x, scale_y) <= 0.0:
        raise ValueError("Screen units per output pixel must be positive")
    camera_to_screen = np.asarray(camera_to_screen_3x3, dtype=np.float64)
    correction = np.asarray(correction_output_3x3, dtype=np.float64)
    if camera_to_screen.shape != (3, 3) or correction.shape != (3, 3):
        raise ValueError("Panel-axis refinement requires finite 3x3 transforms")
    if not np.isfinite(camera_to_screen).all() or not np.isfinite(correction).all():
        raise ValueError("Panel-axis refinement requires finite 3x3 transforms")
    screen_to_output = np.asarray(
        [
            [1.0 / scale_x, 0.0, -origin_x / scale_x],
            [0.0, 1.0 / scale_y, -origin_y / scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    refined = (
        np.linalg.inv(screen_to_output)
        .dot(correction)
        .dot(screen_to_output)
        .dot(camera_to_screen)
    )
    return refined / refined[2, 2]


def render_panel_axis_evidence(
    rectified_bgr: np.ndarray, measurement: Mapping[str, Any]
) -> np.ndarray:
    """Overlay detected samples and fitted axes in the measured output space."""

    canvas = np.asarray(rectified_bgr).copy()
    height, width = canvas.shape[:2]
    colors = {"horizontal": (0, 220, 255), "vertical": (255, 180, 0)}
    for line in measurement.get("representative_lines", measurement.get("lines", [])):
        color = colors.get(str(line.get("axis")), (255, 255, 255))
        for point in line.get("detected_points_output_xy", []):
            cv2.circle(
                canvas,
                tuple(np.rint(point).astype(int)),
                1,
                color,
                -1,
                cv2.LINE_AA,
            )
        point = np.asarray(line.get("point_xy"), dtype=np.float64)
        direction = np.asarray(line.get("direction_unit_xy"), dtype=np.float64)
        if point.shape == (2,) and direction.shape == (2,):
            extent = float(max(width, height))
            first = np.rint(point - extent * direction).astype(int)
            second = np.rint(point + extent * direction).astype(int)
            cv2.line(canvas, tuple(first), tuple(second), color, 2, cv2.LINE_AA)
    label = "Panel residual {} deg CW | {}".format(
        (
            "{:+.4f}".format(float(measurement["residual_clockwise_degrees"]))
            if measurement.get("residual_clockwise_degrees") is not None
            else "unavailable"
        ),
        measurement.get("status", "unknown"),
    )
    cv2.rectangle(canvas, (0, 0), (min(width - 1, 650), 34), (16, 16, 16), -1)
    cv2.putText(
        canvas,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas
