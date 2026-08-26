"""Pixel-faithful inspection and review-image helpers."""

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .contracts import GeometryEstimate


def extract_one_to_one_patch(
    image: np.ndarray,
    center_xy: Sequence[float],
    size_px: Sequence[int],
    border_value: Sequence[int] = (0, 0, 0),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Crop source samples without resampling, padding only outside the raster."""

    if image is None or image.size == 0:
        raise ValueError("Inspection image is empty")
    width, height = map(int, size_px)
    if width <= 0 or height <= 0:
        raise ValueError("Inspection size must be positive")
    center_x, center_y = map(float, center_xy)
    left = int(round(center_x - (width - 1) / 2.0))
    top = int(round(center_y - (height - 1) / 2.0))
    right = left + width
    bottom = top + height
    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(image.shape[1], right)
    source_bottom = min(image.shape[0], bottom)
    if image.ndim == 2:
        fill = int(border_value[0]) if border_value else 0
        patch = np.full((height, width), fill, dtype=image.dtype)
    else:
        channels = image.shape[2]
        fill = np.asarray(list(border_value)[:channels], dtype=image.dtype)
        if fill.size != channels:
            fill = np.zeros(channels, dtype=image.dtype)
        patch = np.empty((height, width, channels), dtype=image.dtype)
        patch[...] = fill
    if source_right > source_left and source_bottom > source_top:
        destination_left = source_left - left
        destination_top = source_top - top
        patch[
            destination_top : destination_top + source_bottom - source_top,
            destination_left : destination_left + source_right - source_left,
        ] = image[source_top:source_bottom, source_left:source_right]
    return patch, {
        "source_space_rect_xywh": [left, top, width, height],
        "copied_source_rect_xyxy": [
            source_left,
            source_top,
            source_right,
            source_bottom,
        ],
        "zoom": "1:1",
        "interpolation": "none",
    }


def nearest_neighbor_magnify(patch: np.ndarray, zoom: int) -> np.ndarray:
    if int(zoom) != zoom or zoom < 1:
        raise ValueError("Inspection zoom must be a positive integer")
    if zoom == 1:
        return patch.copy()
    return cv2.resize(
        patch,
        (patch.shape[1] * int(zoom), patch.shape[0] * int(zoom)),
        interpolation=cv2.INTER_NEAREST,
    )


def render_geometry_overlay(
    image: np.ndarray,
    geometry: GeometryEstimate,
    camera_points_xy: Optional[Sequence[Sequence[float]]] = None,
) -> np.ndarray:
    """Render the fitted screen boundary, corner inliers, and core metrics."""

    if image.ndim == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()
    polygon = np.round(geometry.screen_polygon_input_xy).astype(np.int32)
    cv2.polylines(overlay, [polygon], True, (70, 230, 110), 2, cv2.LINE_AA)
    if camera_points_xy is not None:
        points = np.asarray(camera_points_xy, dtype=np.float64).reshape((-1, 2))
        for index, point in enumerate(points):
            inlier = index < len(geometry.inlier_mask) and geometry.inlier_mask[index]
            color = (80, 220, 255) if inlier else (70, 70, 255)
            cv2.circle(overlay, tuple(np.round(point).astype(int)), 3, color, -1, cv2.LINE_AA)
    labels = [
        "coverage {:.1%}".format(geometry.metrics["screen_coverage"]),
        "utilization {:.1%}".format(geometry.metrics["camera_utilization"]),
        "IoU {:.1%}".format(geometry.metrics["screen_view_iou"]),
        "RMSE {:.2f}px".format(geometry.metrics["reprojection_rmse_px"]),
        "confidence {:.1%}".format(geometry.confidence),
    ]
    for row, label in enumerate(labels):
        cv2.putText(
            overlay,
            label,
            (12, 24 + row * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return overlay


def render_matchability_curve(
    result: Mapping[str, Any], size_px: Sequence[int] = (900, 480)
) -> np.ndarray:
    width, height = map(int, size_px)
    canvas = np.full((height, width, 3), 22, dtype=np.uint8)
    summary = result.get("scale_summary", [])
    if not summary:
        return canvas
    margin = 58
    cells = np.asarray([item["detail_cells_across"] for item in summary], dtype=float)
    rates = np.asarray([item["success_rate"] for item in summary], dtype=float)
    x_min, x_max = float(np.min(cells)), float(np.max(cells))
    points = []
    for cell, rate in zip(cells, rates):
        x = margin + int(round((cell - x_min) / max(x_max - x_min, 1.0) * (width - 2 * margin)))
        y = height - margin - int(round(rate * (height - 2 * margin)))
        points.append((x, y))
    cv2.line(canvas, (margin, margin), (margin, height - margin), (170, 170, 170), 1)
    cv2.line(canvas, (margin, height - margin), (width - margin, height - margin), (170, 170, 170), 1)
    threshold = float(result.get("reliability_threshold", 0.95))
    threshold_y = height - margin - int(round(threshold * (height - 2 * margin)))
    cv2.line(canvas, (margin, threshold_y), (width - margin, threshold_y), (70, 190, 255), 1)
    if len(points) > 1:
        cv2.polylines(canvas, [np.asarray(points, np.int32)], False, (90, 230, 120), 2, cv2.LINE_AA)
    for point, cell, rate in zip(points, cells, rates):
        cv2.circle(canvas, point, 4, (90, 230, 120), -1, cv2.LINE_AA)
        cv2.putText(canvas, "{}:{:.0%}".format(int(cell), rate), (point[0] - 18, point[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, str(result.get("metric", "matchability")), (margin, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def render_latency_timeline(
    result: Mapping[str, Any], size_px: Sequence[int] = (1000, 500)
) -> np.ndarray:
    width, height = map(int, size_px)
    canvas = np.full((height, width, 3), 22, dtype=np.uint8)
    transitions = result.get("transitions", [])
    if not transitions:
        return canvas
    start = min(item["control_time_ns"] for item in transitions)
    end = max(item["observation_time_ns"] for item in transitions)
    margin = 55

    def x_position(time_ns: int) -> int:
        return margin + int(round((time_ns - start) / max(end - start, 1) * (width - 2 * margin)))

    for index, transition in enumerate(transitions):
        y = margin + int(round(index / max(len(transitions) - 1, 1) * (height - 2 * margin)))
        control_x = x_position(int(transition["control_time_ns"]))
        observation_x = x_position(int(transition["observation_time_ns"]))
        cv2.line(canvas, (control_x, y), (observation_x, y), (80, 180, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, (control_x, y), 2, (80, 80, 255), -1)
        cv2.circle(canvas, (observation_x, y), 2, (80, 230, 120), -1)
    cv2.putText(canvas, "control", (margin, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "perception", (margin + 110, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 230, 120), 1, cv2.LINE_AA)
    cv2.putText(canvas, "median {:.1f} ms  p95 {:.1f} ms".format(float(result["median_ns"]) / 1.0e6, float(result["p95_ns"]) / 1.0e6), (width - 360, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas
