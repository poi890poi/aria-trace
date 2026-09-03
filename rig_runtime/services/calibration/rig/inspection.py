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
    quality_region_screen_xywh: Optional[Sequence[float]] = None,
    canonical_screen_size_px: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Render the inferred screen, atlas inliers, visible quality ROI, and IoU."""

    if image.ndim == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()
    polygon = np.round(geometry.screen_polygon_input_xy).astype(np.int32)
    cv2.polylines(overlay, [polygon], True, (70, 230, 110), 2, cv2.LINE_AA)
    if quality_region_screen_xywh is not None:
        x, y, width, height = map(float, quality_region_screen_xywh)
        screen_patch = np.asarray(
            [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
            dtype=np.float64,
        )
        homogeneous = np.column_stack(
            [screen_patch, np.ones(len(screen_patch), dtype=np.float64)]
        )
        projected = homogeneous.dot(geometry.inverse_matrix_3x3.T)
        projected = projected[:, :2] / projected[:, 2:3]
        cv2.polylines(
            overlay,
            [np.round(projected).astype(np.int32)],
            True,
            (255, 190, 70),
            2,
            cv2.LINE_AA,
        )
    if camera_points_xy is not None:
        points = np.asarray(camera_points_xy, dtype=np.float64).reshape((-1, 2))
        for index, point in enumerate(points):
            inlier = index < len(geometry.inlier_mask) and geometry.inlier_mask[index]
            color = (80, 220, 255) if inlier else (70, 70, 255)
            cv2.circle(overlay, tuple(np.round(point).astype(int)), 3, color, -1, cv2.LINE_AA)
    if (
        canonical_screen_size_px is not None
        and overlay.shape[1] >= 320
        and overlay.shape[0] >= 240
    ):
        screen_width, screen_height = map(float, canonical_screen_size_px)
        screen = np.asarray(
            [[0.0, 0.0], [screen_width, 0.0], [screen_width, screen_height], [0.0, screen_height]],
            dtype=np.float64,
        )
        viewport = np.asarray(
            geometry.viewport_polygon_screen_xy, dtype=np.float64
        ).reshape((-1, 2))
        union = np.vstack([screen, viewport])
        low = np.min(union, axis=0)
        high = np.max(union, axis=0)
        extent = np.maximum(high - low, 1.0)
        panel_width = min(300, max(170, int(round(overlay.shape[1] * 0.34))))
        panel_height = min(250, max(150, int(round(overlay.shape[0] * 0.44))))
        panel = np.full((panel_height, panel_width, 3), 22, dtype=np.uint8)
        margin, label_height = 10, 28
        scale = min(
            (panel_width - margin * 2) / extent[0],
            (panel_height - label_height - margin) / extent[1],
        )
        offset = np.asarray(
            [
                (panel_width - extent[0] * scale) / 2.0,
                label_height + (panel_height - label_height - extent[1] * scale) / 2.0,
            ],
            dtype=np.float64,
        )

        def panel_points(points: np.ndarray) -> np.ndarray:
            return np.round((points - low) * scale + offset).astype(np.int32)

        screen_panel = panel_points(screen)
        viewport_panel = panel_points(viewport)
        fills = panel.copy()
        cv2.fillConvexPoly(fills, screen_panel, (35, 88, 50), cv2.LINE_AA)
        cv2.fillConvexPoly(fills, viewport_panel, (110, 72, 35), cv2.LINE_AA)
        panel = cv2.addWeighted(panel, 0.30, fills, 0.70, 0.0)
        cv2.polylines(panel, [screen_panel], True, (70, 230, 110), 2, cv2.LINE_AA)
        cv2.polylines(panel, [viewport_panel], True, (70, 175, 255), 2, cv2.LINE_AA)
        if quality_region_screen_xywh is not None:
            x, y, width, height = map(float, quality_region_screen_xywh)
            quality = np.asarray(
                [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
                dtype=np.float64,
            )
            cv2.polylines(
                panel,
                [panel_points(quality)],
                True,
                (255, 190, 70),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            panel,
            "screen / camera viewport / quality patch",
            (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        top, left = 8, overlay.shape[1] - panel_width - 8
        overlay[top : top + panel_height, left : left + panel_width] = panel
    labels = [
        "coverage {:.1%}".format(geometry.metrics["screen_coverage"]),
        "utilization {:.1%}".format(geometry.metrics["camera_utilization"]),
        "IoU {:.1%}".format(geometry.metrics["screen_view_iou"]),
        "quality ROI = visible task intersection",
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


def render_esfr_curve(
    result: Mapping[str, Any], size_px: Sequence[int] = (1000, 560)
) -> np.ndarray:
    """Render display-referred conservative and median e-SFR curves."""

    width, height = map(int, size_px)
    canvas = np.full((height, width, 3), 22, dtype=np.uint8)
    display = result.get("display_referred", {})
    frequency = np.asarray(display.get("frequency", []), dtype=np.float64)
    conservative = np.asarray(
        display.get("mtf_conservative", display.get("mtf", [])), dtype=np.float64
    )
    median = np.asarray(display.get("mtf_median", []), dtype=np.float64)
    if not len(frequency) or len(conservative) != len(frequency):
        return canvas
    margin_left, margin_right, margin_top, margin_bottom = 72, 28, 48, 62
    plot_width = max(1, width - margin_left - margin_right)
    plot_height = max(1, height - margin_top - margin_bottom)

    def point(freq: float, response: float) -> Tuple[int, int]:
        x = margin_left + int(round(np.clip(freq / 0.5, 0.0, 1.0) * plot_width))
        y = margin_top + int(round((1.0 - np.clip(response, 0.0, 1.2) / 1.2) * plot_height))
        return x, y

    cv2.rectangle(
        canvas,
        (margin_left, margin_top),
        (margin_left + plot_width, margin_top + plot_height),
        (110, 110, 110),
        1,
    )
    for level in (0.1, 0.5, 1.0):
        first = point(0.0, level)
        second = point(0.5, level)
        cv2.line(canvas, first, second, (62, 62, 62), 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "{:.1f}".format(level),
            (12, first[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
    for freq in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        location = point(freq, 0.0)
        cv2.putText(
            canvas,
            "{:.1f}".format(freq),
            (location[0] - 12, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (190, 190, 190),
            1,
            cv2.LINE_AA,
        )
    if len(median) == len(frequency):
        median_points = np.asarray(
            [point(float(x), float(y)) for x, y in zip(frequency, median)],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [median_points], False, (90, 190, 255), 2, cv2.LINE_AA)
    conservative_points = np.asarray(
        [point(float(x), float(y)) for x, y in zip(frequency, conservative)],
        dtype=np.int32,
    )
    cv2.polylines(canvas, [conservative_points], False, (80, 230, 120), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Display-referred e-SFR / MTF (cy/dpx)",
        (margin_left, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    mtf50 = display.get("mtf50_conservative", display.get("mtf50"))
    mtf10 = display.get("mtf10_conservative", display.get("mtf10"))
    label = "MTF50 {}   MTF10 {}   display Nyquist 0.500".format(
        "{:.3f}".format(float(mtf50)) if mtf50 is not None else ">0.500",
        "{:.3f}".format(float(mtf10)) if mtf10 is not None else ">0.500",
    )
    cv2.putText(
        canvas,
        label,
        (margin_left, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    return canvas


def render_feature_matching_curve(
    result: Mapping[str, Any], size_px: Sequence[int] = (1000, 560)
) -> np.ndarray:
    """Render repeatability, matching score, and MMA by display-pixel error."""

    width, height = map(int, size_px)
    canvas = np.full((height, width, 3), 22, dtype=np.uint8)
    series = (
        ("repeatability", result.get("repeatability_by_threshold_px", {}), (80, 230, 120)),
        ("matching score", result.get("matching_score_by_threshold_px", {}), (90, 190, 255)),
        ("MMA", result.get("mma_by_threshold_px", {}), (235, 130, 80)),
    )
    thresholds = sorted(
        {int(key) for _, values, _ in series for key in values.keys()}
    )
    if not thresholds:
        return canvas
    margin_left, margin_right, margin_top, margin_bottom = 72, 28, 56, 62
    plot_width = max(1, width - margin_left - margin_right)
    plot_height = max(1, height - margin_top - margin_bottom)

    def point(threshold: int, value: float) -> Tuple[int, int]:
        position = (threshold - thresholds[0]) / max(thresholds[-1] - thresholds[0], 1)
        return (
            margin_left + int(round(position * plot_width)),
            margin_top + int(round((1.0 - np.clip(value, 0.0, 1.0)) * plot_height)),
        )

    cv2.rectangle(
        canvas,
        (margin_left, margin_top),
        (margin_left + plot_width, margin_top + plot_height),
        (110, 110, 110),
        1,
    )
    for level in (0.0, 0.5, 1.0):
        y = margin_top + int(round((1.0 - level) * plot_height))
        cv2.line(canvas, (margin_left, y), (margin_left + plot_width, y), (62, 62, 62), 1)
        cv2.putText(canvas, "{:.1f}".format(level), (12, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, cv2.LINE_AA)
    for threshold in thresholds:
        x, _ = point(threshold, 0.0)
        cv2.putText(canvas, str(threshold), (x - 5, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, cv2.LINE_AA)
    for index, (label, values, color) in enumerate(series):
        points = np.asarray(
            [point(threshold, float(values.get(threshold, values.get(str(threshold), 0.0)))) for threshold in thresholds],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (margin_left + index * 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "Ground-truth reprojection threshold (display px)",
        (margin_left, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (225, 225, 225),
        1,
        cv2.LINE_AA,
    )
    return canvas


def render_feature_matching_overlay(
    reference_display_image: np.ndarray,
    observed_camera_image: np.ndarray,
    result: Mapping[str, Any],
    maximum_size_px: Sequence[int] = (1200, 620),
) -> np.ndarray:
    """Render inspectable correct/incorrect ground-truth match examples."""

    def bgr(image: np.ndarray) -> np.ndarray:
        return (
            cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            if image.ndim == 2
            else image.copy()
        )

    reference = bgr(reference_display_image)
    observed = bgr(observed_camera_image)
    maximum_width, maximum_height = map(int, maximum_size_px)
    target_height = min(
        maximum_height - 54,
        max(120, min(reference.shape[0], observed.shape[0])),
    )

    def resize(image: np.ndarray) -> Tuple[np.ndarray, float]:
        scale = target_height / float(image.shape[0])
        width = max(1, int(round(image.shape[1] * scale)))
        return cv2.resize(image, (width, target_height), interpolation=cv2.INTER_AREA), scale

    reference_view, reference_scale = resize(reference)
    observed_view, observed_scale = resize(observed)
    gap = 18
    combined_width = reference_view.shape[1] + gap + observed_view.shape[1]
    if combined_width > maximum_width:
        reduction = maximum_width / float(combined_width)
        target_height = max(80, int(round(target_height * reduction)))
        reference_view, reference_scale = resize(reference)
        observed_view, observed_scale = resize(observed)
        combined_width = reference_view.shape[1] + gap + observed_view.shape[1]
    canvas = np.full((target_height + 54, combined_width, 3), 22, dtype=np.uint8)
    canvas[44 : 44 + target_height, : reference_view.shape[1]] = reference_view
    observed_left = reference_view.shape[1] + gap
    canvas[
        44 : 44 + target_height,
        observed_left : observed_left + observed_view.shape[1],
    ] = observed_view
    for item in result.get("match_examples", []):
        reference_xy = np.asarray(item["reference_display_xy"], dtype=np.float64)
        camera_xy = np.asarray(item["camera_xy"], dtype=np.float64)
        first = (
            int(round(reference_xy[0] * reference_scale)),
            44 + int(round(reference_xy[1] * reference_scale)),
        )
        second = (
            observed_left + int(round(camera_xy[0] * observed_scale)),
            44 + int(round(camera_xy[1] * observed_scale)),
        )
        color = (70, 225, 100) if item["correct_at_primary_threshold"] else (70, 70, 245)
        cv2.line(canvas, first, second, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, first, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, second, 2, color, -1, cv2.LINE_AA)
    cv2.putText(canvas, "display reference", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, "native camera", (observed_left + 8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
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
