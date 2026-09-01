"""Coarse Android mini-map discovery before the verified boundary fitter."""

from __future__ import annotations

from typing import Mapping, Optional

import cv2
import numpy as np


DEFAULT_ANDROID_DISCOVERY = {
    "center_search_fraction_xy": [0.35, 0.35],
    "radius_fraction_of_short_side": [0.07, 0.22],
    "minimum_visible_circumference_fraction": 0.85,
    "minimum_priority_margin_fraction": 0.04,
}


def discover_android_minimap_crop(
    frames: np.ndarray,
    config: Optional[Mapping[str, object]] = None,
) -> Mapping[str, object]:
    """Return one rough crop/config; never performs the final boundary fit."""

    if frames.ndim != 4 or len(frames) < 4:
        raise ValueError("Automatic mini-map discovery requires at least four frames")
    settings = dict(DEFAULT_ANDROID_DISCOVERY)
    settings.update(dict(config or {}))
    height, width = frames.shape[1:3]
    search_fx, search_fy = [
        float(value) for value in settings["center_search_fraction_xy"]
    ]
    radius_low, radius_high = [
        float(value) for value in settings["radius_fraction_of_short_side"]
    ]
    if not (0.0 < search_fx <= 1.0 and 0.0 < search_fy <= 1.0):
        raise ValueError("Android mini-map center search fractions must be in (0, 1]")
    if not (0.0 < radius_low < radius_high < 0.5):
        raise ValueError("Android mini-map radius fractions are invalid")
    short = min(width, height)
    min_radius = max(6, int(round(radius_low * short)))
    max_radius = max(min_radius + 2, int(round(radius_high * short)))
    search_width = min(width, int(round(width * search_fx)) + max_radius)
    search_height = min(height, int(round(height * search_fy)) + max_radius)

    gray = np.stack(
        [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
    ).astype(np.float32)
    temporal = np.std(gray, axis=0)
    consecutive = np.mean(np.abs(np.diff(gray, axis=0)), axis=0)
    response = 0.55 * temporal + 0.45 * consecutive
    roi = response[:search_height, :search_width]
    normalized = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    normalized = cv2.GaussianBlur(normalized, (7, 7), 1.4)
    high = int(max(20, np.percentile(normalized, 82)))
    low = max(8, int(round(high * 0.45)))
    edges = cv2.Canny(normalized, low, high)
    circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=max(12, min_radius // 2),
        param1=max(40, high),
        param2=12,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        raise ValueError(
            "No circular mini-map candidate was found in the configured Android search area"
        )
    gradient_x = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    candidates = []
    angles = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    center_limit_x = width * search_fx
    center_limit_y = height * search_fy
    minimum_visible = float(settings["minimum_visible_circumference_fraction"])
    for x, y, radius in np.asarray(circles[0], dtype=np.float64):
        if x > center_limit_x or y > center_limit_y:
            continue
        sample_x = np.rint(x + radius * np.cos(angles)).astype(np.int32)
        sample_y = np.rint(y + radius * np.sin(angles)).astype(np.int32)
        visible = (
            (sample_x >= 0) & (sample_x < width)
            & (sample_y >= 0) & (sample_y < height)
        )
        visible_fraction = float(np.mean(visible))
        if visible_fraction < minimum_visible:
            continue
        local = (
            (sample_x >= 0) & (sample_x < search_width)
            & (sample_y >= 0) & (sample_y < search_height)
        )
        if int(np.count_nonzero(local)) < 64:
            continue
        ring_strength = float(np.median(gradient[sample_y[local], sample_x[local]]))
        inner_radius = max(2, int(round(radius * 0.72)))
        mask = np.zeros_like(normalized, dtype=np.uint8)
        cv2.circle(mask, (int(round(x)), int(round(y))), inner_radius, 255, -1)
        inner_values = normalized[mask > 0]
        activity = float(np.std(inner_values)) if len(inner_values) else 0.0
        score = ring_strength + 0.20 * activity
        candidates.append(
            {
                "center_xy": [float(x), float(y)],
                "radius_px": float(radius),
                "visible_circumference_fraction": visible_fraction,
                "ring_strength": ring_strength,
                "interior_activity": activity,
                "score": score,
            }
        )
    if not candidates:
        raise ValueError(
            "Circular candidates failed the configured center/circumference bounds"
        )
    average_gray = cv2.GaussianBlur(
        cv2.cvtColor(
            np.mean(frames, axis=0).astype(np.uint8), cv2.COLOR_BGR2GRAY
        ),
        (5, 5),
        1.1,
    )[:search_height, :search_width]
    average_circles = cv2.HoughCircles(
        average_gray,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(20.0, float(min_radius)),
        param1=70.0,
        param2=18.0,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    average_rows = [] if average_circles is None else average_circles[0]
    for item in candidates:
        x, y = item["center_xy"]
        radius = float(item["radius_px"])
        matches = []
        for average_row in average_rows:
            ax, ay, ar = [float(value) for value in average_row]
            center_delta = float(np.hypot(ax - x, ay - y))
            radius_delta = abs(ar - radius)
            if center_delta <= 0.60 * radius and radius_delta <= 0.28 * radius:
                matches.append(
                    (
                        center_delta / radius + radius_delta / radius,
                        [ax, ay, ar],
                    )
                )
        if matches:
            item["average_image_circle_match"] = min(matches)[1]
    for item in candidates:
        x, y = item["center_xy"]
        radius = float(item["radius_px"])
        top_gap = max(0.0, y - radius)
        item["discovery_priority"] = float(
            (item["score"] / np.sqrt(max(1.0, radius / float(min_radius))))
            * np.exp(-top_gap / max(1.0, 0.20 * search_height))
        )
    candidates.sort(key=lambda item: item["discovery_priority"], reverse=True)

    def hypothesis(item):
        x, y = item["center_xy"]
        radius = float(item["radius_px"])
        config_radius = radius
        positioning_radius = radius
        average_match = item.get("average_image_circle_match")
        if average_match is not None:
            x = float(average_match[0])
            y = float(average_match[1])
            positioning_radius = max(radius, float(average_match[2]))
        # A near-top circle has a shortened Hough arc and commonly biases its
        # coarse center upward. Keep this bounded correction inside the stated
        # >=85% visible-circumference policy; the backend still owns the fit.
        seed_y = float(y)
        if abs(seed_y - positioning_radius) <= 0.40 * positioning_radius:
            seed_y = min(float(height - 1), positioning_radius * 1.05)
        margin = 1.35 * positioning_radius
        left = max(0, int(np.floor(x - margin)))
        top = max(0, int(np.floor(seed_y - margin)))
        right = min(width, int(np.ceil(x + margin)))
        bottom = min(height, int(np.ceil(seed_y + margin)))
        return {
            "crop_xywh": [left, top, right - left, bottom - top],
            "boundary_config": {
                "expected_center_xy": [float(x - left), float(seed_y - top)],
                "center_search_radius_px": float(max(8.0, config_radius * 0.20)),
                "radius_range_px": [
                    float(config_radius * 0.85),
                    float(config_radius * 1.18),
                ],
            },
            "coarse_candidate": item,
        }

    hypotheses = [hypothesis(item) for item in candidates[:12]]
    requested_index = settings.get("candidate_index")
    operator_selected = requested_index is not None
    if operator_selected:
        requested_index = int(requested_index)
        if not 0 <= requested_index < len(hypotheses):
            raise ValueError(
                "Android mini-map candidate_index {} is outside 0..{}".format(
                    requested_index, len(hypotheses) - 1
                )
            )
        hypotheses.insert(0, hypotheses.pop(requested_index))
    best = hypotheses[0]["coarse_candidate"]
    crop = hypotheses[0]["crop_xywh"]
    runner_priority = (
        float(candidates[1]["discovery_priority"])
        if len(candidates) > 1 else 0.0
    )
    priority_margin_fraction = float(
        (float(best["discovery_priority"]) - runner_priority)
        / max(float(best["discovery_priority"]), 1.0e-9)
    ) if not operator_selected else 1.0
    return {
        "schema_version": "1.0",
        "method": "bounded_android_temporal_heatmap_hough_seed",
        "frame_size_px": [width, height],
        "search_bounds": settings,
        "crop_xywh": crop,
        "boundary_config": hypotheses[0]["boundary_config"],
        "selected_candidate": best,
        "operator_selected_candidate": operator_selected,
        "priority_margin_fraction": priority_margin_fraction,
        "candidate_count": len(candidates),
        "review_candidates": candidates[:32],
        "rough_hypotheses": hypotheses,
        "diagnostics": {
            "temporal_heatmap": response,
            "normalized_heatmap": normalized,
            "edges": edges,
        },
    }


__all__ = ["DEFAULT_ANDROID_DISCOVERY", "discover_android_minimap_crop"]
