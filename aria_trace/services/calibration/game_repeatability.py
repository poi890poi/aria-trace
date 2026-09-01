"""Lighting-tolerant static geometry checks for a prepared Android game."""

from __future__ import annotations

import math
from typing import Mapping, Sequence, Tuple

import cv2
import numpy as np

from aria_trace.domain.spatial import require_spatial_geometry


def fixed_static_features(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return fixed-threshold structure after robust contrast normalization."""

    if image is None or image.size == 0:
        raise ValueError("A non-empty app image is required")
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else np.asarray(image, dtype=np.uint8)
    )
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    low, high = np.percentile(blurred, [5.0, 95.0])
    if high - low < 8.0:
        normalized = blurred.copy()
    else:
        normalized = np.clip(
            (blurred.astype(np.float32) - float(low))
            * (255.0 / float(high - low)),
            0,
            255,
        ).astype(np.uint8)
    binary = cv2.threshold(normalized, 128, 255, cv2.THRESH_BINARY)[1]
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    edges = cv2.Canny(binary, 40, 120)
    return binary, edges


def evaluate_minimap_static_geometry(
    image: np.ndarray,
    crop_xywh: Sequence[int],
    boundary: Mapping[str, object],
    *,
    radial_tolerance_px: float = 6.0,
    minimum_angular_coverage: float = 0.35,
) -> Tuple[dict, dict]:
    """Check only the calibrated static mini-map rim in the current app image."""

    x, y, width, height = map(int, crop_xywh)
    image_height, image_width = image.shape[:2]
    if (
        min(x, y) < 0
        or min(width, height) <= 0
        or x + width > image_width
        or y + height > image_height
    ):
        raise ValueError(
            "Mini-map crop {} exceeds app raster {}x{}".format(
                [x, y, width, height], image_width, image_height
            )
        )
    crop = image[y : y + height, x : x + width].copy()
    binary, edges = fixed_static_features(crop)
    boundary = require_spatial_geometry(
        boundary, "circle", expected_space_id="current_minimap_crop_pixels"
    )
    if boundary["space"]["size_px"] != [width, height]:
        raise ValueError("Mini-map boundary space does not match the current crop")
    center_x = float(boundary["center_x"])
    center_y = float(boundary["center_y"])
    radius = float(boundary["radius"])
    if radius <= 0.0:
        raise ValueError("Mini-map boundary radius must be positive")
    tolerance = max(float(radial_tolerance_px), radius * 0.035)
    angles = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
    radial_offsets = np.arange(-math.ceil(tolerance), math.ceil(tolerance) + 1)
    residuals = []
    detected_points = []
    missed_points = []
    for angle in angles:
        cosine, sine = math.cos(angle), math.sin(angle)
        candidates = []
        for offset in radial_offsets:
            sample_radius = radius + float(offset)
            px = int(round(center_x + sample_radius * cosine))
            py = int(round(center_y + sample_radius * sine))
            if 0 <= px < width and 0 <= py < height and edges[py, px] != 0:
                candidates.append((abs(float(offset)), px, py))
        if candidates:
            residual, px, py = min(candidates)
            residuals.append(residual)
            detected_points.append((px, py))
        else:
            missed_points.append(
                (
                    int(round(center_x + radius * cosine)),
                    int(round(center_y + radius * sine)),
                )
            )
    coverage = len(residuals) / float(len(angles))
    median = float(np.median(residuals)) if residuals else float("inf")
    p95 = float(np.percentile(residuals, 95)) if residuals else float("inf")
    residual_score = (
        math.exp(-median / max(1.0, tolerance * 0.5))
        if math.isfinite(median)
        else 0.0
    )
    score = float(coverage * residual_score)
    matches = bool(
        coverage >= float(minimum_angular_coverage)
        and p95 <= tolerance
    )
    overlay = crop.copy()
    cv2.circle(
        overlay,
        (int(round(center_x)), int(round(center_y))),
        int(round(radius)),
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )
    for point in detected_points:
        cv2.circle(overlay, point, 1, (0, 255, 0), -1)
    for point in missed_points[::6]:
        if 0 <= point[0] < width and 0 <= point[1] < height:
            cv2.circle(overlay, point, 1, (0, 0, 255), -1)
    result = {
        "method": "fixed_threshold_static_minimap_rim_geometry",
        "lighting_invariant": True,
        "dynamic_game_content_used": False,
        "crop_xywh": [x, y, width, height],
        "expected_boundary": {
            "center_xy": [center_x, center_y],
            "radius_px": radius,
        },
        "angular_sample_count": int(len(angles)),
        "detected_sample_count": int(len(residuals)),
        "angular_coverage": float(coverage),
        "minimum_angular_coverage": float(minimum_angular_coverage),
        "radial_residual_median_px": median,
        "radial_residual_p95_px": p95,
        "radial_tolerance_px": float(tolerance),
        "score": score,
        "matches": matches,
    }
    return result, {
        "minimap_current_crop.png": crop,
        "static_features_binary.png": binary,
        "static_features_edges.png": edges,
        "static_geometry_overlay.png": overlay,
    }


def compare_thresholded_app_geometry(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    maximum_shift_px: float = 12.0,
    minimum_score: float = 0.55,
) -> Tuple[dict, dict]:
    """Diagnostic full-app comparison for apps without mini-map profiles."""

    if reference.shape[:2] != current.shape[:2]:
        return (
            {
                "method": "fixed_threshold_full_app_diagnostic",
                "status": "shape_mismatch",
                "reference_size_px": [reference.shape[1], reference.shape[0]],
                "current_size_px": [current.shape[1], current.shape[0]],
                "matches": False,
                "score": 0.0,
            },
            {},
        )
    reference_binary, reference_edges = fixed_static_features(reference)
    current_binary, current_edges = fixed_static_features(current)
    shift, response = cv2.phaseCorrelate(
        reference_edges.astype(np.float32), current_edges.astype(np.float32)
    )
    shift_magnitude = float(math.hypot(shift[0], shift[1]))
    aligned = cv2.warpAffine(
        current_edges,
        np.asarray([[1.0, 0.0, -shift[0]], [0.0, 1.0, -shift[1]]]),
        (current_edges.shape[1], current_edges.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    distance_to_reference = cv2.distanceTransform(
        255 - reference_edges, cv2.DIST_L2, 3
    )
    distance_to_current = cv2.distanceTransform(255 - aligned, cv2.DIST_L2, 3)
    current_points = aligned > 0
    reference_points = reference_edges > 0
    distances = []
    if np.any(current_points):
        distances.extend(distance_to_reference[current_points].tolist())
    if np.any(reference_points):
        distances.extend(distance_to_current[reference_points].tolist())
    median_distance = float(np.median(distances)) if distances else float("inf")
    score = (
        float(max(0.0, response)) * math.exp(-median_distance / 4.0)
        if math.isfinite(median_distance)
        else 0.0
    )
    matches = bool(
        score >= float(minimum_score)
        and shift_magnitude <= float(maximum_shift_px)
    )
    overlay = np.zeros((*reference_edges.shape, 3), np.uint8)
    overlay[reference_edges > 0] = (255, 0, 255)
    overlay[aligned > 0] = np.maximum(overlay[aligned > 0], (0, 255, 0))
    return (
        {
            "method": "fixed_threshold_full_app_diagnostic",
            "status": "compared",
            "lighting_invariant": True,
            "reference_size_px": [reference.shape[1], reference.shape[0]],
            "current_size_px": [current.shape[1], current.shape[0]],
            "estimated_shift_xy_px": [float(shift[0]), float(shift[1])],
            "estimated_shift_magnitude_px": shift_magnitude,
            "maximum_shift_px": float(maximum_shift_px),
            "phase_response": float(response),
            "bidirectional_edge_distance_median_px": median_distance,
            "score": float(score),
            "minimum_score": float(minimum_score),
            "matches": matches,
        },
        {
            "diagnostic_reference_binary.png": reference_binary,
            "diagnostic_current_binary.png": current_binary,
            "diagnostic_reference_edges.png": reference_edges,
            "diagnostic_current_edges.png": current_edges,
            "diagnostic_geometry_overlay.png": overlay,
        },
    )


__all__ = [
    "compare_thresholded_app_geometry",
    "evaluate_minimap_static_geometry",
    "fixed_static_features",
]
