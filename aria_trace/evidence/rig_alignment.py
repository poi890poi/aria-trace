"""Review evidence for already-aligned phone and rig-camera images."""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


def cross_source_alignment_evidence(
    adb_crop: np.ndarray,
    hik_rectified: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Compare already-aligned ADB and HIK views without fitting a transform."""

    if adb_crop is None or hik_rectified is None:
        raise ValueError("Cross-source images are required")
    if adb_crop.shape[:2] != hik_rectified.shape[:2]:
        raise ValueError(
            "Cross-source image sizes differ: {} versus {}".format(
                adb_crop.shape[:2], hik_rectified.shape[:2]
            )
        )
    if valid_mask.shape[:2] != adb_crop.shape[:2]:
        raise ValueError("Cross-source valid mask has the wrong size")
    selected = np.asarray(valid_mask) > 0
    if int(np.count_nonzero(selected)) < 64:
        raise ValueError("Cross-source valid mask contains too few pixels")

    adb_gray = cv2.cvtColor(adb_crop, cv2.COLOR_BGR2GRAY)
    hik_gray = cv2.cvtColor(hik_rectified, cv2.COLOR_BGR2GRAY)
    adb_blur = cv2.GaussianBlur(adb_gray, (5, 5), 0.8)
    hik_blur = cv2.GaussianBlur(hik_gray, (5, 5), 0.8)
    adb_values = adb_blur[selected].astype(np.float64)
    hik_values = hik_blur[selected].astype(np.float64)
    if min(float(np.std(adb_values)), float(np.std(hik_values))) <= 1.0e-6:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(adb_values, hik_values)[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0

    # Cross-source game evidence can differ substantially in exposure, gamma,
    # and white balance.  Percentile masks compare geometry instead of DN, and
    # several occupancies prevent one sparse/full threshold from dominating.
    threshold_agreements = []
    threshold_occupancies = []
    for percentile in (30, 50, 70):
        adb_threshold = float(np.percentile(adb_values, percentile))
        hik_threshold = float(np.percentile(hik_values, percentile))
        adb_binary = adb_blur >= adb_threshold
        hik_binary = hik_blur >= hik_threshold
        adb_occupancy = float(np.mean(adb_binary[selected]))
        hik_occupancy = float(np.mean(hik_binary[selected]))
        threshold_occupancies.append([adb_occupancy, hik_occupancy])
        threshold_agreements.append(
            float(np.mean(adb_binary[selected] == hik_binary[selected]))
        )
    binary_agreement = float(np.median(threshold_agreements))

    def adaptive_edges(image, values):
        median = float(np.median(values))
        lower = int(max(8, min(180, round(0.55 * median))))
        upper = int(max(lower + 8, min(245, round(1.45 * median))))
        return cv2.Canny(image, lower, upper) > 0

    adb_edges = adaptive_edges(adb_blur, adb_values)
    hik_edges = adaptive_edges(hik_blur, hik_values)
    kernel = np.ones((3, 3), np.uint8)
    adb_dilated = cv2.dilate(adb_edges.astype(np.uint8), kernel) > 0
    hik_dilated = cv2.dilate(hik_edges.astype(np.uint8), kernel) > 0
    adb_edge_count = int(np.count_nonzero(adb_edges & selected))
    hik_edge_count = int(np.count_nonzero(hik_edges & selected))
    adb_supported = (
        float(np.count_nonzero(adb_edges & hik_dilated & selected)) / adb_edge_count
        if adb_edge_count
        else 0.0
    )
    hik_supported = (
        float(np.count_nonzero(hik_edges & adb_dilated & selected)) / hik_edge_count
        if hik_edge_count
        else 0.0
    )
    edge_overlap = 0.5 * (adb_supported + hik_supported)
    adb_edge_density = float(adb_edge_count) / float(np.count_nonzero(selected))
    hik_edge_density = float(hik_edge_count) / float(np.count_nonzero(selected))
    occupancy_valid = all(
        0.015 <= item <= 0.985
        for pair in threshold_occupancies
        for item in pair
    )
    edge_density_valid = (
        0.001 <= adb_edge_density <= 0.35
        and 0.001 <= hik_edge_density <= 0.35
    )
    information_quality = float(
        np.clip(
            min(adb_edge_density, hik_edge_density) / 0.015,
            0.0,
            1.0,
        )
    ) if occupancy_valid and edge_density_valid else 0.0
    confidence = float(
        np.clip(
            (0.45 * max(0.0, correlation)
            + 0.25 * binary_agreement
            + 0.30 * edge_overlap),
            0.0,
            1.0,
        )
    )

    adb_normalized = cv2.normalize(adb_blur, None, 0, 255, cv2.NORM_MINMAX)
    hik_normalized = cv2.normalize(hik_blur, None, 0, 255, cv2.NORM_MINMAX)
    difference = cv2.absdiff(adb_normalized, hik_normalized)
    heatmap = cv2.applyColorMap(difference, cv2.COLORMAP_TURBO)
    overlay = np.zeros((*adb_gray.shape, 3), np.uint8)
    overlay[:, :, 2] = adb_edges.astype(np.uint8) * 255
    overlay[:, :, 0] = hik_edges.astype(np.uint8) * 255
    overlay[:, :, 1] = hik_edges.astype(np.uint8) * 255
    invalid = ~selected
    heatmap[invalid] = 0
    overlay[invalid] = 0
    side_by_side = np.hstack([adb_crop, hik_rectified])
    metrics = {
        "confidence": confidence,
        "grayscale_correlation": correlation,
        "binary_agreement": binary_agreement,
        "threshold_agreements": threshold_agreements,
        "threshold_occupancies": threshold_occupancies,
        "edge_overlap": edge_overlap,
        "adb_edge_supported_fraction": adb_supported,
        "hik_edge_supported_fraction": hik_supported,
        "valid_pixel_fraction": float(np.mean(selected)),
        "adb_edge_density": adb_edge_density,
        "hik_edge_density": hik_edge_density,
        "information_quality": information_quality,
        "information_eligible": bool(information_quality > 0.0),
    }
    images = {
        "adb_visible_crop.png": adb_crop,
        "hik_rectified.png": hik_rectified,
        "edge_overlay_adb_red_hik_cyan.png": overlay,
        "normalized_difference_heatmap.png": heatmap,
        "side_by_side_adb_then_hik.png": side_by_side,
        "valid_mask.png": np.asarray(valid_mask, dtype=np.uint8),
    }
    return metrics, images
