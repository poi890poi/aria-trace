"""Review evidence for already-aligned phone and rig-camera images."""

from __future__ import annotations

from typing import Dict, Sequence

import cv2
import numpy as np


DEFAULT_MAXIMUM_RESIDUAL_TRANSLATION_PX = 3.0


def _multi_threshold_residual_translation(
    adb_blur: np.ndarray,
    hik_blur: np.ndarray,
    selected: np.ndarray,
    adb_values: np.ndarray,
    hik_values: np.ndarray,
    percentiles: Sequence[int] = (30, 50, 70),
) -> dict:
    """Measure HIK's residual XY offset from ADB without changing the mapping.

    The independent percentile masks make this insensitive to exposure, gamma,
    and most color differences.  Phase correlation is used only as a
    diagnostic: callers must repair the owning spatial transform rather than
    applying this residual as an untracked crop correction.
    """

    points = cv2.findNonZero(selected.astype(np.uint8))
    if points is None:
        return {"status": "ineligible", "reason": "empty_valid_mask"}
    x, y, width, height = cv2.boundingRect(points)
    if min(width, height) < 16:
        return {"status": "ineligible", "reason": "valid_region_too_small"}
    support = selected[y : y + height, x : x + width].astype(np.float32)
    window = cv2.createHanningWindow((width, height), cv2.CV_32F) * support
    estimates = []
    for percentile in percentiles:
        adb_threshold = float(np.percentile(adb_values, percentile))
        hik_threshold = float(np.percentile(hik_values, percentile))
        adb_binary = (
            adb_blur[y : y + height, x : x + width] > adb_threshold
        ).astype(np.float32)
        hik_binary = (
            hik_blur[y : y + height, x : x + width] > hik_threshold
        ).astype(np.float32)
        active = support > 0
        adb_occupancy = float(np.mean(adb_binary[active]))
        hik_occupancy = float(np.mean(hik_binary[active]))
        if not (
            0.015 <= adb_occupancy <= 0.985
            and 0.015 <= hik_occupancy <= 0.985
        ):
            continue
        adb_signal = (adb_binary - adb_occupancy) * window
        hik_signal = (hik_binary - hik_occupancy) * window
        shift, response = cv2.phaseCorrelate(adb_signal, hik_signal)
        shift_xy = np.asarray(shift, dtype=np.float64)
        if not np.all(np.isfinite(shift_xy)) or not np.isfinite(response):
            continue
        estimates.append(
            {
                "percentile": int(percentile),
                "hik_offset_xy_px_from_adb": shift_xy.tolist(),
                "response": float(response),
            }
        )
    if len(estimates) < 2:
        return {
            "status": "ineligible",
            "reason": "fewer_than_two_usable_threshold_levels",
            "threshold_estimates": estimates,
        }
    shifts = np.asarray(
        [item["hik_offset_xy_px_from_adb"] for item in estimates],
        dtype=np.float64,
    )
    median_shift = np.median(shifts, axis=0)
    deviations = np.linalg.norm(shifts - median_shift, axis=1)
    response = float(np.median([item["response"] for item in estimates]))
    magnitude = float(np.linalg.norm(median_shift))
    consensus_spread = float(np.median(deviations))
    reliable = response >= 0.03 and consensus_spread <= 3.0
    return {
        "status": "measured" if reliable else "inconclusive",
        "hik_offset_xy_px_from_adb": median_shift.tolist(),
        "hik_correction_xy_px_to_adb": (-median_shift).tolist(),
        "magnitude_px": magnitude,
        "threshold_consensus_spread_px": consensus_spread,
        "median_phase_response": response,
        "threshold_estimates": estimates,
    }


def cross_source_alignment_warning(
    metrics: dict,
    maximum_residual_translation_px: float = DEFAULT_MAXIMUM_RESIDUAL_TRANSLATION_PX,
) -> str:
    """Return a clear spatial-contract warning, or an empty string."""

    residual = dict(metrics.get("residual_translation") or {})
    if residual.get("status") == "measured":
        magnitude = float(residual.get("magnitude_px", 0.0))
        if magnitude > float(maximum_residual_translation_px):
            offset = residual.get("hik_offset_xy_px_from_adb") or [0.0, 0.0]
            return (
                "Declared ADB/HIK space conversion is displaced by "
                "{:.2f}px (HIK relative to ADB: dx={:.2f}, dy={:.2f}); "
                "repair or recompose the owning rig/game transform before use"
            ).format(magnitude, float(offset[0]), float(offset[1]))
    if residual.get("status") in ("ineligible", "inconclusive"):
        return (
            "ADB/HIK residual translation is {} ({}); the declared space "
            "conversion was not independently verified from image features"
        ).format(residual.get("status"), residual.get("reason", "weak consensus"))
    return ""


def cross_source_alignment_evidence(
    adb_crop: np.ndarray,
    hik_rectified: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[dict, Dict[str, np.ndarray]]:
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
        adb_binary = adb_blur > adb_threshold
        hik_binary = hik_blur > hik_threshold
        adb_occupancy = float(np.mean(adb_binary[selected]))
        hik_occupancy = float(np.mean(hik_binary[selected]))
        threshold_occupancies.append([adb_occupancy, hik_occupancy])
        threshold_agreements.append(
            float(np.mean(adb_binary[selected] == hik_binary[selected]))
        )
    binary_agreement = float(np.median(threshold_agreements))

    residual_translation = _multi_threshold_residual_translation(
        adb_blur,
        hik_blur,
        selected,
        adb_values,
        hik_values,
    )

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
    appearance_confidence = float(
        np.clip(
            (0.45 * max(0.0, correlation)
            + 0.25 * binary_agreement
            + 0.30 * edge_overlap),
            0.0,
            1.0,
        )
    )
    spatial_alignment_factor = 1.0
    if residual_translation.get("status") == "measured":
        residual_px = float(residual_translation["magnitude_px"])
        spatial_alignment_factor = float(
            np.exp(
                -max(
                    0.0,
                    residual_px - DEFAULT_MAXIMUM_RESIDUAL_TRANSLATION_PX,
                )
                / DEFAULT_MAXIMUM_RESIDUAL_TRANSLATION_PX
            )
        )
    confidence = appearance_confidence * spatial_alignment_factor

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
    translation_overlay = overlay.copy()
    if residual_translation.get("status") == "measured":
        dx, dy = residual_translation["hik_offset_xy_px_from_adb"]
        selected_points = cv2.findNonZero(selected.astype(np.uint8))
        box_x, box_y, box_width, box_height = cv2.boundingRect(selected_points)
        origin = (box_x + box_width // 2, box_y + box_height // 2)
        endpoint = (
            int(round(origin[0] + float(dx))),
            int(round(origin[1] + float(dy))),
        )
        cv2.arrowedLine(
            translation_overlay,
            origin,
            endpoint,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
            tipLength=0.25,
        )
        cv2.putText(
            translation_overlay,
            "HIK offset dx={:.1f} dy={:.1f}px".format(float(dx), float(dy)),
            (max(4, box_x), max(20, box_y + box_height - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    side_by_side = np.hstack([adb_crop, hik_rectified])
    metrics = {
        "confidence": confidence,
        "appearance_confidence": appearance_confidence,
        "spatial_alignment_factor": spatial_alignment_factor,
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
        "residual_translation": residual_translation,
    }
    images = {
        "adb_visible_crop.png": adb_crop,
        "hik_rectified.png": hik_rectified,
        "edge_overlay_adb_red_hik_cyan.png": overlay,
        "residual_translation_overlay.png": translation_overlay,
        "normalized_difference_heatmap.png": heatmap,
        "side_by_side_adb_then_hik.png": side_by_side,
        "valid_mask.png": np.asarray(valid_mask, dtype=np.uint8),
    }
    return metrics, images
