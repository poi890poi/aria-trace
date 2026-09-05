"""Translation-based full-map stitching with inspectable registration evidence."""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from rig_runtime.services.calibration.minimap.spatial import (
    minimap_crop_space,
    normalize_minimap_geometry,
)

from rig_runtime.adapters.filesystem.session import SessionReader


MAP_ORIENTATION_MODEL = "fixed_north_up"
MAX_LOCALIZATION_REFERENCE_FRAMES = 9
ORIENTED_GRADIENT_MIN_SCORE = 0.35
ORIENTED_GRADIENT_MIN_MARGIN = 0.10
MAX_RIGID_TRANSLATION_SPREAD_PX = 2.0
POSE_GRAPH_LOOP_GATE_PX = 18.0
POSE_GRAPH_MAX_LOOP_EDGES = 900
POSE_GRAPH_MIN_LOOP_EDGES = 5


def _oriented_gradient_channels(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return (
        np.maximum(gx, 0.0),
        np.maximum(-gx, 0.0),
        np.maximum(gy, 0.0),
        np.maximum(-gy, 0.0),
    )


def _masked_oriented_gradient_zncc(
    source: np.ndarray, template: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Correlate signed edge directions with a bounded normalized dot product."""
    source_channels = _oriented_gradient_channels(source)
    template_channels = _oriented_gradient_channels(template)
    weights = (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)
    weight_sum = max(float(np.sum(weights)), 1.0)
    numerator = None
    source_energy = None
    template_energy = 0.0
    for source_channel, template_channel in zip(
        source_channels, template_channels
    ):
        template_mean = float(np.sum(template_channel * weights) / weight_sum)
        centered_template = (template_channel - template_mean) * weights
        channel_numerator = cv2.matchTemplate(
            source_channel, centered_template, cv2.TM_CCORR
        )
        source_square_sum = cv2.matchTemplate(
            source_channel * source_channel, weights, cv2.TM_CCORR
        )
        channel_source_energy = np.maximum(source_square_sum, 0.0)
        numerator = (
            channel_numerator
            if numerator is None
            else numerator + channel_numerator
        )
        source_energy = (
            channel_source_energy
            if source_energy is None
            else source_energy + channel_source_energy
        )
        template_energy += float(np.sum(centered_template * centered_template))
    minimum_source_energy = max(template_energy * 1.0e-6, 1.0e-6)
    valid = source_energy >= minimum_source_energy
    denominator = np.sqrt(
        np.maximum(source_energy * template_energy, 1.0e-12)
    )
    response = np.full(numerator.shape, -1.0, np.float32)
    np.divide(numerator, denominator, out=response, where=valid)
    response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
    return np.clip(response, -1.0, 1.0)


def _correlation_heatmap(response, best_location, second_location) -> np.ndarray:
    normalized = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_TURBO)
    cv2.circle(heatmap, best_location, 7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(heatmap, second_location, 7, (30, 30, 30), 2, cv2.LINE_AA)
    return heatmap


def _minimap_reference(image: np.ndarray, calibration: dict):
    """Extract the calibrated circular map content used for cross-space scale."""
    calibration = normalize_minimap_geometry(
        calibration,
        minimap_crop_space([image.shape[1], image.shape[0]]),
        allow_legacy=True,
    )
    boundary = calibration["outer_boundary"]
    center_x = float(boundary["center_x"])
    center_y = float(boundary["center_y"])
    radius = float(boundary["radius"])
    left = max(0, int(round(center_x - radius)))
    top = max(0, int(round(center_y - radius)))
    right = min(image.shape[1], int(round(center_x + radius)))
    bottom = min(image.shape[0], int(round(center_y + radius)))
    patch = image[top:bottom, left:right].copy()
    if min(patch.shape[:2]) < 48:
        raise RuntimeError("Mini-map scale reference is too small")
    mask = np.zeros(patch.shape[:2], np.uint8)
    patch_center = (patch.shape[1] // 2, patch.shape[0] // 2)
    usable_radius = max(8, min(patch.shape[:2]) // 2 - 3)
    cv2.circle(mask, patch_center, usable_radius, 255, -1)
    cv2.circle(mask, patch_center, max(5, int(round(radius * 0.28))), 0, -1)
    return patch, mask


def load_localization_reference_candidates(
    calibration_root: Path,
    calibration: dict,
    maximum_frames: int = MAX_LOCALIZATION_REFERENCE_FRAMES,
):
    """Load persisted endpoints and evenly spaced observations from forward motion."""
    calibration_root = Path(calibration_root)
    references = []
    seen_indices = set()
    verification = calibration.get("forward_verification") or {}
    source_frames = verification.get("source_frames") or {}
    endpoint_names = {"start": "forward_start.png", "end": "forward_end.png"}
    for role, name in endpoint_names.items():
        path = calibration_root / name
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        frame_index = (source_frames.get(role) or {}).get("frame_index")
        if frame_index is not None:
            seen_indices.add(int(frame_index))
        references.append(
            {
                "image": image,
                "source_image_name": name,
                "source_frame_index": frame_index,
                "source_kind": "persisted_endpoint",
            }
        )
    session_path = verification.get("source_session_path")
    start_index = (source_frames.get("start") or {}).get("frame_index")
    end_index = (source_frames.get("end") or {}).get("frame_index")
    crop = (calibration.get("config") or {}).get("crop_xywh")
    if (
        session_path
        and start_index is not None
        and end_index is not None
        and crop
        and Path(session_path).is_dir()
        and maximum_frames > len(references)
    ):
        reader = SessionReader(Path(session_path))
        capture = cv2.VideoCapture(str(reader.video_path("main")))
        try:
            sample_count = max(2, int(maximum_frames))
            indices = np.linspace(
                int(start_index), int(end_index), sample_count, dtype=np.int32
            )
            x, y, width, height = [int(value) for value in crop]
            for frame_index in indices:
                frame_index = int(frame_index)
                if frame_index in seen_indices:
                    continue
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    continue
                image = frame[y : y + height, x : x + width]
                if image.shape[:2] != (height, width):
                    continue
                references.append(
                    {
                        "image": image.copy(),
                        "source_image_name": "forward_frame_{:06d}".format(frame_index),
                        "source_frame_index": frame_index,
                        "source_kind": "sampled_forward_session",
                    }
                )
                seen_indices.add(frame_index)
        finally:
            capture.release()
    references.sort(
        key=lambda item: (
            item.get("source_frame_index") is None,
            item.get("source_frame_index") or 0,
        )
    )
    return references[: max(1, int(maximum_frames))]


def _root_sift(descriptors: np.ndarray) -> np.ndarray:
    """Map SIFT histograms into the Hellinger (RootSIFT) feature space."""
    descriptors = descriptors.astype(np.float32, copy=False)
    normalization = np.maximum(np.sum(descriptors, axis=1, keepdims=True), 1.0e-12)
    return np.sqrt(descriptors / normalization)


def _prepare_localization_mosaic(mosaic: np.ndarray, coverage: np.ndarray) -> dict:
    """Compute reusable full-map features once for all mini-map references."""
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.005, edgeThreshold=15)
    mosaic_gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    map_mask = cv2.erode(coverage, np.ones((5, 5), np.uint8))
    map_points, map_descriptors = sift.detectAndCompute(mosaic_gray, map_mask)
    if map_descriptors is None:
        raise RuntimeError("Mini-map/full-map scale calibration found no map descriptors")
    return {
        "sift": sift,
        "gray": mosaic_gray,
        "mask": map_mask,
        "points": map_points,
        "descriptors": _root_sift(map_descriptors),
    }


def _fit_fixed_north_up_similarity(source: np.ndarray, target: np.ndarray):
    """Robustly fit target = uniform_scale * source + translation."""
    if len(source) < 2:
        return None, None
    best_inliers = None
    best_error = float("inf")
    for first in range(len(source) - 1):
        for second in range(first + 1, len(source)):
            source_delta = source[second] - source[first]
            denominator = float(np.dot(source_delta, source_delta))
            if denominator < 16.0:
                continue
            target_delta = target[second] - target[first]
            scale = float(np.dot(source_delta, target_delta) / denominator)
            if not 0.5 <= scale <= 16.0:
                continue
            translation = np.mean(
                target[[first, second]] - scale * source[[first, second]], axis=0
            )
            errors = np.linalg.norm(source * scale + translation - target, axis=1)
            inliers = errors <= 6.0
            count = int(np.count_nonzero(inliers))
            median = float(np.median(errors[inliers])) if count else float("inf")
            if best_inliers is None or (count, -median) > (
                int(np.count_nonzero(best_inliers)),
                -best_error,
            ):
                best_inliers = inliers
                best_error = median
    if best_inliers is None or np.count_nonzero(best_inliers) < 2:
        return None, None
    inliers = best_inliers
    for _ in range(2):
        source_inliers = source[inliers]
        target_inliers = target[inliers]
        source_center = np.mean(source_inliers, axis=0)
        target_center = np.mean(target_inliers, axis=0)
        centered_source = source_inliers - source_center
        denominator = float(np.sum(centered_source * centered_source))
        if denominator <= 1.0e-9:
            return None, None
        scale = float(
            np.sum(centered_source * (target_inliers - target_center)) / denominator
        )
        if not 0.5 <= scale <= 16.0:
            return None, None
        translation = target_center - scale * source_center
        errors = np.linalg.norm(source * scale + translation - target, axis=1)
        updated = errors <= 6.0
        if np.array_equal(updated, inliers):
            break
        if np.count_nonzero(updated) < 2:
            break
        inliers = updated
    matrix = np.asarray(
        [[scale, 0.0, translation[0]], [0.0, scale, translation[1]]],
        dtype=np.float64,
    )
    return matrix, inliers.astype(np.uint8).reshape((-1, 1))


def _estimate_minimap_similarity(
    reference: np.ndarray,
    mask: np.ndarray,
    mosaic: np.ndarray,
    coverage: np.ndarray,
    mosaic_cache=None,
    minimum_ratio_matches: int = 6,
):
    """Estimate mini-map pixels to original-mosaic pixels with SIFT geometry."""
    mosaic_cache = mosaic_cache or _prepare_localization_mosaic(mosaic, coverage)
    sift = mosaic_cache["sift"]
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    reference_points, reference_descriptors = sift.detectAndCompute(reference_gray, mask)
    map_points = mosaic_cache["points"]
    map_descriptors = mosaic_cache["descriptors"]
    if reference_descriptors is None:
        raise RuntimeError("Mini-map/full-map scale calibration found no SIFT descriptors")
    reference_descriptors = _root_sift(reference_descriptors)
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        reference_descriptors, map_descriptors, k=2
    )
    matches = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.80 * second.distance
    ]
    if len(matches) < minimum_ratio_matches:
        raise RuntimeError(
            "Mini-map/full-map scale calibration needs at least {} ratio-test matches; got {}"
            .format(minimum_ratio_matches, len(matches))
        )
    source = np.float32([reference_points[item.queryIdx].pt for item in matches])
    target = np.float32([map_points[item.trainIdx].pt for item in matches])
    diagnostic_matrix, _ = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=6.0,
        maxIters=20000,
        confidence=0.999,
    )
    matrix, inlier_mask = _fit_fixed_north_up_similarity(source, target)
    if matrix is None or inlier_mask is None:
        raise RuntimeError(
            "Mini-map/full-map matches have no fixed-north-up scale and translation"
        )
    inliers = inlier_mask.ravel().astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count < 2:
        raise RuntimeError(
            "Mini-map/full-map similarity has fewer than 2 geometric inliers"
        )
    predicted = cv2.transform(source.reshape((-1, 1, 2)), matrix).reshape((-1, 2))
    errors = np.linalg.norm(predicted - target, axis=1)
    scale = math.hypot(float(matrix[0, 0]), float(matrix[0, 1]))
    residual_rotation = (
        math.degrees(
            math.atan2(float(diagnostic_matrix[1, 0]), float(diagnostic_matrix[0, 0]))
        )
        if diagnostic_matrix is not None
        else None
    )
    center = cv2.transform(
        np.float32([[[reference.shape[1] / 2.0, reference.shape[0] / 2.0]]]),
        matrix,
    ).reshape(2)
    return {
        "matrix": matrix,
        "source_points": source,
        "target_points": target,
        "inlier_mask": inliers,
        "ratio_match_count": len(matches),
        "inlier_count": inlier_count,
        "inlier_ratio": float(np.mean(inliers)),
        "reprojection_median_px": float(np.median(errors[inliers])),
        "reprojection_p95_px": float(np.percentile(errors[inliers], 95)),
        "map_pixels_per_minimap_pixel": float(scale),
        "minimap_to_map_rotation_deg": 0.0,
        "diagnostic_residual_rotation_deg": residual_rotation,
        "reference_center_map_xy": center,
    }


def _scale_consensus(
    estimates,
    relative_tolerance: float = 0.035,
    preferred_name=None,
    minimum_inliers: int = 2,
) -> dict:
    """Choose the largest mutually consistent scale cluster and its best member."""
    estimates = [
        item
        for item in estimates
        if item["estimate"]["inlier_count"] >= minimum_inliers
    ]
    if not estimates:
        raise RuntimeError("Mini-map/full-map scale calibration found no usable references")
    scales = np.asarray(
        [item["estimate"]["map_pixels_per_minimap_pixel"] for item in estimates],
        dtype=np.float64,
    )
    clusters = []
    for center in scales:
        tolerance = max(0.02, abs(float(center)) * relative_tolerance)
        members = np.flatnonzero(np.abs(scales - center) <= tolerance)
        member_scales = scales[members]
        clusters.append(
            (
                len(members),
                -float(np.median(np.abs(member_scales - np.median(member_scales)))),
                members,
            )
        )
    _, _, member_indices = max(clusters, key=lambda item: (item[0], item[1]))
    members = [estimates[int(index)] for index in member_indices]
    for _ in range(3):
        consensus_scale = float(
            np.median(
                [
                    item["estimate"]["map_pixels_per_minimap_pixel"]
                    for item in members
                ]
            )
        )
        tolerance = max(0.02, abs(consensus_scale) * relative_tolerance)
        refined = [
            item
            for item in members
            if abs(
                item["estimate"]["map_pixels_per_minimap_pixel"] - consensus_scale
            )
            <= tolerance
        ]
        if len(refined) == len(members):
            break
        members = refined
    preferred = [
        item
        for item in members
        if item["candidate"].get("source_image_name") == preferred_name
    ]
    best = preferred[0] if preferred else max(
        members,
        key=lambda item: (
            item["estimate"]["inlier_count"],
            item["estimate"]["inlier_ratio"],
            -item["estimate"]["reprojection_p95_px"],
        ),
    )
    deviations = np.abs(
        np.asarray(
            [item["estimate"]["map_pixels_per_minimap_pixel"] for item in members]
        )
        - consensus_scale
    )
    return {
        "scale": consensus_scale,
        "members": members,
        "selected": best,
        "median_absolute_deviation": float(np.median(deviations)),
        "maximum_relative_deviation": float(
            np.max(deviations) / max(abs(consensus_scale), 1.0e-9)
        ),
    }


def _apply_consensus_scale(estimate: dict, scale: float, reference_shape) -> dict:
    """Refit translation and errors for one reference at the agreed scale."""
    source = estimate["source_points"]
    target = estimate["target_points"]
    seed_inliers = estimate["inlier_mask"]
    translation = np.median(target[seed_inliers] - source[seed_inliers] * scale, axis=0)
    predicted = source * scale + translation
    errors = np.linalg.norm(predicted - target, axis=1)
    inliers = errors <= 6.0
    if np.count_nonzero(inliers) >= 2:
        translation = np.median(target[inliers] - source[inliers] * scale, axis=0)
        predicted = source * scale + translation
        errors = np.linalg.norm(predicted - target, axis=1)
        inliers = errors <= 6.0
    matrix = np.asarray(
        [[scale, 0.0, translation[0]], [0.0, scale, translation[1]]],
        dtype=np.float64,
    )
    center = cv2.transform(
        np.float32([[[reference_shape[1] / 2.0, reference_shape[0] / 2.0]]]),
        matrix,
    ).reshape(2)
    updated = dict(estimate)
    updated.update(
        {
            "matrix": matrix,
            "inlier_mask": inliers,
            "inlier_count": int(np.count_nonzero(inliers)),
            "inlier_ratio": float(np.mean(inliers)),
            "reprojection_median_px": float(np.median(errors[inliers])),
            "reprojection_p95_px": float(np.percentile(errors[inliers], 95)),
            "map_pixels_per_minimap_pixel": float(scale),
            "reference_center_map_xy": center,
        }
    )
    return updated


def _render_reference_consensus(rows, selected_name: str) -> np.ndarray:
    width, height = 250, 190
    canvas = np.full((height * 3, width * 3, 3), 18, np.uint8)
    accepted_names = {
        row["candidate"].get("source_image_name")
        for row in rows
        if row.get("consensus_member")
    }
    for index, row in enumerate(rows[:9]):
        x = (index % 3) * width
        y = (index // 3) * height
        image = cv2.resize(row["candidate"]["image"], (220, 150))
        canvas[y + 28 : y + 178, x + 15 : x + 235] = image
        name = row["candidate"].get("source_image_name") or "reference"
        estimate = row.get("estimate")
        if estimate:
            label = "{}  s={:.3f}  in={}".format(
                name.replace("forward_frame_", "f"),
                estimate["map_pixels_per_minimap_pixel"],
                estimate["inlier_count"],
            )
        else:
            label = "{}  rejected".format(name.replace("forward_frame_", "f"))
        color = (70, 235, 100) if name in accepted_names else (80, 80, 255)
        thickness = 3 if name == selected_name else 1
        cv2.rectangle(canvas, (x + 14, y + 27), (x + 236, y + 179), color, thickness)
        cv2.putText(
            canvas,
            label,
            (x + 8, y + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _reference_consensus_result(row: dict) -> dict:
    candidate = row["candidate"]
    estimate = row.get("estimate")
    result = {
        "source_image_name": candidate.get("source_image_name"),
        "source_frame_index": candidate.get("source_frame_index"),
        "source_kind": candidate.get("source_kind"),
        "status": "estimated" if estimate else "rejected",
        "consensus_member": bool(row.get("consensus_member")),
    }
    if estimate:
        result.update(
            {
                "map_pixels_per_minimap_pixel": estimate[
                    "map_pixels_per_minimap_pixel"
                ],
                "ratio_match_count": estimate["ratio_match_count"],
                "inlier_count": estimate["inlier_count"],
                "inlier_ratio": estimate["inlier_ratio"],
                "reprojection_p95_original_map_px": estimate[
                    "reprojection_p95_px"
                ],
                "diagnostic_residual_rotation_deg": estimate[
                    "diagnostic_residual_rotation_deg"
                ],
            }
        )
    else:
        result["reason"] = row.get("error")
    return result


def _build_localization_derivative(
    mosaic: np.ndarray,
    coverage: np.ndarray,
    output_path: Path,
    reference: dict,
    mosaic_cache=None,
) -> dict:
    """Create a reversible map raster normalized to calibrated mini-map scale."""
    mosaic_cache = mosaic_cache or _prepare_localization_mosaic(mosaic, coverage)
    candidates = reference.get("candidates") or [reference]
    estimate_rows = []
    review_rows = []
    minimum_matches = 3 if len(candidates) > 1 else 6
    for candidate in candidates:
        patch, mask = _minimap_reference(candidate["image"], reference["calibration"])
        try:
            candidate_estimate = _estimate_minimap_similarity(
                patch,
                mask,
                mosaic,
                coverage,
                mosaic_cache=mosaic_cache,
                minimum_ratio_matches=minimum_matches,
            )
            row = {
                "candidate": candidate,
                "patch": patch,
                "mask": mask,
                "estimate": candidate_estimate,
            }
            estimate_rows.append(row)
            review_rows.append(row)
        except RuntimeError as error:
            review_rows.append({"candidate": candidate, "error": str(error)})
    consensus = _scale_consensus(
        estimate_rows,
        preferred_name=reference.get("source_image_name"),
        minimum_inliers=3 if len(candidates) > 1 else 2,
    )
    consensus_names = {
        item["candidate"].get("source_image_name") for item in consensus["members"]
    }
    for row in review_rows:
        row["consensus_member"] = (
            row["candidate"].get("source_image_name") in consensus_names
        )
    selected = consensus["selected"]
    patch, mask = selected["patch"], selected["mask"]
    estimate = (
        selected["estimate"]
        if len(candidates) == 1
        else _apply_consensus_scale(
            selected["estimate"], consensus["scale"], patch.shape
        )
    )
    scale = estimate["map_pixels_per_minimap_pixel"]
    if not 0.5 <= scale <= 16.0:
        raise RuntimeError("Mini-map/full-map scale {:.3f} is implausible".format(scale))
    requested_factor = 1.0 / scale
    if requested_factor > 1.0 + 1.0e-3:
        raise RuntimeError(
            "The stitched map would need {:.3f}x image enlargement to match the "
            "mini-map. Record the full map at a larger rendered-map scale; "
            "upscaling cannot create matchable detail.".format(requested_factor)
        )
    localization_size = (
        max(64, int(round(mosaic.shape[1] * requested_factor))),
        max(64, int(round(mosaic.shape[0] * requested_factor))),
    )
    interpolation = cv2.INTER_AREA if requested_factor < 1.0 else cv2.INTER_NEAREST
    localization_mosaic = cv2.resize(mosaic, localization_size, interpolation=interpolation)
    localization_coverage = cv2.resize(
        coverage, localization_size, interpolation=cv2.INTER_NEAREST
    )
    factor_x = localization_size[0] / float(mosaic.shape[1])
    factor_y = localization_size[1] / float(mosaic.shape[0])
    original_to_localization = np.asarray(
        [[factor_x, 0.0, 0.0], [0.0, factor_y, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    localization_to_original = np.linalg.inv(original_to_localization)

    residual_rotation = estimate["diagnostic_residual_rotation_deg"]
    applied_rotation = 0.0
    response = _masked_oriented_gradient_zncc(
        localization_mosaic,
        patch,
        mask,
    )
    _, score, _, location = cv2.minMaxLoc(response)
    suppressed = response.copy()
    cv2.circle(
        suppressed,
        location,
        max(8, min(patch.shape[:2]) // 3),
        -1.0,
        -1,
    )
    _, second_score, _, second_location = cv2.minMaxLoc(suppressed)
    correlation_center = np.asarray(
        [location[0] + patch.shape[1] / 2.0, location[1] + patch.shape[0] / 2.0],
        dtype=np.float64,
    )
    sift_center = np.asarray(
        [
            estimate["reference_center_map_xy"][0] * factor_x,
            estimate["reference_center_map_xy"][1] * factor_y,
        ],
        dtype=np.float64,
    )
    center_agreement = float(np.linalg.norm(correlation_center - sift_center))
    consensus_ready = len(candidates) == 1 or len(consensus["members"]) >= 3
    ready = bool(
        consensus_ready
        and estimate["inlier_count"] >= 6
        and estimate["inlier_ratio"] >= 0.60
        and estimate["reprojection_p95_px"] <= 4.0
        and score >= ORIENTED_GRADIENT_MIN_SCORE
        and score - second_score >= ORIENTED_GRADIENT_MIN_MARGIN
        and center_agreement <= 8.0
    )

    evidence = localization_mosaic.copy()
    for point, accepted in zip(estimate["target_points"], estimate["inlier_mask"]):
        if accepted:
            location_xy = (
                int(round(float(point[0]) * factor_x)),
                int(round(float(point[1]) * factor_y)),
            )
            cv2.circle(evidence, location_xy, 3, (40, 220, 245), -1, cv2.LINE_AA)
    cv2.circle(
        evidence,
        tuple(np.round(sift_center).astype(int)),
        max(8, min(patch.shape[:2]) // 2),
        (70, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.circle(
        evidence,
        tuple(np.round(correlation_center).astype(int)),
        max(5, min(patch.shape[:2]) // 2 - 5),
        (70, 235, 100),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        evidence,
        "scale {:.3f} map px/mini px  SIFT {}/{}  corr {:.3f} margin {:.3f}".format(
            scale,
            estimate["inlier_count"],
            estimate["ratio_match_count"],
            score,
            score - second_score,
        ),
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (250, 250, 250),
        2,
        cv2.LINE_AA,
    )
    _write_image(output_path / "localization_mosaic.png", localization_mosaic)
    _write_image(output_path / "localization_coverage.png", localization_coverage)
    _write_image(output_path / "localization_scale_evidence.png", evidence)
    _write_image(
        output_path / "localization_correlation_heatmap.png",
        _correlation_heatmap(response, location, second_location),
    )
    selected_name = selected["candidate"].get("source_image_name")
    _write_image(
        output_path / "localization_reference_consensus.png",
        _render_reference_consensus(review_rows, selected_name),
    )
    return {
        "schema_version": "1.0",
        "status": "ready" if ready else "review_required",
        "method": "rootsift_fixed_north_up_multi_reference_with_oriented_gradient_zncc",
        "map_orientation_model": MAP_ORIENTATION_MODEL,
        "north_normalization_applied": False,
        "source_minimap_calibration_id": reference.get("calibration_id"),
        "source_minimap_image": selected_name,
        "reference_candidate_count": len(reference.get("candidates") or [reference]),
        "reference_candidate_names": [
            item.get("source_image_name")
            for item in (reference.get("candidates") or [reference])
        ],
        "scale_consensus": {
            "required_reference_count": 3 if len(candidates) > 1 else 1,
            "accepted_reference_count": len(consensus["members"]),
            "accepted_reference_names": sorted(consensus_names),
            "median_absolute_deviation": consensus["median_absolute_deviation"],
            "maximum_relative_deviation": consensus["maximum_relative_deviation"],
        },
        "reference_results": [
            _reference_consensus_result(row) for row in review_rows
        ],
        "coordinate_space": "derived_localization_map_px",
        "mosaic_file": "localization_mosaic.png",
        "coverage_file": "localization_coverage.png",
        "size_wh": list(localization_size),
        "map_pixels_per_minimap_pixel": scale,
        "resample_factor": float(requested_factor),
        "resampling": (
            "area_downsample" if requested_factor < 1.0 else "native_scale"
        ),
        "minimap_to_map_rotation_deg": applied_rotation,
        "applied_map_rotation_deg": applied_rotation,
        "diagnostic_residual_rotation_deg": residual_rotation,
        "original_map_to_localization_3x3": original_to_localization.tolist(),
        "localization_to_original_map_3x3": localization_to_original.tolist(),
        "reference_center_original_map_xy": estimate["reference_center_map_xy"].tolist(),
        "reference_center_localization_xy": sift_center.tolist(),
        "quality": {
            "ratio_match_count": estimate["ratio_match_count"],
            "inlier_count": estimate["inlier_count"],
            "inlier_ratio": estimate["inlier_ratio"],
            "reprojection_median_original_map_px": estimate["reprojection_median_px"],
            "reprojection_p95_original_map_px": estimate["reprojection_p95_px"],
            "gradient_correlation_score": float(score),
            "gradient_correlation_margin": float(score - second_score),
            "gradient_correlation_method": "masked_oriented_gradient_zero_mean_template_normalized_cross_correlation",
            "gradient_correlation_minimum_score": ORIENTED_GRADIENT_MIN_SCORE,
            "gradient_correlation_minimum_margin": ORIENTED_GRADIENT_MIN_MARGIN,
            "sift_correlation_center_agreement_localization_px": center_agreement,
            "scale_consensus_reference_count": len(consensus["members"]),
        },
        "evidence_file": "localization_scale_evidence.png",
        "reference_consensus_evidence_file": "localization_reference_consensus.png",
    }


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError("Could not write map-stitch evidence: {}".format(path))


def _composition_quality_evidence(
    mosaic: np.ndarray,
    coverage: np.ndarray,
    source_owner: np.ndarray,
) -> dict:
    """Render diagnostics for the existing hard-owner composition result."""

    valid = source_owner >= 0
    seam = np.zeros(source_owner.shape, np.uint8)
    horizontal = (
        valid[:, :-1]
        & valid[:, 1:]
        & (source_owner[:, :-1] != source_owner[:, 1:])
    )
    vertical = (
        valid[:-1, :]
        & valid[1:, :]
        & (source_owner[:-1, :] != source_owner[1:, :])
    )
    seam[:, :-1][horizontal] = 255
    seam[:, 1:][horizontal] = 255
    seam[:-1, :][vertical] = 255
    seam[1:, :][vertical] = 255

    seam_overlay = mosaic.copy()
    seam_overlay[seam > 0] = (30, 30, 255)

    source_owner_image = np.zeros_like(mosaic)
    owner_ids = np.unique(source_owner[valid])
    if owner_ids.size:
        values = np.arange(int(owner_ids.max()) + 1, dtype=np.int64)
        palette = np.column_stack(
            (
                35 + (values * 67) % 205,
                35 + (values * 131) % 205,
                35 + (values * 193) % 205,
            )
        ).astype(np.uint8)
        source_owner_image[valid] = palette[source_owner[valid]]

    gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    low_frequency = cv2.GaussianBlur(
        gray.astype(np.float32), (0, 0), 8.0
    )
    step = np.zeros(gray.shape, np.float32)
    horizontal_step = np.abs(
        low_frequency[:, :-1] - low_frequency[:, 1:]
    )
    vertical_step = np.abs(
        low_frequency[:-1, :] - low_frequency[1:, :]
    )
    step[:, :-1][horizontal] = np.maximum(
        step[:, :-1][horizontal], horizontal_step[horizontal]
    )
    step[:, 1:][horizontal] = np.maximum(
        step[:, 1:][horizontal], horizontal_step[horizontal]
    )
    step[:-1, :][vertical] = np.maximum(
        step[:-1, :][vertical], vertical_step[vertical]
    )
    step[1:, :][vertical] = np.maximum(
        step[1:, :][vertical], vertical_step[vertical]
    )
    low_frequency_step = cv2.applyColorMap(
        np.uint8(np.clip(step / 32.0 * 255.0, 0, 255)),
        cv2.COLORMAP_TURBO,
    )
    low_frequency_step[seam == 0] = 0

    protected_features = cv2.dilate(
        cv2.Canny(gray, 50, 140), np.ones((5, 5), np.uint8)
    ) > 0
    protected_seam = (seam > 0) & protected_features
    protected_overlay = mosaic.copy()
    protected_overlay[seam > 0] = (30, 30, 255)
    protected_overlay[protected_seam] = (255, 70, 220)

    outside = coverage == 0
    component_count, labels = cv2.connectedComponents(
        outside.astype(np.uint8), connectivity=8
    )
    border_labels = set(labels[0, :]) | set(labels[-1, :])
    border_labels |= set(labels[:, 0]) | set(labels[:, -1])
    hole_labels = [
        label for label in range(1, component_count) if label not in border_labels
    ]
    interior_holes = np.isin(labels, hole_labels)
    seam_values = step[seam > 0]
    seam_pixels = int(np.count_nonzero(seam))
    return {
        "images": {
            "source_ownership.png": source_owner_image,
            "seam_overlay.png": seam_overlay,
            "low_frequency_tile_step.png": low_frequency_step,
            "protected_feature_seams.png": protected_overlay,
        },
        "metrics": {
            "source_owner_count": int(len(owner_ids)),
            "seam_pixel_count": seam_pixels,
            "protected_feature_seam_fraction": (
                float(np.count_nonzero(protected_seam) / seam_pixels)
                if seam_pixels
                else 0.0
            ),
            "low_frequency_step_luma": {
                "median": float(np.median(seam_values)) if seam_values.size else 0.0,
                "p95": (
                    float(np.percentile(seam_values, 95))
                    if seam_values.size
                    else 0.0
                ),
                "worst": float(np.max(seam_values)) if seam_values.size else 0.0,
            },
            "interior_hole_count": len(hole_labels),
            "interior_hole_pixel_count": int(np.count_nonzero(interior_holes)),
        },
    }


def _registration_image(first: np.ndarray, second: np.ndarray, shift_xy) -> np.ndarray:
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    aligned = cv2.warpAffine(
        second,
        np.float32([[1, 0, -shift_xy[0]], [0, 1, -shift_xy[1]]]),
        (second.shape[1], second.shape[0]),
    )
    second_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    image = np.zeros_like(first)
    image[:, :, 2] = first_gray
    image[:, :, 1] = second_gray
    return image


def _estimate_translation(first: np.ndarray, second: np.ndarray):
    height, width = first.shape[:2]
    first_small = cv2.resize(first, (width // 2, height // 2))
    second_small = cv2.resize(second, (width // 2, height // 2))
    first_gray = cv2.cvtColor(first_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    second_gray = cv2.cvtColor(second_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    first_gray -= cv2.GaussianBlur(first_gray, (0, 0), 3.0)
    second_gray -= cv2.GaussianBlur(second_gray, (0, 0), 3.0)
    window = cv2.createHanningWindow(
        (first_gray.shape[1], first_gray.shape[0]), cv2.CV_32F
    )
    shift, response = cv2.phaseCorrelate(first_gray, second_gray, window)
    return (float(shift[0] * 2.0), float(shift[1] * 2.0)), float(response)


def _estimate_rigid_translation(first: np.ndarray, second: np.ndarray) -> dict:
    """Estimate translation and reject screen tears that violate map rigidity."""
    shift, response = _estimate_translation(first, second)
    height = first.shape[0]
    band_height = max(48, int(round(height * 0.40)))
    starts = (0, max(0, (height - band_height) // 2), height - band_height)
    bands = []
    for start in starts:
        band_shift, band_response = _estimate_translation(
            first[start : start + band_height],
            second[start : start + band_height],
        )
        if band_response >= 0.06:
            bands.append(
                {
                    "shift_xy_px": [float(band_shift[0]), float(band_shift[1])],
                    "response": float(band_response),
                }
            )
    spread = None
    if len(bands) >= 2:
        band_shifts = np.asarray(
            [item["shift_xy_px"] for item in bands], dtype=np.float64
        )
        center = np.median(band_shifts, axis=0)
        spread = float(
            np.max(np.linalg.norm(band_shifts - center[None, :], axis=1))
        )
    return {
        "shift_xy_px": [float(shift[0]), float(shift[1])],
        "response": float(response),
        "translation_spread_px": spread,
        "band_observations": bands,
        "spatially_coherent": spread is None
        or spread <= MAX_RIGID_TRANSLATION_SPREAD_PX,
    }


def _translation_loop_pairs(positions: np.ndarray, viewport_wh, per_frame=3):
    """Return bounded non-neighbour overlaps proposed by the chain placement."""

    width, height = viewport_wh
    pairs = set()
    for first in range(len(positions)):
        candidates = []
        for second in range(first + 12, len(positions)):
            delta = np.abs(positions[second] - positions[first])
            if delta[0] >= width * 0.72 or delta[1] >= height * 0.72:
                continue
            overlap_fraction = (1.0 - delta[0] / width) * (
                1.0 - delta[1] / height
            )
            if overlap_fraction < 0.32:
                continue
            candidates.append((float(np.linalg.norm(delta)), second))
        for _, second in sorted(candidates)[:per_frame]:
            pairs.add((first, second))
    return sorted(pairs)


def _discover_translation_loop_edges(viewports, positions) -> list:
    """Measure loop closures without granting the chain result pose authority."""

    height, width = viewports[0].shape[:2]
    pairs = _translation_loop_pairs(positions, (width, height))
    if len(pairs) > POSE_GRAPH_MAX_LOOP_EDGES:
        indexes = np.unique(
            np.linspace(0, len(pairs) - 1, POSE_GRAPH_MAX_LOOP_EDGES)
            .round()
            .astype(int)
        )
        pairs = [pairs[index] for index in indexes]
    edges = []
    for ordinal, (first, second) in enumerate(pairs):
        rigid = _estimate_rigid_translation(viewports[first], viewports[second])
        shift = np.asarray(rigid["shift_xy_px"], dtype=np.float64)
        expected = positions[first] - positions[second]
        gate_error = float(np.linalg.norm(shift - expected))
        accepted = bool(
            rigid["response"] >= 0.10
            and rigid["spatially_coherent"]
            and gate_error <= POSE_GRAPH_LOOP_GATE_PX
        )
        edges.append(
            {
                "first": int(first),
                "second": int(second),
                "delta_xy_px": (-shift).tolist(),
                "response": float(rigid["response"]),
                "discovery_gate_error_px": gate_error,
                "spatially_coherent": bool(rigid["spatially_coherent"]),
                "accepted": accepted,
                "held_out": bool(
                    (first * 1009 + second * 9176 + ordinal) % 5 == 0
                ),
            }
        )
    return edges


def _fit_translation_pose_graph(count: int, edges, iterations=5) -> np.ndarray:
    """Fit translation-only tile poses with robust iteratively weighted LS."""

    training = [
        edge
        for edge in edges
        if edge.get("accepted", True) and not edge.get("held_out", False)
    ]
    if not training:
        raise ValueError("Translation pose graph has no training constraints")
    base_weights = np.asarray(
        [max(float(edge.get("response", 1.0)), 0.05) for edge in training],
        dtype=np.float64,
    )
    weights = base_weights.copy()
    positions = np.zeros((count, 2), np.float64)
    for _ in range(max(1, int(iterations))):
        matrix = np.zeros((len(training) + 1, count), np.float64)
        target = np.zeros((len(training) + 1, 2), np.float64)
        for row, (edge, weight) in enumerate(zip(training, weights)):
            scale = float(np.sqrt(weight))
            matrix[row, int(edge["first"])] = -scale
            matrix[row, int(edge["second"])] = scale
            target[row] = np.asarray(edge["delta_xy_px"], np.float64) * scale
        matrix[-1, 0] = 10.0
        positions = np.linalg.lstsq(matrix, target, rcond=None)[0]
        residuals = np.asarray(
            [
                np.linalg.norm(
                    positions[int(edge["second"])]
                    - positions[int(edge["first"])]
                    - np.asarray(edge["delta_xy_px"], np.float64)
                )
                for edge in training
            ],
            dtype=np.float64,
        )
        huber = np.minimum(1.0, 2.0 / np.maximum(residuals, 1.0e-6))
        weights = base_weights * huber
    return positions


def _translation_residual_summary(positions: np.ndarray, edges, held_out: bool):
    values = []
    for edge in edges:
        if not edge.get("accepted", True):
            continue
        if bool(edge.get("held_out", False)) != bool(held_out):
            continue
        residual = (
            positions[int(edge["second"])]
            - positions[int(edge["first"])]
            - np.asarray(edge["delta_xy_px"], np.float64)
        )
        values.append(float(np.linalg.norm(residual)))
    if not values:
        return {"count": 0, "median_px": None, "p95_px": None, "worst_px": None}
    return {
        "count": len(values),
        "median_px": float(np.median(values)),
        "p95_px": float(np.percentile(values, 95)),
        "worst_px": float(np.max(values)),
    }


def _render_placement_closure(
    initial: np.ndarray,
    optimized: np.ndarray,
    before: dict,
    after: dict,
) -> np.ndarray:
    image = np.full((620, 1100, 3), 18, np.uint8)
    panel_width = 760
    combined = np.vstack((initial, optimized))
    minimum = combined.min(axis=0)
    maximum = combined.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    scale = min((panel_width - 70) / span[0], 540 / span[1])

    def points(values):
        normalized = (values - minimum) * scale
        normalized[:, 0] += 30
        normalized[:, 1] += 40
        normalized[:, 1] = 590 - normalized[:, 1]
        return np.rint(normalized).astype(int)

    initial_points = points(initial)
    optimized_points = points(optimized)
    cv2.polylines(image, [initial_points], False, (70, 90, 240), 2, cv2.LINE_AA)
    cv2.polylines(image, [optimized_points], False, (70, 225, 120), 2, cv2.LINE_AA)
    for index in range(0, len(initial_points), max(1, len(initial_points) // 40)):
        cv2.line(
            image,
            tuple(initial_points[index]),
            tuple(optimized_points[index]),
            (100, 100, 100),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(image, "Sequential chain", (790, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 90, 240), 2, cv2.LINE_AA)
    cv2.putText(image, "Robust pose graph", (790, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 225, 120), 2, cv2.LINE_AA)
    rows = (
        "Held-out closure residual",
        "before median: {:.2f} px".format(before["median_px"] or 0.0),
        "before P95: {:.2f} px".format(before["p95_px"] or 0.0),
        "before worst: {:.2f} px".format(before["worst_px"] or 0.0),
        "after median: {:.2f} px".format(after["median_px"] or 0.0),
        "after P95: {:.2f} px".format(after["p95_px"] or 0.0),
        "after worst: {:.2f} px".format(after["worst_px"] or 0.0),
    )
    for index, row in enumerate(rows):
        cv2.putText(
            image,
            row,
            (790, 200 + index * 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    return image


def _optimize_selected_translation_poses(
    viewports,
    selected,
    initial_positions: np.ndarray,
    registrations,
):
    selected_viewports = [viewports[index] for index in selected]
    node_for_local = {local: node for node, local in enumerate(selected)}
    sequential = []
    for registration in registrations:
        first = node_for_local.get(registration["_from_local"])
        second = node_for_local.get(registration["_to_local"])
        if first is None or second is None or not registration["accepted"]:
            continue
        sequential.append(
            {
                "first": first,
                "second": second,
                "delta_xy_px": (
                    -np.asarray(registration["content_shift_xy_px"], np.float64)
                ).tolist(),
                "response": float(registration["response"]),
                "accepted": True,
                "held_out": False,
            }
        )
    loops = _discover_translation_loop_edges(selected_viewports, initial_positions)
    accepted_loops = [edge for edge in loops if edge["accepted"]]
    training_loops = [edge for edge in accepted_loops if not edge["held_out"]]
    heldout_loops = [edge for edge in accepted_loops if edge["held_out"]]
    before = _translation_residual_summary(
        initial_positions, accepted_loops, held_out=True
    )
    diagnostics = {
        "method": "sequential_translation_chain",
        "status": "not_applied",
        "loop_candidate_count": len(loops),
        "accepted_loop_count": len(accepted_loops),
        "training_loop_count": len(training_loops),
        "heldout_loop_count": len(heldout_loops),
        "loop_gate_px": POSE_GRAPH_LOOP_GATE_PX,
        "baseline_heldout_residual": before,
        "optimized_heldout_residual": None,
        "maximum_tile_adjustment_px": 0.0,
        "loops": loops,
    }
    if (
        len(accepted_loops) < POSE_GRAPH_MIN_LOOP_EDGES
        or not training_loops
        or not heldout_loops
    ):
        diagnostics["status"] = "insufficient-independent-loop-evidence"
        return initial_positions, diagnostics, None
    optimized = _fit_translation_pose_graph(
        len(selected), sequential + accepted_loops
    )
    after = _translation_residual_summary(optimized, accepted_loops, held_out=True)
    adjustment = np.linalg.norm(optimized - initial_positions, axis=1)
    maximum_adjustment = float(np.max(adjustment))
    diagnostics["optimized_heldout_residual"] = after
    diagnostics["maximum_tile_adjustment_px"] = maximum_adjustment
    improved = bool(
        after["p95_px"] is not None
        and before["p95_px"] is not None
        and after["p95_px"] < before["p95_px"]
        and after["worst_px"] <= before["worst_px"]
        and maximum_adjustment <= POSE_GRAPH_LOOP_GATE_PX
    )
    if not improved:
        diagnostics["status"] = "rejected-no-heldout-improvement"
        return initial_positions, diagnostics, None
    diagnostics["method"] = "robust_translation_pose_graph_irls"
    diagnostics["status"] = "applied"
    evidence = _render_placement_closure(initial_positions, optimized, before, after)
    return optimized, diagnostics, evidence


def stitch_map_frames(
    frames,
    output_path: Path,
    provenance=None,
    progress=None,
    localization_reference=None,
    source_frame_indices=None,
    source_frame_count=None,
) -> dict:
    """Register overlapping map-view frames and write a reviewable mosaic."""
    if len(frames) < 2:
        raise ValueError("Map stitching needs at least two frames")
    source_frame_indices = list(
        range(len(frames)) if source_frame_indices is None else source_frame_indices
    )
    if len(source_frame_indices) != len(frames):
        raise ValueError("Source frame indices must match the supplied map frames")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    margins = {
        "left": min(180, width // 5),
        "top": min(65, height // 10),
        "right": min(100, width // 8),
        "bottom": min(55, height // 10),
    }
    x0, y0 = margins["left"], margins["top"]
    x1, y1 = width - margins["right"], height - margins["bottom"]
    if progress:
        progress("Cropping map viewports from {} decoded frames".format(len(frames)))
    viewports = [frame[y0:y1, x0:x1] for frame in frames]
    positions = [np.array([0.0, 0.0])]
    selected = [0]
    registrations = []
    reference = viewports[0]
    reference_index = 0
    reference_position = np.array([0.0, 0.0])
    for index in range(1, len(viewports)):
        if progress and (
            index == 1 or index % 25 == 0 or index == len(viewports) - 1
        ):
            progress(
                "Registering map frames: {} / {}".format(
                    index, len(viewports) - 1
                )
            )
        rigid = _estimate_rigid_translation(reference, viewports[index])
        shift = rigid["shift_xy_px"]
        response = rigid["response"]
        magnitude = float(np.linalg.norm(shift))
        accepted = bool(
            response >= 0.06
            and abs(shift[0]) < (x1 - x0) * 0.45
            and abs(shift[1]) < (y1 - y0) * 0.45
            and rigid["spatially_coherent"]
        )
        registrations.append(
            {
                "from_frame": source_frame_indices[reference_index],
                "to_frame": source_frame_indices[index],
                "_from_local": reference_index,
                "_to_local": index,
                "content_shift_xy_px": [shift[0], shift[1]],
                "magnitude_px": magnitude,
                "response": response,
                "translation_spread_px": rigid["translation_spread_px"],
                "spatially_coherent": rigid["spatially_coherent"],
                "accepted": accepted,
            }
        )
        if accepted:
            position = reference_position - np.asarray(shift)
            if magnitude >= 28.0:
                selected.append(index)
                reference = viewports[index]
                reference_index = index
                reference_position = position.copy()
        else:
            position = positions[-1].copy()
        positions.append(position)
    if (
        selected[-1] != len(positions) - 1
        and np.linalg.norm(positions[-1] - positions[selected[-1]]) >= 5.0
    ):
        selected.append(len(positions) - 1)
    selected_positions = np.asarray([positions[index] for index in selected])
    viewport_height, viewport_width = viewports[0].shape[:2]
    if progress:
        progress("Validating global tile placement with non-neighbour loop closures")
    selected_positions, placement_optimization, placement_evidence = (
        _optimize_selected_translation_poses(
            viewports,
            selected,
            selected_positions,
            registrations,
        )
    )
    minimum = np.floor(selected_positions.min(axis=0)).astype(int)
    maximum = np.ceil(selected_positions.max(axis=0)).astype(int)
    canvas_width = int(maximum[0] - minimum[0] + viewport_width)
    canvas_height = int(maximum[1] - minimum[1] + viewport_height)
    if canvas_width * canvas_height > 120_000_000:
        raise RuntimeError("Estimated map mosaic is implausibly large")
    if progress:
        progress(
            "Composing the observed {} x {} map mosaic".format(
                canvas_width, canvas_height
            )
        )
    # A map is rigid source artwork.  Averaging dozens of almost-aligned
    # viewports destroys its small text and line detail, so retain the best
    # source sample at every canvas pixel instead.  Subpixel placement is
    # applied inside a one-pixel-padded ROI rather than rounded away.
    mosaic = np.zeros((canvas_height, canvas_width, 3), np.uint8)
    best_weights = np.zeros((canvas_height, canvas_width), np.float32)
    overlap_count = np.zeros((canvas_height, canvas_width), np.uint16)
    source_owner = np.full((canvas_height, canvas_width), -1, np.int32)
    feather = cv2.createHanningWindow(
        (viewport_width, viewport_height), cv2.CV_32F
    )
    feather = np.maximum(feather, 0.08)
    for owner_id, (index, selected_position) in enumerate(
        zip(selected, selected_positions)
    ):
        origin = selected_position - minimum
        base = np.floor(origin).astype(int)
        ox, oy = int(base[0]), int(base[1])
        fraction = origin - base
        roi_width = min(viewport_width + 1, canvas_width - ox)
        roi_height = min(viewport_height + 1, canvas_height - oy)
        transform = np.float32(
            [[1.0, 0.0, float(fraction[0])], [0.0, 1.0, float(fraction[1])]]
        )
        warped = cv2.warpAffine(
            viewports[index], transform, (roi_width, roi_height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )
        warped_weight = cv2.warpAffine(
            feather, transform, (roi_width, roi_height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )
        target_weights = best_weights[oy : oy + roi_height, ox : ox + roi_width]
        replace = warped_weight > target_weights
        target = mosaic[oy : oy + roi_height, ox : ox + roi_width]
        target[replace] = warped[replace]
        target_weights[replace] = warped_weight[replace]
        target_owner = source_owner[
            oy : oy + roi_height, ox : ox + roi_width
        ]
        target_owner[replace] = owner_id
        overlap_count[oy : oy + roi_height, ox : ox + roi_width] += (
            warped_weight > 0
        ).astype(np.uint16)
    coverage = (best_weights > 0).astype(np.uint8) * 255
    coverage_heatmap = cv2.applyColorMap(
        np.uint8(
            np.clip(
                overlap_count / max(float(overlap_count.max()), 1.0) * 255,
                0,
                255,
            )
        ),
        cv2.COLORMAP_TURBO,
    )
    composition_quality = _composition_quality_evidence(
        mosaic, coverage, source_owner
    )
    if progress:
        progress("Building registration-quality and coverage evidence")
    quality = np.full((360, 900, 3), 18, np.uint8)
    responses = [float(item["response"]) for item in registrations]
    for index in range(1, len(responses)):
        x_prev = 25 + int((index - 1) / max(len(responses) - 1, 1) * 850)
        x_now = 25 + int(index / max(len(responses) - 1, 1) * 850)
        y_prev = 320 - int(np.clip(responses[index - 1], 0, 1) * 280)
        y_now = 320 - int(np.clip(responses[index], 0, 1) * 280)
        color = (
            (80, 230, 120)
            if registrations[index - 1]["accepted"]
            else (80, 80, 255)
        )
        cv2.line(quality, (x_prev, y_prev), (x_now, y_now), color, 1, cv2.LINE_AA)
    cv2.putText(quality, "Pairwise registration response", (22, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 1, cv2.LINE_AA)
    sample_rows = [item for item in registrations if item["accepted"]]
    sample_rows = sample_rows[:: max(1, len(sample_rows) // 3)][:3]
    samples = []
    for item in sample_rows:
        overlay = _registration_image(
            viewports[item["_from_local"]],
            viewports[item["_to_local"]],
            item["content_shift_xy_px"],
        )
        samples.append(cv2.resize(overlay, (320, 180)))
    alignment_samples = (
        np.hstack(samples) if samples else np.zeros((180, 320, 3), np.uint8)
    )
    evidence = [
        {"name": "mosaic.png", "title": "Observed full-map mosaic", "category": "map"},
        {"name": "coverage.png", "title": "Observed viewport coverage", "category": "coverage"},
        {"name": "coverage_heatmap.png", "title": "Coverage overlap heatmap", "category": "coverage"},
        {"name": "registration_quality.png", "title": "Pairwise registration quality", "category": "quality"},
        {"name": "alignment_samples.png", "title": "Representative frame alignments", "category": "quality"},
        {"name": "source_ownership.png", "title": "Hard source owner per output pixel", "category": "composition"},
        {"name": "seam_overlay.png", "title": "Hard-owner seam locations", "category": "composition"},
        {"name": "low_frequency_tile_step.png", "title": "Low-frequency luma change across seams", "category": "composition"},
        {"name": "protected_feature_seams.png", "title": "Seams crossing generic line features", "category": "composition"},
    ]
    if placement_evidence is not None:
        evidence.append(
            {
                "name": "placement_closure.png",
                "title": "Sequential versus loop-constrained tile placement",
                "category": "quality",
            }
        )
    if progress:
        progress("Writing the mosaic and composition review images")
    _write_image(output_path / "mosaic.png", mosaic)
    _write_image(output_path / "coverage.png", coverage)
    _write_image(output_path / "coverage_heatmap.png", coverage_heatmap)
    _write_image(output_path / "registration_quality.png", quality)
    _write_image(output_path / "alignment_samples.png", alignment_samples)
    for name, image in composition_quality["images"].items():
        _write_image(output_path / name, image)
    if placement_evidence is not None:
        _write_image(output_path / "placement_closure.png", placement_evidence)
    localization = None
    if localization_reference is not None:
        if progress:
            progress("Calibrating full-map pixels to the reviewed mini-map scale")
        localization = _build_localization_derivative(
            mosaic,
            coverage,
            output_path,
            localization_reference,
        )
        evidence.extend(
            [
                {
                    "name": "localization_mosaic.png",
                    "title": "Derived mosaic normalized to mini-map scale",
                    "category": "localization",
                },
                {
                    "name": "localization_coverage.png",
                    "title": "Valid coverage in localization coordinates",
                    "category": "localization",
                },
                {
                    "name": "localization_scale_evidence.png",
                    "title": "Mini-map/full-map scale and position verification",
                    "category": "localization",
                },
                {
                    "name": "localization_reference_consensus.png",
                    "title": "Independent mini-map scale-reference consensus",
                    "category": "localization",
                },
                {
                    "name": "localization_correlation_heatmap.png",
                    "title": "Oriented-gradient correlation response",
                    "category": "localization",
                },
            ]
        )
    accepted = sum(1 for item in registrations if item["accepted"])
    incoherent = sum(
        1 for item in registrations if not item["spatially_coherent"]
    )
    public_registrations = [
        {key: value for key, value in item.items() if not key.startswith("_")}
        for item in registrations
    ]
    observed_coverage = float(np.count_nonzero(coverage) / coverage.size)
    result = {
        "schema_version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "review_required" if accepted else "failed",
        "coverage_scope": "observed_viewports_only",
        "provenance": provenance or {},
        "source_frame_count": int(source_frame_count or len(frames)),
        "selected_frame_indices": [source_frame_indices[index] for index in selected],
        "selected_frame_count": len(selected),
        "accepted_registrations": accepted,
        "rejected_registrations": len(registrations) - accepted,
        "spatially_incoherent_registrations": incoherent,
        "maximum_rigid_translation_spread_px": MAX_RIGID_TRANSLATION_SPREAD_PX,
        "median_registration_response": float(np.median(responses)) if responses else 0.0,
        "observed_canvas_coverage": observed_coverage,
        "composition_method": "subpixel_highest_feather_weight_source",
        "composition_quality": composition_quality["metrics"],
        "placement_method": placement_optimization["method"],
        "placement_optimization": placement_optimization,
        "viewport_crop_xywh": [x0, y0, viewport_width, viewport_height],
        "mosaic_size_wh": [canvas_width, canvas_height],
        "localization": localization,
        "registrations": public_registrations,
        "warnings": [
            "Observed coverage does not certify every game region or layer."
        ],
        "evidence": evidence,
    }
    _atomic_json(output_path / "map_stitch.json", result)
    return result


def _select_session_keyframes(
    capture, expected_count: int, progress=None, diagnostics=None
):
    """Stream a long recording and retain only overlapping, displaced keyframes."""
    selected_frames = []
    selected_indices = []
    reference = None
    reference_position = np.array([0.0, 0.0])
    last_position = reference_position.copy()
    last_accepted_frame = None
    last_accepted_index = None
    decoded_count = 0
    spatially_incoherent_indices = []
    viewport_size = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        source_index = decoded_count
        decoded_count += 1
        height, width = frame.shape[:2]
        margins = {
            "left": min(180, width // 5),
            "top": min(65, height // 10),
            "right": min(100, width // 8),
            "bottom": min(55, height // 10),
        }
        x0, y0 = margins["left"], margins["top"]
        x1, y1 = width - margins["right"], height - margins["bottom"]
        viewport = frame[y0:y1, x0:x1]
        if reference is None:
            reference = viewport.copy()
            viewport_size = (x1 - x0, y1 - y0)
            selected_frames.append(frame.copy())
            selected_indices.append(source_index)
            last_accepted_frame = frame.copy()
            last_accepted_index = source_index
        else:
            rigid = _estimate_rigid_translation(reference, viewport)
            shift = rigid["shift_xy_px"]
            response = rigid["response"]
            magnitude = float(np.linalg.norm(shift))
            accepted = bool(
                response >= 0.06
                and abs(shift[0]) < viewport_size[0] * 0.45
                and abs(shift[1]) < viewport_size[1] * 0.45
                and rigid["spatially_coherent"]
            )
            if not rigid["spatially_coherent"]:
                spatially_incoherent_indices.append(source_index)
            if accepted:
                last_position = reference_position - np.asarray(shift)
                last_accepted_frame = frame.copy()
                last_accepted_index = source_index
                if magnitude >= 28.0:
                    selected_frames.append(frame.copy())
                    selected_indices.append(source_index)
                    reference = viewport.copy()
                    reference_position = last_position.copy()
        if progress and decoded_count % 90 == 0:
            progress(
                "Selecting map keyframes: {} / {} frames · {} retained".format(
                    decoded_count, expected_count or "?", len(selected_frames)
                )
            )
    if (
        last_accepted_frame is not None
        and last_accepted_index != selected_indices[-1]
        and np.linalg.norm(last_position - reference_position) >= 5.0
    ):
        selected_frames.append(last_accepted_frame)
        selected_indices.append(last_accepted_index)
    if diagnostics is not None:
        diagnostics.update(
            {
                "spatially_incoherent_frame_indices": spatially_incoherent_indices,
                "spatially_incoherent_frame_count": len(
                    spatially_incoherent_indices
                ),
                "maximum_rigid_translation_spread_px": (
                    MAX_RIGID_TRANSLATION_SPREAD_PX
                ),
            }
        )
    return selected_frames, selected_indices, decoded_count


def stitch_map_session(
    session_path: Path,
    output_path: Path,
    progress=None,
    localization_reference=None,
) -> dict:
    reader = SessionReader(session_path)
    records = reader.frames_by_stream.get("main", [])
    if progress:
        progress("Streaming the full-map recording and selecting keyframes")
    capture = cv2.VideoCapture(str(reader.video_path("main")))
    selection_diagnostics = {}
    try:
        frames, source_indices, decoded_count = _select_session_keyframes(
            capture, len(records), progress, diagnostics=selection_diagnostics
        )
    finally:
        capture.release()
    result = stitch_map_frames(
        frames,
        output_path,
        provenance={
            "source_session_path": str(Path(session_path).resolve()),
            "source_session_id": reader.manifest.get("session_id"),
            "source_frame_records": records,
        },
        progress=progress,
        localization_reference=localization_reference,
        source_frame_indices=source_indices,
        source_frame_count=decoded_count,
    )
    result["source_selection"] = selection_diagnostics
    _atomic_json(Path(output_path) / "map_stitch.json", result)
    return result
