"""Ground-truth feature repeatability and matching evaluation for a display rig."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import cv2
import numpy as np

from .contracts import matrix_3x3
from .geometry import transform_points


def _rect_polygon(rect_xywh: Sequence[float]) -> np.ndarray:
    x, y, width, height = map(float, rect_xywh)
    if min(width, height) <= 0:
        raise ValueError("Feature target rectangle must have positive size")
    return np.asarray(
        [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
        dtype=np.float64,
    )


def generate_feature_target(
    screen_size_px: Sequence[int],
    rect_screen_xywh: Sequence[float],
    seed: int = 7319,
) -> np.ndarray:
    """Generate a deterministic multiscale target with features across the ROI."""

    screen_width, screen_height = map(int, screen_size_px)
    if min(screen_width, screen_height) <= 0:
        raise ValueError("Screen size must be positive")
    x, y, width, height = map(int, rect_screen_xywh)
    if min(width, height) <= 0 or x < 0 or y < 0:
        raise ValueError("Feature target rectangle is invalid")
    if x + width > screen_width or y + height > screen_height:
        raise ValueError("Feature target rectangle lies outside the display")
    random = np.random.RandomState(int(seed))
    image = np.full((screen_height, screen_width, 3), 24, dtype=np.uint8)
    cell = max(6, min(width, height) // 24)
    grid_width = int(math.ceil(width / float(cell)))
    grid_height = int(math.ceil(height / float(cell)))
    palette = np.asarray(
        [
            [32, 32, 32],
            [224, 224, 224],
            [42, 156, 230],
            [210, 88, 52],
            [70, 210, 110],
            [190, 70, 200],
        ],
        dtype=np.uint8,
    )
    indices = random.randint(0, len(palette), size=(grid_height, grid_width))
    coarse = palette[indices]
    patch = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_NEAREST)

    shape_count = max(36, int(width * height / 5000))
    for index in range(shape_count):
        center = (
            int(random.randint(4, max(5, width - 4))),
            int(random.randint(4, max(5, height - 4))),
        )
        radius = int(random.randint(2, max(3, min(width, height) // 22)))
        color = tuple(int(value) for value in palette[random.randint(0, len(palette))])
        if index % 3 == 0:
            cv2.circle(patch, center, radius, color, -1, cv2.LINE_8)
            cv2.circle(patch, center, max(1, radius // 2), (255, 255, 255), 1, cv2.LINE_8)
        elif index % 3 == 1:
            end = (
                int(np.clip(center[0] + random.randint(-radius * 2, radius * 2 + 1), 0, width - 1)),
                int(np.clip(center[1] + random.randint(-radius * 2, radius * 2 + 1), 0, height - 1)),
            )
            cv2.line(patch, center, end, color, max(1, radius // 3), cv2.LINE_8)
        else:
            cv2.rectangle(
                patch,
                (max(0, center[0] - radius), max(0, center[1] - radius)),
                (min(width - 1, center[0] + radius), min(height - 1, center[1] + radius)),
                color,
                max(1, radius // 4),
                cv2.LINE_8,
            )
    cv2.putText(
        patch,
        "ARIA {:04X}".format(int(seed) & 0xFFFF),
        (max(4, width // 20), max(24, height // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.45, min(width, height) / 500.0),
        (250, 250, 250),
        2,
        cv2.LINE_8,
    )
    image[y : y + height, x : x + width] = patch
    return image


def _create_detector(name: str, maximum_features: int):
    normalized = str(name).strip().lower()
    if normalized in ("auto", "sift") and hasattr(cv2, "SIFT_create"):
        return "SIFT", cv2.SIFT_create(nfeatures=int(maximum_features)), cv2.NORM_L2
    if normalized not in ("auto", "orb", "sift"):
        raise ValueError("Unsupported feature detector {}".format(name))
    return (
        "ORB",
        cv2.ORB_create(nfeatures=int(maximum_features), fastThreshold=8),
        cv2.NORM_HAMMING,
    )


def _mask_polygon(shape: Sequence[int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros((int(shape[0]), int(shape[1])), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 255, cv2.LINE_8)
    return mask


def _mutual_geometric_correspondences(
    reference_points: np.ndarray,
    observed_points_screen: np.ndarray,
    thresholds: Sequence[float],
) -> Dict[int, int]:
    if not len(reference_points) or not len(observed_points_screen):
        return {int(value): 0 for value in thresholds}
    first = np.asarray(reference_points, dtype=np.float32)
    second = np.asarray(observed_points_screen, dtype=np.float32)
    matches = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(first, second)
    distances = np.asarray([item.distance for item in matches], dtype=np.float64)
    return {
        int(threshold): int(np.count_nonzero(distances <= float(threshold)))
        for threshold in thresholds
    }


def _grid_coverage(points: np.ndarray, rect_xywh: Sequence[float], grid: int = 6) -> float:
    if not len(points):
        return 0.0
    x, y, width, height = map(float, rect_xywh)
    normalized = (points - [x, y]) / [max(width, 1.0), max(height, 1.0)]
    cells = np.floor(normalized * grid).astype(np.int32)
    valid = np.all((cells >= 0) & (cells < grid), axis=1)
    occupied = {(int(item[0]), int(item[1])) for item in cells[valid]}
    return float(len(occupied) / float(grid * grid))


def _homography_pose_error(
    estimated_camera_to_screen: np.ndarray,
    ground_truth_camera_to_screen: np.ndarray,
    rect_screen_xywh: Sequence[float],
) -> Dict[str, Any]:
    polygon = _rect_polygon(rect_screen_xywh)
    center = np.mean(polygon, axis=0)
    probes_screen = np.vstack([polygon, center, center + [10.0, 0.0]])
    camera_probes = transform_points(probes_screen, np.linalg.inv(ground_truth_camera_to_screen))
    estimated_screen = transform_points(camera_probes, estimated_camera_to_screen)
    errors = np.linalg.norm(estimated_screen[:-1] - probes_screen[:-1], axis=1)
    expected_vector = probes_screen[-1] - probes_screen[-2]
    measured_vector = estimated_screen[-1] - estimated_screen[-2]
    expected_angle = math.degrees(math.atan2(expected_vector[1], expected_vector[0]))
    measured_angle = math.degrees(math.atan2(measured_vector[1], measured_vector[0]))
    rotation_error = abs((measured_angle - expected_angle + 180.0) % 360.0 - 180.0)
    return {
        "probe_count": int(len(errors)),
        "reprojection_error_median_display_px": float(np.median(errors)),
        "reprojection_error_p95_display_px": float(np.percentile(errors, 95)),
        "reprojection_error_max_display_px": float(np.max(errors)),
        "rotation_error_deg": float(rotation_error),
    }


def evaluate_feature_matching(
    reference_display_image: np.ndarray,
    observed_camera_image: np.ndarray,
    camera_to_screen_3x3: Sequence[Sequence[float]],
    rect_screen_xywh: Sequence[float],
    reference_mode: str = "generated_to_camera",
    detector: str = "auto",
    maximum_features: int = 2000,
    thresholds_display_px: Sequence[int] = tuple(range(1, 11)),
    geometry_confidence: float = 1.0,
) -> Dict[str, Any]:
    """Evaluate detector and descriptor behavior against ChArUco geometry."""

    if reference_display_image is None or reference_display_image.size == 0:
        raise ValueError("Feature reference image is empty")
    if observed_camera_image is None or observed_camera_image.size == 0:
        raise ValueError("Observed camera image is empty")
    transform = matrix_3x3(camera_to_screen_3x3)
    rect = _rect_polygon(rect_screen_xywh)
    inverse = np.linalg.inv(transform)
    camera_rect = transform_points(rect, inverse)
    reference_mask = _mask_polygon(reference_display_image.shape, rect)
    camera_mask = _mask_polygon(observed_camera_image.shape, camera_rect)
    reference_gray = (
        cv2.cvtColor(reference_display_image, cv2.COLOR_BGR2GRAY)
        if reference_display_image.ndim == 3
        else reference_display_image
    )
    camera_gray = (
        cv2.cvtColor(observed_camera_image, cv2.COLOR_BGR2GRAY)
        if observed_camera_image.ndim == 3
        else observed_camera_image
    )
    detector_name, implementation, norm = _create_detector(detector, maximum_features)
    reference_keypoints, reference_descriptors = implementation.detectAndCompute(
        reference_gray, reference_mask
    )
    camera_keypoints, camera_descriptors = implementation.detectAndCompute(
        camera_gray, camera_mask
    )
    reference_keypoints = reference_keypoints or []
    camera_keypoints = camera_keypoints or []
    if reference_descriptors is None or camera_descriptors is None:
        raise ValueError("Feature detector returned no descriptors in the visible target ROI")
    reference_points = np.asarray([item.pt for item in reference_keypoints], dtype=np.float64)
    camera_points = np.asarray([item.pt for item in camera_keypoints], dtype=np.float64)
    observed_points_screen = transform_points(camera_points, transform)
    thresholds = sorted({int(value) for value in thresholds_display_px if int(value) > 0})
    if not thresholds:
        raise ValueError("At least one positive reprojection threshold is required")

    repeatable = _mutual_geometric_correspondences(
        reference_points, observed_points_screen, thresholds
    )
    denominator = max(1, min(len(reference_points), len(observed_points_screen)))
    matcher = cv2.BFMatcher(norm, crossCheck=True)
    descriptor_matches = sorted(
        matcher.match(reference_descriptors, camera_descriptors), key=lambda item: item.distance
    )
    match_errors = np.asarray(
        [
            np.linalg.norm(
                reference_points[item.queryIdx] - observed_points_screen[item.trainIdx]
            )
            for item in descriptor_matches
        ],
        dtype=np.float64,
    )
    mma = {
        int(threshold): float(np.mean(match_errors <= float(threshold)))
        if len(match_errors)
        else 0.0
        for threshold in thresholds
    }
    correct_counts = {
        int(threshold): int(np.count_nonzero(match_errors <= float(threshold)))
        for threshold in thresholds
    }
    primary_threshold = 3 if 3 in thresholds else thresholds[min(2, len(thresholds) - 1)]
    correct_primary = correct_counts[primary_threshold]
    correct_reference_points = np.asarray(
        [
            reference_points[item.queryIdx]
            for item, error in zip(descriptor_matches, match_errors)
            if error <= primary_threshold
        ],
        dtype=np.float64,
    ).reshape((-1, 2))
    coverage = _grid_coverage(correct_reference_points, rect_screen_xywh)

    downstream: Dict[str, Any]
    if len(descriptor_matches) >= 4:
        matched_camera = np.asarray(
            [camera_points[item.trainIdx] for item in descriptor_matches], dtype=np.float64
        )
        matched_reference = np.asarray(
            [reference_points[item.queryIdx] for item in descriptor_matches], dtype=np.float64
        )
        estimated, inliers = cv2.findHomography(
            matched_camera,
            matched_reference,
            cv2.RANSAC,
            float(primary_threshold),
        )
    else:
        estimated, inliers = None, None
    if estimated is None or inliers is None or int(np.count_nonzero(inliers)) < 4:
        downstream = {
            "status": "insufficient_consistent_matches",
            "homography_inlier_count": 0,
        }
    else:
        downstream = {
            "status": "estimated",
            "homography_inlier_count": int(np.count_nonzero(inliers)),
            "estimated_camera_to_screen_3x3": (
                estimated / estimated[2, 2]
            ).tolist(),
        }
        downstream.update(_homography_pose_error(estimated, transform, rect_screen_xywh))

    repeatability_at_primary = repeatable[primary_threshold] / float(denominator)
    matching_score = correct_primary / float(denominator)
    count_quality = min(1.0, correct_primary / 80.0)
    confidence = float(
        0.20 * float(np.clip(geometry_confidence, 0.0, 1.0))
        + 0.25 * min(1.0, repeatability_at_primary)
        + 0.25 * min(1.0, matching_score)
        + 0.15 * min(1.0, coverage / 0.50)
        + 0.15 * count_quality
    )
    match_examples = []
    for item, error in list(zip(descriptor_matches, match_errors))[:250]:
        match_examples.append(
            {
                "reference_display_xy": reference_points[item.queryIdx].tolist(),
                "camera_xy": camera_points[item.trainIdx].tolist(),
                "projected_camera_display_xy": observed_points_screen[item.trainIdx].tolist(),
                "reprojection_error_display_px": float(error),
                "correct_at_primary_threshold": bool(error <= primary_threshold),
            }
        )
    return {
        "protocol": "planar_homography_ground_truth",
        "aggregation": "macro_average_over_image_pairs",
        "reference_mode": str(reference_mode),
        "detector_descriptor": detector_name,
        "threshold_space": "canonical_display_px",
        "primary_threshold_display_px": int(primary_threshold),
        "reference_feature_count": int(len(reference_points)),
        "observed_feature_count": int(len(observed_points_screen)),
        "evaluated_match_count": int(len(descriptor_matches)),
        "repeatable_count_by_threshold_px": repeatable,
        "repeatability_by_threshold_px": {
            threshold: float(count / float(denominator))
            for threshold, count in repeatable.items()
        },
        "correct_match_count_by_threshold_px": correct_counts,
        "mma_by_threshold_px": mma,
        "matching_score_at_primary_threshold": float(matching_score),
        "correct_match_spatial_coverage": float(coverage),
        "match_error_median_display_px": (
            float(np.median(match_errors)) if len(match_errors) else None
        ),
        "match_error_p95_display_px": (
            float(np.percentile(match_errors, 95)) if len(match_errors) else None
        ),
        "downstream_geometry": downstream,
        "match_examples": match_examples,
        "confidence": confidence,
        "confidence_components": {
            "geometry": float(np.clip(geometry_confidence, 0.0, 1.0)),
            "repeatability": float(repeatability_at_primary),
            "matching_score": float(matching_score),
            "spatial_coverage": float(coverage),
            "correct_match_count": float(count_quality),
        },
    }


def aggregate_feature_matching(
    trials: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate counts across held-out trials while retaining every trial."""

    rows = [dict(item) for item in trials]
    if not rows:
        raise ValueError("At least one feature-matching trial is required")
    thresholds = sorted(
        {int(key) for row in rows for key in row["mma_by_threshold_px"].keys()}
    )
    evaluated_total = sum(int(row["evaluated_match_count"]) for row in rows)
    denominator_total = sum(
        min(int(row["reference_feature_count"]), int(row["observed_feature_count"]))
        for row in rows
    )
    correct_totals = {
        threshold: sum(
            int(row["correct_match_count_by_threshold_px"].get(threshold, 0))
            for row in rows
        )
        for threshold in thresholds
    }
    primary = int(rows[0]["primary_threshold_display_px"])
    downstream_p95 = [
        float(row["downstream_geometry"]["reprojection_error_p95_display_px"])
        for row in rows
        if row.get("downstream_geometry", {}).get("status") == "estimated"
    ]
    catastrophic = [
        row.get("downstream_geometry", {}).get("status") != "estimated"
        or float(row["downstream_geometry"].get("reprojection_error_p95_display_px", 1.0e9))
        > 10.0
        for row in rows
    ]
    return {
        "protocol": "planar_homography_ground_truth",
        "threshold_space": "canonical_display_px",
        "reference_modes": sorted({str(row["reference_mode"]) for row in rows}),
        "detector_descriptors": sorted(
            {str(row["detector_descriptor"]) for row in rows}
        ),
        "trial_count": len(rows),
        "primary_threshold_display_px": primary,
        "evaluated_match_count": int(evaluated_total),
        "denominator_feature_count": int(denominator_total),
        "repeatability_by_threshold_px": {
            threshold: float(
                np.mean(
                    [
                        row["repeatability_by_threshold_px"].get(threshold, 0.0)
                        for row in rows
                    ]
                )
            )
            for threshold in thresholds
        },
        "matching_score_by_threshold_px": {
            threshold: float(
                np.mean(
                    [
                        row["correct_match_count_by_threshold_px"].get(threshold, 0)
                        / float(
                            max(
                                1,
                                min(
                                    int(row["reference_feature_count"]),
                                    int(row["observed_feature_count"]),
                                ),
                            )
                        )
                        for row in rows
                    ]
                )
            )
            for threshold in thresholds
        },
        "mma_by_threshold_px": {
            threshold: float(
                np.mean(
                    [row["mma_by_threshold_px"].get(threshold, 0.0) for row in rows]
                )
            )
            for threshold in thresholds
        },
        "correct_match_count_by_threshold_px": correct_totals,
        "spatial_coverage_min": float(
            min(float(row["correct_match_spatial_coverage"]) for row in rows)
        ),
        "spatial_coverage_median": float(
            np.median([row["correct_match_spatial_coverage"] for row in rows])
        ),
        "downstream_reprojection_p95_display_px": (
            float(np.percentile(downstream_p95, 95)) if downstream_p95 else None
        ),
        "catastrophic_mismatch_rate": float(np.mean(catastrophic)),
        "confidence": float(min(float(row.get("confidence", 0.0)) for row in rows)),
        "trials": rows,
    }
