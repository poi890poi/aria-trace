"""Evidence-gated lens distortion calibration and remap composition."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .contracts import matrix_3x3, points_xy
from .geometry import calibrate_intrinsics_from_views, transform_points


def undistort_pixel_points(
    points: Sequence[Sequence[float]], lens_model: Mapping[str, Any]
) -> np.ndarray:
    camera_matrix = matrix_3x3(lens_model["camera_matrix_3x3"])
    distortion = np.asarray(
        lens_model["distortion_coefficients"], dtype=np.float64
    ).reshape(-1)
    value = points_xy(points).reshape((-1, 1, 2)).astype(np.float64)
    return cv2.undistortPoints(
        value, camera_matrix, distortion, P=camera_matrix
    ).reshape((-1, 2))


def distort_pixel_points(
    points: Sequence[Sequence[float]], lens_model: Mapping[str, Any]
) -> np.ndarray:
    """Map ideal full-sensor pixel coordinates back to the raw distorted raster."""

    camera_matrix = matrix_3x3(lens_model["camera_matrix_3x3"])
    distortion = np.asarray(
        lens_model["distortion_coefficients"], dtype=np.float64
    ).reshape(-1)
    ideal = points_xy(points)
    inverse = np.linalg.inv(camera_matrix)
    homogeneous = np.column_stack([ideal, np.ones(len(ideal))]).dot(inverse.T)
    objects = np.column_stack(
        [
            homogeneous[:, 0] / homogeneous[:, 2],
            homogeneous[:, 1] / homogeneous[:, 2],
            np.ones(len(ideal)),
        ]
    )
    projected, _ = cv2.projectPoints(
        objects,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        camera_matrix,
        distortion,
    )
    return projected.reshape((-1, 2))


def _validation_errors(
    camera_points: np.ndarray,
    screen_points: np.ndarray,
    lens_model: Mapping[str, Any] | None,
) -> np.ndarray:
    count = len(camera_points)
    validation = np.arange(count) % 3 == 0
    training = ~validation
    if np.count_nonzero(training) < 4 or np.count_nonzero(validation) < 4:
        raise ValueError("Held-out ChArUco view needs at least 12 distributed corners")
    source = (
        undistort_pixel_points(camera_points, lens_model)
        if lens_model is not None
        else camera_points
    )
    homography, _ = cv2.findHomography(
        source[training].astype(np.float64),
        screen_points[training].astype(np.float64),
        0,
    )
    if homography is None:
        raise RuntimeError("Cannot fit held-out ChArUco homography")
    predicted = transform_points(source[validation], homography)
    return np.linalg.norm(predicted - screen_points[validation], axis=1)


def fit_evidence_gated_distortion(
    camera_points_by_view: Sequence[Sequence[Sequence[float]]],
    screen_points_by_view: Sequence[Sequence[Sequence[float]]],
    camera_size_px: Sequence[int],
    minimum_relative_p95_improvement: float = 0.05,
) -> dict[str, Any]:
    """Fit on all but one view and accept only an independent holdout improvement."""

    if len(camera_points_by_view) < 4:
        raise ValueError("Distortion calibration needs at least four distinct views")
    if not 0.0 <= float(minimum_relative_p95_improvement) <= 1.0:
        raise ValueError("Minimum relative improvement must be within 0..1")
    training_camera = list(camera_points_by_view[:-1])
    training_screen = list(screen_points_by_view[:-1])
    candidate = calibrate_intrinsics_from_views(
        training_camera, training_screen, camera_size_px
    )
    holdout_camera = points_xy(camera_points_by_view[-1], minimum=12)
    holdout_screen = points_xy(screen_points_by_view[-1], minimum=12)
    baseline_errors = _validation_errors(
        holdout_camera, holdout_screen, lens_model=None
    )
    candidate_errors = _validation_errors(
        holdout_camera, holdout_screen, lens_model=candidate
    )
    baseline_p95 = float(np.percentile(baseline_errors, 95))
    candidate_p95 = float(np.percentile(candidate_errors, 95))
    relative = (baseline_p95 - candidate_p95) / max(baseline_p95, 1.0e-12)
    accepted = bool(
        candidate_p95 < baseline_p95
        and float(np.max(candidate_errors)) <= float(np.max(baseline_errors))
        and relative >= float(minimum_relative_p95_improvement)
    )
    result: dict[str, Any] = {
        "source": "measured" if accepted else "rejected_holdout",
        "model": "opencv_radtan",
        "accepted": accepted,
        "training_view_count": len(training_camera),
        "holdout_view_count": 1,
        "minimum_relative_p95_improvement": float(
            minimum_relative_p95_improvement
        ),
        "holdout": {
            "baseline_homography_p95_screen_px": baseline_p95,
            "candidate_p95_screen_px": candidate_p95,
            "baseline_max_screen_px": float(np.max(baseline_errors)),
            "candidate_max_screen_px": float(np.max(candidate_errors)),
            "relative_p95_improvement": float(relative),
            "validation_corner_count": int(len(candidate_errors)),
        },
    }
    if accepted:
        final = calibrate_intrinsics_from_views(
            camera_points_by_view, screen_points_by_view, camera_size_px
        )
        result.update(final)
        result["source"] = "measured"
        result["accepted"] = True
    return result


def combined_output_to_raw_maps(
    undistorted_camera_to_output_3x3: Sequence[Sequence[float]],
    output_size_px: Sequence[int],
    lens_model: Mapping[str, Any] | None = None,
    chunk_size: int = 250_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Precompose homography and optional distortion into one runtime remap."""

    width, height = map(int, output_size_px)
    if min(width, height) <= 0:
        raise ValueError("Output size must be positive")
    inverse = np.linalg.inv(matrix_3x3(undistorted_camera_to_output_3x3))
    yy, xx = np.mgrid[0:height, 0:width]
    output = np.column_stack([xx.reshape(-1), yy.reshape(-1)]).astype(np.float64)
    ideal_camera = transform_points(output, inverse)
    if lens_model is None or lens_model.get("source") != "measured":
        raw_camera = ideal_camera
    else:
        pieces = []
        for start in range(0, len(ideal_camera), max(1, int(chunk_size))):
            pieces.append(
                distort_pixel_points(
                    ideal_camera[start : start + int(chunk_size)], lens_model
                )
            )
        raw_camera = np.vstack(pieces)
    return (
        raw_camera[:, 0].reshape((height, width)).astype(np.float32),
        raw_camera[:, 1].reshape((height, width)).astype(np.float32),
    )


def distorted_screen_region_mask(
    camera_size_px: Sequence[int],
    screen_region_xywh: Sequence[int],
    screen_to_undistorted_camera_3x3: Sequence[Sequence[float]],
    lens_model: Mapping[str, Any],
    inset_screen_px: int = 8,
) -> np.ndarray:
    """Rasterize a display rectangle in raw camera pixels through lens distortion."""

    camera_width, camera_height = map(int, camera_size_px)
    x, y, width, height = map(float, screen_region_xywh)
    inset = float(max(0, int(inset_screen_px)))
    left, top = x + inset, y + inset
    right, bottom = x + width - inset, y + height - inset
    if right <= left or bottom <= top:
        raise ValueError("Screen region is too small for its inset")
    count = 33
    xs = np.linspace(left, right, count)
    ys = np.linspace(top, bottom, count)
    boundary = np.vstack(
        [
            np.column_stack([xs, np.full(count, top)]),
            np.column_stack([np.full(count, right), ys]),
            np.column_stack([xs[::-1], np.full(count, bottom)]),
            np.column_stack([np.full(count, left), ys[::-1]]),
        ]
    )
    ideal = transform_points(boundary, screen_to_undistorted_camera_3x3)
    raw = distort_pixel_points(ideal, lens_model)
    polygon = np.rint(raw).astype(np.int32).reshape((-1, 1, 2))
    mask = np.zeros((camera_height, camera_width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def distorted_screen_region_roi(
    camera_size_px: Sequence[int],
    screen_region_xywh: Sequence[int],
    screen_to_undistorted_camera_3x3: Sequence[Sequence[float]],
    lens_model: Mapping[str, Any],
    margin_px: int = 4,
) -> list[int]:
    """Bound a screen rectangle in the raw distorted camera raster."""

    camera_width, camera_height = map(int, camera_size_px)
    x, y, width, height = map(float, screen_region_xywh)
    count = 33
    xs = np.linspace(x, x + width, count)
    ys = np.linspace(y, y + height, count)
    screen_boundary = np.vstack(
        [
            np.column_stack([xs, np.full(count, y)]),
            np.column_stack([np.full(count, x + width), ys]),
            np.column_stack([xs[::-1], np.full(count, y + height)]),
            np.column_stack([np.full(count, x), ys[::-1]]),
        ]
    )
    ideal = transform_points(
        screen_boundary, matrix_3x3(screen_to_undistorted_camera_3x3)
    )
    raw = distort_pixel_points(ideal, lens_model)
    margin = max(0, int(margin_px))
    left = max(0, int(np.floor(float(np.min(raw[:, 0])))) - margin)
    top = max(0, int(np.floor(float(np.min(raw[:, 1])))) - margin)
    right = min(camera_width, int(np.ceil(float(np.max(raw[:, 0])))) + margin)
    bottom = min(camera_height, int(np.ceil(float(np.max(raw[:, 1])))) + margin)
    if right <= left or bottom <= top:
        raise RuntimeError("Phone region does not map to a valid raw camera ROI")
    return [left, top, right - left, bottom - top]


def raw_roi_for_maps(
    map_x: np.ndarray,
    map_y: np.ndarray,
    full_sensor_size_px: Sequence[int],
    margin_px: int = 2,
) -> list[int]:
    width, height = map(int, full_sensor_size_px)
    valid = (
        np.isfinite(map_x)
        & np.isfinite(map_y)
        & (map_x >= 0)
        & (map_y >= 0)
        & (map_x <= width - 1)
        & (map_y <= height - 1)
    )
    if not np.any(valid):
        raise RuntimeError("Rectification map does not sample the HIK sensor")
    margin = max(0, int(margin_px))
    left = max(0, int(np.floor(float(np.min(map_x[valid])))) - margin)
    top = max(0, int(np.floor(float(np.min(map_y[valid])))) - margin)
    right = min(width, int(np.ceil(float(np.max(map_x[valid])))) + margin + 1)
    bottom = min(height, int(np.ceil(float(np.max(map_y[valid])))) + margin + 1)
    return [left, top, right - left, bottom - top]


def raw_sensor_viewport_in_screen(
    camera_size_px: Sequence[int],
    undistorted_camera_to_screen_3x3: Sequence[Sequence[float]],
    lens_model: Mapping[str, Any],
    samples_per_edge: int = 65,
) -> np.ndarray:
    """Project the distorted sensor boundary into screen space.

    A four-corner homography viewport is insufficient once raw pixels pass
    through a nonlinear lens model. Sampling the physical raw-raster boundary
    keeps visible-region selection conservative without affecting runtime.
    """

    width, height = map(int, camera_size_px)
    if min(width, height) <= 0:
        raise ValueError("Camera size must be positive")
    count = max(3, int(samples_per_edge))
    xs = np.linspace(0.0, float(width - 1), count)
    ys = np.linspace(0.0, float(height - 1), count)
    raw_boundary = np.vstack(
        [
            np.column_stack([xs, np.zeros(count)]),
            np.column_stack([np.full(count, width - 1.0), ys]),
            np.column_stack([xs[::-1], np.full(count, height - 1.0)]),
            np.column_stack([np.zeros(count), ys[::-1]]),
        ]
    )
    ideal_boundary = undistort_pixel_points(raw_boundary, lens_model)
    screen_boundary = transform_points(
        ideal_boundary, matrix_3x3(undistorted_camera_to_screen_3x3)
    )
    return cv2.convexHull(screen_boundary.astype(np.float32)).reshape((-1, 2))
