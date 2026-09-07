"""Unattended distortion correction for a stationary camera/display plane.

The centered, square camera matrix is a normalization gauge, not an estimate of
physical intrinsics. A free homography absorbs the plane projection. Spatial
corners withheld from every fit and a later frame gate the exact saved model.
"""
from __future__ import annotations

import cv2
import numpy as np

from .contracts import points_xy
from .distortion import distort_pixel_points, undistort_pixel_points
from .geometry import transform_points


def fixed_plane_pose_matches(lens_model, camera_points, screen_points):
    """Check that later geometry frames still describe the measured placement."""
    evidence = lens_model["observations"]
    reference = dict(zip(map(tuple, evidence["display_points_xy"]), evidence["camera_points_by_frame"][-1]))
    displacements = [np.linalg.norm(np.asarray(raw) - reference[tuple(np.round(screen, 4))])
                     for raw, screen in zip(camera_points, screen_points)
                     if tuple(np.round(screen, 4)) in reference]
    return len(displacements) >= 36 and float(np.percentile(displacements, 95)) <= 2.0


def _fit(camera, screen, size):
    scale = float(max(size))
    center = (np.asarray(size, dtype=float) - 1) / 2
    xy = (screen - screen.mean(axis=0)) / np.ptp(screen, axis=0).max()
    target = (camera - center) / scale
    homography, _ = cv2.findHomography(xy, target, 0)
    if homography is None:
        raise ValueError("Cannot initialize fixed-plane distortion fit")
    parameters = np.r_[homography.ravel()[:8] / homography[2, 2], np.zeros(4)]
    homogeneous = np.c_[xy, np.ones(len(xy))]

    def project(values):
        projected = homogeneous @ np.r_[values[:8], 1].reshape(3, 3).T
        if np.any(np.abs(projected[:, 2]) < 1e-10):
            return np.full(target.shape, np.inf)
        x, y = (projected[:, :2] / projected[:, 2:]).T
        r2 = x * x + y * y
        k1, k2, p1, p2 = values[8:]
        radial = 1 + k1 * r2 + k2 * r2 * r2
        return np.c_[x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                     y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y]

    damping = 1e-3
    # Bounded damped least squares; no new runtime dependency or unbounded search.
    for _ in range(100):
        prediction = project(parameters)
        residual = (prediction - target).ravel()
        jacobian = np.empty((residual.size, parameters.size))
        for column in range(parameters.size):
            moved = parameters.copy()
            step = 1e-6 * (1 + abs(parameters[column]))
            moved[column] += step
            jacobian[:, column] = ((project(moved) - prediction) / step).ravel()
        normal = jacobian.T @ jacobian
        update = np.linalg.solve(
            normal + damping * np.diag(np.maximum(np.diag(normal), 1e-12)),
            -jacobian.T @ residual,
        )
        trial = parameters + update
        if np.sum((project(trial) - target) ** 2) < residual @ residual:
            parameters = trial
            damping = max(damping / 3, 1e-12)
            if np.linalg.norm(update) < 1e-9:
                break
        else:
            damping = min(damping * 10, 1e12)
    if not np.isfinite(parameters).all():
        raise ValueError("Non-finite fixed-plane distortion fit")
    return {
        "camera_matrix_3x3": [[scale, 0, float(center[0])],
                              [0, scale, float(center[1])], [0, 0, 1]],
        "distortion_coefficients": [*parameters[8:].tolist(), 0.0],
    }


def fit_fixed_plane_distortion(camera_points_by_frame, screen_points_by_frame,
                               camera_size_px, minimum_relative_p95_improvement=0.05):
    """Fit stationary observations; never fit on held-out corner identities.

    The last frame is reserved for evaluation. All observations must share at
    least 36 non-collinear corners. Improvement must exceed 0.05 display pixels
    as well as the requested relative threshold, without worsening max error.
    """
    size = np.asarray(camera_size_px, dtype=int)
    if size.shape != (2,) or np.min(size) <= 0:
        raise ValueError("Camera size must contain two positive dimensions")
    if not 0 <= minimum_relative_p95_improvement <= 1:
        raise ValueError("Minimum relative improvement must be within 0..1")
    if len(camera_points_by_frame) != len(screen_points_by_frame):
        raise ValueError("Camera and display frame counts differ")
    result = {
        "source": "unavailable", "accepted": False, "model": "opencv_radtan",
        "calibration_method": "fixed_plane_joint_homography_radtan",
        "camera_matrix_role": "normalization_gauge_not_physical_intrinsics",
        "scope": "fixed_camera_and_display_plane",
        "training_frame_count": max(0, len(camera_points_by_frame) - 1),
        "holdout_frame_count": 1,
        "minimum_relative_p95_improvement": float(minimum_relative_p95_improvement),
        "minimum_absolute_p95_improvement_screen_px": 0.05,
    }
    if len(camera_points_by_frame) < 4:
        return {**result, "reason": "At least four detected stationary frames are required"}
    frames = []
    for camera, screen in zip(camera_points_by_frame, screen_points_by_frame):
        camera, screen = points_xy(camera), points_xy(screen)
        if camera.shape != screen.shape:
            raise ValueError("Camera and display corner counts differ")
        frames.append({tuple(np.round(point, 4)): value for point, value in zip(screen, camera)})
    keys = sorted(set.intersection(*(set(frame) for frame in frames)))
    if len(keys) < 36:
        return {**result, "reason": "Fewer than 36 shared ChArUco corners; insufficient spatial evidence"}
    screen = np.asarray(keys)
    camera = np.asarray([[frame[key] for key in keys] for frame in frames])
    if np.linalg.matrix_rank(screen - screen.mean(axis=0)) < 2:
        return {**result, "reason": "ChArUco corners do not span a two-dimensional region"}
    movement = float(np.max(np.percentile(np.linalg.norm(camera - camera[0], axis=2), 95, axis=1)))
    result["stationary_p95_displacement_camera_px"] = movement
    if movement > 2.0:
        return {**result, "reason": "Target moved or corner measurements were unstable during automatic collection"}
    training = np.arange(len(keys)) % 3 != 0
    result["training_corner_count"] = int(training.sum())
    result["holdout_corner_count"] = int((~training).sum())
    result["holdout_display_points_xy"] = screen[~training].tolist()
    result["observations"] = {"display_points_xy": screen.tolist(), "camera_points_by_frame": camera.tolist()}
    averaged = np.median(camera[:-1], axis=0)
    try:
        candidate = _fit(averaged[training], screen[training], size)
        result["candidate"] = candidate
        # Reject folding or an unstable inverse across the raw sensor before
        # using the model in viewport/ROI planning outside detected corners.
        xx, yy = np.meshgrid(np.linspace(0, size[0] - 1, 17), np.linspace(0, size[1] - 1, 17))
        probe = np.c_[xx.ravel(), yy.ravel()]
        ideal = undistort_pixel_points(probe, candidate)
        roundtrip = distort_pixel_points(ideal, candidate)
        result["sensor_inverse_roundtrip_max_px"] = float(np.max(np.linalg.norm(roundtrip - probe, axis=1)))
        mapped = distort_pixel_points(probe, candidate)
        dx = distort_pixel_points(probe + [0.1, 0], candidate) - mapped
        dy = distort_pixel_points(probe + [0, 0.1], candidate) - mapped
        if (not np.isfinite(roundtrip).all() or result["sensor_inverse_roundtrip_max_px"] > 0.25
                or np.any(dx[:, 0] * dy[:, 1] - dx[:, 1] * dy[:, 0] <= 0)):
            return {**result, "reason": "Candidate distortion map folds or has an unstable sensor inverse"}

        def errors(lens):
            source = camera[-1] if lens is None else undistort_pixel_points(camera[-1], lens)
            homography, _ = cv2.findHomography(source[training], screen[training], 0)
            if homography is None:
                raise ValueError("Cannot fit holdout homography")
            return np.linalg.norm(transform_points(source[~training], homography) - screen[~training], axis=1)

        baseline, corrected = errors(None), errors(candidate)
        baseline_p95, candidate_p95 = float(np.percentile(baseline, 95)), float(np.percentile(corrected, 95))
        relative = (baseline_p95 - candidate_p95) / max(baseline_p95, 1e-12)
        accepted = bool(np.isfinite(corrected).all() and baseline_p95 - candidate_p95 >= 0.05
                        and relative >= minimum_relative_p95_improvement and max(corrected) <= max(baseline))
        result["holdout"] = {
            "baseline_homography_p95_screen_px": baseline_p95, "candidate_p95_screen_px": candidate_p95,
            "relative_p95_improvement": relative,
            "baseline_max_screen_px": float(max(baseline)), "candidate_max_screen_px": float(max(corrected)),
            "validation_corner_count": int((~training).sum()),
        }
        result.update(source="measured" if accepted else "rejected_holdout", accepted=accepted,
                      reason="Independent spatial/frame holdout improved" if accepted else
                      "Correction did not improve held-out geometry enough; retaining homography-only")
        if accepted:
            # Keep the evaluated candidate exactly; do not refit on holdout data.
            result.update(candidate)
        return result
    except (ValueError, np.linalg.LinAlgError, cv2.error) as exc:
        return {**result, "reason": "Automatic distortion fit unavailable: {}".format(exc)}
