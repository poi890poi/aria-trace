"""Color-independent pivot fitting from an incomplete temporal cold core."""

import cv2
import numpy as np


def fit_temporal_cold_circle(frames, initial, search_radius):
    """Vote along cold-facing edge normals near the known mini-map center.

    The per-pixel range ignores dwell time and frame order. A rotating cursor
    commonly leaves a quiet central disc; only its visible inward boundary
    needs to be circular, not the whole motion envelope or direction histogram.
    The initial point constrains the search but is never substituted as a fit.
    """
    height, width = frames.shape[1:3]
    initial = np.asarray(initial, dtype=np.float64)
    if search_radius <= 0 or not np.isfinite(search_radius) or not np.isfinite(initial).all():
        raise RuntimeError("Cursor rotation-center search requires a finite positive radius and center")
    max_radius = min(15.0, max(3.0, search_radius))
    extent = int(np.ceil(search_radius + max_radius + 4))
    x0, y0 = np.maximum(0, np.floor(initial - extent).astype(int))
    x1, y1 = np.minimum([width, height], np.ceil(initial + extent + 1).astype(int))
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("Cursor rotation-center search region does not fit the frame")
    local = frames[:, y0:y1, x0:x1]
    # Subtract after conversion to avoid uint8 subtraction wrapping. No frame
    # multiplicity weights or ordered frame differences enter this measurement.
    raw = (local.max(axis=0).astype(np.float32) - local.min(axis=0)).mean(axis=2)
    temporal = cv2.GaussianBlur(raw, (5, 5), 0.8)
    yy, xx = np.indices(raw.shape)
    origin = initial - [x0, y0]
    region = (xx - origin[0]) ** 2 + (yy - origin[1]) ** 2 <= (search_radius + max_radius) ** 2
    floor = float(np.percentile(temporal[region], 35))
    contrast = float(np.percentile(temporal[region], 99) - floor)
    if contrast < 2.0:
        raise RuntimeError("Cursor rotation center has no observable temporal signal near the mini-map center")
    # Clipping the background floor also suppresses spatially uniform flicker.
    temporal = np.maximum(temporal - floor, 0)
    gx = cv2.Sobel(temporal, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(temporal, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    peak = float(magnitude[region].max())
    eligible = region & (magnitude > max(1e-6, peak * 0.18))
    y, x = np.nonzero(eligible)
    if len(x) < 6:
        raise RuntimeError("Cursor rotation center has too few observable cold-core edges")
    strength = magnitude[y, x] / peak
    ux, uy = gx[y, x] / magnitude[y, x], gy[y, x] / magnitude[y, x]
    grid_y, grid_x = np.indices((raw.shape[0] * 2, raw.shape[1] * 2))
    allowed = (grid_x / 2 - origin[0]) ** 2 + (grid_y / 2 - origin[1]) ** 2 <= search_radius ** 2
    scores = np.zeros(allowed.shape, np.float32)
    best = None
    for radius in np.arange(2.0, max_radius + 0.01, 0.5):
        # The gradient points from cold to hot. Vote on its cold side only,
        # avoiding the outer hot-envelope circle, which need not be complete.
        ix = np.rint(2 * (x - radius * ux)).astype(int)
        iy = np.rint(2 * (y - radius * uy)).astype(int)
        valid = (ix >= 0) & (iy >= 0) & (ix < scores.shape[1]) & (iy < scores.shape[0])
        votes = np.zeros_like(scores)
        np.add.at(votes, (iy[valid], ix[valid]), strength[valid])
        votes = cv2.GaussianBlur(votes, (9, 9), 1.5) / np.sqrt(radius)
        votes[~allowed] = 0
        np.maximum(scores, votes, out=scores)
        j, i = np.unravel_index(np.argmax(votes), votes.shape)
        if best is None or votes[j, i] > best[0]:
            best = (float(votes[j, i]), i / 2, j / 2, float(radius))
    score, cx, cy, radius = best
    distance = np.hypot(x - cx, y - cy)
    alignment = ((x - cx) * ux + (y - cy) * uy) / np.maximum(distance, 1e-6)
    inliers = (np.abs(distance - radius) < 1.3) & (alignment > 0.92)
    if score <= 0 or np.count_nonzero(inliers) < 6:
        raise RuntimeError("Cursor rotation center has no supported cold-core circle")
    # Subpixel intersection of the observed normals does not fill missing arcs
    # or reward opposite-direction balance. Conditioning measures localization.
    normals = np.column_stack([-uy[inliers], ux[inliers]])
    points = np.column_stack([x[inliers], y[inliers]])
    weights = np.sqrt(strength[inliers])
    design = normals * weights[:, None]
    eigenvalues = np.linalg.eigvalsh(design.T @ design)
    conditioning = float(eigenvalues[0] / max(eigenvalues[1], 1e-6))
    if conditioning < 0.04:
        raise RuntimeError("Cursor cold-core arc does not constrain a rotation center in two dimensions")
    fitted = np.linalg.lstsq(design, (normals * points).sum(axis=1) * weights, rcond=None)[0]
    if np.linalg.norm(fitted - [cx, cy]) <= 2 and np.linalg.norm(fitted - origin) <= search_radius:
        cx, cy = map(float, fitted)
    radii = np.linalg.norm(points - [cx, cy], axis=1)
    radius = float(np.average(radii, weights=weights ** 2))
    residual = float(np.sqrt(np.average((radii - radius) ** 2, weights=weights ** 2)))
    angles = np.mod(np.arctan2(points[:, 1] - cy, points[:, 0] - cx), 2 * np.pi)
    occupied = len(np.unique(np.floor(angles * 36 / (2 * np.pi)).astype(int))) / 36
    components = {
        "temporal_signal": float(np.clip(contrast / 20, 0, 1)),
        "normal_conditioning": float(np.sqrt(conditioning)),
        "circle_fit": float(np.exp(-(residual / 1.3) ** 2)),
    }
    confidence = float(np.prod(list(components.values())) ** (1 / len(components)))
    heatmap = np.zeros((height, width), np.float32)
    heatmap[y0:y1, x0:x1] = temporal
    score_map = np.zeros((height, width), np.float32)
    score_map[y0:y1, x0:x1] = scores[::2, ::2]
    return {
        "temporal_heatmap": heatmap,
        "center_score_map": score_map,
        "metrics": {
            "x": cx + int(x0), "y": cy + int(y0),
            "method": "color_agnostic_temporal_cold_circle",
            "centroid_orbit_radius_px": None,
            "temporal_signal_radius_px": radius,
            "cold_core_radius_px": radius,
            "cold_core_edge_count": int(inliers.sum()),
            "cold_core_radial_residual_px": residual,
            "normal_conditioning": conditioning,
            "hough_score": score,
            "analyzed_frames": int(len(frames)), "total_frames": int(len(frames)),
            # Descriptive evidence only: never a balanced-coverage gate.
            "angular_coverage_10deg_bins": occupied,
            "confidence": confidence, "confidence_components": components,
        },
    }
