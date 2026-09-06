"""Bounded fractional peak refinement with unchanged template acceptance."""

import math

import cv2
import numpy as np


def quadratic_peak(response, location):
    """Fit a concave peak; flat responses and image edges stay on the grid."""
    x, y = location

    def offset(a, b, c):
        denominator = float(a) - 2 * float(b) + float(c)
        if denominator >= -1e-8 or not all(math.isfinite(float(v)) for v in (a, b, c)):
            return 0.0
        return float(np.clip(.5 * (float(a) - float(c)) / denominator, -.5, .5))

    dx = offset(*response[y, x-1:x+2]) if 0 < x < response.shape[1]-1 else 0.0
    dy = offset(*response[y-1:y+2, x]) if 0 < y < response.shape[0]-1 else 0.0
    return dx, dy


def fractional_match(localizer, observation_gradient, mask, canonical_xy, search_radius_px):
    """Match current pixels near a proposal; refine XY only after integer scoring.

    The integer winning footprint owns coverage and score, exactly as in the
    existing matcher. Fractional fitting does not weaken either acceptance gate.
    """
    height, width = observation_gradient.shape[:2]
    center_x, center_y = localizer._localization_xy(canonical_xy)
    transform = localizer.original_to_localization
    scale_x = math.hypot(transform[0, 0], transform[1, 0])
    scale_y = math.hypot(transform[0, 1], transform[1, 1])
    radius = max(4., float(search_radius_px) * (scale_x + scale_y) / 2)
    left = max(0, int(math.floor(center_x-radius-width/2)))
    top = max(0, int(math.floor(center_y-radius-height/2)))
    right = min(localizer.map_gradient.shape[1], int(math.ceil(center_x+radius+width/2)))
    bottom = min(localizer.map_gradient.shape[0], int(math.ceil(center_y+radius+height/2)))
    search = localizer.map_gradient[top:bottom, left:right]
    if search.shape[0] < height or search.shape[1] < width:
        return {"valid": False, "score": 0., "coverage_fraction": 0.,
                "reason": "insufficient-local-map-area"}
    response = cv2.matchTemplate(search, observation_gradient, cv2.TM_CCORR_NORMED, mask=mask)
    response = np.nan_to_num(response, nan=-1., posinf=-1., neginf=-1.)
    _, score, _, location = cv2.minMaxLoc(response)
    match_left, match_top = left+location[0], top+location[1]
    coverage = localizer.coverage[match_top:match_top+height, match_left:match_left+width]
    selected = mask > 0
    fraction = float(np.mean(coverage[selected] > 0) if np.any(selected) else 0.)
    dx, dy = quadratic_peak(response, location)
    center = localizer._original_xy((match_left+width/2+dx, match_top+height/2+dy))
    valid = bool(math.isfinite(score) and score >= 0 and fraction >= .75)
    return {
        "valid": valid, "score": max(0., float(score)) if valid else 0.,
        "coverage_fraction": fraction,
        "best_offset_canonical_xy": [float(center[0]-canonical_xy[0]), float(center[1]-canonical_xy[1])],
        "search_bounds_localization_xyxy": [left, top, right, bottom],
        "reason": None if valid else "insufficient-observed-coverage",
        "subpixel_offset_localization_xy": [dx, dy],
    }
