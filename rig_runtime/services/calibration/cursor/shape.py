"""Continuous symmetry-constrained rigid cursor geometry."""

import math

import cv2
import numpy as np


def reflection_matrix(center, axis_angle_deg):
    """Return an affine reflection through an axis passing through center."""
    angle = math.radians(float(axis_angle_deg))
    cosine = math.cos(2.0 * angle)
    sine = math.sin(2.0 * angle)
    linear = np.array([[cosine, sine], [sine, -cosine]], dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    translation = center - linear @ center
    return np.column_stack([linear, translation])


def reflect_image(image, center, axis_angle_deg):
    matrix = reflection_matrix(center, axis_angle_deg)
    return cv2.warpAffine(
        image.astype(np.float32),
        matrix.astype(np.float32),
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def reflect_points(points, center, axis_angle_deg):
    points = np.asarray(points, dtype=np.float64)
    matrix = reflection_matrix(center, axis_angle_deg)
    return points @ matrix[:, :2].T + matrix[:, 2]


def _soft_iou(left, right):
    intersection = float(np.minimum(left, right).sum())
    union = float(np.maximum(left, right).sum())
    return intersection / union if union else 0.0


def _axis_score(probability, center, angle_deg):
    reflected = reflect_image(probability, center, angle_deg)
    return _soft_iou(probability, reflected)


def fit_symmetric_polygon(probability, threshold, center, hint_angle_deg=None):
    """Fit an exactly mirror-symmetric convex polygon to a probability mask."""
    probability = np.asarray(probability, dtype=np.float32)
    center = np.asarray(center, dtype=np.float64)
    coarse_angles = np.arange(0.0, 180.0, 1.0)
    coarse_scores = np.array(
        [_axis_score(probability, center, angle) for angle in coarse_angles]
    )
    if hint_angle_deg is None:
        best_coarse = float(coarse_angles[int(np.argmax(coarse_scores))])
    else:
        hint_axis = float(hint_angle_deg) % 180.0
        top = np.argsort(coarse_scores)[-8:]
        best_coarse = float(
            min(
                coarse_angles[top],
                key=lambda angle: abs((angle - hint_axis + 90.0) % 180.0 - 90.0),
            )
        )
    fine_angles = np.arange(best_coarse - 1.0, best_coarse + 1.001, 0.05)
    fine_scores = np.array(
        [_axis_score(probability, center, angle) for angle in fine_angles]
    )
    axis_angle = float(fine_angles[int(np.argmax(fine_scores))] % 180.0)
    reflected = reflect_image(probability, center, axis_angle)
    symmetric_probability = 0.5 * (probability + reflected)
    symmetric_binary = (symmetric_probability >= float(threshold)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(symmetric_binary, 8)
    if count <= 1:
        raise RuntimeError("Symmetric cursor model has no connected component")
    selected = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    symmetric_binary = (labels == selected).astype(np.uint8)
    contours, _ = cv2.findContours(
        symmetric_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        raise RuntimeError("Symmetric cursor model has no contour")
    contour = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    mirrored_contour = reflect_points(contour, center, axis_angle)
    combined = np.vstack([contour, mirrored_contour]).astype(np.float32)
    hull = cv2.convexHull(combined.reshape(-1, 1, 2))
    simplified = cv2.approxPolyDP(hull, 0.70, True)[:, 0, :].astype(np.float64)
    # Re-add every reflected vertex before the final hull so symmetry is exact.
    paired = np.vstack(
        [simplified, reflect_points(simplified, center, axis_angle)]
    ).astype(np.float32)
    polygon = cv2.convexHull(paired.reshape(-1, 1, 2))[:, 0, :].astype(np.float64)

    axis = np.array(
        [math.cos(math.radians(axis_angle)), math.sin(math.radians(axis_angle))]
    )
    projections = (polygon - center) @ axis
    if float(projections.max()) < abs(float(projections.min())):
        direction_angle = (axis_angle + 180.0) % 360.0
    else:
        direction_angle = axis_angle % 360.0
    rendered = render_polygon(
        polygon - center,
        probability.shape[0],
        0.0,
        supersample=4,
    )
    observed = probability >= float(threshold)
    predicted = rendered >= 0.5
    union = np.logical_or(observed, predicted).sum()
    polygon_iou = (
        float(np.logical_and(observed, predicted).sum() / union) if union else 0.0
    )
    residual = np.abs(probability - reflected)
    return {
        "axis_angle_deg": float(direction_angle),
        "axis_angle_mod_180_deg": float(axis_angle),
        "axis_fit_soft_iou": float(np.max(fine_scores)),
        "axis_coarse_scores": coarse_scores,
        "reflected_probability": reflected,
        "symmetric_probability": symmetric_probability,
        "symmetric_binary": symmetric_binary,
        "symmetry_residual": residual,
        "polygon_xy": polygon,
        "polygon_relative_xy": polygon - center,
        "polygon_iou": polygon_iou,
    }


def render_polygon(relative_polygon, size, rotation_deg, supersample=4):
    """Rasterize a relative polygon continuously with supersampling."""
    relative_polygon = np.asarray(relative_polygon, dtype=np.float64)
    angle = math.radians(float(rotation_deg))
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    center = (float(size) - 1.0) / 2.0
    vertices = relative_polygon @ rotation.T + center
    high_size = int(size * supersample)
    high_center_offset = (supersample - 1.0) / 2.0
    high_vertices = np.round(
        vertices * supersample + high_center_offset
    ).astype(np.int32)
    canvas = np.zeros((high_size, high_size), dtype=np.uint8)
    cv2.fillPoly(canvas, [high_vertices], 255, cv2.LINE_AA)
    return cv2.resize(
        canvas.astype(np.float32) / 255.0,
        (size, size),
        interpolation=cv2.INTER_AREA,
    )


def polygon_edge(mask):
    binary = (np.asarray(mask) >= 0.5).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel) > 0


def edge_distance_transform(edge):
    return cv2.distanceTransform((~np.asarray(edge, dtype=bool)).astype(np.uint8), cv2.DIST_L2, 3)
