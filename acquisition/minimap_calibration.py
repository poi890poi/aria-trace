"""Evidence-linked circular mini-map and cursor calibration.

The calibration deliberately separates three observations:

* camera rotation keeps the mini-map boundary and Genshin cursor screen-fixed;
* the rotating map supplies radial evidence for the outer boundary;
* movement-only direction changes rotate the cursor and expose its pivot.

Every fitted value is accompanied by the intermediate image that produced it.
"""

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from .cursor_shape_model import fit_symmetric_polygon
from .session import SessionReader


SCHEMA_VERSION = "1.0"
DEFAULT_CONFIG = {
    "crop_xywh": [0, 0, 220, 180],
    "boundary": {
        "expected_center_xy": [111.0, 83.0],
        "center_search_radius_px": 14.0,
        "radius_range_px": [62.0, 75.0],
    },
    "cursor": {
        "hsv_lower": [88, 160, 175],
        "hsv_upper": [98, 255, 255],
        "search_radius_px": 25.0,
        "component_area_px": [20, 240],
        "shape_persistence_threshold": 0.72,
    },
}


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


def _merged_config(value: Optional[dict]) -> dict:
    result = json.loads(json.dumps(DEFAULT_CONFIG))
    for key, item in (value or {}).items():
        if isinstance(item, dict) and isinstance(result.get(key), dict):
            result[key].update(item)
        else:
            result[key] = item
    return result


def _validate_segments(segments: dict) -> dict:
    required = ("rotation_only", "movement_only")
    cleaned = {}
    for name in required:
        interval = segments.get(name)
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise ValueError("{} must be [start_s, end_s]".format(name))
        start_s, end_s = map(float, interval)
        if start_s < 0 or end_s <= start_s:
            raise ValueError("{} has an invalid time interval".format(name))
        cleaned[name] = [start_s, end_s]
    return cleaned


def _read_video_segment(
    video_path: Path,
    interval: Sequence[float],
    crop_xywh: Sequence[int],
    frame_records: Optional[Sequence[dict]] = None,
) -> Tuple[np.ndarray, float]:
    x, y, width, height = map(int, crop_xywh)
    if min(x, y) < 0 or min(width, height) <= 0:
        raise ValueError("Invalid mini-map crop")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Cannot open calibration video: {}".format(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.set(cv2.CAP_PROP_POS_MSEC, float(interval[0]) * 1000.0)
    records_by_index = {
        int(record["frame_index"]): record for record in (frame_records or [])
    }
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded_index = max(0, int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1)
            record = records_by_index.get(decoded_index)
            time_s = (
                float(record["session_time_ns"]) / 1e9
                if record is not None
                else float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            )
            if time_s > float(interval[1]) + 1.0e-6:
                break
            if time_s + 1.0e-6 < float(interval[0]):
                continue
            if y + height > frame.shape[0] or x + width > frame.shape[1]:
                raise ValueError(
                    "Mini-map crop {} exceeds video frame {}x{}".format(
                        list(crop_xywh), frame.shape[1], frame.shape[0]
                    )
                )
            frames.append(frame[y : y + height, x : x + width].copy())
    finally:
        capture.release()
    if len(frames) < 12:
        raise ValueError("Calibration segment contains fewer than 12 decoded frames")
    return np.stack(frames), fps

def _row_robust_z(values: np.ndarray) -> np.ndarray:
    median = np.median(values, axis=1, keepdims=True)
    mad = np.median(np.abs(values - median), axis=1, keepdims=True) + 1.0e-4
    return np.clip((values - median) / (1.4826 * mad), 0.0, 12.0)


def _circle_from_three(points: np.ndarray) -> Optional[np.ndarray]:
    (x1, y1), (x2, y2), (x3, y3) = points
    denominator = 2.0 * (
        x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    )
    if abs(denominator) < 1.0e-8:
        return None
    ux = (
        (x1 * x1 + y1 * y1) * (y2 - y3)
        + (x2 * x2 + y2 * y2) * (y3 - y1)
        + (x3 * x3 + y3 * y3) * (y1 - y2)
    ) / denominator
    uy = (
        (x1 * x1 + y1 * y1) * (x3 - x2)
        + (x2 * x2 + y2 * y2) * (x1 - x3)
        + (x3 * x3 + y3 * y3) * (x2 - x1)
    ) / denominator
    return np.array([ux, uy, math.hypot(ux - x1, uy - y1)], dtype=np.float64)


def _algebraic_circle(points: np.ndarray, weights=None) -> np.ndarray:
    design = np.column_stack([points[:, 0], points[:, 1], np.ones(len(points))])
    target = -(points[:, 0] ** 2 + points[:, 1] ** 2)
    if weights is not None:
        scale = np.sqrt(np.maximum(np.asarray(weights), 1.0e-6))
        design = design * scale[:, None]
        target = target * scale
    d, e, f = np.linalg.lstsq(design, target, rcond=None)[0]
    center_x, center_y = -d / 2.0, -e / 2.0
    radius = math.sqrt(max(1.0e-9, center_x ** 2 + center_y ** 2 - f))
    return np.array([center_x, center_y, radius], dtype=np.float64)


def _initial_circle(average: np.ndarray, config: dict) -> np.ndarray:
    expected = np.asarray(config["expected_center_xy"], dtype=np.float64)
    search_radius = float(config["center_search_radius_px"])
    min_radius, max_radius = map(float, config["radius_range_px"])
    gray = cv2.cvtColor(average, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.1)
    detected = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.0,
        minDist=max(20.0, min_radius),
        param1=70.0,
        param2=18.0,
        minRadius=int(math.floor(min_radius)),
        maxRadius=int(math.ceil(max_radius)),
    )
    candidates = [] if detected is None else detected[0]
    plausible = [
        np.asarray(item, dtype=np.float64)
        for item in candidates
        if np.linalg.norm(np.asarray(item[:2]) - expected) <= search_radius
        and min_radius <= item[2] <= max_radius
    ]
    if plausible:
        return min(
            plausible,
            key=lambda item: np.linalg.norm(item[:2] - expected)
            + 0.15 * abs(item[2] - np.mean([min_radius, max_radius])),
        )
    return np.array(
        [expected[0], expected[1], (min_radius + max_radius) / 2.0],
        dtype=np.float64,
    )


def _radial_observations(
    average: np.ndarray, temporal_std: np.ndarray, initial: np.ndarray
) -> dict:
    gray = cv2.GaussianBlur(
        cv2.cvtColor(average, cv2.COLOR_BGR2GRAY), (5, 5), 1.0
    ).astype(np.float32)
    std_blur = cv2.GaussianBlur(temporal_std.astype(np.float32), (5, 5), 1.0)
    angles = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    radii = np.linspace(initial[2] - 10.5, initial[2] + 10.5, 169)
    x_map = initial[0] + np.cos(angles)[:, None] * radii[None, :]
    y_map = initial[1] + np.sin(angles)[:, None] * radii[None, :]

    def sample(image):
        return cv2.remap(
            image.astype(np.float32),
            x_map.astype(np.float32),
            y_map.astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )

    average_gradient = np.abs(np.gradient(sample(gray), radii, axis=1))
    std_gradient = np.abs(np.gradient(sample(std_blur), radii, axis=1))
    average_z = _row_robust_z(average_gradient)
    std_z = _row_robust_z(std_gradient)
    response = 0.48 * average_z + 0.52 * std_z
    search = np.abs(radii - initial[2]) <= 4.75
    search_radii = radii[search]
    search_response = response[:, search]
    indices = search_response.argmax(axis=1)
    observed_radii = search_radii[indices]
    peaks = search_response[np.arange(len(angles)), indices]
    second = np.empty_like(peaks)
    for row_index, radius_index in enumerate(indices):
        row = search_response[row_index].copy()
        row[max(0, radius_index - 8) : radius_index + 9] = -np.inf
        second[row_index] = np.max(row)
    prominence = peaks - np.maximum(second, 0.0)
    points = np.column_stack(
        [
            initial[0] + np.cos(angles) * observed_radii,
            initial[1] + np.sin(angles) * observed_radii,
        ]
    )
    return {
        "angles": angles,
        "radii": radii,
        "response": response,
        "average_z": average_z,
        "std_z": std_z,
        "observed_radii": observed_radii,
        "peaks": peaks,
        "prominence": prominence,
        "points": points,
    }


def _fit_radial_circle(observation: dict, initial: np.ndarray, seed=826) -> dict:
    points = observation["points"]
    peaks = observation["peaks"]
    prominence = observation["prominence"]
    observed_radii = observation["observed_radii"]
    weights = np.clip(peaks, 0.0, 10.0) * (
        0.4 + np.clip(prominence, 0.0, 5.0)
    )
    eligible = (
        (peaks >= np.percentile(peaks, 38))
        & (prominence > 0.12)
        & (np.abs(observed_radii - initial[2]) <= 4.5)
    )
    candidate_points = points[eligible]
    candidate_weights = weights[eligible]
    rng = np.random.default_rng(seed)
    best = None
    best_score = -np.inf
    for _ in range(2500):
        if len(candidate_points) < 3:
            break
        circle = _circle_from_three(
            candidate_points[rng.choice(len(candidate_points), 3, replace=False)]
        )
        if circle is None:
            continue
        if np.linalg.norm(circle[:2] - initial[:2]) > 7.0:
            continue
        if abs(circle[2] - initial[2]) > 4.75:
            continue
        residual = np.abs(
            np.linalg.norm(candidate_points - circle[:2], axis=1) - circle[2]
        )
        inliers = residual < 1.35
        score = candidate_weights[inliers].sum() - 0.15 * np.minimum(
            residual, 4.0
        ).sum()
        if score > best_score:
            best_score = score
            best = inliers
    circle = (
        _algebraic_circle(candidate_points[best], candidate_weights[best])
        if best is not None and best.sum() >= 3
        else initial.copy()
    )
    residual = np.abs(np.linalg.norm(points - circle[:2], axis=1) - circle[2])
    accepted = eligible & (residual < 1.5)
    for _ in range(2):
        neighbors = sum(
            np.roll(accepted, shift).astype(np.int32) for shift in (-2, -1, 1, 2)
        )
        accepted &= neighbors >= 1
    return {
        "circle": circle,
        "accepted": accepted,
        "eligible": eligible,
        "residual": residual,
    }


def _fit_one_channel(observation: dict, initial: np.ndarray, key: str) -> np.ndarray:
    replacement = dict(observation)
    response = observation[key]
    radii = observation["radii"]
    search = np.abs(radii - initial[2]) <= 4.75
    values = response[:, search]
    indices = values.argmax(axis=1)
    observed = radii[search][indices]
    peaks = values[np.arange(len(values)), indices]
    points = np.column_stack(
        [
            initial[0] + np.cos(observation["angles"]) * observed,
            initial[1] + np.sin(observation["angles"]) * observed,
        ]
    )
    replacement.update(
        points=points,
        observed_radii=observed,
        peaks=peaks,
        prominence=np.maximum(peaks - np.median(values, axis=1), 0.0),
    )
    return _fit_radial_circle(replacement, initial)["circle"]


def _boundary_model(rotation_frames: np.ndarray, config: dict) -> dict:
    average = rotation_frames.mean(axis=0).astype(np.uint8)
    temporal_std = rotation_frames.astype(np.float32).std(axis=0).mean(axis=2)
    initial = _initial_circle(average, config)
    observation = _radial_observations(average, temporal_std, initial)
    fitted = _fit_radial_circle(observation, initial)
    circle = fitted["circle"]
    accepted = fitted["accepted"]
    residual = fitted["residual"]
    angle_bins = np.arange(len(accepted)) * 72 // len(accepted)
    coverage = float(
        np.mean([accepted[angle_bins == index].sum() >= 2 for index in range(72)])
    )
    accepted_residual = residual[accepted]
    rmse = (
        float(np.sqrt(np.mean(accepted_residual ** 2)))
        if len(accepted_residual)
        else float("inf")
    )
    median_residual = (
        float(np.median(accepted_residual))
        if len(accepted_residual)
        else float("inf")
    )

    average_circle = _fit_one_channel(observation, initial, "average_z")
    std_circle = _fit_one_channel(observation, initial, "std_z")
    channel_center_delta = float(np.linalg.norm(average_circle[:2] - std_circle[:2]))
    channel_radius_delta = float(abs(average_circle[2] - std_circle[2]))

    chunk_circles = []
    for chunk_index, chunk in enumerate(np.array_split(rotation_frames, 9)):
        if len(chunk) < 3:
            continue
        chunk_average = chunk.mean(axis=0).astype(np.uint8)
        chunk_std = chunk.astype(np.float32).std(axis=0).mean(axis=2)
        chunk_observation = _radial_observations(chunk_average, chunk_std, initial)
        chunk_fit = _fit_radial_circle(
            chunk_observation, initial, seed=826 + chunk_index
        )["circle"]
        if np.linalg.norm(chunk_fit[:2] - circle[:2]) <= 5.0:
            chunk_circles.append(chunk_fit)
    chunk_circles = np.asarray(chunk_circles)
    if len(chunk_circles):
        chunk_center_std = float(
            math.sqrt(np.var(chunk_circles[:, 0]) + np.var(chunk_circles[:, 1]))
        )
        chunk_radius_std = float(np.std(chunk_circles[:, 2]))
        chunk_agreement = float(
            np.mean(
                (np.linalg.norm(chunk_circles[:, :2] - circle[:2], axis=1) < 2.5)
                & (np.abs(chunk_circles[:, 2] - circle[2]) < 2.0)
            )
        )
    else:
        chunk_center_std = chunk_radius_std = float("inf")
        chunk_agreement = 0.0

    median_prominence = (
        float(np.median(observation["prominence"][accepted]))
        if accepted.any()
        else 0.0
    )
    component_scores = {
        "angular_coverage": float(np.clip(coverage / 0.70, 0.0, 1.0)),
        "radial_residual": float(math.exp(-((rmse / 1.25) ** 2))),
        "transition_signal": float(
            np.clip((median_prominence - 0.10) / 1.20, 0.0, 1.0)
        ),
        "temporal_stability": float(
            chunk_agreement
            * math.exp(
                -((chunk_center_std / 3.0) ** 2)
                - ((chunk_radius_std / 2.5) ** 2)
            )
        ),
        "channel_center_agreement": float(
            math.exp(-((channel_center_delta / 3.5) ** 2))
        ),
        # The two channels can identify the inner and outer edge of a translucent rim.
        "rim_band_compatibility": float(
            math.exp(-((max(0.0, channel_radius_delta - 4.0) / 1.5) ** 2))
        ),
    }
    confidence = float(
        np.prod([max(value, 0.02) for value in component_scores.values()])
        ** (1.0 / len(component_scores))
    )
    return {
        "average": average,
        "temporal_std": temporal_std,
        "initial": initial,
        "observation": observation,
        "fit": fitted,
        "chunk_circles": chunk_circles,
        "metrics": {
            "center_x": float(circle[0]),
            "center_y": float(circle[1]),
            "radius": float(circle[2]),
            "initial_hough": {
                "center_x": float(initial[0]),
                "center_y": float(initial[1]),
                "radius": float(initial[2]),
            },
            "accepted_samples": int(accepted.sum()),
            "total_angular_samples": int(len(accepted)),
            "angular_coverage_5deg_bins": coverage,
            "radial_rmse_px": rmse,
            "radial_median_residual_px": median_residual,
            "median_transition_prominence": median_prominence,
            "channel_center_disagreement_px": channel_center_delta,
            "channel_radius_band_px": channel_radius_delta,
            "valid_temporal_chunks": int(len(chunk_circles)),
            "chunk_agreement_rate": chunk_agreement,
            "chunk_center_std_px": chunk_center_std,
            "chunk_radius_std_px": chunk_radius_std,
            "confidence": confidence,
            "confidence_level": _confidence_level(confidence),
            "confidence_components": component_scores,
        },
    }


def _cursor_components(
    frames: np.ndarray, center: np.ndarray, config: dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lower = np.asarray(config["hsv_lower"], dtype=np.uint8)
    upper = np.asarray(config["hsv_upper"], dtype=np.uint8)
    search_radius = float(config["search_radius_px"])
    min_area, max_area = map(int, config["component_area_px"])
    height, width = frames.shape[1:3]
    yy, xx = np.ogrid[:height, :width]
    search_mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= search_radius ** 2
    masks, centroids, indices = [], [], []
    for frame_index, frame in enumerate(frames):
        binary = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lower, upper)
        binary[~search_mask] = 0
        count, labels, stats, component_centroids = cv2.connectedComponentsWithStats(
            binary, 8
        )
        candidates = []
        for index in range(1, count):
            area = stats[index, cv2.CC_STAT_AREA]
            distance = np.linalg.norm(component_centroids[index] - center)
            if min_area <= area <= max_area and distance < search_radius * 0.5:
                candidates.append((distance, index))
        if not candidates:
            continue
        selected = min(candidates)[1]
        masks.append((labels == selected).astype(np.uint8))
        centroids.append(component_centroids[selected])
        indices.append(frame_index)
    if not masks:
        raise RuntimeError("No cursor-colored components were detected")
    return np.stack(masks), np.asarray(centroids), np.asarray(indices)


def _rotation_center(movement_frames: np.ndarray, boundary: dict, config: dict) -> dict:
    initial = np.array([boundary["center_x"], boundary["center_y"]])
    masks, centroids, indices = _cursor_components(movement_frames, initial, config)
    circle = _algebraic_circle(centroids)
    distances = np.linalg.norm(centroids - circle[:2], axis=1)
    residual = distances - circle[2]
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    angles = np.mod(np.arctan2(centroids[:, 1] - circle[1], centroids[:, 0] - circle[0]), 2*np.pi)
    occupied = len(np.unique(np.floor(angles / (2 * np.pi) * 36).astype(int))) / 36.0
    rng = np.random.default_rng(826)
    bootstrap = []
    for _ in range(500):
        sample = centroids[rng.integers(0, len(centroids), len(centroids))]
        try:
            bootstrap.append(_algebraic_circle(sample)[:2])
        except np.linalg.LinAlgError:
            pass
    bootstrap = np.asarray(bootstrap)
    bootstrap_sigma = (
        float(math.sqrt(np.var(bootstrap[:, 0]) + np.var(bootstrap[:, 1])))
        if len(bootstrap)
        else float("inf")
    )
    detection_rate = len(masks) / len(movement_frames)
    components = {
        "detection": float(detection_rate),
        "angular_coverage": float(occupied),
        "fit_residual": float(math.exp(-((rmse / 0.5) ** 2))),
        "bootstrap": float(math.exp(-((bootstrap_sigma / 0.35) ** 2))),
    }
    confidence = float(np.prod([max(v, 0.02) for v in components.values()]) ** 0.25)
    return {
        "masks": masks,
        "centroids": centroids,
        "indices": indices,
        "metrics": {
            "x": float(circle[0]),
            "y": float(circle[1]),
            "centroid_orbit_radius_px": float(circle[2]),
            "circle_fit_rmse_px": rmse,
            "detected_frames": int(len(masks)),
            "total_frames": int(len(movement_frames)),
            "detection_rate": float(detection_rate),
            "angular_coverage_10deg_bins": float(occupied),
            "bootstrap_center_sigma_px": bootstrap_sigma,
            "confidence": confidence,
            "confidence_level": _confidence_level(confidence),
            "confidence_components": components,
        },
    }


def _cursor_shape(
    rotation_frames: np.ndarray, center_metrics: dict, config: dict
) -> dict:
    pivot = np.array([center_metrics["x"], center_metrics["y"]])
    masks, centroids, indices = _cursor_components(rotation_frames, pivot, config)
    size = 41
    half = size // 2
    patches = np.stack(
        [
            cv2.getRectSubPix(mask.astype(np.float32), (size, size), tuple(pivot))
            for mask in masks
        ]
    )
    probability = patches.mean(axis=0)
    threshold = float(config["shape_persistence_threshold"])
    binary = (probability >= threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count > 1:
        selected = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        binary = (labels == selected).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Persistent cursor template has no contour")
    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    centroid = np.array(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
    )
    contour_points = contour[:, 0, :].astype(np.float64)
    farthest = contour_points[
        np.argmax(np.linalg.norm(contour_points - np.array([half, half]), axis=1))
    ]
    farthest_angle = math.degrees(
        math.atan2(farthest[1] - half, farthest[0] - half)
    )
    symmetric_model = fit_symmetric_polygon(
        probability,
        threshold,
        np.array([half, half], dtype=np.float64),
        hint_angle_deg=farthest_angle,
    )
    ious = []
    for patch in patches:
        observed = patch >= 0.5
        union = np.logical_or(observed, binary).sum()
        ious.append(np.logical_and(observed, binary).sum() / union if union else 0.0)
    ious = np.asarray(ious)

    theta = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
    radii = np.linspace(0.5, 15.0, 36)
    x_map = half + np.cos(theta)[:, None] * radii[None, :]
    y_map = half + np.sin(theta)[:, None] * radii[None, :]
    template_polar = cv2.remap(
        binary.astype(np.float32), x_map.astype(np.float32), y_map.astype(np.float32), cv2.INTER_LINEAR
    )
    template_energy = math.sqrt(float(np.sum(template_polar ** 2))) + 1.0e-6
    correlations, shifts, peaks, margins, polar_rows = [], [], [], [], []
    for patch in patches:
        polar = cv2.remap(
            patch, x_map.astype(np.float32), y_map.astype(np.float32), cv2.INTER_LINEAR
        )
        polar_rows.append(polar.sum(axis=1))
        correlation = np.zeros(360, dtype=np.float32)
        for radius_index in range(polar.shape[1]):
            correlation += np.fft.ifft(
                np.fft.fft(polar[:, radius_index])
                * np.conj(np.fft.fft(template_polar[:, radius_index]))
            ).real.astype(np.float32)
        correlation /= math.sqrt(float(np.sum(polar ** 2))) * template_energy + 1.0e-6
        peak_index = int(np.argmax(correlation))
        signed_shift = peak_index if peak_index <= 180 else peak_index - 360
        excluded = correlation.copy()
        for offset in range(-5, 6):
            excluded[(peak_index + offset) % 360] = -np.inf
        correlations.append(correlation)
        shifts.append(signed_shift)
        peaks.append(correlation[peak_index])
        margins.append(correlation[peak_index] - np.max(excluded))
    correlations = np.stack(correlations)
    shifts = np.asarray(shifts)
    peaks = np.asarray(peaks)
    margins = np.asarray(margins)
    polar_rows = np.stack(polar_rows)
    component_scores = {
        "detection": float(len(masks) / len(rotation_frames)),
        "temporal_iou": float(np.clip(np.median(ious) / 0.90, 0.0, 1.0)),
        "tail_iou": float(np.clip(np.percentile(ious, 10) / 0.78, 0.0, 1.0)),
        "pose_stability": float(math.exp(-((np.std(shifts) / 1.5) ** 2))),
        "correlation_margin": float(np.clip(np.median(margins) / 0.025, 0.0, 1.0)),
    }
    confidence = float(np.prod([max(v, 0.02) for v in component_scores.values()]) ** 0.2)
    return {
        "masks": masks,
        "indices": indices,
        "patches": patches,
        "probability": probability,
        "binary": binary,
        "contour": contour,
        "centroid": centroid,
        "farthest": farthest,
        "symmetric_model": symmetric_model,
        "polar_rows": polar_rows,
        "template_polar": template_polar,
        "correlations": correlations,
        "shifts": shifts,
        "metrics": {
            "source": "screen_fixed_hsv_persistence_during_camera_rotation",
            "detected_frames": int(len(masks)),
            "total_frames": int(len(rotation_frames)),
            "detection_rate": float(len(masks) / len(rotation_frames)),
            "persistence_threshold": threshold,
            "contour_area_px": float(cv2.contourArea(contour)),
            "contour_perimeter_px": float(cv2.arcLength(contour, True)),
            "centroid_offset_from_pivot_px": {
                "dx": float(centroid[0] - half),
                "dy": float(centroid[1] - half),
                "magnitude": float(np.linalg.norm(centroid - np.array([half, half]))),
            },
            "farthest_contour_point_angle_screen_deg": float(farthest_angle),
            "model_type": "symmetry_constrained_rigid_polygon",
            "symmetry_constraint": "mirror axis through fitted cursor rotation center",
            "symmetry_axis_screen_deg": float(
                symmetric_model["axis_angle_deg"]
            ),
            "symmetry_axis_mod_180_deg": float(
                symmetric_model["axis_angle_mod_180_deg"]
            ),
            "symmetry_fit_soft_iou": float(
                symmetric_model["axis_fit_soft_iou"]
            ),
            "polygon_template_iou": float(symmetric_model["polygon_iou"]),
            "polygon_vertex_count": int(len(symmetric_model["polygon_xy"])),
            "polygon_vertices_relative_xy": symmetric_model[
                "polygon_relative_xy"
            ].tolist(),
            "median_frame_template_iou": float(np.median(ious)),
            "p10_frame_template_iou": float(np.percentile(ious, 10)),
            "pose_shift_std_deg": float(np.std(shifts)),
            "pose_abs_shift_p95_deg": float(np.percentile(np.abs(shifts), 95)),
            "median_rotation_correlation_peak": float(np.median(peaks)),
            "median_rotation_correlation_margin": float(np.median(margins)),
            "confidence": confidence,
            "confidence_level": _confidence_level(confidence),
            "confidence_components": component_scores,
            "direction_semantics": "pointed end of fitted symmetry axis in screen coordinates; game heading offset unresolved",
        },
    }


def _confidence_level(value: float) -> str:
    if value >= 0.85:
        return "high"
    if value >= 0.65:
        return "moderate"
    return "low"


def _color_heatmap(values: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(values, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def _stacked_difference_heatmap(frames: np.ndarray) -> np.ndarray:
    """Accumulate actual consecutive-frame differences without a fitted model."""
    if len(frames) < 2:
        return np.zeros(frames.shape[1:3], dtype=np.float32)
    stacked = np.zeros(frames.shape[1:3], dtype=np.float32)
    previous = frames[0]
    for current in frames[1:]:
        difference = cv2.absdiff(previous, current).mean(axis=2)
        stacked += difference.astype(np.float32)
        previous = current
    return stacked / float(len(frames) - 1)


def _write_evidence(
    output: Path,
    rotation_frames: np.ndarray,
    movement_frames: np.ndarray,
    boundary: dict,
    center: dict,
    shape: dict,
) -> Sequence[dict]:
    output.mkdir(parents=True, exist_ok=True)
    files = []

    def save(name, image, title, category):
        path = output / name
        if not cv2.imwrite(str(path), image):
            raise RuntimeError("Could not write calibration evidence: {}".format(path))
        files.append({"name": name, "title": title, "category": category})

    bmetrics = boundary["metrics"]
    fit = boundary["fit"]
    observation = boundary["observation"]
    average = boundary["average"]
    save(
        "minimap_stacked_difference_heatmap.png",
        _color_heatmap(_stacked_difference_heatmap(rotation_frames)),
        "Stacked consecutive-frame difference heatmap",
        "boundary",
    )
    save(
        "boundary_temporal_heatmap.png",
        _color_heatmap(boundary["temporal_std"]),
        "Boundary temporal heatmap",
        "boundary",
    )
    radial = _color_heatmap(observation["response"])
    radial = cv2.resize(radial, (760, 720), interpolation=cv2.INTER_NEAREST)
    for row, radius in enumerate(observation["observed_radii"]):
        x = int(
            round(
                (radius - observation["radii"][0])
                / (observation["radii"][-1] - observation["radii"][0])
                * (radial.shape[1] - 1)
            )
        )
        color = (255, 255, 255) if fit["accepted"][row] else (0, 0, 0)
        cv2.circle(radial, (x, row), 1, color, -1)
    cv2.putText(radial, "radius left-to-right / angle top-to-bottom", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .52, (255,255,255), 1, cv2.LINE_AA)
    save("boundary_radial_heatmap.png", radial, "Boundary radial response", "boundary")

    point_binary = np.zeros_like(average)
    for accepted, point in zip(fit["accepted"], observation["points"]):
        if accepted:
            cv2.circle(point_binary, tuple(np.round(point).astype(int)), 1, (255,255,255), -1)
    save("boundary_points_binary.png", point_binary, "Accepted boundary points", "boundary")

    fit_overlay = average.copy()
    fit_center = (round(bmetrics["center_x"]), round(bmetrics["center_y"]))
    fit_radius = round(bmetrics["radius"])
    cv2.circle(
        fit_overlay,
        fit_center,
        fit_radius,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.drawMarker(
        fit_overlay,
        fit_center,
        (255, 255, 255),
        cv2.MARKER_CROSS,
        9,
        1,
    )
    save(
        "boundary_fitted_circle.png",
        fit_overlay,
        "Complete fitted mini-map boundary",
        "boundary",
    )

    overlay = average.copy()
    for index in range(0, len(observation["points"]), 4):
        point = tuple(np.round(observation["points"][index]).astype(int))
        color = (0,255,0) if fit["accepted"][index] else (0,0,255)
        cv2.circle(overlay, point, 1, color, -1)
    cv2.circle(overlay, (round(bmetrics["center_x"]), round(bmetrics["center_y"])), round(bmetrics["radius"]), (255,255,0), 1, cv2.LINE_AA)
    cv2.putText(overlay, "green accepted / red rejected / cyan fit", (4, overlay.shape[0]-6), cv2.FONT_HERSHEY_SIMPLEX, .38, (255,255,255), 1, cv2.LINE_AA)
    save("boundary_evidence_overlay.png", overlay, "Boundary evidence overlay", "boundary")

    confidence = np.full((520, 760, 3), 18, np.uint8)
    cv2.putText(confidence, "Boundary confidence {:.3f} ({})".format(bmetrics["confidence"], bmetrics["confidence_level"]), (20,35), cv2.FONT_HERSHEY_SIMPLEX, .82, (245,245,245), 2, cv2.LINE_AA)
    for index, (label, value) in enumerate(bmetrics["confidence_components"].items()):
        y = 88 + index * 66
        cv2.putText(confidence, label.replace("_", " "), (20,y), cv2.FONT_HERSHEY_SIMPLEX, .50, (225,225,225), 1, cv2.LINE_AA)
        cv2.rectangle(confidence, (235,y-18), (665,y+5), (60,60,60), -1)
        cv2.rectangle(confidence, (235,y-18), (235+int(430*value),y+5), (30,190,235), -1)
        cv2.putText(confidence, "{:.2f}".format(value), (675,y+2), cv2.FONT_HERSHEY_SIMPLEX, .48, (245,245,245), 1)
    save("boundary_confidence.png", confidence, "Boundary confidence breakdown", "quality")

    cm = center["metrics"]
    occupancy = center["masks"].mean(axis=0)
    orbit_heatmap = _color_heatmap(occupancy * (1.0 - occupancy))
    save("cursor_center_heatmap.png", orbit_heatmap, "Stepping cursor rotation heatmap", "center")
    orbit = movement_frames.mean(axis=0).astype(np.uint8)
    for point in center["centroids"]:
        cv2.circle(orbit, tuple(np.round(point).astype(int)), 1, (0,255,255), -1)
    cv2.circle(orbit, (round(cm["x"]), round(cm["y"])), max(1, round(cm["centroid_orbit_radius_px"])), (0,255,0), 1, cv2.LINE_AA)
    cv2.drawMarker(orbit, (round(cm["x"]), round(cm["y"])), (0,0,255), cv2.MARKER_CROSS, 12, 1)
    save("cursor_center_orbit.png", orbit, "Cursor centroid orbit and fitted pivot", "center")

    persistence = shape["masks"].mean(axis=0)
    save("cursor_shape_persistence_heatmap.png", _color_heatmap(persistence), "Screen-fixed cursor persistence", "cursor_shape")
    probability = _color_heatmap(shape["probability"])
    probability = cv2.resize(probability, (410,410), interpolation=cv2.INTER_NEAREST)
    save("cursor_shape_probability.png", probability, "Cursor shape probability", "cursor_shape")
    binary = cv2.resize((shape["binary"]*255).astype(np.uint8), (410,410), interpolation=cv2.INTER_NEAREST)
    save("cursor_shape_binary.png", binary, "Cursor shape binary", "cursor_shape")
    edges = cv2.Canny((shape["binary"]*255).astype(np.uint8), 50, 150)
    edges = cv2.resize(edges, (410,410), interpolation=cv2.INTER_NEAREST)
    save("cursor_shape_edge.png", edges, "Cursor shape edge", "cursor_shape")

    overlay = np.zeros((410,410,3), np.uint8)
    overlay[binary > 0] = (235,235,235)
    overlay[edges > 0] = (0,255,0)
    cv2.drawMarker(overlay, (205,205), (0,0,255), cv2.MARKER_CROSS, 20, 2)
    centroid = tuple(np.round(shape["centroid"]*10).astype(int))
    farthest = tuple(np.round(shape["farthest"]*10).astype(int))
    cv2.drawMarker(overlay, centroid, (255,255,0), cv2.MARKER_TILTED_CROSS, 18, 2)
    cv2.line(overlay, (205,205), farthest, (0,200,255), 2)
    cv2.circle(overlay, farthest, 5, (0,200,255), -1)
    save("cursor_shape_overlay.png", overlay, "Cursor contour, pivot, and geometric tip", "cursor_shape")

    symmetric = shape["symmetric_model"]
    symmetric_view = cv2.resize(
        _color_heatmap(symmetric["symmetric_probability"]),
        (410, 410),
        interpolation=cv2.INTER_NEAREST,
    )
    polygon = np.round((symmetric["polygon_xy"] * 10)).astype(np.int32)
    cv2.polylines(symmetric_view, [polygon], True, (0,255,0), 2, cv2.LINE_AA)
    axis_angle = math.radians(symmetric["axis_angle_deg"])
    axis_start = np.array([205.0,205.0]) - 120.0*np.array([math.cos(axis_angle),math.sin(axis_angle)])
    axis_end = np.array([205.0,205.0]) + 120.0*np.array([math.cos(axis_angle),math.sin(axis_angle)])
    cv2.arrowedLine(
        symmetric_view,
        tuple(np.round(axis_start).astype(int)),
        tuple(np.round(axis_end).astype(int)),
        (0,220,255),
        2,
        cv2.LINE_AA,
        tipLength=0.12,
    )
    cv2.drawMarker(symmetric_view, (205,205), (255,255,255), cv2.MARKER_CROSS, 18, 2)
    save(
        "cursor_shape_symmetric_polygon.png",
        symmetric_view,
        "Symmetry-constrained rigid cursor polygon",
        "cursor_shape",
    )
    residual_view = cv2.resize(
        _color_heatmap(symmetric["symmetry_residual"]),
        (410,410),
        interpolation=cv2.INTER_NEAREST,
    )
    save(
        "cursor_shape_symmetry_residual.png",
        residual_view,
        "Cursor mirror-symmetry residual",
        "quality",
    )
    symmetric_binary_view = cv2.resize(
        symmetric["symmetric_binary"] * 255,
        (410,410),
        interpolation=cv2.INTER_NEAREST,
    )
    save(
        "cursor_shape_symmetric_binary.png",
        symmetric_binary_view,
        "Symmetrized cursor silhouette",
        "cursor_shape",
    )

    polar_rows = shape["polar_rows"] / (shape["polar_rows"].max(axis=1, keepdims=True) + 1e-6)
    polar_occupancy = _color_heatmap(np.sqrt(np.clip(polar_rows, 0, 1)))
    polar_occupancy = cv2.resize(polar_occupancy, (720, max(360, 2*len(polar_rows))), interpolation=cv2.INTER_NEAREST)
    save("cursor_shape_polar_occupancy.png", polar_occupancy, "Cursor polar occupancy over time", "polar")
    template_polar = _color_heatmap(shape["template_polar"])
    template_polar = cv2.resize(template_polar, (500,720), interpolation=cv2.INTER_NEAREST)
    save("cursor_shape_polar_template.png", template_polar, "Canonical cursor polar template", "polar")
    correlation = _color_heatmap(shape["correlations"])
    correlation = cv2.resize(correlation, (720, max(360, 2*len(shape["correlations"]))), interpolation=cv2.INTER_NEAREST)
    save("cursor_shape_polar_correlation.png", correlation, "Cursor rotation correlation", "polar")

    model_mask = np.zeros(average.shape[:2], dtype=np.uint8)
    cv2.circle(model_mask, (round(bmetrics["center_x"]), round(bmetrics["center_y"])), round(bmetrics["radius"]), 255, -1)
    np.savez_compressed(
        output / "model.npz",
        boundary=np.array([bmetrics["center_x"], bmetrics["center_y"], bmetrics["radius"]]),
        rotation_center=np.array([cm["x"], cm["y"]]),
        minimap_mask=model_mask,
        cursor_probability=shape["probability"],
        cursor_binary=shape["binary"],
        cursor_symmetric_probability=shape["symmetric_model"]["symmetric_probability"],
        cursor_symmetric_binary=shape["symmetric_model"]["symmetric_binary"],
        cursor_polygon_relative_xy=shape["symmetric_model"]["polygon_relative_xy"],
        cursor_symmetry_axis_deg=np.array(
            [shape["symmetric_model"]["axis_angle_deg"]], dtype=np.float64
        ),
    )
    return files


def calibrate_minimap_frames(
    rotation_frames: np.ndarray,
    movement_frames: np.ndarray,
    output_path: Path,
    config: Optional[dict] = None,
    provenance: Optional[dict] = None,
    progress=None,
) -> dict:
    """Calibrate from already labeled frame arrays and persist review evidence."""
    config = _merged_config(config)
    output_path = Path(output_path)
    if rotation_frames.ndim != 4 or movement_frames.ndim != 4:
        raise ValueError("Calibration frames must be N x H x W x C arrays")
    if progress:
        progress("Fitting the circular mini-map boundary")
    boundary = _boundary_model(rotation_frames, config["boundary"])
    if progress:
        progress("Fitting the cursor pivot from movement frames")
    center = _rotation_center(movement_frames, boundary["metrics"], config["cursor"])
    if progress:
        progress("Fitting the cursor shape and polar template")
    shape = _cursor_shape(rotation_frames, center["metrics"], config["cursor"])
    if progress:
        progress("Rendering boundary, cursor, and polar evidence")
    evidence = _write_evidence(
        output_path, rotation_frames, movement_frames, boundary, center, shape
    )
    outer = boundary["metrics"]
    pivot = center["metrics"]
    offset_x = pivot["x"] - outer["center_x"]
    offset_y = pivot["y"] - outer["center_y"]
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "review_required",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance or {},
        "config": config,
        "outer_boundary": outer,
        "rotation_center": pivot,
        "center_offset": {
            "dx": float(offset_x),
            "dy": float(offset_y),
            "magnitude_px": float(math.hypot(offset_x, offset_y)),
        },
        "cursor_shape": shape["metrics"],
        "overall_confidence": float(
            (
                outer["confidence"]
                * pivot["confidence"]
                * shape["metrics"]["confidence"]
            )
            ** (1.0 / 3.0)
        ),
        "confidence_scale": "0..1 experimental; human evidence review is authoritative",
        "model_file": "model.npz",
        "evidence": evidence,
    }
    _atomic_json(output_path / "calibration.json", result)
    # Validate pose estimation on the reviewed movement-only segment using the
    # just-written calibration model. This remains screen pose until a separate
    # world-heading reference is supplied.
    from .cursor_pose import estimate_cursor_pose_frames

    if progress:
        progress("Verifying cursor pose across the movement session")
    pose_fps = float((provenance or {}).get("container_fps") or 30.0)
    movement_start_s = float(
        ((provenance or {}).get("segments") or {}).get("movement_only", [0.0])[0]
    )
    pose_times_ns = [
        int((movement_start_s + index / pose_fps) * 1e9)
        for index in range(len(movement_frames))
    ]
    pose_summary = estimate_cursor_pose_frames(
        output_path / "calibration.json",
        movement_frames,
        output_path,
        frame_indices=list(range(len(movement_frames))),
        session_times_ns=pose_times_ns,
        provenance={
            "source": "reviewed_movement_only_segment",
            "calibration_provenance": provenance or {},
        },
    )
    result["cursor_pose_validation"] = pose_summary
    result["evidence"].extend(pose_summary["evidence"])
    if progress:
        progress("Writing the calibrated model and pose evidence")
    _atomic_json(output_path / "calibration.json", result)
    return result


def calibrate_session(
    session_path: Path,
    output_path: Path,
    segments: dict,
    config: Optional[dict] = None,
    progress=None,
) -> dict:
    """Decode approved segment intervals from one acquisition session."""
    session_path = Path(session_path).resolve()
    segments = _validate_segments(segments)
    config = _merged_config(config)
    reader = SessionReader(session_path)
    video_path = reader.video_path("main")
    frame_records = reader.frames_by_stream.get("main", [])
    if progress:
        progress("Decoding the rotation-only calibration frames")
    rotation_frames, fps = _read_video_segment(
        video_path,
        segments["rotation_only"],
        config["crop_xywh"],
        frame_records=frame_records,
    )
    if progress:
        progress("Decoding the movement-only calibration frames")
    movement_frames, _ = _read_video_segment(
        video_path,
        segments["movement_only"],
        config["crop_xywh"],
        frame_records=frame_records,
    )
    provenance = {
        "session_path": str(session_path),
        "session_id": reader.manifest.get("session_id"),
        "video_path": str(video_path),
        "stream_id": "main",
        "segments": segments,
        "segment_label_source": "human_reviewed",
        "rotation_frame_count": int(len(rotation_frames)),
        "movement_frame_count": int(len(movement_frames)),
        "container_fps": fps,
        "input_evidence_used": False,
    }
    return calibrate_minimap_frames(
        rotation_frames,
        movement_frames,
        output_path,
        config=config,
        provenance=provenance,
        progress=progress,
    )


def calibrate_segment_sessions(
    rotation_session_path: Path,
    movement_session_path: Path,
    output_path: Path,
    config: Optional[dict] = None,
    forward_session_path: Optional[Path] = None,
    progress=None,
) -> dict:
    """Calibrate from short sessions whose capture plan supplies each label."""
    config = _merged_config(config)

    def read_session(path: Path, expected_label: str):
        path = Path(path).resolve()
        reader = SessionReader(path)
        context = reader.manifest.get("context") or {}
        actual_label = context.get("segment_label")
        metadata_path = path / "session_metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if "label" in metadata:
                actual_label = metadata.get("label") or None
        if actual_label != expected_label:
            raise ValueError(
                "Expected a {} segment, got {} from {}".format(
                    expected_label, actual_label or "unlabeled", path
                )
            )
        duration_s = float(reader.manifest.get("duration_ns") or 0) / 1.0e9
        if duration_s <= 0.0:
            raise ValueError("Calibration segment has no recorded duration: {}".format(path))
        frames, fps = _read_video_segment(
            reader.video_path("main"),
            [0.0, duration_s],
            config["crop_xywh"],
            frame_records=reader.frames_by_stream.get("main", []),
        )
        return reader, frames, fps

    if progress:
        progress("Decoding the rotation-only calibration session")
    rotation_reader, rotation_frames, rotation_fps = read_session(
        rotation_session_path, "rotation_only"
    )
    if progress:
        progress("Decoding the movement-only calibration session")
    movement_reader, movement_frames, movement_fps = read_session(
        movement_session_path, "movement_only"
    )
    forward_reference = None
    if forward_session_path is not None:
        forward_reader, _forward_frames, forward_fps = read_session(
            forward_session_path, "forward_no_turn"
        )
        forward_reference = {
            "session_path": str(Path(forward_session_path).resolve()),
            "session_id": forward_reader.manifest.get("session_id"),
            "frame_count": len(_forward_frames),
            "container_fps": forward_fps,
            "segment_semantics": (
                forward_reader.manifest.get("context") or {}
            ).get("segment_semantics"),
        }
    provenance = {
        "segment_label_source": "capture_plan",
        "input_evidence_used": False,
        "container_fps": movement_fps,
        "segments": {
            "rotation_only": [0.0, float(rotation_reader.manifest.get("duration_ns") or 0) / 1.0e9],
            "movement_only": [0.0, float(movement_reader.manifest.get("duration_ns") or 0) / 1.0e9],
        },
        "segment_sessions": {
            "rotation_only": {
                "session_path": str(Path(rotation_session_path).resolve()),
                "session_id": rotation_reader.manifest.get("session_id"),
                "frame_count": len(rotation_frames),
                "container_fps": rotation_fps,
            },
            "movement_only": {
                "session_path": str(Path(movement_session_path).resolve()),
                "session_id": movement_reader.manifest.get("session_id"),
                "frame_count": len(movement_frames),
                "container_fps": movement_fps,
            },
        },
        "forward_heading_motion_reference": forward_reference,
    }
    return calibrate_minimap_frames(
        rotation_frames,
        movement_frames,
        output_path,
        config=config,
        provenance=provenance,
        progress=progress,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rotation", nargs=2, type=float, required=True, metavar=("START", "END"))
    parser.add_argument("--movement", nargs=2, type=float, required=True, metavar=("START", "END"))
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
    result = calibrate_session(
        args.session,
        args.output,
        {"rotation_only": args.rotation, "movement_only": args.movement},
        config,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
