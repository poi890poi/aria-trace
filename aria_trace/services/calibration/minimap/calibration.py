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

from aria_trace.services.calibration.cursor.shape import fit_symmetric_polygon
from aria_trace.adapters.filesystem.session import SessionReader
from aria_trace.domain.spatial import (
    SPATIAL_SCHEMA_VERSION,
    bind_geometry,
    raster_space,
    require_same_space,
    require_spatial_geometry,
    validate_raster_space,
)
from aria_trace.services.calibration.minimap.spatial import minimap_crop_space


SCHEMA_VERSION = "2.0"
ORDINARY_MOTION_SEGMENT_LABELS = ("ordinary_cruise", "route")
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


def _prepared_frame_space(
    frames: np.ndarray,
    supplied: Optional[dict],
    default_space_id: str,
) -> dict:
    height, width = frames.shape[1:3]
    if supplied is None:
        return raster_space(default_space_id, [width, height])
    space = validate_raster_space(supplied)
    if space["size_px"] != [width, height]:
        raise ValueError(
            "Prepared mini-map frames are {}x{}, but their coordinate space "
            "declares {}x{}".format(
                width,
                height,
                space["size_px"][0],
                space["size_px"][1],
            )
        )
    return space

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


def _read_session_segment(
    reader: SessionReader,
    stream_id: str,
    interval: Sequence[float],
    crop_xywh: Sequence[int],
) -> Tuple[np.ndarray, float]:
    """Read a timed segment from either a traceable image series or video."""

    records = list(reader.frames_by_stream.get(stream_id, []))
    storage_kinds = {
        str((record.get("storage") or {}).get("kind") or "video")
        for record in records
    }
    if storage_kinds != {"image_series"}:
        if storage_kinds not in ({"video"}, set()):
            raise ValueError(
                "Stream {} mixes incompatible frame storage: {}".format(
                    stream_id, sorted(storage_kinds)
                )
            )
        return _read_video_segment(
            reader.video_path(stream_id),
            interval,
            crop_xywh,
            frame_records=records,
        )

    start_ns = int(float(interval[0]) * 1.0e9)
    end_ns = int(float(interval[1]) * 1.0e9)
    selected = [
        record
        for record in records
        if start_ns <= int(record["session_time_ns"]) <= end_ns
    ]
    if not selected:
        raise ValueError(
            "No {} image-series frames fall inside {:.3f}..{:.3f} seconds".format(
                stream_id, float(interval[0]), float(interval[1])
            )
        )
    if len(selected) < 12:
        raise ValueError("Calibration segment contains fewer than 12 image frames")
    frames = reader.read_image_frames(selected)
    x, y, width, height = map(int, crop_xywh)
    if min(x, y) < 0 or min(width, height) <= 0:
        raise ValueError("Invalid mini-map crop")
    if x + width > frames.shape[2] or y + height > frames.shape[1]:
        raise ValueError(
            "Mini-map crop {} exceeds image-series frame {}x{}".format(
                list(map(int, crop_xywh)), frames.shape[2], frames.shape[1]
            )
        )
    frames = frames[:, y : y + height, x : x + width]
    times = np.asarray(
        [int(record["session_time_ns"]) for record in selected], dtype=np.int64
    )
    if len(times) > 1:
        median_delta = float(np.median(np.diff(times)))
        fps = 1.0e9 / median_delta if median_delta > 0 else 1.0
    else:
        fps = 1.0
    return frames, float(fps)


def _frame_record_raster_space(records: Sequence[dict]) -> Optional[dict]:
    """Preserve a producer-declared raster identity for derived geometry."""

    declared = [
        (record.get("metadata") or {}).get("image_space") for record in records
    ]
    present = [value for value in declared if value]
    if not present:
        return None
    if len(present) != len(records):
        raise ValueError("Session stream mixes frames with and without image-space metadata")
    first = present[0]
    if any(value != first for value in present[1:]):
        raise ValueError("Session stream changes image space between calibration frames")
    size = [int(value) for value in first.get("stored_size_px") or []]
    if len(size) != 2:
        raise ValueError("Session image-space metadata has no stored_size_px")
    for record in records:
        if [int(record["width"]), int(record["height"])] != size:
            raise ValueError("Session frame dimensions disagree with image-space metadata")
    space_id = str(first.get("space_id") or "")
    if not space_id:
        raise ValueError("Session image-space metadata has no space_id")
    canonical_id = first.get("canonical_space_id")
    local_to_canonical = first.get("local_to_canonical_3x3")
    if canonical_id and local_to_canonical and str(canonical_id) != space_id:
        return raster_space(
            space_id,
            size,
            parent_space_id=str(canonical_id),
            local_to_parent_3x3=local_to_canonical,
        )
    return raster_space(space_id, size)

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
    thresholded_frame_count = 0
    component_areas = []
    component_distances = []
    rejected_small = 0
    rejected_large = 0
    rejected_distance = 0
    distance_limit = search_radius * 0.5
    for frame_index, frame in enumerate(frames):
        binary = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), lower, upper)
        binary[~search_mask] = 0
        if np.any(binary):
            thresholded_frame_count += 1
        count, labels, stats, component_centroids = cv2.connectedComponentsWithStats(
            binary, 8
        )
        candidates = []
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            distance = float(np.linalg.norm(component_centroids[index] - center))
            component_areas.append(area)
            component_distances.append(distance)
            if area < min_area:
                rejected_small += 1
            elif area > max_area:
                rejected_large += 1
            elif distance >= distance_limit:
                rejected_distance += 1
            else:
                candidates.append((distance, index))
        if not candidates:
            continue
        selected = min(candidates)[1]
        masks.append((labels == selected).astype(np.uint8))
        centroids.append(component_centroids[selected])
        indices.append(frame_index)
    if not masks:
        observed_area = (
            "{}..{} px".format(min(component_areas), max(component_areas))
            if component_areas else "none"
        )
        nearest = (
            "{:.2f} px".format(min(component_distances))
            if component_distances else "none"
        )
        raise RuntimeError(
            "Cursor component gate accepted 0/{} frames. Required HSV {}..{}, "
            "component area {}..{} px, and centroid distance < {:.2f} px from "
            "expected center [{:.2f}, {:.2f}]. Observed: threshold pixels in "
            "{}/{} frames, {} connected components, area range {}, nearest "
            "centroid {}; rejected {} too small, {} too large, {} too far. "
            "This gate uses cursor-color components; mini-map boundary heatmaps "
            "do not by themselves satisfy it.".format(
                len(frames), lower.tolist(), upper.tolist(), min_area, max_area,
                distance_limit, float(center[0]), float(center[1]),
                thresholded_frame_count, len(frames), len(component_areas),
                observed_area, nearest, rejected_small, rejected_large,
                rejected_distance,
            )
        )
    return np.stack(masks), np.asarray(centroids), np.asarray(indices)


def _rotation_center(movement_frames: np.ndarray, boundary: dict, config: dict) -> dict:
    initial = np.array([boundary["center_x"], boundary["center_y"]])
    masks, centroids, indices = _cursor_components(movement_frames, initial, config)
    if len(centroids) < 3:
        raise RuntimeError(
            "Cursor rotation-center circle fit needs at least 3 eligible frames; "
            "the component gate accepted {}/{}. Center and shape were not guessed."
            .format(len(centroids), len(movement_frames))
        )
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
    polygon_relative = np.asarray(
        symmetric_model["polygon_relative_xy"], dtype=np.float64
    )
    rotating_envelope_radius = float(
        np.max(np.linalg.norm(polygon_relative, axis=1))
    )
    polygon_span = float(
        np.max(
            np.linalg.norm(
                polygon_relative[:, None, :] - polygon_relative[None, :, :],
                axis=2,
            )
        )
    )
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
            "rotating_cursor_envelope_radius_px": rotating_envelope_radius,
            "rotating_cursor_envelope_diameter_px": float(
                2.0 * rotating_envelope_radius
            ),
            "cursor_polygon_max_span_px": polygon_span,
            "size_definition": (
                "rotating_cursor_envelope_diameter_px is twice the greatest "
                "distance from the fitted rotation center to the persistent "
                "cursor polygon; it contains the cursor at every rotation angle"
            ),
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
    boundary_only: bool = False,
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

    if boundary_only:
        return files

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
        geometry_spatial_schema_version=np.array(
            [SPATIAL_SCHEMA_VERSION], dtype=np.int32
        ),
        boundary_space_id=np.array(str(bmetrics["space"]["space_id"])),
        rotation_center_space_id=np.array(str(cm["space"]["space_id"])),
        minimap_mask_space_id=np.array(str(bmetrics["space"]["space_id"])),
        cursor_polygon_parent_space_id=np.array(str(cm["space"]["space_id"])),
        cursor_polygon_coordinate_convention=np.array(
            "offset_xy_from_rotation_center"
        ),
        cursor_polygon_origin_xy=np.asarray([cm["x"], cm["y"]], dtype=np.float64),
        geometry_space_size_px=np.asarray(
            bmetrics["space"]["size_px"], dtype=np.int32
        ),
    )
    return files


def calibrate_minimap_boundary_frames(
    frames: np.ndarray,
    output_path: Path,
    config: Optional[dict] = None,
    progress=None,
    write_evidence: bool = True,
    frame_space: Optional[dict] = None,
) -> dict:
    """Run the verified mini-map boundary calibration on prepared frame arrays.

    Game control, image acquisition, frame selection, and cropping belong
    outside this function. The boundary fit and review evidence always come
    from the same implementation used by :func:`calibrate_minimap_frames`.
    """

    if frames.ndim != 4:
        raise ValueError("Calibration frames must be N x H x W x C arrays")
    unknown_config = set(config or {}) - set(DEFAULT_CONFIG["boundary"])
    if unknown_config:
        raise ValueError(
            "Unsupported mini-map boundary config keys: {}".format(
                ", ".join(sorted(unknown_config))
            )
        )
    boundary_config = json.loads(json.dumps(DEFAULT_CONFIG["boundary"]))
    boundary_config.update(config or {})
    if progress:
        progress("Fitting the circular mini-map boundary")
    boundary = _boundary_model(frames, boundary_config)
    space = _prepared_frame_space(
        frames, frame_space, "minimap_boundary_input_pixels"
    )
    boundary["metrics"] = bind_geometry(
        boundary["metrics"], "circle", space
    )
    evidence = []
    if write_evidence:
        if progress:
            progress("Rendering mini-map boundary evidence")
        evidence = list(
            _write_evidence(
                Path(output_path),
                frames,
                frames,
                boundary,
                None,
                None,
                boundary_only=True,
            )
        )
    return {
        "model": boundary,
        "outer_boundary": boundary["metrics"],
        "geometry_space": space,
        "config": boundary_config,
        "evidence": evidence,
    }


def _cursor_orbit_aligned_frames(
    frames: np.ndarray, center: dict
) -> Tuple[np.ndarray, Sequence[float]]:
    """Render detected cursor masks in a common screen direction."""

    pivot = np.asarray(
        [center["metrics"]["x"], center["metrics"]["y"]], dtype=np.float64
    )
    hsv_color = np.uint8([[[93, 255, 255]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
    aligned = []
    angles = []
    height, width = frames.shape[1:3]
    for mask, centroid in zip(center["masks"], center["centroids"]):
        vector = np.asarray(centroid, dtype=np.float64) - pivot
        angle = math.degrees(math.atan2(vector[1], vector[0]))
        matrix = cv2.getRotationMatrix2D(tuple(pivot), angle, 1.0)
        rotated = cv2.warpAffine(
            mask.astype(np.uint8),
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[rotated > 0] = bgr_color
        aligned.append(image)
        angles.append(float(angle))
    if not aligned:
        raise RuntimeError("No cursor masks were available for orbit alignment")
    return np.stack(aligned), angles


def _write_cursor_orbit_evidence(
    output: Path,
    frames: np.ndarray,
    boundary: dict,
    center: dict,
    shape: Optional[dict],
    alignment_angles: Sequence[float],
) -> Sequence[dict]:
    output.mkdir(parents=True, exist_ok=True)
    files = []

    def save(name, image, title, category):
        path = output / name
        if not cv2.imwrite(str(path), image):
            raise RuntimeError("Could not write calibration evidence: {}".format(path))
        files.append({"name": name, "title": title, "category": category})

    metrics = center["metrics"]
    occupancy = center["masks"].mean(axis=0)
    save(
        "cursor_center_heatmap.png",
        _color_heatmap(occupancy * (1.0 - occupancy)),
        "Cursor orbit temporal heatmap",
        "center",
    )
    orbit = frames.mean(axis=0).astype(np.uint8)
    for point in center["centroids"]:
        cv2.circle(orbit, tuple(np.round(point).astype(int)), 2, (0, 255, 255), -1)
    cv2.circle(
        orbit,
        (round(metrics["x"]), round(metrics["y"])),
        max(1, round(metrics["centroid_orbit_radius_px"])),
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.drawMarker(
        orbit,
        (round(metrics["x"]), round(metrics["y"])),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        12,
        1,
    )
    save(
        "cursor_center_orbit.png",
        orbit,
        "Cursor centroids and fitted rotation center",
        "center",
    )

    if shape is not None:
        probability = cv2.resize(
            _color_heatmap(shape["probability"]),
            (410, 410),
            interpolation=cv2.INTER_NEAREST,
        )
        binary = cv2.resize(
            (shape["binary"] * 255).astype(np.uint8),
            (410, 410),
            interpolation=cv2.INTER_NEAREST,
        )
        edges = cv2.resize(
            cv2.Canny((shape["binary"] * 255).astype(np.uint8), 50, 150),
            (410, 410),
            interpolation=cv2.INTER_NEAREST,
        )
        save("cursor_shape_probability.png", probability, "Aligned cursor probability", "cursor_shape")
        save("cursor_shape_binary.png", binary, "Aligned cursor binary shape", "cursor_shape")
        save("cursor_shape_edge.png", edges, "Aligned cursor shape edge", "cursor_shape")

        overlay = np.zeros((410, 410, 3), np.uint8)
        overlay[binary > 0] = (235, 235, 235)
        overlay[edges > 0] = (0, 255, 0)
        cv2.drawMarker(overlay, (205, 205), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        save("cursor_shape_overlay.png", overlay, "Aligned cursor contour and pivot", "cursor_shape")

        symmetric = shape["symmetric_model"]
        symmetric_view = cv2.resize(
            _color_heatmap(symmetric["symmetric_probability"]),
            (410, 410),
            interpolation=cv2.INTER_NEAREST,
        )
        polygon = np.round(symmetric["polygon_xy"] * 10).astype(np.int32)
        cv2.polylines(symmetric_view, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)
        save(
            "cursor_shape_symmetric_polygon.png",
            symmetric_view,
            "Symmetry-constrained cursor polygon",
            "cursor_shape",
        )
        template_polar = cv2.resize(
            _color_heatmap(shape["template_polar"]),
            (500, 720),
            interpolation=cv2.INTER_NEAREST,
        )
        correlation = cv2.resize(
            _color_heatmap(shape["correlations"]),
            (720, max(360, 2 * len(shape["correlations"]))),
            interpolation=cv2.INTER_NEAREST,
        )
        save("cursor_shape_polar_template.png", template_polar, "Cursor polar template", "polar")
        save("cursor_shape_polar_correlation.png", correlation, "Cursor polar correlation", "polar")

        angle_plot = np.full((240, 720, 3), 18, np.uint8)
        if alignment_angles:
            points = []
            for index, angle in enumerate(alignment_angles):
                x = int(round(index * 719 / max(1, len(alignment_angles) - 1)))
                y = int(round(120 + ((angle + 180.0) % 360.0 - 180.0) * 0.55))
                points.append([x, int(np.clip(y, 5, 234))])
            cv2.polylines(angle_plot, [np.asarray(points, np.int32)], False, (0, 220, 255), 2)
        save("cursor_orbit_alignment.png", angle_plot, "Per-frame cursor alignment angle", "quality")

    model_mask = np.zeros(frames.shape[1:3], dtype=np.uint8)
    cv2.circle(
        model_mask,
        (round(boundary["center_x"]), round(boundary["center_y"])),
        round(boundary["radius"]),
        255,
        -1,
    )
    model = dict(
        boundary=np.asarray(
            [boundary["center_x"], boundary["center_y"], boundary["radius"]]
        ),
        rotation_center=np.asarray([metrics["x"], metrics["y"]]),
        minimap_mask=model_mask,
        geometry_spatial_schema_version=np.asarray([SPATIAL_SCHEMA_VERSION]),
        boundary_space_id=np.asarray(str(boundary["space"]["space_id"])),
        rotation_center_space_id=np.asarray(str(metrics["space"]["space_id"])),
        geometry_space_size_px=np.asarray(boundary["space"]["size_px"], dtype=np.int32),
    )
    if shape is not None:
        symmetric = shape["symmetric_model"]
        model.update(
            cursor_probability=shape["probability"],
            cursor_binary=shape["binary"],
            cursor_symmetric_probability=symmetric["symmetric_probability"],
            cursor_symmetric_binary=symmetric["symmetric_binary"],
            cursor_polygon_relative_xy=symmetric["polygon_relative_xy"],
            cursor_symmetry_axis_deg=np.asarray([symmetric["axis_angle_deg"]]),
        )
    np.savez_compressed(output / "model.npz", **model)
    return files


def calibrate_cursor_orbit_frames(
    frames: np.ndarray,
    output_path: Path,
    *,
    outer_boundary: dict,
    config: Optional[dict] = None,
    provenance: Optional[dict] = None,
    progress=None,
    frame_space: Optional[dict] = None,
) -> dict:
    """Fit cursor pivot and shape while preserving an existing map boundary."""

    if frames.ndim != 4:
        raise ValueError("Cursor-orbit frames must be N x H x W x C arrays")
    if len(frames) < 4:
        raise ValueError("Cursor-orbit calibration requires at least four frames")
    merged = _merged_config(config)
    space = _prepared_frame_space(frames, frame_space, "cursor_orbit_crop_pixels")
    boundary = require_spatial_geometry(outer_boundary, "circle")
    if boundary["space"] != space:
        raise ValueError("Cursor-orbit frames and mini-map boundary must share one space")
    if progress:
        progress("Fitting cursor rotation center from balanced joystick pulses")
    center = _rotation_center(frames, boundary, merged["cursor"])
    center["metrics"] = bind_geometry(center["metrics"], "point", space)
    require_same_space(boundary, center["metrics"])
    shape = None
    alignment_angles = []
    shape_failure = None
    try:
        aligned_frames, alignment_angles = _cursor_orbit_aligned_frames(frames, center)
        if progress:
            progress("Fitting cursor shape after direction normalization")
        shape = _cursor_shape(aligned_frames, center["metrics"], merged["cursor"])
        shape["metrics"]["source"] = (
            "cursor_orbit_masks_aligned_to_canonical_direction"
        )
        shape["metrics"]["alignment_angles_screen_deg"] = alignment_angles
    except Exception as exc:
        shape_failure = {
            "stage": "cursor_shape_after_rotation_center",
            "exception_type": type(exc).__name__,
            "reason": str(exc),
            "preserved_result": "rotation_center",
        }
        if progress:
            progress(
                "Cursor rotation center retained; shape unavailable: {}: {}"
                .format(type(exc).__name__, exc)
            )
    output_path = Path(output_path)
    evidence = _write_cursor_orbit_evidence(
        output_path, frames, boundary, center, shape, alignment_angles
    )
    pivot = center["metrics"]
    offset_x = pivot["x"] - boundary["center_x"]
    offset_y = pivot["y"] - boundary["center_y"]
    model_geometry = {
        "minimap_mask": bind_geometry({"array_name": "minimap_mask"}, "mask", space),
    }
    if shape is not None:
        model_geometry["cursor_polygon"] = bind_geometry(
            {
                "points_xy": (
                    np.asarray(shape["symmetric_model"]["polygon_relative_xy"])
                    + np.asarray([pivot["x"], pivot["y"]])
                ).tolist(),
                "model_array_name": "cursor_polygon_relative_xy",
                "model_coordinate_convention": "offset_xy_from_rotation_center",
            },
            "polygon",
            space,
        )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "review_required" if shape is not None else "partial",
        "result_level": (
            "rotation_center_and_shape" if shape is not None
            else "rotation_center_only"
        ),
        "calibration_kind": "cursor_orbit",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance or {},
        "config": merged,
        "outer_boundary": boundary,
        "rotation_center": pivot,
        "center_offset": bind_geometry(
            {
                "dx": float(offset_x),
                "dy": float(offset_y),
                "magnitude_px": float(math.hypot(offset_x, offset_y)),
            },
            "vector",
            space,
        ),
        "geometry_space": space,
        "model_geometry": model_geometry,
        "cursor_shape": shape["metrics"] if shape is not None else None,
        "capabilities": {
            "rotation_center": {
                "status": "available",
                "confidence": float(pivot["confidence"]),
                "detected_frames": int(pivot["detected_frames"]),
                "total_frames": int(pivot["total_frames"]),
            },
            "cursor_shape": (
                {
                    "status": "available",
                    "confidence": float(shape["metrics"]["confidence"]),
                }
                if shape is not None else {
                    "status": "unavailable",
                    **shape_failure,
                }
            ),
        },
        "failure_reasons": [shape_failure] if shape_failure is not None else [],
        "overall_confidence": float(
            math.sqrt(max(0.0, pivot["confidence"] * shape["metrics"]["confidence"]))
            if shape is not None else pivot["confidence"]
        ),
        "confidence_scale": "0..1 experimental; human evidence review is authoritative",
        "model_file": "model.npz",
        "evidence": evidence,
    }
    _atomic_json(output_path / "calibration.json", result)
    return result


def calibrate_cursor_static_frames(
    frames: np.ndarray,
    output_path: Path,
    *,
    outer_boundary: dict,
    existing_rotation_center: Optional[dict] = None,
    config: Optional[dict] = None,
    provenance: Optional[dict] = None,
    progress=None,
    frame_space: Optional[dict] = None,
) -> dict:
    """Fit observable cursor shape without inventing rotation-center evidence."""

    if frames.ndim != 4:
        raise ValueError("Static cursor frames must be N x H x W x C arrays")
    if len(frames) < 4:
        raise ValueError("Static cursor calibration requires at least four frames")
    merged = _merged_config(config)
    space = _prepared_frame_space(frames, frame_space, "cursor_static_crop_pixels")
    boundary = require_spatial_geometry(outer_boundary, "circle")
    if boundary["space"] != space:
        raise ValueError("Static cursor frames and mini-map boundary must share one space")
    center_status = "existing_verified_rotation_center"
    if existing_rotation_center is not None:
        center = require_spatial_geometry(existing_rotation_center, "point")
        require_same_space(boundary, center)
    else:
        initial = np.asarray(
            [boundary["center_x"], boundary["center_y"]], dtype=np.float64
        )
        _, centroids, _ = _cursor_components(frames, initial, merged["cursor"])
        observed = np.median(centroids, axis=0)
        center = bind_geometry(
            {
                "x": float(observed[0]),
                "y": float(observed[1]),
                "confidence": 0.0,
                "confidence_level": "unavailable",
            },
            "point",
            space,
        )
        center_status = "not_observable_from_static_cursor"
    if progress:
        progress("Fitting static cursor shape from persistent color components")
    shape = _cursor_shape(frames, center, merged["cursor"])
    shape["metrics"]["source"] = "static_cursor_persistence"
    shape["metrics"]["observed_static_cursor_max_span_px"] = float(
        shape["metrics"]["cursor_polygon_max_span_px"]
    )
    if existing_rotation_center is None:
        shape["metrics"]["rotating_cursor_envelope_radius_px"] = None
        shape["metrics"]["rotating_cursor_envelope_diameter_px"] = None
        shape["metrics"]["size_definition"] = (
            "Only observed_static_cursor_max_span_px is supported by static data; "
            "a rotating envelope requires a verified rotation center"
        )

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    probability = _color_heatmap(shape["probability"])
    binary = (shape["binary"] * 255).astype(np.uint8)
    edge = cv2.Canny(binary, 50, 150)
    overlay = frames.mean(axis=0).astype(np.uint8)
    polygon = np.round(
        np.asarray(shape["symmetric_model"]["polygon_relative_xy"])
        + np.asarray([center["x"], center["y"]])
    ).astype(np.int32)
    cv2.polylines(overlay, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.drawMarker(
        overlay,
        (int(round(center["x"])), int(round(center["y"]))),
        (0, 0, 255),
        cv2.MARKER_CROSS,
        14,
        1,
    )
    evidence_images = {
        "cursor_static_probability.png": probability,
        "cursor_static_binary.png": binary,
        "cursor_static_edge.png": edge,
        "cursor_static_shape_overlay.png": overlay,
    }
    evidence = []
    for name, image in evidence_images.items():
        if not cv2.imwrite(str(output_path / name), image):
            raise RuntimeError("Could not write cursor evidence {}".format(name))
        evidence.append({"name": name, "category": "cursor_shape"})
    np.savez_compressed(
        output_path / "model.npz",
        cursor_probability=shape["probability"],
        cursor_binary=shape["binary"],
        cursor_polygon_relative_xy=shape["symmetric_model"]["polygon_relative_xy"],
        shape_center=np.asarray([center["x"], center["y"]]),
        geometry_spatial_schema_version=np.asarray([SPATIAL_SCHEMA_VERSION]),
        shape_center_space_id=np.asarray(str(space["space_id"])),
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "review_required",
        "result_level": (
            "rotation_center_and_shape"
            if existing_rotation_center is not None else "shape_only"
        ),
        "calibration_kind": "cursor_static",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance or {},
        "config": merged,
        "outer_boundary": boundary,
        "rotation_center": center if existing_rotation_center is not None else None,
        "rotation_center_status": center_status,
        "shape_center": center,
        "cursor_shape": shape["metrics"],
        "capabilities": {
            "rotation_center": {
                "status": (
                    "available" if existing_rotation_center is not None
                    else "unavailable"
                ),
                "reason": (
                    None if existing_rotation_center is not None
                    else "A static cursor series cannot identify a rotation center"
                ),
            },
            "cursor_shape": {
                "status": "available",
                "confidence": float(shape["metrics"]["confidence"]),
            },
        },
        "failure_reasons": [],
        "geometry_space": space,
        "overall_confidence": float(shape["metrics"]["confidence"]),
        "confidence_scale": "0..1 experimental; human evidence review is authoritative",
        "model_file": "model.npz",
        "evidence": evidence,
    }
    _atomic_json(output_path / "calibration.json", result)
    return result


def calibrate_minimap_frames(
    rotation_frames: np.ndarray,
    movement_frames: np.ndarray,
    output_path: Path,
    config: Optional[dict] = None,
    provenance: Optional[dict] = None,
    progress=None,
    ordinary_frames: Optional[np.ndarray] = None,
    frame_space: Optional[dict] = None,
) -> dict:
    """Calibrate from already labeled frame arrays and persist review evidence."""
    config = _merged_config(config)
    output_path = Path(output_path)
    if rotation_frames.ndim != 4 or movement_frames.ndim != 4:
        raise ValueError("Calibration frames must be N x H x W x C arrays")
    if rotation_frames.shape[1:3] != movement_frames.shape[1:3]:
        raise ValueError("Rotation and movement frames must share one raster space")
    space = _prepared_frame_space(
        rotation_frames, frame_space, "minimap_calibration_crop_pixels"
    )
    if progress:
        progress("Fitting the circular mini-map boundary")
    boundary = _boundary_model(rotation_frames, config["boundary"])
    boundary["metrics"] = bind_geometry(
        boundary["metrics"], "circle", space
    )
    if progress:
        progress("Fitting the cursor pivot from movement frames")
    center = _rotation_center(movement_frames, boundary["metrics"], config["cursor"])
    center["metrics"] = bind_geometry(center["metrics"], "point", space)
    require_same_space(boundary["metrics"], center["metrics"])
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
        "center_offset": bind_geometry(
            {
                "dx": float(offset_x),
                "dy": float(offset_y),
                "magnitude_px": float(math.hypot(offset_x, offset_y)),
            },
            "vector",
            space,
        ),
        "geometry_space": space,
        "model_geometry": {
            "minimap_mask": bind_geometry(
                {"array_name": "minimap_mask"}, "mask", space
            ),
            "cursor_polygon": bind_geometry(
                {
                    "points_xy": (
                        np.asarray(
                            shape["symmetric_model"]["polygon_relative_xy"],
                            dtype=np.float64,
                        )
                        + np.asarray([pivot["x"], pivot["y"]], dtype=np.float64)
                    ).tolist(),
                    "model_array_name": "cursor_polygon_relative_xy",
                    "model_coordinate_convention": (
                        "offset_xy_from_rotation_center"
                    ),
                },
                "polygon",
                space,
            ),
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
    from aria_trace.services.calibration.cursor.pose import estimate_cursor_pose_frames

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
    if ordinary_frames is not None:
        if progress:
            progress("Measuring cursor turning dynamics from standard sessions")
        from aria_trace.services.calibration.cursor.dynamics import summarize_cursor_dynamics
        from aria_trace.services.calibration.cursor.pose import (
            CursorPoseEstimator,
            estimate_cursor_pose_sequence,
            timing_summary_ms,
        )

        movement_poses = [
            json.loads(line)
            for line in (output_path / "cursor_poses.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        ordinary_source = (
            ((provenance or {}).get("segment_sessions") or {}).get(
                "ordinary_cruise", {}
            )
        )
        ordinary_fps = float(ordinary_source.get("container_fps") or pose_fps)
        ordinary_times_ns = [
            int(index * 1.0e9 / ordinary_fps)
            for index in range(len(ordinary_frames))
        ]
        dynamics_estimator = CursorPoseEstimator(output_path / "calibration.json")
        ordinary_private, ordinary_durations_ns = estimate_cursor_pose_sequence(
            dynamics_estimator,
            ordinary_frames,
            frame_indices=list(range(len(ordinary_frames))),
            session_times_ns=ordinary_times_ns,
        )
        ordinary_poses = [
            dynamics_estimator.public_result(item) for item in ordinary_private
        ]
        dynamics = summarize_cursor_dynamics(
            {
                "ordinary_cruise": ordinary_poses,
                "movement_only": movement_poses,
            },
            source_provenance={
                role: dict(value)
                for role, value in (
                    ((provenance or {}).get("segment_sessions") or {}).items()
                )
                if role in ("ordinary_cruise", "movement_only")
            },
        )
        dynamics["ordinary_pose_estimation_benchmark"] = timing_summary_ms(
            ordinary_durations_ns,
            "standard ordinary-cruise cursor pose measurements",
        )
        dynamics["measurement_files"] = {
            "ordinary_cruise": "cursor_dynamics_ordinary_poses.jsonl",
            "movement_only": "cursor_poses.jsonl",
        }
        with (output_path / "cursor_dynamics_ordinary_poses.jsonl").open(
            "w", encoding="utf-8"
        ) as stream:
            for pose in ordinary_poses:
                stream.write(json.dumps(pose, separators=(",", ":")) + "\n")
        result["cursor_temporal_dynamics"] = dynamics
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
    stream_id = (
        "main" if reader.frames_by_stream.get("main") else "android_phone"
    )
    frame_records = reader.frames_by_stream.get(stream_id, [])
    if not frame_records:
        raise ValueError("Calibration session has no main or android_phone frames")
    if progress:
        progress("Decoding the rotation-only calibration frames")
    rotation_frames, fps = _read_session_segment(
        reader,
        stream_id,
        segments["rotation_only"],
        config["crop_xywh"],
    )
    if progress:
        progress("Decoding the movement-only calibration frames")
    movement_frames, _ = _read_session_segment(
        reader,
        stream_id,
        segments["movement_only"],
        config["crop_xywh"],
    )
    source_space = _frame_record_raster_space(frame_records)
    crop_space = minimap_crop_space(
        config["crop_xywh"][2:],
        parent_space_id=(source_space["space_id"] if source_space else None),
        crop_xywh=(config["crop_xywh"] if source_space else None),
    )
    provenance = {
        "session_path": str(session_path),
        "session_id": reader.manifest.get("session_id"),
        "frame_storage": (
            (frame_records[0].get("storage") or {}).get("kind") or "video"
        ),
        "stream_id": stream_id,
        "source_image_space": source_space,
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
        frame_space=crop_space,
    )


def calibrate_segment_sessions(
    rotation_session_path: Path,
    movement_session_path: Path,
    output_path: Path,
    config: Optional[dict] = None,
    forward_session_path: Optional[Path] = None,
    progress=None,
    ordinary_session_path: Optional[Path] = None,
) -> dict:
    """Calibrate from short sessions whose capture plan supplies each label."""
    config = _merged_config(config)

    def read_session(path: Path, expected_labels):
        if isinstance(expected_labels, str):
            expected_labels = (expected_labels,)
        else:
            expected_labels = tuple(expected_labels)
        path = Path(path).resolve()
        reader = SessionReader(path)
        context = reader.manifest.get("context") or {}
        actual_label = context.get("segment_label")
        metadata_path = path / "session_metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if "label" in metadata:
                actual_label = metadata.get("label") or None
        if actual_label not in expected_labels:
            expected_text = " or ".join(expected_labels)
            article = "an" if expected_text[:1].lower() in "aeiou" else "a"
            raise ValueError(
                "Expected {} {} segment, got {} from {}".format(
                    article, expected_text, actual_label or "unlabeled", path
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
        return reader, frames, fps, actual_label

    ordinary_reader = None
    ordinary_frames = None
    ordinary_fps = None
    if ordinary_session_path is not None:
        if progress:
            progress("Decoding the ordinary-cruise calibration session")
        (
            ordinary_reader,
            ordinary_frames,
            ordinary_fps,
            ordinary_recorded_label,
        ) = read_session(
            ordinary_session_path, ORDINARY_MOTION_SEGMENT_LABELS
        )
    if progress:
        progress("Decoding the rotation-only calibration session")
    rotation_reader, rotation_frames, rotation_fps, _ = read_session(
        rotation_session_path, "rotation_only"
    )
    if progress:
        progress("Decoding the movement-only calibration session")
    movement_reader, movement_frames, movement_fps, _ = read_session(
        movement_session_path, "movement_only"
    )
    forward_reference = None
    if forward_session_path is not None:
        forward_reader, _forward_frames, forward_fps, _ = read_session(
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
    if ordinary_reader is not None:
        provenance["segments"]["ordinary_cruise"] = [
            0.0,
            float(ordinary_reader.manifest.get("duration_ns") or 0) / 1.0e9,
        ]
        provenance["segment_sessions"]["ordinary_cruise"] = {
            "session_path": str(Path(ordinary_session_path).resolve()),
            "session_id": ordinary_reader.manifest.get("session_id"),
            "recorded_label": ordinary_recorded_label,
            "frame_count": len(ordinary_frames),
            "container_fps": ordinary_fps,
        }
    return calibrate_minimap_frames(
        rotation_frames,
        movement_frames,
        output_path,
        config=config,
        provenance=provenance,
        progress=progress,
        ordinary_frames=ordinary_frames,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rotation", nargs=2, type=float, required=True, metavar=("START", "END"))
    parser.add_argument("--movement", nargs=2, type=float, required=True, metavar=("START", "END"))
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
    result = calibrate_session(
        args.session,
        args.output,
        {"rotation_only": args.rotation, "movement_only": args.movement},
        config,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
