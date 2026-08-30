"""Learn a destination-oriented teleport behavior from synchronized evidence."""

import bisect
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from aria_trace.services.tracking.runtime import GlobalMapLocalizer, MinimapExtractor
from .session import SessionReader
from .teleport_behavior import make_teleport_behavior_sample, save_teleport_behavior_sample


TELEPORT_BEHAVIOR_MODEL = {
    "model": "visually_guarded_state_machine",
    "initial_state": "world_ready",
    "terminal_state": "destination_ready",
    "states": [
        "world_ready",
        "map_open",
        "map_navigation",
        "target_selected",
        "teleport_requested",
        "confirmation_ready",
        "loading",
        "destination_ready",
    ],
    "transitions": [
        {
            "from": "world_ready",
            "to": "map_open",
            "actions": ["press_open_map", "click_minimap"],
            "guard": "global_map_visible",
        },
        {
            "from": "map_open",
            "to": "map_navigation",
            "actions": ["zoom_out", "pan", "zoom_in"],
            "guard": "target_visible_and_separable",
            "policy": "choose pan and zoom from the localized viewport; do not replay fixed counts",
        },
        {
            "from": "map_navigation",
            "to": "target_selected",
            "actions": ["click_teleport_target"],
            "guard": "selected_target_panel_matches_expected_target",
        },
        {
            "from": "target_selected",
            "to": "teleport_requested",
            "actions": ["click_teleport_button"],
            "guard": "loading_started_or_confirmation_visible",
        },
        {
            "from": "teleport_requested",
            "to": "confirmation_ready",
            "actions": ["wait_for_confirmation"],
            "guard": "confirmation_visible",
            "optional": True,
        },
        {
            "from": "confirmation_ready",
            "to": "loading",
            "actions": ["click_confirm"],
            "guard": "loading_started",
            "optional": True,
        },
        {
            "from": "teleport_requested",
            "to": "loading",
            "actions": ["wait"],
            "guard": "loading_started_without_confirmation",
            "optional": True,
        },
        {
            "from": "loading",
            "to": "destination_ready",
            "actions": ["wait"],
            "guard": "stable_world_hud_and_destination_localization_consensus",
        },
    ],
    "scope": "explicit learned game behavior; never infer teleportation from a tracker position jump",
}


def _time_s(item: Mapping) -> float:
    return float(item.get("session_time_ns") or 0) / 1.0e9


def _finite_xy(value: Sequence[float]) -> List[float]:
    result = [float(value[0]), float(value[1])]
    if not all(math.isfinite(item) for item in result):
        raise RuntimeError("Teleport analysis produced non-finite coordinates")
    return result


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError("Could not write teleport evidence image: {}".format(path))


def parse_teleport_inputs(inputs: Sequence[Mapping]) -> dict:
    """Summarize semantic input episodes without requiring absolute mouse XY."""
    map_open = []
    wheel_events = []
    left_episodes = []
    active = None
    for item in sorted(inputs, key=lambda row: int(row.get("session_time_ns") or 0)):
        payload = item.get("payload") or {}
        time_s = _time_s(item)
        if (
            item.get("kind") == "pc_raw_keyboard"
            and payload.get("pressed")
            and str(payload.get("key_name") or "").upper() == "M"
        ):
            map_open.append(
                {
                    "time_s": time_s,
                    "trigger": "keyboard",
                    "key": "M",
                }
            )
        if item.get("kind") != "pc_raw_mouse":
            continue
        wheel = int(payload.get("wheel_delta") or 0)
        if wheel:
            wheel_events.append({"time_s": time_s, "delta": wheel})
        transitions = payload.get("button_transitions") or []
        if "left_down" in transitions:
            active = {
                "start_s": time_s,
                "end_s": time_s,
                "delta_xy": [0.0, 0.0],
                "path_length_px": 0.0,
                "motion_event_count": 0,
            }
        if active is not None:
            dx = float(payload.get("delta_x") or 0.0)
            dy = float(payload.get("delta_y") or 0.0)
            active["delta_xy"][0] += dx
            active["delta_xy"][1] += dy
            active["path_length_px"] += math.hypot(dx, dy)
            active["motion_event_count"] += 1
            active["end_s"] = time_s
        if "left_up" in transitions and active is not None:
            active["duration_s"] = active["end_s"] - active["start_s"]
            active["classification"] = (
                "drag" if active["path_length_px"] >= 12.0 else "stationary_click"
            )
            active["delta_xy"] = [
                float(active["delta_xy"][0]),
                float(active["delta_xy"][1]),
            ]
            active["path_length_px"] = float(active["path_length_px"])
            left_episodes.append(active)
            active = None

    wheel_bursts = []
    for item in wheel_events:
        direction = 1 if item["delta"] > 0 else -1
        if (
            not wheel_bursts
            or item["time_s"] - wheel_bursts[-1]["end_s"] > 0.35
            or direction != wheel_bursts[-1]["direction"]
        ):
            wheel_bursts.append(
                {
                    "start_s": item["time_s"],
                    "end_s": item["time_s"],
                    "direction": direction,
                    "notches": 0.0,
                    "event_count": 0,
                }
            )
        burst = wheel_bursts[-1]
        burst["end_s"] = item["time_s"]
        burst["notches"] += abs(item["delta"]) / 120.0
        burst["event_count"] += 1
    for burst in wheel_bursts:
        burst["action"] = "zoom_in" if burst["direction"] > 0 else "zoom_out"
        burst["notches"] = float(burst["notches"])

    return {
        "map_open_events": map_open,
        "wheel_bursts": wheel_bursts,
        "left_button_episodes": left_episodes,
        "drag_episodes": [
            dict(item) for item in left_episodes if item["classification"] == "drag"
        ],
        "stationary_clicks": [
            dict(item)
            for item in left_episodes
            if item["classification"] == "stationary_click"
        ],
    }


class _RecordedFrames:
    def __init__(self, reader: SessionReader) -> None:
        self.records = list(reader.frames_by_stream.get("main", []))
        if not self.records:
            raise ValueError("Teleport session has no main-stream frames")
        self.times_s = [_time_s(item) for item in self.records]
        self.capture = cv2.VideoCapture(str(reader.video_path("main")))
        if not self.capture.isOpened():
            raise RuntimeError("Cannot open teleport session video")

    def close(self) -> None:
        self.capture.release()

    def at(self, time_s: float) -> Tuple[np.ndarray, dict]:
        position = bisect.bisect_left(self.times_s, float(time_s))
        position = max(0, min(position, len(self.records) - 1))
        index = int(self.records[position].get("frame_index", position))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.capture.read()
        if not ok:
            raise RuntimeError("Could not decode teleport frame {}".format(index))
        return frame, dict(self.records[position])

    def iter_from(self, time_s: float, stride: int = 10):
        position = bisect.bisect_left(self.times_s, float(time_s))
        position = max(0, min(position, len(self.records) - 1))
        frame_index = int(self.records[position].get("frame_index", position))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        decoded = frame_index
        while decoded < len(self.records):
            ok, frame = self.capture.read()
            if not ok:
                break
            if (decoded - frame_index) % max(1, int(stride)) == 0:
                yield frame, dict(self.records[decoded])
            decoded += 1


class _MapViewLocalizer:
    def __init__(self, mosaic: np.ndarray, coverage: Optional[np.ndarray] = None) -> None:
        self.mosaic = mosaic
        self.sift = cv2.SIFT_create(
            nfeatures=8000, contrastThreshold=0.005, edgeThreshold=15
        )
        mask = None
        if coverage is not None:
            mask = cv2.erode(np.uint8(coverage > 0) * 255, np.ones((5, 5), np.uint8))
        self.points, self.descriptors = self.sift.detectAndCompute(
            cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY), mask
        )
        if self.descriptors is None or len(self.points) < 12:
            raise ValueError("Map mosaic has too few features for teleport target localization")

    def localize(self, frame: np.ndarray, crop_xywh: Sequence[int]) -> dict:
        x, y, width, height = [int(item) for item in crop_xywh]
        query = frame[y : y + height, x : x + width]
        if query.shape[:2] != (height, width):
            raise ValueError("Map viewport crop exceeds the teleport frame")
        points, descriptors = self.sift.detectAndCompute(
            cv2.cvtColor(query, cv2.COLOR_BGR2GRAY), None
        )
        if descriptors is None or len(points) < 12:
            raise RuntimeError("Teleport map view has too few localizable features")
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
            descriptors, self.descriptors, k=2
        )
        matches = [
            first
            for pair in pairs
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < 0.75 * second.distance
        ]
        if len(matches) < 8:
            raise RuntimeError("Teleport map view has too few ratio matches")
        source = np.float32([points[item.queryIdx].pt for item in matches])
        target = np.float32([self.points[item.trainIdx].pt for item in matches])
        affine, inlier_mask = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
            maxIters=20000,
            confidence=0.999,
        )
        if affine is None or inlier_mask is None:
            raise RuntimeError("Teleport map view has no consistent similarity")
        inliers = inlier_mask.ravel().astype(bool)
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < 8:
            raise RuntimeError("Teleport map view has too few geometric inliers")
        predicted = cv2.transform(source.reshape((-1, 1, 2)), affine).reshape((-1, 2))
        errors = np.linalg.norm(predicted - target, axis=1)
        scale = math.hypot(float(affine[0, 0]), float(affine[0, 1]))
        angle = math.degrees(math.atan2(float(affine[1, 0]), float(affine[0, 0])))
        return {
            "affine_viewport_to_original_map_2x3": affine.tolist(),
            "ratio_match_count": len(matches),
            "inlier_count": inlier_count,
            "inlier_ratio": float(np.mean(inliers)),
            "reprojection_p95_px": float(np.percentile(errors[inliers], 95)),
            "scale_original_map_px_per_screen_px": scale,
            "rotation_deg": angle,
            "viewport_crop_xywh": [x, y, width, height],
        }


def _frame_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    difference = cv2.absdiff(first, second)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _click_panel_change(frames: _RecordedFrames, click: Mapping) -> float:
    first, _ = frames.at(max(0.0, float(click["start_s"]) - 0.18))
    second, _ = frames.at(float(click["end_s"]) + 0.35)
    difference = _frame_difference(first, second)
    height, width = difference.shape
    panel = difference[int(height * 0.08) : int(height * 0.94), int(width * 0.74) :]
    return float(np.mean(panel)) if panel.size else 0.0


def _target_change_component(first: np.ndarray, second: np.ndarray) -> dict:
    difference = _frame_difference(first, second)
    height, width = difference.shape
    allowed = np.zeros_like(difference, dtype=np.uint8)
    allowed[
        max(80, int(height * 0.12)) : int(height * 0.94),
        max(70, int(width * 0.07)) : int(width * 0.74),
    ] = 1
    allowed_values = difference[allowed > 0]
    threshold = max(
        18.0,
        float(np.percentile(allowed_values, 95.0)) if allowed_values.size else 18.0,
    )
    binary = np.uint8(difference > threshold) * 255
    binary[: max(80, int(height * 0.12)), :] = 0
    binary[int(height * 0.94) :, :] = 0
    binary[:, : max(70, int(width * 0.07))] = 0
    binary[:, int(width * 0.74) :] = 0
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
    candidates = []
    for index in range(1, count):
        x, y, component_width, component_height, area = [
            int(item) for item in stats[index]
        ]
        if (
            area < 25
            or area > 6000
            or component_width > 140
            or component_height > 140
        ):
            continue
        cx, cy = [float(item) for item in centroids[index]]
        strength = float(np.mean(difference[y : y + component_height, x : x + component_width]))
        candidates.append(
            {
                "screen_xy": [cx, cy],
                "bounding_box_xywh": [x, y, component_width, component_height],
                "area_px": area,
                "mean_difference": strength,
                "score": strength * math.sqrt(float(area)),
            }
        )
    if not candidates:
        raise RuntimeError("Could not isolate the visually selected teleport target")
    candidates.sort(key=lambda item: item["score"], reverse=True)
    selected = dict(candidates[0])
    selected["score_margin"] = (
        float(selected["score"] - candidates[1]["score"])
        if len(candidates) > 1
        else float(selected["score"])
    )
    selected["binary"] = binary
    selected["difference"] = difference
    return selected


def _map_xy(screen_xy, map_localization: Mapping) -> List[float]:
    x0, y0, _, _ = map_localization["viewport_crop_xywh"]
    point = np.float32(
        [[[float(screen_xy[0]) - x0, float(screen_xy[1]) - y0]]]
    )
    affine = np.asarray(
        map_localization["affine_viewport_to_original_map_2x3"], dtype=np.float32
    )
    return _finite_xy(cv2.transform(point, affine).reshape(2))


def _is_loading_frame(frame: np.ndarray) -> bool:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    saturation = float(np.mean(hsv[:, :, 1]))
    brightness = float(np.mean(gray))
    edges = cv2.Canny(gray, 60, 140)
    edge_fraction = float(np.count_nonzero(edges) / edges.size)
    return brightness >= 165.0 and saturation <= 35.0 and edge_fraction <= 0.10


def _circular_summary_deg(values: Sequence[float]) -> dict:
    radians = np.radians(np.asarray(values, dtype=np.float64))
    sine = float(np.mean(np.sin(radians)))
    cosine = float(np.mean(np.cos(radians)))
    mean = math.degrees(math.atan2(sine, cosine))
    concentration = min(1.0, max(1.0e-9, math.hypot(sine, cosine)))
    std = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(concentration))))
    return {"mean_deg": mean, "circular_std_deg": std}


def _arrival_observation(fix, record: Mapping) -> Optional[dict]:
    """Return strict localization or a tightly guarded offline geometric candidate."""

    source = "strict_feature_correlation"
    x = float(fix.x)
    y = float(fix.y)
    if not fix.valid:
        allowed_rejections = {
            "low-correlation",
            "ambiguous-correlation",
            "feature-correlation-disagreement",
        }
        reasons = set(fix.rejection_reasons)
        diagnostics = fix.diagnostics or {}
        feature_xy = diagnostics.get("feature_center_original_xy")
        reprojection = fix.reprojection_p95_px
        if (
            not reasons
            or not reasons.issubset(allowed_rejections)
            or fix.inlier_count < 8
            or fix.inlier_ratio < 0.75
            or reprojection is None
            or not math.isfinite(float(reprojection))
            or float(reprojection) > 2.0
            or not diagnostics.get("feature_center_covered")
            or not isinstance(feature_xy, (list, tuple))
            or len(feature_xy) != 2
        ):
            return None
        x, y = _finite_xy(feature_xy)
        source = "geometric_consensus_fallback"
    return {
        "time_s": _time_s(record),
        "frame_index": int(record["frame_index"]),
        "x": x,
        "y": y,
        "yaw_deg": float(fix.yaw_deg),
        "score": float(fix.score),
        "margin": float(fix.margin),
        "inlier_count": int(fix.inlier_count),
        "inlier_ratio": float(fix.inlier_ratio),
        "reprojection_p95_px": (
            float(fix.reprojection_p95_px)
            if fix.reprojection_p95_px is not None
            else None
        ),
        "localization_source": source,
        "strict_fix_valid": bool(fix.valid),
        "strict_rejection_reasons": list(fix.rejection_reasons),
    }


def _arrival_consensus(
    frames: _RecordedFrames,
    start_s: float,
    extractor: MinimapExtractor,
    localizer: GlobalMapLocalizer,
) -> dict:
    observations = []
    loading = []
    for frame, record in frames.iter_from(start_s, stride=10):
        time_s = _time_s(record)
        if _is_loading_frame(frame):
            loading.append({"time_s": time_s, "frame_index": record["frame_index"]})
        try:
            observation, mask = extractor.extract(frame)
            fix = localizer.localize(observation, mask)
        except (RuntimeError, ValueError):
            continue
        candidate = _arrival_observation(fix, record)
        if candidate is not None:
            observations.append(candidate)
    consensus_start = None
    for index in range(2, len(observations)):
        window = observations[index - 2 : index + 1]
        points = np.asarray([[item["x"], item["y"]] for item in window])
        center = np.median(points, axis=0)
        spread = float(np.max(np.linalg.norm(points - center, axis=1)))
        if spread <= 12.0 and window[-1]["time_s"] - window[0]["time_s"] <= 2.0:
            consensus_start = index - 2
            break
    if consensus_start is None:
        raise RuntimeError("No stable post-load destination localization consensus")
    world_ready = observations[consensus_start]
    arrival = [
        item
        for item in observations[consensus_start:]
        if item["time_s"] <= world_ready["time_s"] + 2.0
    ]
    points = np.asarray([[item["x"], item["y"]] for item in arrival], dtype=np.float64)
    center = np.median(points, axis=0)
    covariance = (
        np.cov(points.T, ddof=1)
        if len(points) > 1
        else np.zeros((2, 2), dtype=np.float64)
    )
    radii = np.linalg.norm(points - center, axis=1)
    source_counts = {}
    for item in arrival:
        source = str(item["localization_source"])
        source_counts[source] = source_counts.get(source, 0) + 1
    arrival_reprojection = [
        item["reprojection_p95_px"]
        for item in arrival
        if item["reprojection_p95_px"] is not None
    ]
    return {
        "destination_global_xy": _finite_xy(center),
        "world_ready": dict(world_ready),
        "loading_start": dict(loading[0]) if loading else None,
        "arrival_observations": arrival,
        "arrival_model": {
            "model": "first_stable_post_load_localization_consensus",
            "sample_count": len(arrival),
            "center_global_xy": _finite_xy(center),
            "covariance_global_xy_px2": covariance.tolist(),
            "radial_p95_px": float(np.percentile(radii, 95)) if len(radii) else 0.0,
            "yaw": _circular_summary_deg([item["yaw_deg"] for item in arrival]),
            "median_score": float(np.median([item["score"] for item in arrival])),
            "median_margin": float(np.median([item["margin"] for item in arrival])),
            "median_inlier_count": float(
                np.median([item["inlier_count"] for item in arrival])
            ),
            "median_inlier_ratio": float(
                np.median([item["inlier_ratio"] for item in arrival])
            ),
            "median_reprojection_p95_px": float(
                np.median(arrival_reprojection)
            ) if arrival_reprojection else None,
            "localization_source_counts": source_counts,
        },
    }


def _labeled_panel(image: np.ndarray, label: str, point=None) -> np.ndarray:
    panel = image.copy()
    if point is not None:
        cv2.circle(panel, (int(round(point[0])), int(round(point[1]))), 14, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 38), (10, 18, 26), -1)
    cv2.putText(panel, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 1, cv2.LINE_AA)
    return panel


def _render_target_evidence(
    first: np.ndarray,
    second: np.ndarray,
    target_change: Mapping,
    target_xy: Sequence[float],
) -> np.ndarray:
    size = (480, 270)
    point = target_change["screen_xy"]
    scale_x = size[0] / float(first.shape[1])
    scale_y = size[1] / float(first.shape[0])
    resized_point = (point[0] * scale_x, point[1] * scale_y)
    before = _labeled_panel(cv2.resize(first, size), "Before target selection", resized_point)
    after = _labeled_panel(cv2.resize(second, size), "Selected target panel", resized_point)
    binary = cv2.cvtColor(target_change["binary"], cv2.COLOR_GRAY2BGR)
    binary = cv2.resize(binary, size, interpolation=cv2.INTER_NEAREST)
    binary = _labeled_panel(
        binary,
        "Selection change -> map ({:.1f}, {:.1f})".format(target_xy[0], target_xy[1]),
        resized_point,
    )
    return np.hstack([before, after, binary])


def _render_path_evidence(
    mosaic: np.ndarray,
    target_xy: Sequence[float],
    destination_xy: Sequence[float],
) -> np.ndarray:
    canvas = mosaic.copy()
    target = tuple(int(round(item)) for item in target_xy)
    destination = tuple(int(round(item)) for item in destination_xy)
    cv2.arrowedLine(canvas, target, destination, (0, 220, 255), 5, cv2.LINE_AA, tipLength=0.25)
    cv2.circle(canvas, target, 15, (0, 170, 255), 4, cv2.LINE_AA)
    cv2.circle(canvas, destination, 12, (80, 255, 100), 4, cv2.LINE_AA)
    cv2.putText(canvas, "selected target", (target[0] + 18, target[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 110, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "observed arrival", (destination[0] + 18, destination[1] + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 180, 40), 2, cv2.LINE_AA)
    x0 = max(0, min(target[0], destination[0]) - 260)
    y0 = max(0, min(target[1], destination[1]) - 220)
    x1 = min(canvas.shape[1], max(target[0], destination[0]) + 360)
    y1 = min(canvas.shape[0], max(target[1], destination[1]) + 260)
    crop = canvas[y0:y1, x0:x1]
    return cv2.resize(crop, (1000, max(360, int(round(crop.shape[0] * 1000.0 / crop.shape[1])))))


def _render_timeline(frames: _RecordedFrames, phases: Sequence[Mapping]) -> np.ndarray:
    panels = []
    for phase in phases:
        frame, _ = frames.at(float(phase["start_s"]))
        panel = cv2.resize(frame, (320, 180))
        panels.append(_labeled_panel(panel, phase["state"].replace("_", " ")))
    return np.hstack(panels)


def analyze_teleport_session(
    session_path: Path,
    output_path: Path,
    *,
    game_profile_id: str,
    minimap_config: Mapping,
    minimap_calibration: Mapping,
    map_stitch: Mapping,
    map_stitch_root: Path,
    progress=None,
) -> dict:
    """Analyze one complete map-selected teleport and persist review evidence."""
    session_path = Path(session_path)
    output_path = Path(output_path)
    map_stitch_root = Path(map_stitch_root)
    reader = SessionReader(session_path)
    context = reader.manifest.get("context") or {}
    if context.get("game_profile_id") != game_profile_id:
        raise ValueError("Teleport session belongs to another game profile")
    if progress:
        progress("Reading synchronized teleport inputs and frame times")
    inputs = parse_teleport_inputs(reader.inputs)
    clicks = inputs["stationary_clicks"]
    if len(clicks) < 2:
        raise ValueError("Teleport analysis needs target and activation click evidence")
    frames = _RecordedFrames(reader)
    try:
        scored = []
        for click in clicks[:-1]:
            row = dict(click)
            row["right_panel_change"] = _click_panel_change(frames, click)
            scored.append(row)
        target_click = max(scored, key=lambda item: item["right_panel_change"])
        later_clicks = [
            dict(item) for item in clicks if item["start_s"] > target_click["start_s"]
        ]
        if not later_clicks:
            raise RuntimeError("No teleport activation click follows target selection")
        activation_click = later_clicks[-1]
        target_before, target_before_record = frames.at(
            max(0.0, target_click["start_s"] - 0.18)
        )
        target_after, target_after_record = frames.at(target_click["end_s"] + 0.50)

        if progress:
            progress("Localizing the selected target in the stitched global map")
        mosaic = cv2.imread(str(map_stitch_root / "mosaic.png"), cv2.IMREAD_COLOR)
        coverage = cv2.imread(str(map_stitch_root / "coverage.png"), cv2.IMREAD_GRAYSCALE)
        if mosaic is None:
            raise ValueError("Map stitch has no readable original mosaic")
        view_localizer = _MapViewLocalizer(mosaic, coverage)
        view_crop = map_stitch.get("viewport_crop_xywh") or [180, 65, 1000, 600]
        map_localization = view_localizer.localize(target_before, view_crop)
        target_change = _target_change_component(target_before, target_after)
        target_xy = _map_xy(target_change["screen_xy"], map_localization)

        if progress:
            progress("Finding the first stable destination localization after loading")
        localization = map_stitch.get("localization") or {}
        localization_mosaic = cv2.imread(
            str(map_stitch_root / str(localization.get("mosaic_file") or "localization_mosaic.png")),
            cv2.IMREAD_COLOR,
        )
        localization_coverage = cv2.imread(
            str(map_stitch_root / str(localization.get("coverage_file") or "localization_coverage.png")),
            cv2.IMREAD_GRAYSCALE,
        )
        if localization_mosaic is None:
            raise ValueError("Map stitch has no readable localization mosaic")
        localizer = GlobalMapLocalizer(
            localization_mosaic,
            localization_coverage,
            localization.get("localization_to_original_map_3x3"),
        )
        extractor = MinimapExtractor(
            minimap_config["crop_xywh"], dict(minimap_calibration)
        )
        try:
            arrival = _arrival_consensus(
                frames, activation_click["end_s"] + 0.15, extractor, localizer
            )
        finally:
            localizer.close()
        destination_xy = arrival["destination_global_xy"]
        loading_start = arrival.get("loading_start") or {
            "time_s": activation_click["end_s"] + 0.15,
            "frame_index": target_after_record["frame_index"],
        }
        world_ready = arrival["world_ready"]

        post_target_clicks = later_clicks
        confirmation_observed = len(post_target_clicks) >= 2
        phases = [
            {
                "state": "map_open",
                "start_s": float(
                    (inputs["map_open_events"] or [{"time_s": 0.0}])[0]["time_s"]
                ),
                "end_s": float(
                    min(
                        [item["start_s"] for item in inputs["wheel_bursts"] + inputs["drag_episodes"]]
                        or [target_click["start_s"]]
                    )
                ),
                "guard": "global_map_visible",
            },
            {
                "state": "map_navigation",
                "start_s": float(
                    min(
                        [item["start_s"] for item in inputs["wheel_bursts"] + inputs["drag_episodes"]]
                        or [target_click["start_s"]]
                    )
                ),
                "end_s": float(target_click["start_s"]),
                "guard": "target_visible_and_separable",
            },
            {
                "state": "target_selected",
                "start_s": float(target_click["start_s"]),
                "end_s": float(post_target_clicks[0]["start_s"]),
                "guard": "selected_target_panel_visible",
            },
            {
                "state": "teleport_requested",
                "start_s": float(post_target_clicks[0]["start_s"]),
                "end_s": float(loading_start["time_s"]),
                "guard": "loading_started_or_confirmation_visible",
                "confirmation_branch_observed": confirmation_observed,
            },
            {
                "state": "loading",
                "start_s": float(loading_start["time_s"]),
                "end_s": float(world_ready["time_s"]),
                "guard": "destination_localization_not_yet_ready",
            },
            {
                "state": "destination_ready",
                "start_s": float(world_ready["time_s"]),
                "end_s": float(reader.manifest.get("duration_ns", 0)) / 1.0e9,
                "guard": "stable_world_hud_and_destination_localization_consensus",
            },
        ]
        phases = [item for item in phases if item["end_s"] >= item["start_s"]]

        output_path.mkdir(parents=True, exist_ok=True)
        target_evidence = _render_target_evidence(
            target_before, target_after, target_change, target_xy
        )
        path_evidence = _render_path_evidence(mosaic, target_xy, destination_xy)
        timeline = _render_timeline(frames, phases)
        _write_image(output_path / "teleport_target_localization.png", target_evidence)
        _write_image(output_path / "teleport_target_to_arrival.png", path_evidence)
        _write_image(output_path / "teleport_phase_timeline.png", timeline)

        offset = np.asarray(destination_xy) - np.asarray(target_xy)
        map_stitch_id = str(map_stitch.get("stitch_id") or map_stitch_root.name)
        calibration_id = str(
            minimap_calibration.get("calibration_id")
            or map_stitch.get("source_minimap_calibration_id")
            or ""
        )
        coordinate_space_id = "map-stitch:{}:original-map-px".format(map_stitch_id)
        source_session_id = str(reader.manifest.get("session_id") or session_path.name)
        evidence = {
            "map_open_events": inputs["map_open_events"],
            "wheel_bursts": inputs["wheel_bursts"],
            "drag_episodes": inputs["drag_episodes"],
            "stationary_clicks": inputs["stationary_clicks"],
            "target_click": target_click,
            "activation_clicks": post_target_clicks,
            "target_click_frames": {
                "before": int(target_before_record["frame_index"]),
                "after": int(target_after_record["frame_index"]),
            },
            "loading_start_frame": int(loading_start["frame_index"]),
            "world_ready_frame": int(world_ready["frame_index"]),
            "target_selection_screen_xy": _finite_xy(target_change["screen_xy"]),
            "target_selection_component": {
                key: value
                for key, value in target_change.items()
                if key not in {"binary", "difference"}
            },
            "map_view_localization": map_localization,
            "arrival_observations": arrival["arrival_observations"],
            "review_images": [
                "teleport_phase_timeline.png",
                "teleport_target_localization.png",
                "teleport_target_to_arrival.png",
            ],
        }
        provenance = {
            "source_session_path": str(session_path.resolve()),
            "source_session_id": source_session_id,
            "map_stitch_id": map_stitch_id,
            "minimap_calibration_id": calibration_id,
            "map_coordinate_space": "original_map_px",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        }
        quality = {
            "status": "review_required",
            "target_map_inliers": int(map_localization["inlier_count"]),
            "target_map_inlier_ratio": float(map_localization["inlier_ratio"]),
            "target_map_reprojection_p95_px": float(
                map_localization["reprojection_p95_px"]
            ),
            "target_selection_score_margin": float(target_change["score_margin"]),
            "arrival_sample_count": int(arrival["arrival_model"]["sample_count"]),
            "arrival_radial_p95_px": float(arrival["arrival_model"]["radial_p95_px"]),
            "arrival_localization_median_score": float(
                arrival["arrival_model"]["median_score"]
            ),
            "arrival_localization_median_margin": float(
                arrival["arrival_model"]["median_margin"]
            ),
            "arrival_localization_source_counts": dict(
                arrival["arrival_model"]["localization_source_counts"]
            ),
            "generalization_status": "single_observed_episode",
            "human_review_required": True,
        }
        sample = make_teleport_behavior_sample(
            game_profile_id=game_profile_id,
            session_id=source_session_id,
            coordinate_space_id=coordinate_space_id,
            teleport_target_global_xy=target_xy,
            destination_global_xy=destination_xy,
            portal_id=None,
            phases=phases,
            behavior_model=TELEPORT_BEHAVIOR_MODEL,
            arrival_model=arrival["arrival_model"],
            evidence=evidence,
            provenance=provenance,
            quality=quality,
        )
        result = sample.to_dict()
        result.update(
            {
                "status": "review_required",
                "target_to_destination_offset_xy": _finite_xy(offset),
                "evidence_files": [
                    {
                        "name": "teleport_phase_timeline.png",
                        "title": "Observed teleport phase timeline",
                        "category": "phases",
                    },
                    {
                        "name": "teleport_target_localization.png",
                        "title": "Selected target and map localization",
                        "category": "target",
                    },
                    {
                        "name": "teleport_target_to_arrival.png",
                        "title": "Selected target to observed arrival",
                        "category": "spatial",
                    },
                ],
            }
        )
        if progress:
            progress("Saving reusable teleport behavior and review evidence")
        save_teleport_behavior_sample(sample, output_path / "teleport.json")
        # Preserve analyzer-level fields alongside the reusable sample contract.
        temporary = output_path / "teleport.analysis.json.tmp"
        final = output_path / "teleport.analysis.json"
        import json

        temporary.write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
        )
        temporary.replace(final)
        return result
    finally:
        frames.close()
