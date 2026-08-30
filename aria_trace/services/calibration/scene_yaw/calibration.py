"""Scene-relative yaw calibration from a stationary horizontal full turn."""

import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from aria_trace.services.vision import KltAngularYawEstimator, camera_matrix

from aria_trace.services.calibration.cursor.pose import timing_summary_ms
from acquisition.session import SessionReader


DEFAULT_CONFIG = {
    "sample_fps": 15.0,
    "focal_ratio_min": 0.55,
    "focal_ratio_max": 1.45,
    "focal_ratio_steps": 10,
    "min_tracks": 20,
    "max_corners": 1000,
    "use_essential_gate": True,
    "excluded_rects": (),
    "minimum_closure_fraction": 0.40,
}


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


def _merged_config(config=None) -> dict:
    value = dict(DEFAULT_CONFIG)
    value.update(config or {})
    value["sample_fps"] = float(value["sample_fps"])
    value["focal_ratio_steps"] = int(value["focal_ratio_steps"])
    value["min_tracks"] = int(value["min_tracks"])
    value["max_corners"] = int(value["max_corners"])
    value["excluded_rects"] = tuple(tuple(row) for row in value["excluded_rects"])
    if value["sample_fps"] <= 0:
        raise ValueError("Scene-yaw sample rate must be positive")
    if value["focal_ratio_steps"] < 2:
        raise ValueError("Scene-yaw focal search needs at least two steps")
    if value["focal_ratio_min"] <= 0 or value["focal_ratio_max"] <= value["focal_ratio_min"]:
        raise ValueError("Invalid scene-yaw focal-ratio range")
    return value


def _mask(shape, excluded_rects) -> np.ndarray:
    height, width = shape[:2]
    mask = np.full((height, width), 255, np.uint8)
    for x0, y0, x1, y1 in excluded_rects:
        cv2.rectangle(
            mask,
            (int(x0 * width), int(y0 * height)),
            (int(x1 * width), int(y1 * height)),
            0,
            -1,
        )
    return mask


def _loop_match(first: np.ndarray, candidate: np.ndarray, excluded_rects=()):
    gray0 = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
    feature_mask = _mask(gray0.shape, excluded_rects)
    small0 = cv2.resize(gray0, (192, 108), interpolation=cv2.INTER_AREA).astype(np.float32)
    small1 = cv2.resize(gray1, (192, 108), interpolation=cv2.INTER_AREA).astype(np.float32)
    small_mask = cv2.resize(feature_mask, (192, 108), interpolation=cv2.INTER_NEAREST) > 0
    values0, values1 = small0[small_mask], small1[small_mask]
    values0 = (values0 - values0.mean()) / (values0.std() + 1.0e-6)
    values1 = (values1 - values1.mean()) / (values1.std() + 1.0e-6)
    correlation = float(np.mean(values0 * values1))

    orb = cv2.ORB_create(nfeatures=1600, fastThreshold=10)
    points0, descriptors0 = orb.detectAndCompute(gray0, feature_mask)
    points1, descriptors1 = orb.detectAndCompute(gray1, feature_mask)
    good = []
    inlier_mask = None
    if descriptors0 is not None and descriptors1 is not None:
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors0, descriptors1, k=2)
        good = [first_match for first_match, second_match in pairs if first_match.distance < 0.75 * second_match.distance]
        if len(good) >= 8:
            source = np.float32([points0[item.queryIdx].pt for item in good])
            target = np.float32([points1[item.trainIdx].pt for item in good])
            _, inlier_mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    inliers = int(np.count_nonzero(inlier_mask)) if inlier_mask is not None else 0
    ratio = float(inliers / max(len(good), 1))
    score = correlation + min(inliers, 100) / 100.0 + ratio * 0.25
    return {
        "correlation": correlation,
        "match_count": len(good),
        "inlier_count": inliers,
        "inlier_ratio": ratio,
        "score": float(score),
        "keypoints_first": points0,
        "keypoints_candidate": points1,
        "matches": good,
        "inlier_mask": inlier_mask,
    }


def _estimate(frames, focal_ratio: float, config: dict):
    height, width = frames[0].shape[:2]
    estimator = KltAngularYawEstimator(
        camera_matrix(width, height, focal_ratio),
        max_corners=config["max_corners"],
        min_tracks=config["min_tracks"],
        use_essential_gate=bool(config["use_essential_gate"]),
        excluded_rects=config["excluded_rects"],
    )
    rows = []
    durations_ns = []
    for index, frame in enumerate(frames):
        started_ns = time.perf_counter_ns()
        estimate = estimator.update(frame)
        durations_ns.append(time.perf_counter_ns() - started_ns)
        rows.append(
            {
                "frame_index": index,
                "delta_yaw_deg": float(estimate.delta_deg),
                "relative_yaw_deg": float(estimate.total_deg),
                "tracks": int(estimate.tracks),
                "inliers": int(estimate.inliers),
                "confidence": float(estimate.confidence),
                "status": estimate.status,
                "elapsed_ms": float(estimate.elapsed_ms),
            }
        )
    return rows, durations_ns


def _measure_tracks(frames, config: dict):
    """Measure KLT correspondences once; camera intrinsics are not used here."""
    height, width = frames[0].shape[:2]
    estimator = KltAngularYawEstimator(
        camera_matrix(width, height, 0.9),
        max_corners=config["max_corners"],
        min_tracks=config["min_tracks"],
        use_essential_gate=bool(config["use_essential_gate"]),
        excluded_rects=config["excluded_rects"],
    )
    return [estimator.measure(frame) for frame in frames]


def _estimate_measurements(measurements, shape, focal_ratio: float, config: dict):
    """Evaluate reusable KLT correspondences for one candidate camera scale."""
    height, width = shape[:2]
    estimator = KltAngularYawEstimator(
        camera_matrix(width, height, focal_ratio),
        max_corners=config["max_corners"],
        min_tracks=config["min_tracks"],
        use_essential_gate=bool(config["use_essential_gate"]),
        excluded_rects=config["excluded_rects"],
    )
    rows = []
    durations_ns = []
    for index, measurement in enumerate(measurements):
        estimate = estimator.update_measurement(measurement)
        durations_ns.append(int(round(estimate.elapsed_ms * 1.0e6)))
        rows.append(
            {
                "frame_index": index,
                "delta_yaw_deg": float(estimate.delta_deg),
                "relative_yaw_deg": float(estimate.total_deg),
                "tracks": int(estimate.tracks),
                "inliers": int(estimate.inliers),
                "confidence": float(estimate.confidence),
                "status": estimate.status,
                "elapsed_ms": float(estimate.elapsed_ms),
            }
        )
    return rows, durations_ns


def _curve_image(rows, closure_index: int, target_yaw: float) -> np.ndarray:
    canvas = np.full((500, 1000, 3), 18, np.uint8)
    values = np.asarray([row["relative_yaw_deg"] for row in rows], np.float64)
    low = min(float(values.min()), target_yaw, 0.0)
    high = max(float(values.max()), target_yaw, 0.0)
    span = max(high - low, 1.0)

    def point(index, value):
        x = 55 + int(index / max(len(rows) - 1, 1) * 910)
        y = 450 - int((value - low) / span * 400)
        return x, y

    for index in range(1, len(rows)):
        color = (80, 230, 120) if rows[index]["status"] == "ok" else (70, 70, 230)
        cv2.line(canvas, point(index - 1, values[index - 1]), point(index, values[index]), color, 2, cv2.LINE_AA)
    cv2.line(canvas, point(0, target_yaw), point(len(rows) - 1, target_yaw), (220, 180, 70), 1, cv2.LINE_AA)
    closure_x = point(closure_index, values[closure_index])[0]
    cv2.line(canvas, (closure_x, 35), (closure_x, 460), (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Scene-relative yaw / detected full-turn closure", (24, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, "target {:+.1f} deg".format(target_yaw), (65, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 180, 70), 1, cv2.LINE_AA)
    return canvas


def _quality_image(rows) -> np.ndarray:
    canvas = np.full((360, 1000, 3), 18, np.uint8)
    for index in range(1, len(rows)):
        x0 = 45 + int((index - 1) / max(len(rows) - 1, 1) * 920)
        x1 = 45 + int(index / max(len(rows) - 1, 1) * 920)
        confidence0 = float(rows[index - 1]["confidence"])
        confidence1 = float(rows[index]["confidence"])
        cv2.line(canvas, (x0, 320 - int(confidence0 * 280)), (x1, 320 - int(confidence1 * 280)), (80, 220, 230), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Per-frame tracking confidence", (24, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 1, cv2.LINE_AA)
    return canvas


def calibrate_scene_yaw_frames(frames, output_path: Path, config=None, provenance=None, progress=None) -> dict:
    """Fit relative scene yaw against the observed one-revolution loop closure."""
    if len(frames) < 20:
        raise ValueError("Scene-yaw calibration needs at least 20 frames")
    config = _merged_config(config)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if progress:
        progress("Tracking scene features once for angular-scale fitting")
    measurements = _measure_tracks(frames, config)
    coarse_rows, _ = _estimate_measurements(
        measurements, frames[0].shape, 0.9, config
    )
    start = max(2, int(math.ceil(len(frames) * config["minimum_closure_fraction"])))
    if progress:
        progress("Searching for the first full-turn visual loop closure")
    candidates = []
    for index in range(start, len(frames)):
        match = _loop_match(frames[0], frames[index], config["excluded_rects"])
        match["frame_index"] = index
        match["coarse_yaw_deg"] = coarse_rows[index]["relative_yaw_deg"]
        candidates.append(match)
    plausible = [item for item in candidates if abs(item["coarse_yaw_deg"]) >= 220.0]
    closure = max(plausible or candidates, key=lambda item: item["score"])
    closure_index = int(closure["frame_index"])
    coarse_sign = -1.0 if coarse_rows[closure_index]["relative_yaw_deg"] < 0 else 1.0
    target_yaw = coarse_sign * 360.0

    ratios = np.linspace(
        float(config["focal_ratio_min"]),
        float(config["focal_ratio_max"]),
        int(config["focal_ratio_steps"]),
    )
    searches = []
    candidate_count = len(ratios) + 9
    candidate_index = 0
    closure_measurements = measurements[: closure_index + 1]
    fit_cache = {}

    def evaluate_ratio(ratio):
        key = round(float(ratio), 12)
        if key not in fit_cache:
            fit_cache[key] = _estimate_measurements(
                closure_measurements, frames[0].shape, float(ratio), config
            )
        return fit_cache[key]

    for ratio in ratios:
        candidate_index += 1
        if progress:
            progress(
                "Fitting camera angular scale: candidate {} of {}".format(
                    candidate_index, candidate_count
                )
            )
        rows, _ = evaluate_ratio(ratio)
        final_yaw = float(rows[-1]["relative_yaw_deg"])
        searches.append((abs(final_yaw - target_yaw), float(ratio), final_yaw))
    _, best_ratio, _ = min(searches)
    step = float(ratios[1] - ratios[0])
    refinements = np.linspace(max(0.1, best_ratio - step), best_ratio + step, 9)
    for ratio in refinements:
        candidate_index += 1
        if progress:
            progress(
                "Fitting camera angular scale: candidate {} of {}".format(
                    candidate_index, candidate_count
                )
            )
        rows, _ = evaluate_ratio(ratio)
        final_yaw = float(rows[-1]["relative_yaw_deg"])
        searches.append((abs(final_yaw - target_yaw), float(ratio), final_yaw))
    _, best_ratio, _ = min(searches)

    if progress:
        progress("Applying the fitted angular scale to cached scene tracks")
    rows, durations_ns = _estimate_measurements(
        measurements, frames[0].shape, best_ratio, config
    )
    closure_yaw = float(rows[closure_index]["relative_yaw_deg"])
    closure_error = abs(closure_yaw - target_yaw)
    valid = [row for row in rows[1:] if row["status"] == "ok"]
    deltas = np.asarray([row["delta_yaw_deg"] for row in valid], np.float64)
    dominant = -1.0 if float(np.median(deltas)) < 0 else 1.0
    reversals = int(np.count_nonzero(deltas * dominant < -0.03)) if len(deltas) else 0
    reversal_fraction = float(reversals / max(len(deltas), 1))
    valid_rate = float(len(valid) / max(len(rows) - 1, 1))
    accepted = bool(
        (closure["correlation"] >= 0.55 or closure["inlier_count"] >= 20)
        and closure_error <= 12.0
        and valid_rate >= 0.60
        and reversal_fraction <= 0.20
    )

    if progress:
        progress("Writing loop-closure, yaw-curve, and tracking-quality evidence")
    curve = _curve_image(rows, closure_index, target_yaw)
    quality = _quality_image(rows)
    blend = np.zeros_like(frames[0])
    blend[:, :, 2] = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    blend[:, :, 1] = cv2.cvtColor(frames[closure_index], cv2.COLOR_BGR2GRAY)
    inlier_flags = (
        closure["inlier_mask"].ravel().astype(np.uint8).tolist()
        if closure["inlier_mask"] is not None
        else None
    )
    matches_image = cv2.drawMatches(
        frames[0],
        closure["keypoints_first"],
        frames[closure_index],
        closure["keypoints_candidate"],
        closure["matches"][:100],
        None,
        matchColor=(80, 230, 120),
        singlePointColor=(80, 80, 230),
        matchesMask=(inlier_flags[:100] if inlier_flags is not None else None),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    evidence = [
        {"name": "scene_yaw_curve.png", "title": "Accumulated relative yaw and full-turn closure", "category": "yaw"},
        {"name": "scene_yaw_confidence.png", "title": "Per-frame scene tracking confidence", "category": "quality"},
        {"name": "scene_yaw_loop_closure.png", "title": "First frame vs detected full-turn frame", "category": "closure"},
        {"name": "scene_yaw_loop_matches.png", "title": "Loop-closure feature inliers", "category": "closure"},
    ]
    for name, image in (
        ("scene_yaw_curve.png", curve),
        ("scene_yaw_confidence.png", quality),
        ("scene_yaw_loop_closure.png", blend),
        ("scene_yaw_loop_matches.png", matches_image),
    ):
        if not cv2.imwrite(str(output_path / name), image):
            raise RuntimeError("Could not write scene-yaw evidence: {}".format(name))
    with (output_path / "scene_yaw_estimates.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    result = {
        "schema_version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "review_required" if accepted else "failed",
        "method": "klt_robust_horizontal_angular_flow_with_visual_360_loop_closure",
        "config": {
            "sample_fps": config["sample_fps"],
            "min_tracks": config["min_tracks"],
            "max_corners": config["max_corners"],
            "use_essential_gate": bool(config["use_essential_gate"]),
            "excluded_rects": [list(row) for row in config["excluded_rects"]],
        },
        "provenance": provenance or {},
        "frame_count": len(frames),
        "focal_ratio": float(best_ratio),
        "camera_matrix": camera_matrix(frames[0].shape[1], frames[0].shape[0], best_ratio).tolist(),
        "direction": "clockwise_screen" if dominant > 0 else "counterclockwise_screen",
        "closure_frame_index": closure_index,
        "closure_target_yaw_deg": target_yaw,
        "closure_estimated_yaw_deg": closure_yaw,
        "closure_error_deg": float(closure_error),
        "closure_correlation": float(closure["correlation"]),
        "closure_feature_matches": int(closure["match_count"]),
        "closure_feature_inliers": int(closure["inlier_count"]),
        "closure_feature_inlier_ratio": float(closure["inlier_ratio"]),
        "total_observed_yaw_deg": float(rows[-1]["relative_yaw_deg"]),
        "valid_frame_rate": valid_rate,
        "reversal_fraction": reversal_fraction,
        "median_confidence": float(np.median([row["confidence"] for row in valid])) if valid else 0.0,
        "estimator_benchmark": timing_summary_ms(
            durations_ns,
            "observed per-frame calibrated scene-yaw estimator wall time",
        ),
        "focal_search": [
            {"focal_ratio": ratio, "closure_yaw_deg": yaw, "absolute_error_deg": error}
            for error, ratio, yaw in sorted(searches, key=lambda item: item[1])
        ],
        "evidence": evidence,
    }
    _atomic_json(output_path / "scene_yaw_calibration.json", result)
    return result


def calibrate_scene_yaw_session(session_path: Path, output_path: Path, config=None, progress=None) -> dict:
    config = _merged_config(config)
    reader = SessionReader(session_path)
    records = reader.frames_by_stream.get("main", [])
    capture = cv2.VideoCapture(str(reader.video_path("main")))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    stride = max(1, int(round(source_fps / config["sample_fps"])))
    frames = []
    source_indices = []
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                frames.append(frame)
                source_indices.append(index)
            index += 1
    finally:
        capture.release()
    if progress:
        progress("Decoded {} scene frames at approximately {:.1f} Hz".format(len(frames), source_fps / stride))
    return calibrate_scene_yaw_frames(
        frames,
        output_path,
        config=config,
        provenance={
            "source_session_path": str(Path(session_path).resolve()),
            "source_session_id": reader.manifest.get("session_id"),
            "source_frame_count": len(records),
            "sampled_source_frame_indices": source_indices,
            "source_fps": source_fps,
            "sample_stride": stride,
        },
        progress=progress,
    )
