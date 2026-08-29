"""Learn scale-transition timing from a recorded continuous traversal."""

import math
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from replay.session_tools import decode_frames

from .minimap_transition import ModeObservation, learn_transition_model
from .session import SessionReader


class MapScaleMatcher:
    """Estimate mini-map scale and center against one preserved global mosaic."""

    def __init__(self, mosaic: np.ndarray, coverage: np.ndarray) -> None:
        self.sift = cv2.SIFT_create(
            nfeatures=12000, contrastThreshold=0.004, edgeThreshold=15
        )
        gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
        self.map_points, self.map_descriptors = self.sift.detectAndCompute(
            gray, coverage
        )
        if self.map_descriptors is None or len(self.map_points) < 8:
            raise RuntimeError("Global mosaic has too few features for scale analysis")

    def estimate(self, observation: np.ndarray, mask: np.ndarray) -> dict:
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        points, descriptors = self.sift.detectAndCompute(gray, mask)
        if descriptors is None or len(points) < 8:
            raise RuntimeError("Mini-map observation has too few features")
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
            descriptors, self.map_descriptors, k=2
        )
        matches = [
            first
            for pair in pairs
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < 0.78 * second.distance
        ]
        if len(matches) < 8:
            raise RuntimeError("Mini-map observation has fewer than 8 map matches")
        source = np.float32([points[item.queryIdx].pt for item in matches])
        target = np.float32([self.map_points[item.trainIdx].pt for item in matches])
        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=20000,
            confidence=0.999,
        )
        if matrix is None or inlier_mask is None:
            raise RuntimeError("Mini-map observation has no consistent map transform")
        accepted = inlier_mask.ravel().astype(bool)
        inlier_count = int(np.count_nonzero(accepted))
        inlier_ratio = float(np.mean(accepted))
        predicted = cv2.transform(source.reshape((-1, 1, 2)), matrix).reshape(
            (-1, 2)
        )
        errors = np.linalg.norm(predicted - target, axis=1)
        reprojection_p95 = (
            float(np.percentile(errors[accepted], 95))
            if inlier_count
            else float("inf")
        )
        if inlier_count < 8 or inlier_ratio < 0.30 or reprojection_p95 > 6.0:
            raise RuntimeError("Mini-map observation has weak map geometry")
        scale = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
        center = cv2.transform(
            np.float32(
                [[[observation.shape[1] / 2.0, observation.shape[0] / 2.0]]]
            ),
            matrix,
        ).reshape(2)
        return {
            "map_pixels_per_minimap_pixel": float(scale),
            "canonical_xy": (float(center[0]), float(center[1])),
            "ratio_match_count": len(matches),
            "inlier_count": inlier_count,
            "inlier_ratio": inlier_ratio,
            "reprojection_p95_px": reprojection_p95,
        }


def _sample_records(records, maximum: int):
    if len(records) <= maximum:
        return list(records)
    indexes = np.linspace(0, len(records) - 1, maximum).round().astype(int)
    return [records[int(index)] for index in np.unique(indexes)]


def _likelihoods(scale: float, mode_scales: Mapping[str, float], sigma: float):
    raw = {
        mode_id: math.exp(
            -0.5 * (math.log(scale / float(reference_scale)) / sigma) ** 2
        )
        for mode_id, reference_scale in mode_scales.items()
    }
    total = sum(raw.values()) or 1.0
    return {mode_id: value / total for mode_id, value in raw.items()}


def _render_timeline(samples, model, mode_scales, output: Path) -> None:
    width, height = 1100, 500
    left, right, top, bottom = 90, 30, 45, 70
    canvas = np.full((height, width, 3), 246, np.uint8)
    plot_w, plot_h = width - left - right, height - top - bottom
    scales = [row["scale"] for row in samples] + list(mode_scales.values())
    low, high = min(scales) * 0.90, max(scales) * 1.10
    first_time = samples[0]["session_time_ns"]
    last_time = samples[-1]["session_time_ns"]

    def point(row):
        fraction = (row["session_time_ns"] - first_time) / max(
            1, last_time - first_time
        )
        y_fraction = (math.log(row["scale"]) - math.log(low)) / (
            math.log(high) - math.log(low)
        )
        return (
            int(round(left + fraction * plot_w)),
            int(round(top + (1.0 - y_fraction) * plot_h)),
        )

    cv2.rectangle(canvas, (left, top), (left + plot_w, top + plot_h), (180, 180, 180), 1)
    colors = ((45, 145, 35), (220, 105, 20))
    for (mode_id, scale), color in zip(mode_scales.items(), colors):
        y = point({"session_time_ns": first_time, "scale": scale})[1]
        cv2.line(canvas, (left, y), (left + plot_w, y), color, 2)
        cv2.putText(
            canvas,
            "{} {:.3f} map px / mini-map px".format(mode_id, scale),
            (left + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    for first, second in zip(samples, samples[1:]):
        cv2.line(canvas, point(first), point(second), (55, 75, 180), 2, cv2.LINE_AA)
    for row in samples:
        cv2.circle(canvas, point(row), 4, (55, 75, 180), -1, cv2.LINE_AA)
    center_frame = model["transition"]["center_frame_index"]
    center = min(samples, key=lambda row: abs(row["frame_index"] - center_frame))
    center_x = point(center)[0]
    cv2.line(canvas, (center_x, top), (center_x, top + plot_h), (30, 30, 220), 2)
    cv2.putText(
        canvas,
        "learned switch frame {}".format(center_frame),
        (max(left, center_x - 110), top + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (30, 30, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Recorded mini-map scale transition",
        (left, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (35, 35, 35),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "session time (s)",
        (left + plot_w // 2 - 55, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "{:.1f}".format((last_time - first_time) / 1.0e9),
        (left + plot_w - 25, top + plot_h + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError("Could not write mini-map transition timeline")


def analyze_transition_session(
    session_path: Path,
    extractor,
    mosaic: np.ndarray,
    coverage: np.ndarray,
    mode_scales: Mapping[str, float],
    output_path: Path,
    *,
    source_mode_id: str,
    target_mode_id: str,
    maximum_samples: int = 48,
) -> dict:
    """Fit a reusable temporal and spatial scale-switch model."""

    if set(mode_scales) != {source_mode_id, target_mode_id}:
        raise ValueError("Transition analysis needs exactly the source and target scales")
    source_scale = float(mode_scales[source_mode_id])
    target_scale = float(mode_scales[target_mode_id])
    separation = abs(math.log(source_scale / target_scale))
    if separation < math.log(1.10):
        raise RuntimeError("Recorded endpoints do not show distinct mini-map scales")
    sigma = max(0.04, separation / 6.0)
    reader = SessionReader(Path(session_path))
    records = reader.frames_by_stream.get("main") or []
    selected = _sample_records(records, max(12, int(maximum_samples)))
    images = decode_frames(reader, "main", selected)
    matcher = MapScaleMatcher(mosaic, coverage)
    observations = []
    samples = []
    failures = []
    for record, frame in zip(selected, images):
        observation, mask = extractor.extract(frame)
        try:
            estimate = matcher.estimate(observation, mask)
        except RuntimeError as exc:
            failures.append(
                {"frame_index": int(record["frame_index"]), "reason": str(exc)}
            )
            continue
        likelihoods = _likelihoods(
            estimate["map_pixels_per_minimap_pixel"], mode_scales, sigma
        )
        row = ModeObservation(
            frame_index=int(record["frame_index"]),
            session_time_ns=int(record["session_time_ns"]),
            likelihoods=likelihoods,
            canonical_xy=estimate["canonical_xy"],
        )
        observations.append(row)
        samples.append(
            {
                "frame_index": row.frame_index,
                "session_time_ns": row.session_time_ns,
                "scale": estimate["map_pixels_per_minimap_pixel"],
                "canonical_xy": list(estimate["canonical_xy"]),
                "likelihoods": dict(likelihoods),
                "inlier_count": estimate["inlier_count"],
                "inlier_ratio": estimate["inlier_ratio"],
                "reprojection_p95_px": estimate["reprojection_p95_px"],
            }
        )
    model = learn_transition_model(
        observations,
        source_mode_id,
        target_mode_id,
        stable_count=3,
        stable_margin=0.35,
    )
    model["scale_model"] = {
        "map_pixels_per_minimap_pixel": {
            mode_id: float(value) for mode_id, value in mode_scales.items()
        },
        "scale_ratio": max(source_scale, target_scale)
        / min(source_scale, target_scale),
        "log_likelihood_sigma": sigma,
    }
    model["analysis"] = {
        "requested_sample_count": len(selected),
        "accepted_sample_count": len(samples),
        "failed_sample_count": len(failures),
        "accepted_fraction": len(samples) / max(1, len(selected)),
        "failures": failures,
        "samples": samples,
    }
    model["runtime"] = {
        "confirmation_count": 2,
        "minimum_mode_margin": 0.20,
        "switch_position_policy": "hold_continuous_pose",
        "reset_local_reference": True,
    }
    evidence_file = "transition_scale_timeline.png"
    _render_timeline(samples, model, mode_scales, Path(output_path) / evidence_file)
    model["evidence_file"] = evidence_file
    return model
