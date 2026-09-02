"""Compare cursor-angle estimators against independent forward-motion evidence.

The production estimator is not modified.  Candidate methods consume only one
frame and the existing cursor calibration.  Input telemetry and mini-map motion
are read only by the evaluator after inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from aria_trace.adapters.filesystem.session import SessionReader
from aria_trace.services.calibration.cursor.pose import (
    CursorPoseEstimator,
    circular_difference_degrees,
)
from aria_trace.services.calibration.cursor.shape import (
    edge_distance_transform,
    polygon_edge,
)
from aria_trace.services.calibration.minimap.verification import (
    estimate_masked_shift,
)


DEFAULT_SESSIONS = ("run_03", "run_04", "run_12", "run_13")
DEVELOPMENT_SESSIONS = frozenset(("run_03", "run_04"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*arguments):
    try:
        result = subprocess.run(
            ("git", "-C", str(Path(__file__).resolve().parents[1]), *arguments),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _percentile(values, percentile):
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _timing(values_ns):
    values_ms = np.asarray(values_ns, dtype=np.float64) / 1.0e6
    return {
        "sample_count": int(values_ms.size),
        "median_ms": float(np.median(values_ms)),
        "p95_ms": _percentile(values_ms, 95),
        "mean_ms": float(np.mean(values_ms)),
        "max_ms": float(np.max(values_ms)),
    }


def _parabolic_peak(response):
    response = np.asarray(response, dtype=np.float64)
    index = int(np.argmax(response))
    left = float(response[(index - 1) % len(response)])
    center = float(response[index])
    right = float(response[(index + 1) % len(response)])
    denominator = left - 2.0 * center + right
    offset = 0.0
    if abs(denominator) > 1.0e-12:
        offset = float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))
    return (index + offset) % len(response)


def _local_offsets(response, radius=16):
    response = np.asarray(response, dtype=np.float64)
    peak = int(np.argmax(response))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    values = response[(peak + offsets.astype(np.int64)) % len(response)]
    return peak, offsets, values


def _weighted_centroid_peak(response):
    peak, offsets, values = _local_offsets(response)
    weights = np.maximum(values - float(np.min(values)), 0.0) ** 2
    if float(weights.sum()) <= 1.0e-12:
        return float(peak)
    return float((peak + np.sum(offsets * weights) / np.sum(weights)) % 360.0)


def _softargmax_peak(response):
    peak, offsets, values = _local_offsets(response)
    scale = max(float(np.max(values) - np.median(values)) * 0.15, 1.0e-6)
    weights = np.exp(np.clip((values - np.max(values)) / scale, -60.0, 0.0))
    return float((peak + np.sum(offsets * weights) / np.sum(weights)) % 360.0)


def _von_mises_moment_peak(response):
    """Weighted circular mean, the mean-direction MLE of a von Mises lobe."""
    peak, offsets, values = _local_offsets(response, radius=70)
    weights = np.maximum(values - _percentile(values, 20), 0.0)
    if float(weights.sum()) <= 1.0e-12:
        return float(peak)
    angles = np.radians(peak + offsets)
    vector = np.sum(weights * np.exp(1j * angles))
    return float(np.degrees(np.angle(vector)) % 360.0)


def _phase_peak_1d(template, observed):
    template_fft = np.fft.fft(np.asarray(template, dtype=np.float32))
    observed_fft = np.fft.fft(np.asarray(observed, dtype=np.float32))
    cross = observed_fft * np.conj(template_fft)
    cross /= np.maximum(np.abs(cross), 1.0e-9)
    response = np.fft.ifft(cross).real
    return _parabolic_peak(response), response.astype(np.float32)


def _ncc_response(template, observed):
    response = np.fft.ifft(
        np.fft.fft(observed, axis=0)
        * np.conj(np.fft.fft(template, axis=0)),
        axis=0,
    ).real.sum(axis=1)
    energy = math.sqrt(float(np.sum(template ** 2) * np.sum(observed ** 2)))
    return (response / (energy + 1.0e-9)).astype(np.float32)


def _phase_response_x(template, observed):
    """Circular phase correlation on the angular axis with radius as channels."""
    template_fft = np.fft.fft(template, axis=0)
    observed_fft = np.fft.fft(observed, axis=0)
    # Combine the radial channels before phase normalization.  This preserves
    # their joint geometry while removing magnitude from each angular frequency.
    cross = np.sum(observed_fft * np.conj(template_fft), axis=1)
    cross /= np.maximum(np.abs(cross), 1.0e-9)
    return np.fft.ifft(cross).real.astype(np.float32)


class CandidateEngine:
    def __init__(self, calibration_path):
        self.estimator = CursorPoseEstimator(
            calibration_path,
            gaussian_fit_method="vectorized_grid",
            validation_policy="full",
        )
        model = np.load(self.estimator.root / "model.npz")
        self.average_template = model["cursor_probability"].astype(np.float32)
        self.symmetric_template = model[
            "cursor_symmetric_probability"
        ].astype(np.float32)
        self.average_polar = self._polar(self.average_template)
        self.symmetric_polar = self._polar(self.symmetric_template)
        self.phase_window = cv2.createHanningWindow(
            (self.average_polar.shape[0], self.average_polar.shape[1]),
            cv2.CV_32F,
        )
        self.symmetry_axis_deg = float(self.estimator.symmetry_axis_deg)

    def _polar(self, patch):
        return cv2.remap(
            np.asarray(patch, dtype=np.float32),
            self.estimator.x_map,
            self.estimator.y_map,
            cv2.INTER_LINEAR,
        )

    def extract(self, frame):
        started = time.perf_counter_ns()
        crop = self.estimator._crop(frame)
        mask, centroid, area = self.estimator._cursor_mask(crop)
        if mask is None:
            return None, None, area, time.perf_counter_ns() - started
        patch = cv2.getRectSubPix(
            mask,
            (self.estimator.patch_size, self.estimator.patch_size),
            tuple(self.estimator.pivot),
        )
        return patch, centroid, area, time.perf_counter_ns() - started

    def polygon_response(self, patch):
        started = time.perf_counter_ns()
        observed = patch >= 0.5
        observed_edge = polygon_edge(observed)
        if not np.any(observed_edge):
            raise RuntimeError("Observed cursor polygon has no edge")
        observed_distance = edge_distance_transform(observed_edge)
        chamfer = self.estimator._symmetric_chamfer_curve(
            observed_edge,
            observed_distance,
            self.estimator.polygon_edges,
            self.estimator.polygon_distance_transforms,
        )
        scale = max(float(np.percentile(chamfer, 20)), 0.75)
        response = np.exp(-0.5 * (chamfer / scale) ** 2).astype(np.float32)
        return response, time.perf_counter_ns() - started

    def polar_observation(self, patch):
        started = time.perf_counter_ns()
        value = self._polar(patch)
        return value, time.perf_counter_ns() - started

    def phase_2d(self, template, observed):
        started = time.perf_counter_ns()
        # Transpose so angular displacement is OpenCV's X displacement.
        shift, response = cv2.phaseCorrelate(
            template.T.astype(np.float32, copy=False),
            observed.T.astype(np.float32, copy=False),
            self.phase_window,
        )
        return float(shift[0] % 360.0), float(response), time.perf_counter_ns() - started

    def centroid_angle(self, centroid):
        vector = np.asarray(centroid, dtype=np.float64) - self.estimator.pivot
        angle = math.degrees(math.atan2(float(vector[1]), float(vector[0])))
        relative = circular_difference_degrees(
            angle, self.estimator.canonical_centroid_angle_deg
        )
        return float((self.symmetry_axis_deg + relative) % 360.0)


def _input_audit(session_path):
    kinds = defaultdict(int)
    keys = set()
    mouse_dx = 0
    mouse_dy = 0
    path = Path(session_path) / "inputs.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        kinds[row.get("kind")] += 1
        payload = row.get("payload") or {}
        if row.get("kind") == "pc_raw_keyboard":
            keys.add(str(payload.get("key_name")))
        if row.get("kind") == "pc_raw_mouse":
            mouse_dx += int(payload.get("delta_x") or 0)
            mouse_dy += int(payload.get("delta_y") or 0)
    return {
        "event_count": int(sum(kinds.values())),
        "event_kinds": dict(kinds),
        "keys": sorted(keys),
        "mouse_delta_xy": [mouse_dx, mouse_dy],
        "no_camera_mouse_motion": not kinds.get("pc_raw_mouse"),
        "forward_control_present": "W" in keys,
    }


def _decode_session(session_path):
    reader = SessionReader(session_path)
    capture = cv2.VideoCapture(str(reader.video_path("main")))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    records = reader.frames_by_stream.get("main", [])
    count = min(len(frames), len(records))
    if count < 20:
        raise RuntimeError("Forward session has too few decoded frames")
    return frames[:count], records[:count]


def _motion_reference(engine, frames, first_index, last_index):
    x, y, width, height = engine.estimator.crop_xywh
    first = frames[first_index][y : y + height, x : x + width]
    last = frames[last_index][y : y + height, x : x + width]
    boundary = engine.estimator.calibration["outer_boundary"]
    yy, xx = np.ogrid[:height, :width]
    mask = (
        (xx - float(boundary["center_x"])) ** 2
        + (yy - float(boundary["center_y"])) ** 2
        <= max(float(boundary["radius"]) - 5.0, 1.0) ** 2
    ).astype(np.uint8) * 255
    cursor_hole = (
        (xx - float(engine.estimator.pivot[0])) ** 2
        + (yy - float(engine.estimator.pivot[1])) ** 2
        <= 15.0 ** 2
    )
    mask[cursor_hole] = 0
    shift, response = estimate_masked_shift(first, last, mask)
    shift_angle = math.degrees(math.atan2(shift[1], shift[0])) % 360.0
    travel_angle = (shift_angle + 180.0) % 360.0
    return {
        "first_frame_index": int(first_index),
        "last_frame_index": int(last_index),
        "map_content_shift_xy_px": [float(shift[0]), float(shift[1])],
        "map_content_shift_magnitude_px": float(math.hypot(*shift)),
        "phase_correlation_response": float(response),
        "travel_angle_screen_deg": float(travel_angle),
    }


def _reference(engine, frames):
    count = len(frames)
    margin = max(1, min(8, count // 8))
    return _motion_reference(engine, frames, margin, count - margin - 1)


def _method_row(method, group, session, frame_index, predicted, reference, timing_ns):
    error = abs(float(circular_difference_degrees(predicted, reference)))
    return {
        "method": method,
        "group": group,
        "session": session,
        "split": "development" if session in DEVELOPMENT_SESSIONS else "holdout",
        "frame_index": int(frame_index),
        "predicted_angle_deg": float(predicted),
        "reference_angle_deg": float(reference),
        "absolute_error_deg": error,
        "latency_ns": int(timing_ns),
    }


def _benchmark_frame(engine, frame, frame_index, session, reference, rows):
    patch, centroid, _area, extraction_ns = engine.extract(frame)
    if patch is None:
        return
    polygon, polygon_ns = engine.polygon_response(patch)
    polar, polar_ns = engine.polar_observation(patch)

    refiners = {
        "polygon_argmax": lambda: float(np.argmax(polygon)),
        "polygon_parabolic": lambda: _parabolic_peak(polygon),
        "polygon_weighted_centroid": lambda: _weighted_centroid_peak(polygon),
        "polygon_softargmax": lambda: _softargmax_peak(polygon),
        "polygon_von_mises_moment": lambda: _von_mises_moment_peak(polygon),
        "polygon_gaussian_analytic": lambda: engine.estimator._fit_circular_gaussian_lm(polygon)["center_deg"],
        "polygon_gaussian_fast_grid": lambda: engine.estimator._fit_circular_gaussian_fast(polygon)["center_deg"],
        "polygon_gaussian_cascade": lambda: engine.estimator._fit_circular_gaussian_cascade(polygon)["center_deg"],
        "polygon_gaussian_vectorized": lambda: engine.estimator._fit_circular_gaussian_vectorized(polygon)["center_deg"],
    }
    for name, operation in refiners.items():
        started = time.perf_counter_ns()
        relative = float(operation())
        refine_ns = time.perf_counter_ns() - started
        rows.append(
            _method_row(
                name,
                "polygon_peak_refinement",
                session,
                frame_index,
                (engine.symmetry_axis_deg + relative) % 360.0,
                reference,
                extraction_ns + polygon_ns + refine_ns,
            )
        )

    for name, template in (
        ("average_pixel_phase2d", engine.average_polar),
        ("symmetric_average_pixel_phase2d", engine.symmetric_polar),
    ):
        relative, _response, match_ns = engine.phase_2d(template, polar)
        rows.append(
            _method_row(
                name,
                "averaged_pixel_phase",
                session,
                frame_index,
                (engine.symmetry_axis_deg + relative) % 360.0,
                reference,
                extraction_ns + polar_ns + match_ns,
            )
        )

    for name, template in (
        ("average_pixel_phase_x", engine.average_polar),
        ("symmetric_average_pixel_phase_x", engine.symmetric_polar),
    ):
        started = time.perf_counter_ns()
        response = _phase_response_x(template, polar)
        relative = _parabolic_peak(response)
        match_ns = time.perf_counter_ns() - started
        rows.append(
            _method_row(
                name,
                "averaged_pixel_phase",
                session,
                frame_index,
                (engine.symmetry_axis_deg + relative) % 360.0,
                reference,
                extraction_ns + polar_ns + match_ns,
            )
        )

    feature_methods = []
    average_projection = np.sum(engine.symmetric_polar, axis=1)
    observed_projection = np.sum(polar, axis=1)
    feature_methods.append(
        ("angular_projection_phase1d", average_projection, observed_projection, "phase")
    )
    feature_methods.append(
        ("angular_projection_ncc_parabolic", average_projection, observed_projection, "ncc")
    )
    radial_weights = engine.estimator.radii.astype(np.float32)
    average_moment = np.sum(engine.symmetric_polar * radial_weights[None, :], axis=1)
    observed_moment = np.sum(polar * radial_weights[None, :], axis=1)
    feature_methods.append(
        ("radial_moment_phase1d", average_moment, observed_moment, "phase")
    )
    average_edges = np.sum(np.abs(np.diff(engine.symmetric_polar, axis=1)), axis=1)
    observed_edges = np.sum(np.abs(np.diff(polar, axis=1)), axis=1)
    feature_methods.append(
        ("radial_edge_ncc_parabolic", average_edges, observed_edges, "ncc")
    )
    for name, template_feature, observed_feature, mode in feature_methods:
        started = time.perf_counter_ns()
        if mode == "phase":
            relative, _ = _phase_peak_1d(template_feature, observed_feature)
        else:
            response = np.fft.ifft(
                np.fft.fft(observed_feature)
                * np.conj(np.fft.fft(template_feature))
            ).real
            relative = _parabolic_peak(response)
        match_ns = time.perf_counter_ns() - started
        rows.append(
            _method_row(
                name,
                "polar_x_feature",
                session,
                frame_index,
                (engine.symmetry_axis_deg + relative) % 360.0,
                reference,
                extraction_ns + polar_ns + match_ns,
            )
        )

    started = time.perf_counter_ns()
    pixel_response = _ncc_response(engine.symmetric_polar, polar)
    pixel_response_ns = time.perf_counter_ns() - started
    started = time.perf_counter_ns()
    relative = engine.estimator._fit_circular_gaussian_vectorized(pixel_response)[
        "center_deg"
    ]
    match_ns = pixel_response_ns + time.perf_counter_ns() - started
    rows.append(
        _method_row(
            "symmetric_pixel_fft_ncc_gaussian",
            "pixel_template_correlation",
            session,
            frame_index,
            (engine.symmetry_axis_deg + relative) % 360.0,
            reference,
            extraction_ns + polar_ns + match_ns,
        )
    )
    pixel_refiners = {
        "symmetric_pixel_fft_ncc_argmax": lambda: float(np.argmax(pixel_response)),
        "symmetric_pixel_fft_ncc_parabolic": lambda: _parabolic_peak(pixel_response),
        "symmetric_pixel_fft_ncc_weighted_centroid": lambda: _weighted_centroid_peak(pixel_response),
        "symmetric_pixel_fft_ncc_von_mises": lambda: _von_mises_moment_peak(pixel_response),
    }
    for name, operation in pixel_refiners.items():
        started = time.perf_counter_ns()
        relative = float(operation())
        refine_ns = time.perf_counter_ns() - started
        rows.append(
            _method_row(
                name,
                "pixel_template_correlation",
                session,
                frame_index,
                (engine.symmetry_axis_deg + relative) % 360.0,
                reference,
                extraction_ns + polar_ns + pixel_response_ns + refine_ns,
            )
        )
    started = time.perf_counter_ns()
    centroid_angle = engine.centroid_angle(centroid)
    centroid_ns = time.perf_counter_ns() - started
    rows.append(
        _method_row(
            "centroid_vector",
            "geometric_shortcut",
            session,
            frame_index,
            centroid_angle,
            reference,
            extraction_ns + centroid_ns,
        )
    )


def _exact_baselines(calibration_path, samples, sample_references, rows):
    estimators = (
        (
            "current_accurate_vectorized_full",
            CursorPoseEstimator(calibration_path, "vectorized_grid", "full"),
            "production_baseline",
        ),
        (
            "current_realtime_cascade_ambiguous",
            CursorPoseEstimator(calibration_path, "cascade", "ambiguous"),
            "production_baseline",
        ),
    )
    first = next(iter(samples.values()))[0][1]
    for _name, estimator, _group in estimators:
        estimator.estimate(first)
    for name, estimator, group in estimators:
        for session, session_samples in samples.items():
            for frame_index, frame in session_samples:
                reference = sample_references[session][frame_index][
                    "travel_angle_screen_deg"
                ]
                started = time.perf_counter_ns()
                result = estimator.estimate(frame, frame_index=frame_index)
                elapsed = time.perf_counter_ns() - started
                if result.get("detected") and result.get("angle_screen_deg") is not None:
                    rows.append(
                        _method_row(
                            name,
                            group,
                            session,
                            frame_index,
                            result["angle_screen_deg"],
                            reference,
                            elapsed,
                        )
                    )


def _aggregate(rows, expected_by_split):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["split"])].append(row)
        grouped[(row["method"], "all")].append(row)
    aggregates = []
    for (method, split), method_rows in sorted(grouped.items()):
        errors = [row["absolute_error_deg"] for row in method_rows]
        timings = [row["latency_ns"] for row in method_rows]
        expected = expected_by_split[split]
        aggregates.append(
            {
                "method": method,
                "group": method_rows[0]["group"],
                "split": split,
                "pose_produced_count": len(method_rows),
                "expected_count": expected,
                "pose_production_rate": float(len(method_rows) / expected),
                "mean_abs_error_deg": float(np.mean(errors)),
                "median_abs_error_deg": float(np.median(errors)),
                "p95_abs_error_deg": _percentile(errors, 95),
                "latency": _timing(timings),
            }
        )
    return aggregates


def _decisions(aggregates):
    holdout = {
        row["method"]: row for row in aggregates if row["split"] == "holdout"
    }
    baseline = holdout["current_accurate_vectorized_full"]
    decisions = {}
    for method, row in holdout.items():
        if method == "current_accurate_vectorized_full":
            decisions[method] = "REFERENCE"
            continue
        quality_ok = (
            row["pose_production_rate"] >= baseline["pose_production_rate"] - 0.01
            and row["median_abs_error_deg"] <= baseline["median_abs_error_deg"] + 0.5
            and row["p95_abs_error_deg"] <= baseline["p95_abs_error_deg"] + 1.0
        )
        substantially_faster = (
            row["latency"]["median_ms"]
            <= baseline["latency"]["median_ms"] * 0.8
        )
        clearly_worse = (
            row["p95_abs_error_deg"] > baseline["p95_abs_error_deg"] + 3.0
            or row["pose_production_rate"] < baseline["pose_production_rate"] - 0.05
        )
        decisions[method] = (
            "PROMISING"
            if quality_ok and substantially_faster
            else "REJECT"
            if clearly_worse
            else "HOLD"
        )
    return decisions


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path, result):
    aggregate = [row for row in result["aggregate"] if row["split"] == "holdout"]
    aggregate.sort(key=lambda row: (row["p95_abs_error_deg"], row["latency"]["median_ms"]))
    lines = [
        "# Cursor pose method benchmark",
        "",
        "Independent per-frame reference: centered masked mini-map phase-correlation "
        "travel vector over a multi-second window. Input logs are used only to verify "
        "forward control and absence of camera motion. End-to-end motion remains an audit.",
        "",
        "## Holdout method ranking",
        "",
        "Errors are mean / median / P95. No worst value is reported here because "
        "this is a sampled method layer, not a complete chronological E2E output.",
        "",
    ]
    for row in aggregate:
        lines.extend(
            [
                "### {}".format(row["method"]),
                "",
                "- Group: {}".format(row["group"]),
                "- Pose produced: {:.1%}".format(row["pose_production_rate"]),
                "- Error: {:.2f} / {:.2f} / {:.2f} deg".format(
                    row["mean_abs_error_deg"],
                    row["median_abs_error_deg"],
                    row["p95_abs_error_deg"],
                ),
                "- Latency: {:.3f} / {:.3f} ms median / P95".format(
                    row["latency"]["median_ms"], row["latency"]["p95_ms"]
                ),
                "- Decision: {}".format(
                    result["decisions"].get(row["method"], "")
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Reference audit",
            "",
        ]
    )
    for session, reference in result["references"].items():
        audit = reference["input_audit"]
        lines.extend(
            [
                "### {}".format(session),
                "",
                "- Split: {}".format(
                    "development" if session in DEVELOPMENT_SESSIONS else "holdout"
                ),
                "- Travel angle: {:.2f} deg".format(
                    reference["travel_angle_screen_deg"]
                ),
                "- Shift: {:.2f} px".format(
                    reference["map_content_shift_magnitude_px"]
                ),
                "- Correlation response: {:.3f}".format(
                    reference["phase_correlation_response"]
                ),
                "- Input keys: {}".format(", ".join(audit["keys"])),
                "- Mouse events: {}".format(
                    audit["event_kinds"].get("pc_raw_mouse", 0)
                ),
                "",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Each evaluated frame uses an independent centered motion window; low-response or subpixel-motion references are excluded.",
            "- The window averages short dynamics and therefore does not measure instantaneous rapid turns.",
            "- Decode, model construction, evidence rendering, and file I/O are excluded from latency.",
            "- Learned estimators were not attempted: four headings are insufficient independent labels.",
            "- Decisions use only the holdout split and do not change production code.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_reference_sensitivity(
    calibration_path,
    session_root,
    measurements_csv,
    output_path,
    windows=(30, 45, 60),
):
    """Rescore saved predictions against several independent motion windows."""
    with Path(measurements_csv).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    engine = CandidateEngine(calibration_path)
    decoded = {
        session: _decode_session(Path(session_root) / session)[0]
        for session in DEFAULT_SESSIONS
    }
    indices = {
        session: sorted(
            {int(row["frame_index"]) for row in rows if row["session"] == session}
        )
        for session in DEFAULT_SESSIONS
    }
    references = {}
    aggregates = []
    for window in windows:
        window = int(window)
        by_sample = {}
        for session in DEFAULT_SESSIONS:
            frames = decoded[session]
            for index in indices[session]:
                reference = _motion_reference(
                    engine,
                    frames,
                    max(0, index - window),
                    min(len(frames) - 1, index + window),
                )
                reference["valid"] = bool(
                    reference["phase_correlation_response"] >= 0.05
                    and reference["map_content_shift_magnitude_px"] >= 1.5
                )
                by_sample[(session, index)] = reference
        references[str(window)] = {
            "valid_count": int(sum(item["valid"] for item in by_sample.values())),
            "total_count": int(len(by_sample)),
            "samples": {
                "{}:{}".format(session, index): value
                for (session, index), value in by_sample.items()
            },
        }
        grouped = defaultdict(list)
        for row in rows:
            if row["split"] != "holdout":
                continue
            key = (row["session"], int(row["frame_index"]))
            reference = by_sample[key]
            if not reference["valid"]:
                continue
            error = abs(
                float(
                    circular_difference_degrees(
                        float(row["predicted_angle_deg"]),
                        reference["travel_angle_screen_deg"],
                    )
                )
            )
            grouped[row["method"]].append(error)
        for method, errors in grouped.items():
            aggregates.append(
                {
                    "reference_half_window_frames": window,
                    "method": method,
                    "sample_count": len(errors),
                    "median_abs_error_deg": float(np.median(errors)),
                    "p95_abs_error_deg": _percentile(errors, 95),
                }
            )
    common_keys = set.intersection(
        *[
            {
                key
                for key, value in entry["samples"].items()
                if value["valid"]
            }
            for entry in references.values()
        ]
    )
    angle_spreads = []
    for key in common_keys:
        angles = [
            references[str(int(window))]["samples"][key]["travel_angle_screen_deg"]
            for window in windows
        ]
        center = math.degrees(
            math.atan2(
                float(np.mean(np.sin(np.radians(angles)))),
                float(np.mean(np.cos(np.radians(angles)))),
            )
        )
        angle_spreads.append(
            max(abs(float(circular_difference_degrees(angle, center))) for angle in angles)
        )
    result = {
        "schema_version": "1.0",
        "windows_half_width_frames": list(map(int, windows)),
        "holdout_only": True,
        "references": references,
        "aggregate": aggregates,
        "reference_window_disagreement_deg": {
            "common_sample_count": len(angle_spreads),
            "median_max_deviation": float(np.median(angle_spreads)),
            "p95_max_deviation": _percentile(angle_spreads, 95),
            "max_deviation": float(np.max(angle_spreads)),
        },
    }
    output_path = Path(output_path)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_review_overlay(
    calibration_path,
    session_root,
    results_path,
    measurements_csv,
    output_path,
):
    """Render independently reviewable arrows without changing measurements."""
    result = json.loads(Path(results_path).read_text(encoding="utf-8"))
    with Path(measurements_csv).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_key = {
        (row["session"], int(row["frame_index"]), row["method"]): float(
            row["predicted_angle_deg"]
        )
        for row in rows
    }
    estimator = CursorPoseEstimator(calibration_path, "cascade", "ambiguous")
    methods = (
        ("current_accurate_vectorized_full", (0, 220, 255), "accurate"),
        ("polygon_von_mises_moment", (80, 255, 80), "von Mises"),
        ("symmetric_pixel_fft_ncc_parabolic", (255, 100, 255), "pixel NCC"),
    )
    tiles = []
    for session in DEFAULT_SESSIONS:
        frames = _decode_session(Path(session_root) / session)[0]
        session_indices = sorted(
            {int(row["frame_index"]) for row in rows if row["session"] == session}
        )
        selected = [
            session_indices[index]
            for index in np.linspace(0, len(session_indices) - 1, 4).astype(int)
        ]
        row_tiles = []
        for index in selected:
            crop = estimator._crop(frames[index]).copy()
            reference = result["sample_references"][session][str(index)][
                "travel_angle_screen_deg"
            ]
            angles = [(reference, (255, 255, 255), "motion")]
            angles.extend(
                (
                    by_key[(session, index, method)],
                    color,
                    label,
                )
                for method, color, label in methods
            )
            pivot = np.asarray(estimator.pivot, dtype=np.float64)
            for angle, color, _label in angles:
                radians = math.radians(float(angle))
                direction = np.asarray([math.cos(radians), math.sin(radians)])
                start = tuple(np.round(pivot + direction * 13.0).astype(int))
                end = tuple(np.round(pivot + direction * 33.0).astype(int))
                cv2.arrowedLine(crop, start, end, color, 1, cv2.LINE_AA, tipLength=0.25)
            cv2.putText(
                crop,
                "{} f{} motion {:.1f}".format(session, index, reference),
                (3, crop.shape[0] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.33,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            row_tiles.append(cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST))
        tiles.append(np.hstack(row_tiles))
    body = np.vstack(tiles)
    header = np.full((55, body.shape[1], 3), 18, np.uint8)
    cv2.putText(
        header,
        "white=local motion  yellow=current accurate  green=polygon von Mises  magenta=averaged-pixel NCC",
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), np.vstack([header, body]))


def evaluate_ordinary_cruise_agreement(
    calibration_path,
    session_root,
    output_path,
    sessions=("run_14", "run_15"),
    samples_per_session=90,
):
    """Check candidate continuity on turning data; this is not ground truth."""
    engine = CandidateEngine(calibration_path)
    accurate = CursorPoseEstimator(calibration_path, "vectorized_grid", "full")
    rows = []
    for session in sessions:
        frames, _records = _decode_session(Path(session_root) / session)
        indices = np.unique(
            np.rint(
                np.linspace(8, len(frames) - 9, int(samples_per_session))
            ).astype(int)
        )
        accurate.estimate(frames[int(indices[0])])
        for index in indices:
            index = int(index)
            frame = frames[index]
            started = time.perf_counter_ns()
            baseline = accurate.estimate(frame, frame_index=index)
            baseline_ns = time.perf_counter_ns() - started
            if not baseline.get("detected"):
                continue
            baseline_angle = float(baseline["angle_screen_deg"])
            rows.append(
                {
                    "session": session,
                    "frame_index": index,
                    "method": "current_accurate_vectorized_full",
                    "angle_deg": baseline_angle,
                    "disagreement_deg": 0.0,
                    "latency_ns": baseline_ns,
                }
            )
            patch, _centroid, _area, extraction_ns = engine.extract(frame)
            if patch is None:
                continue
            polygon, polygon_ns = engine.polygon_response(patch)
            polar, polar_ns = engine.polar_observation(patch)
            operations = (
                (
                    "polygon_von_mises_moment",
                    lambda: _von_mises_moment_peak(polygon),
                    extraction_ns + polygon_ns,
                ),
                (
                    "polygon_gaussian_cascade",
                    lambda: engine.estimator._fit_circular_gaussian_cascade(polygon)["center_deg"],
                    extraction_ns + polygon_ns,
                ),
                (
                    "symmetric_pixel_fft_ncc_parabolic",
                    lambda: _parabolic_peak(_ncc_response(engine.symmetric_polar, polar)),
                    extraction_ns + polar_ns,
                ),
                (
                    "angular_projection_ncc_parabolic",
                    lambda: _parabolic_peak(
                        np.fft.ifft(
                            np.fft.fft(np.sum(polar, axis=1))
                            * np.conj(np.fft.fft(np.sum(engine.symmetric_polar, axis=1)))
                        ).real
                    ),
                    extraction_ns + polar_ns,
                ),
                (
                    "symmetric_average_pixel_phase_x",
                    lambda: _parabolic_peak(
                        _phase_response_x(engine.symmetric_polar, polar)
                    ),
                    extraction_ns + polar_ns,
                ),
            )
            for method, operation, shared_ns in operations:
                started = time.perf_counter_ns()
                relative = float(operation())
                elapsed = time.perf_counter_ns() - started
                angle = float((engine.symmetry_axis_deg + relative) % 360.0)
                rows.append(
                    {
                        "session": session,
                        "frame_index": index,
                        "method": method,
                        "angle_deg": angle,
                        "disagreement_deg": abs(
                            float(circular_difference_degrees(angle, baseline_angle))
                        ),
                        "latency_ns": int(shared_ns + elapsed),
                    }
                )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    aggregate = []
    for method, method_rows in sorted(grouped.items()):
        disagreement = [row["disagreement_deg"] for row in method_rows]
        jumps = []
        for session in sessions:
            sequence = sorted(
                (row for row in method_rows if row["session"] == session),
                key=lambda row: row["frame_index"],
            )
            jumps.extend(
                abs(
                    float(
                        circular_difference_degrees(
                            second["angle_deg"], first["angle_deg"]
                        )
                    )
                )
                for first, second in zip(sequence, sequence[1:])
            )
        aggregate.append(
            {
                "method": method,
                "sample_count": len(method_rows),
                "median_disagreement_deg": float(np.median(disagreement)),
                "p95_disagreement_deg": _percentile(disagreement, 95),
                "max_disagreement_deg": float(np.max(disagreement)),
                "catastrophic_disagreement_rate_over_30deg": float(
                    np.mean(np.asarray(disagreement) > 30.0)
                ),
                "sampled_step_angle_p95_deg": _percentile(jumps, 95),
                "sampled_step_angle_max_deg": float(np.max(jumps)),
                "latency": _timing([row["latency_ns"] for row in method_rows]),
            }
        )
    result = {
        "schema_version": "1.0",
        "measurement": "agreement_with_current_accurate_on_ordinary_cruise",
        "accuracy_claim": False,
        "sessions": list(sessions),
        "samples_per_session_requested": int(samples_per_session),
        "aggregate": aggregate,
    }
    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def evaluate_current_fallback_behavior(
    calibration_path,
    session_root,
    forward_results_path,
    forward_measurements_csv,
    output_path,
    ordinary_sessions=("run_14", "run_15"),
    ordinary_samples_per_session=90,
    confidence_min=0.45,
):
    """Measure each deployed real-time fallback/rejection path explicitly."""
    forward_result = json.loads(
        Path(forward_results_path).read_text(encoding="utf-8")
    )
    with Path(forward_measurements_csv).open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        measurement_rows = list(csv.DictReader(stream))
    forward_indices = {
        session: sorted(
            {
                int(row["frame_index"])
                for row in measurement_rows
                if row["session"] == session
            }
        )
        for session in DEFAULT_SESSIONS
    }
    estimator = CursorPoseEstimator(calibration_path, "cascade", "ambiguous")
    rows = []

    def evaluate_session(session, indices, dataset_kind, references=None):
        frames, _records = _decode_session(Path(session_root) / session)
        if len(indices):
            estimator.estimate(frames[int(indices[0])])
        for index in indices:
            index = int(index)
            started = time.perf_counter_ns()
            pose = estimator.estimate(frames[index], frame_index=index)
            elapsed = time.perf_counter_ns() - started
            detected = bool(pose.get("detected"))
            confidence = float(pose.get("confidence") or 0.0)
            low_confidence = bool(not detected or confidence < float(confidence_min))
            reasons = []
            if detected:
                if float(pose.get("gaussian_fit_r_squared") or 0.0) < 0.90:
                    reasons.append("gaussian_r2")
                if float(pose.get("gaussian_center_std_deg") or 999.0) > 1.5:
                    reasons.append("gaussian_center_std")
                if float(pose.get("angular_likelihood_margin") or 0.0) < 0.05:
                    reasons.append("peak_margin")
                if abs(float(pose.get("centroid_agreement_error_deg") or 0.0)) > 15.0:
                    reasons.append("centroid_agreement")
                if float(pose.get("template_aligned_iou") or 0.0) < 0.65:
                    reasons.append("polygon_iou")
                if bool(pose.get("gaussian_fallback_used")):
                    reasons.append("gaussian_fitter_fallback")
            reference_angle = None
            error = None
            if references is not None:
                reference_angle = float(
                    references[session][str(index)]["travel_angle_screen_deg"]
                )
                if detected:
                    error = abs(
                        float(
                            circular_difference_degrees(
                                pose["angle_screen_deg"], reference_angle
                            )
                        )
                    )
            rows.append(
                {
                    "dataset": dataset_kind,
                    "session": session,
                    "frame_index": index,
                    "detected": detected,
                    "confidence": confidence,
                    "accepted_at_profile_threshold": bool(
                        detected and confidence >= float(confidence_min)
                    ),
                    "low_confidence_rejection": low_confidence,
                    "gaussian_fitter_fallback": bool(
                        pose.get("gaussian_fallback_used")
                    ),
                    "pixel_validation_performed": bool(
                        pose.get("pixel_validation_performed")
                    ),
                    "pixel_validation_trigger_reasons": reasons,
                    "angle_deg": float(pose["angle_screen_deg"])
                    if detected
                    else None,
                    "reference_angle_deg": reference_angle,
                    "absolute_error_deg": error,
                    "latency_ns": int(elapsed),
                }
            )

    for session in DEFAULT_SESSIONS:
        evaluate_session(
            session,
            forward_indices[session],
            "forward_independent_reference",
            forward_result["sample_references"],
        )
    for session in ordinary_sessions:
        frames, _records = _decode_session(Path(session_root) / session)
        indices = np.unique(
            np.rint(
                np.linspace(8, len(frames) - 9, int(ordinary_samples_per_session))
            ).astype(int)
        )
        # evaluate_session decodes again to keep this helper simple and deterministic.
        evaluate_session(session, indices, "ordinary_cruise_agreement_only")

    def summarize(selected):
        total = len(selected)
        timings = [row["latency_ns"] for row in selected]
        detected = [row for row in selected if row["detected"]]
        accepted = [row for row in selected if row["accepted_at_profile_threshold"]]
        pixel = [row for row in selected if row["pixel_validation_performed"]]
        gaussian = [row for row in selected if row["gaussian_fitter_fallback"]]
        low = [row for row in selected if row["low_confidence_rejection"]]
        reason_counts = defaultdict(int)
        for row in selected:
            for reason in row["pixel_validation_trigger_reasons"]:
                reason_counts[reason] += 1
        value = {
            "sample_count": total,
            "cursor_detection_rate": float(len(detected) / total),
            "confidence_accepted_measurement_rate": float(len(accepted) / total),
            "low_confidence_rejection_rate": float(len(low) / total),
            "gaussian_fitter_fallback_rate": float(len(gaussian) / total),
            "pixel_validation_invocation_rate": float(len(pixel) / total),
            "pixel_validation_trigger_reason_rate_of_all_samples": {
                reason: float(count / total)
                for reason, count in sorted(reason_counts.items())
            },
            "overall_latency": _timing(timings),
        }
        for label, subset in (
            ("pixel_validation_used", pixel),
            ("pixel_validation_skipped", [row for row in selected if not row["pixel_validation_performed"]]),
            ("gaussian_fallback_used", gaussian),
            ("low_confidence_rejected", low),
            ("accepted", accepted),
        ):
            value[label] = {
                "count": len(subset),
                "rate_of_all_samples": float(len(subset) / total),
                "latency": _timing([row["latency_ns"] for row in subset])
                if subset
                else None,
            }
            errors = [
                row["absolute_error_deg"]
                for row in subset
                if row["absolute_error_deg"] is not None
            ]
            if errors:
                value[label]["independent_reference_error"] = {
                    "sample_count": len(errors),
                    "median_abs_error_deg": float(np.median(errors)),
                    "p95_abs_error_deg": _percentile(errors, 95),
                }
        return value

    summaries = {
        "all": summarize(rows),
        "forward": summarize(
            [row for row in rows if row["dataset"] == "forward_independent_reference"]
        ),
        "ordinary_cruise": summarize(
            [row for row in rows if row["dataset"] == "ordinary_cruise_agreement_only"]
        ),
        "forward_development": summarize(
            [
                row
                for row in rows
                if row["session"] in DEVELOPMENT_SESSIONS
            ]
        ),
        "forward_holdout": summarize(
            [
                row
                for row in rows
                if row["session"] in set(DEFAULT_SESSIONS) - DEVELOPMENT_SESSIONS
            ]
        ),
    }
    result = {
        "schema_version": "1.0",
        "measurement": "current_realtime_cursor_fallback_behavior",
        "profile": {
            "gaussian_fit_method": "cascade",
            "validation_policy": "ambiguous",
            "pose_confidence_min": float(confidence_min),
        },
        "semantics": {
            "gaussian_fitter_fallback": "analytic LM rejected or failed, then fast grid used",
            "pixel_validation_performed": "ambiguous polygon/Gaussian result caused pixel-polar validation; this is validation, not a fallback pose",
            "low_confidence_rejection": "final result below profile confidence threshold after any validation",
        },
        "summaries": summaries,
        "rows": rows,
    }
    Path(output_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run(
    calibration_path,
    session_root,
    output_path,
    samples_per_session=60,
    reference_half_window_frames=45,
):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    engine = CandidateEngine(calibration_path)
    references = {}
    sample_references = {}
    samples = {}
    decoded_by_session = {}
    records_by_session = {}
    for session in DEFAULT_SESSIONS:
        session_path = Path(session_root) / session
        frames, records = _decode_session(session_path)
        decoded_by_session[session] = frames
        records_by_session[session] = records
        reference = _reference(engine, frames)
        reference["input_audit"] = _input_audit(session_path)
        if reference["phase_correlation_response"] < 0.05:
            raise RuntimeError("{} has an unreliable motion reference".format(session))
        references[session] = reference
        indices = np.unique(
            np.rint(
                np.linspace(
                    max(8, int(len(frames) * 0.15)),
                    min(len(frames) - 9, int(len(frames) * 0.85)),
                    int(samples_per_session),
                )
            ).astype(int)
        )
        local_references = {}
        valid_samples = []
        for index in indices:
            index = int(index)
            local = _motion_reference(
                engine,
                frames,
                max(0, index - int(reference_half_window_frames)),
                min(len(frames) - 1, index + int(reference_half_window_frames)),
            )
            local["valid"] = bool(
                local["phase_correlation_response"] >= 0.05
                and local["map_content_shift_magnitude_px"] >= 1.5
            )
            local_references[index] = local
            if local["valid"]:
                valid_samples.append((index, frames[index]))
        sample_references[session] = local_references
        samples[session] = valid_samples
        reference["local_reference_samples"] = len(local_references)
        reference["local_reference_valid_samples"] = len(valid_samples)
        reference["local_reference_valid_rate"] = float(
            len(valid_samples) / max(1, len(local_references))
        )

    # Warm the common OpenCV/NumPy paths before recording latency.
    warm_frame = samples[DEFAULT_SESSIONS[0]][0][1]
    warm_patch, _centroid, _area, _ = engine.extract(warm_frame)
    warm_polygon, _ = engine.polygon_response(warm_patch)
    warm_polar, _ = engine.polar_observation(warm_patch)
    engine.phase_2d(engine.symmetric_polar, warm_polar)
    engine.estimator._fit_circular_gaussian_vectorized(warm_polygon)

    rows = []
    for session, session_samples in samples.items():
        for frame_index, frame in session_samples:
            reference = sample_references[session][frame_index][
                "travel_angle_screen_deg"
            ]
            _benchmark_frame(
                engine,
                frame,
                frame_index,
                session,
                reference,
                rows,
            )
    _exact_baselines(calibration_path, samples, sample_references, rows)
    expected_by_split = {
        "development": sum(len(samples[item]) for item in DEVELOPMENT_SESSIONS),
        "holdout": sum(
            len(rows_) for name, rows_ in samples.items() if name not in DEVELOPMENT_SESSIONS
        ),
        "all": sum(len(rows_) for rows_ in samples.values()),
    }
    aggregates = _aggregate(rows, expected_by_split)
    result = {
        "schema_version": "1.0",
        "experiment": "cursor_pose_method_benchmark",
        "calibration_path": str(Path(calibration_path).resolve()),
        "session_root": str(Path(session_root).resolve()),
        "sessions": list(DEFAULT_SESSIONS),
        "development_sessions": sorted(DEVELOPMENT_SESSIONS),
        "holdout_sessions": sorted(set(DEFAULT_SESSIONS) - DEVELOPMENT_SESSIONS),
        "samples_per_session_requested": int(samples_per_session),
        "reference_half_window_frames": int(reference_half_window_frames),
        "causal_inputs": "one frame plus existing calibration only",
        "evaluation_only_evidence": "forward input audit and end-to-end mini-map shift",
        "references": references,
        "sample_references": sample_references,
        "aggregate": aggregates,
        "provenance": {
            "git_revision": _git_output("rev-parse", "HEAD"),
            "git_branch": _git_output("branch", "--show-current"),
            "tested_source_dirty": bool(
                _git_output(
                    "status", "--porcelain", "--", "poc/benchmark_cursor_pose_methods.py"
                )
            ),
            "benchmark_source_sha256": _sha256(Path(__file__).resolve()),
            "pose_source_sha256": _sha256(
                Path(sys.modules[CursorPoseEstimator.__module__].__file__).resolve()
            ),
            "environment": {
                "python": sys.version,
                "python_executable": sys.executable,
                "platform": platform.platform(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
            },
        },
    }
    result["decisions"] = _decisions(aggregates)
    (output_path / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _write_csv(output_path / "measurements.csv", rows)
    (output_path / "method_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "git_revision": result["provenance"]["git_revision"],
                "benchmark_source_sha256": result["provenance"][
                    "benchmark_source_sha256"
                ],
                "methods": [
                    {
                        "method_id": row["method"],
                        "group": row["group"],
                        "implementation_file": str(Path(__file__).resolve()),
                    }
                    for row in aggregates
                    if row["split"] == "holdout"
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(output_path / "REPORT.md", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("session_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--samples-per-session", type=int, default=60)
    parser.add_argument("--reference-half-window-frames", type=int, default=45)
    arguments = parser.parse_args()
    result = run(
        arguments.calibration,
        arguments.session_root,
        arguments.output,
        samples_per_session=arguments.samples_per_session,
        reference_half_window_frames=arguments.reference_half_window_frames,
    )
    holdout = [
        row for row in result["aggregate"] if row["split"] == "holdout"
    ]
    print(json.dumps(holdout, indent=2))


if __name__ == "__main__":
    main()
