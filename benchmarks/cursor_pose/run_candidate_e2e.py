"""Chronological confidence and fallback benchmark for cursor-pose candidates.

This module is benchmark-only.  Experimental candidates consume one frame and
the saved calibration.  Development sessions select any experimental confidence
threshold; holdout sessions are never used for threshold selection.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from aria_trace.services.calibration.cursor.pose import (
    CursorPoseEstimator,
    circular_difference_degrees,
)
from aria_trace.services.calibration.cursor.shape import render_polygon
from benchmarks.cursor_pose.run_e2e import (
    DEFAULT_DEVELOPMENT_SESSIONS,
    DEFAULT_SESSIONS,
    OUTAGE_SCENARIOS,
    _aggregate,
    _decode_session,
    _file_identity,
    _forward_only_input_audit,
    _gate_config,
    _reference_mask,
    _replay_strategy,
    _repository_identity,
    _session_e2e_reference,
    _source_identity,
    _write_csv,
)
from benchmarks.cursor_pose.stateful import summarize_e2e_rows
from poc.benchmark_cursor_pose_methods import (
    CandidateEngine,
    _ncc_response,
    _parabolic_peak,
    _von_mises_moment_peak,
)


SCHEMA_VERSION = "1.0"
NATIVE_CANDIDATES = {
    "accurate_vectorized_full": {
        "gaussian_fit_method": "vectorized_grid",
        "validation_policy": "full",
        "confidence_threshold": 0.40,
    },
    "realtime_cascade_ambiguous": {
        "gaussian_fit_method": "cascade",
        "validation_policy": "ambiguous",
        "confidence_threshold": 0.45,
    },
    "fast_cascade_minimal": {
        "gaussian_fit_method": "cascade",
        "validation_policy": "minimal",
        "confidence_threshold": 0.50,
    },
    "analytic_lm_ambiguous": {
        "gaussian_fit_method": "analytic_lm",
        "validation_policy": "ambiguous",
        "confidence_threshold": 0.45,
    },
    "fast_grid_ambiguous": {
        "gaussian_fit_method": "fast_grid",
        "validation_policy": "ambiguous",
        "confidence_threshold": 0.45,
    },
}
EXPERIMENTAL_CANDIDATES = (
    "polygon_von_mises_moment",
    "symmetric_pixel_fft_ncc_parabolic",
    "angular_projection_ncc_parabolic",
)
FALLBACKS = {
    "hold": "reuse_previous_state",
    "predict": "constant_velocity_last_2_accepted",
    "unavailable": "no_fallback",
}
CONFIDENCE_MODES = ("selected", "accept_all")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _geometric_mean(values) -> float:
    clipped = [max(float(value), 0.02) for value in values]
    return float(np.prod(clipped) ** (1.0 / len(clipped)))


def _response_prominence(response: np.ndarray, center_deg: float) -> float:
    response = np.asarray(response, dtype=np.float64).reshape(-1)
    positions = np.arange(len(response), dtype=np.float64)
    peak = float(np.interp(center_deg % len(response), positions, response, period=len(response)))
    distance = np.abs(
        circular_difference_degrees(positions * 360.0 / len(response), center_deg)
    )
    outside = response[distance >= 20.0]
    second = float(np.max(outside)) if len(outside) else float(np.median(response))
    scale = max(
        float(np.percentile(response, 95) - np.percentile(response, 20)),
        1.0e-6,
    )
    return float(np.clip((peak - second) / scale, 0.0, 1.0))


class ExperimentalPoseEstimator:
    """Apply one experimental peak estimator plus common geometric confidence."""

    def __init__(self, calibration_path: Path, method_id: str) -> None:
        if method_id not in EXPERIMENTAL_CANDIDATES:
            raise ValueError("Unknown experimental method: {}".format(method_id))
        self.method_id = method_id
        self.engine = CandidateEngine(calibration_path)

    def estimate(self, frame: np.ndarray, **_metadata) -> dict:
        patch, centroid, area, _ = self.engine.extract(frame)
        if patch is None:
            return {
                "detected": False,
                "angle_screen_deg": None,
                "confidence": 0.0,
                "confidence_components": {},
                "failure": "cursor_component_not_detected",
            }

        if self.method_id == "polygon_von_mises_moment":
            response, _ = self.engine.polygon_response(patch)
            relative = float(_von_mises_moment_peak(response))
        else:
            polar, _ = self.engine.polar_observation(patch)
            if self.method_id == "symmetric_pixel_fft_ncc_parabolic":
                response = _ncc_response(self.engine.symmetric_polar, polar)
            else:
                template = np.sum(self.engine.symmetric_polar, axis=1)
                observed = np.sum(polar, axis=1)
                response = np.fft.ifft(
                    np.fft.fft(observed) * np.conj(np.fft.fft(template))
                ).real.astype(np.float32)
            relative = float(_parabolic_peak(response))

        predicted = render_polygon(
            self.engine.estimator.polygon,
            self.engine.estimator.patch_size,
            relative,
            supersample=4,
        ) >= 0.5
        observed = patch >= 0.5
        union = int(np.logical_or(predicted, observed).sum())
        aligned_iou = (
            float(np.logical_and(predicted, observed).sum() / union) if union else 0.0
        )
        centroid_vector = np.asarray(centroid) - self.engine.estimator.pivot
        centroid_angle = math.degrees(
            math.atan2(float(centroid_vector[1]), float(centroid_vector[0]))
        )
        centroid_relative = float(
            circular_difference_degrees(
                centroid_angle,
                self.engine.estimator.canonical_centroid_angle_deg,
            )
        )
        centroid_error = float(circular_difference_degrees(relative, centroid_relative))
        components = {
            "response_prominence": _response_prominence(response, relative),
            "polygon_iou": float(np.clip(aligned_iou / 0.82, 0.0, 1.0)),
            "centroid_agreement": float(math.exp(-((centroid_error / 20.0) ** 2))),
        }
        confidence = _geometric_mean(components.values())
        return {
            "detected": True,
            "angle_screen_deg": float(
                (self.engine.symmetry_axis_deg + relative) % 360.0
            ),
            "confidence": confidence,
            "confidence_components": components,
            "component_area_px": int(area),
            "relative_rotation_deg": relative,
            "template_aligned_iou": aligned_iou,
            "centroid_agreement_error_deg": centroid_error,
        }


def _candidate_factory(calibration_path: Path, method_id: str):
    if method_id in NATIVE_CANDIDATES:
        config = NATIVE_CANDIDATES[method_id]
        return CursorPoseEstimator(
            calibration_path,
            gaussian_fit_method=config["gaussian_fit_method"],
            validation_policy=config["validation_policy"],
        )
    return ExperimentalPoseEstimator(calibration_path, method_id)


def _measure_candidate(
    estimator,
    method_id,
    session,
    frames,
    records,
    reference,
):
    estimator.estimate(frames[0])
    rows = []
    for ordinal, (frame, record) in enumerate(zip(frames, records)):
        started = time.perf_counter_ns()
        result = estimator.estimate(
            frame,
            frame_index=int(record["frame_index"]),
            session_time_ns=int(record["session_time_ns"]),
        )
        latency_ns = time.perf_counter_ns() - started
        produced = bool(
            result.get("detected") and result.get("angle_screen_deg") is not None
        )
        rows.append(
            {
                "profile": method_id,
                "session": session,
                "ordinal": int(ordinal),
                "frame_index": int(record["frame_index"]),
                "session_time_ns": int(record["session_time_ns"]),
                "angle_deg": (
                    float(result["angle_screen_deg"]) if produced else None
                ),
                "confidence": float(result.get("confidence") or 0.0),
                "primary_candidate_produced": produced,
                "confidence_gate_passed": False,
                "natural_primary_accepted": False,
                "natural_rejection_reason": None,
                "confidence_threshold": None,
                "primary_latency_ns": int(latency_ns),
                "pixel_validation_performed": bool(
                    result.get("pixel_validation_performed")
                ),
                "gaussian_fitter_fallback_used": bool(
                    result.get("gaussian_fallback_used")
                ),
                "reference_valid": bool(reference["valid"]),
                "reference_angle_deg": reference["travel_angle_screen_deg"],
                "reference_response": reference["phase_correlation_response"],
                "reference_displacement_px": reference[
                    "map_content_shift_magnitude_px"
                ],
            }
        )
    return rows


def _apply_threshold(rows, threshold):
    output = []
    for source in rows:
        row = dict(source)
        passed = bool(
            row["primary_candidate_produced"]
            and float(row["confidence"]) >= float(threshold)
        )
        row["confidence_gate_passed"] = passed
        row["natural_primary_accepted"] = passed
        row["confidence_threshold"] = float(threshold)
        if not row["primary_candidate_produced"]:
            row["natural_rejection_reason"] = "cursor_not_detected"
        elif not passed:
            row["natural_rejection_reason"] = "confidence_below_threshold"
        else:
            row["natural_rejection_reason"] = None
        output.append(row)
    return output


def _replay_sessions(rows, solution, fallback, gate_config, scenario="natural"):
    output = []
    for session in sorted({row["session"] for row in rows}):
        selected = [row for row in rows if row["session"] == session]
        output.extend(
            _replay_strategy(
                selected,
                scenario,
                solution,
                fallback,
                gate_config,
                False,
                "candidate_experiment",
            )
        )
    return output


def _select_threshold(rows, gate_config):
    candidates = sorted(
        set(
            [0.0, 1.0]
            + [float(value) for value in np.linspace(0.05, 0.95, 19)]
            + [float(np.quantile([row["confidence"] for row in rows], q)) for q in np.linspace(0.05, 0.95, 19)]
        )
    )
    trials = []
    for threshold in candidates:
        thresholded = _apply_threshold(rows, threshold)
        replay = _replay_sessions(
            thresholded,
            "threshold_tuning",
            "reuse_previous_state",
            gate_config,
        )
        summary = summarize_e2e_rows(replay)
        trials.append({"threshold": float(threshold), **summary})
    eligible = [
        row
        for row in trials
        if row["primary_measurement_accepted_rate"] >= 0.95
        and row["final_output_available_rate"] >= 0.995
    ]
    if not eligible:
        eligible = trials
    selected = min(
        eligible,
        key=lambda row: (
            row["e2e_absolute_error_deg"]["worst"]
            if row["e2e_absolute_error_deg"]["worst"] is not None
            else float("inf"),
            row["e2e_absolute_error_deg"]["p95"]
            if row["e2e_absolute_error_deg"]["p95"] is not None
            else float("inf"),
            -row["primary_measurement_accepted_rate"],
            row["threshold"],
        ),
    )
    return float(selected["threshold"]), trials


def _pct(value) -> str:
    return "n/a" if value is None else "{:.2%}".format(float(value))


def _num(value, suffix="") -> str:
    return "n/a" if value is None else "{:.2f}{}".format(float(value), suffix)


def _write_plain_report(path: Path, result: dict) -> None:
    holdout = [row for row in result["aggregate"] if row["split"] == "holdout"]
    natural_hold = [
        row
        for row in holdout
        if row["outage_scenario"] == "natural"
        and row["fallback_strategy"] == "reuse_previous_state"
        and row["confidence_mode"] == "selected"
    ]
    natural_hold.sort(
        key=lambda row: (
            -row["final_output_available_rate"],
            row["e2e_absolute_error_deg"]["worst"],
            row["e2e_absolute_error_deg"]["p95"],
            -row["primary_measurement_accepted_rate"],
            row["e2e_latency_ms"]["median"],
        )
    )
    winner = natural_hold[0] if natural_hold else None
    lines = [
        "CURSOR POSE COMPLETE CHRONOLOGICAL BENCHMARK",
        "",
        "BOTTOM LINE",
        "",
        "Primary order: availability, worst error, P95 error, fresh acceptance, latency.",
        "Held output is never counted as a fresh accepted measurement.",
        "",
    ]
    if winner is not None:
        lines.extend(
            [
                "WINNER  {} plus confidence plus hold".format(winner["profile"]),
                "Available  {}".format(_pct(winner["final_output_available_rate"])),
                "Fresh accepted  {}".format(
                    _pct(winner["primary_measurement_accepted_rate"])
                ),
                "Worst error  {}".format(
                    _num(winner["e2e_absolute_error_deg"]["worst"], " deg")
                ),
                "P95 error  {}".format(
                    _num(winner["e2e_absolute_error_deg"]["p95"], " deg")
                ),
                "Median latency  {}".format(
                    _num(winner["e2e_latency_ms"]["median"], " ms")
                ),
                "",
            ]
        )
    lines.extend(["RANKED NATURAL HOLD STACKS", ""])
    for index, row in enumerate(natural_hold, 1):
        error = row["e2e_absolute_error_deg"]
        lines.extend(
            [
                "{}  {}".format(index, row["profile"]),
                "Available  {}".format(_pct(row["final_output_available_rate"])),
                "Fresh accepted  {}".format(
                    _pct(row["primary_measurement_accepted_rate"])
                ),
                "Held  {}".format(_pct(row["output_provenance_rate"]["held"])),
                "Worst error  {}".format(_num(error["worst"], " deg")),
                "P95 error  {}".format(_num(error["p95"], " deg")),
                "Median error  {}".format(_num(error["median"], " deg")),
                "Median latency  {}".format(
                    _num(row["e2e_latency_ms"]["median"], " ms")
                ),
                "",
            ]
        )
    lines.extend(["CONFIDENCE GATE EFFECT", ""])
    method_ids = sorted({row["profile"] for row in holdout})
    for method_id in method_ids:
        lines.extend([method_id, ""])
        for confidence_mode in CONFIDENCE_MODES:
            selected = next(
                (
                    row
                    for row in holdout
                    if row["profile"] == method_id
                    and row["outage_scenario"] == "natural"
                    and row["fallback_strategy"] == "reuse_previous_state"
                    and row["confidence_mode"] == confidence_mode
                ),
                None,
            )
            if selected is None:
                continue
            lines.extend(
                [
                    confidence_mode.upper(),
                    "Fresh accepted  {}".format(
                        _pct(selected["primary_measurement_accepted_rate"])
                    ),
                    "Held  {}".format(
                        _pct(selected["output_provenance_rate"]["held"])
                    ),
                    "Available  {}".format(
                        _pct(selected["final_output_available_rate"])
                    ),
                    "Worst error  {}".format(
                        _num(selected["e2e_absolute_error_deg"]["worst"], " deg")
                    ),
                    "P95 error  {}".format(
                        _num(selected["e2e_absolute_error_deg"]["p95"], " deg")
                    ),
                    "",
                ]
            )
    lines.extend(["FALLBACK COMPARISON UNDER THREE FRAME OUTAGES", ""])
    for method_id in method_ids:
        lines.extend([method_id, ""])
        for fallback_label, fallback_id in FALLBACKS.items():
            selected = next(
                (
                    row
                    for row in holdout
                    if row["profile"] == method_id
                    and row["outage_scenario"] == "three_frame_burst_every_90"
                    and row["fallback_strategy"] == fallback_id
                    and row["confidence_mode"] == "selected"
                ),
                None,
            )
            if selected is None:
                continue
            lines.extend(
                [
                    fallback_label.upper(),
                    "Available  {}".format(
                        _pct(selected["final_output_available_rate"])
                    ),
                    "Worst error  {}".format(
                        _num(selected["e2e_absolute_error_deg"]["worst"], " deg")
                    ),
                    "P95 error  {}".format(
                        _num(selected["e2e_absolute_error_deg"]["p95"], " deg")
                    ),
                    "Longest unavailable episode  {} frames".format(
                        selected["continuity"]["longest_unavailable_episode_frames"]
                    ),
                    "",
                ]
            )
    lines.extend(
        [
            "CONFIDENCE THRESHOLDS",
            "",
        ]
    )
    for method_id, threshold in result["confidence_thresholds"].items():
        provenance = (
            "fixed production profile"
            if method_id in NATIVE_CANDIDATES
            else "selected on development sessions only"
        )
        lines.append("{}  {:.4f}  {}".format(method_id, threshold, provenance))
    lines.extend(
        [
            "",
            "LIMITATIONS",
            "",
            "The travel reference is functional E2E evidence, not pixel-level cursor truth.",
            "Forward-only sessions can favor stale held poses, so development tuning requires at least 95% fresh acceptance.",
            "A production decision for an experimental estimator still requires pose-labeled turning data.",
            "",
            "TRACEABILITY",
            "",
            "Git revision  {}".format(result["provenance"]["repository"]["revision"]),
            "Tested source dirty  {}".format(
                result["provenance"]["repository"]["tested_source_dirty"]
            ),
            "Machine results  results.json",
            "Threshold trials  threshold_tuning.json",
            "Primary rows  primary_measurements.csv",
            "Chronological outputs  e2e_rows.csv",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    calibration_path: Path,
    session_root: Path,
    output_path: Path,
    sessions=DEFAULT_SESSIONS,
    development_sessions=tuple(DEFAULT_DEVELOPMENT_SESSIONS),
    hash_videos=True,
):
    root = Path(__file__).resolve().parents[2]
    calibration_path = Path(calibration_path).resolve()
    session_root = Path(session_root).resolve()
    output_path = Path(output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    gate_config = _gate_config(calibration)
    method_ids = tuple(NATIVE_CANDIDATES) + tuple(EXPERIMENTAL_CANDIDATES)

    identity = CursorPoseEstimator(calibration_path)
    reference_mask = _reference_mask(identity)
    decoded = {}
    input_manifest = {
        "calibration": _file_identity(calibration_path),
        "model": _file_identity(calibration_path.parent / "model.npz"),
        "sessions": {},
    }
    for session in sessions:
        session_path = session_root / session
        frames, records, reader = _decode_session(session_path)
        audit = _forward_only_input_audit(reader)
        if not audit["valid_forward_only_control"]:
            raise RuntimeError("{} is not forward-only".format(session))
        reference = _session_e2e_reference(identity, reference_mask, frames)
        if not reference["valid"]:
            raise RuntimeError("{} has no valid E2E reference".format(session))
        decoded[session] = (frames, records, reference)
        input_manifest["sessions"][session] = {
            "manifest": _file_identity(session_path / "manifest.json"),
            "frames": _file_identity(session_path / "frames.jsonl"),
            "inputs": _file_identity(session_path / "inputs.jsonl"),
            "video": _file_identity(
                reader.video_path("main"), hash_content=bool(hash_videos)
            ),
            "decoded_frame_count": len(frames),
            "forward_only_input_audit": audit,
            "whole_session_e2e_reference": reference,
        }

    raw_primary = []
    for method_id in method_ids:
        estimator = _candidate_factory(calibration_path, method_id)
        for session in sessions:
            frames, records, reference = decoded[session]
            raw_primary.extend(
                _measure_candidate(
                    estimator,
                    method_id,
                    session,
                    frames,
                    records,
                    reference,
                )
            )

    development_set = set(development_sessions)
    thresholds = {}
    threshold_tuning = {}
    for method_id in method_ids:
        selected = [row for row in raw_primary if row["profile"] == method_id]
        if method_id in NATIVE_CANDIDATES:
            thresholds[method_id] = float(
                NATIVE_CANDIDATES[method_id]["confidence_threshold"]
            )
            threshold_tuning[method_id] = {
                "provenance": "fixed production profile",
                "selected_threshold": thresholds[method_id],
                "trials": [],
            }
        else:
            development = [
                row for row in selected if row["session"] in development_set
            ]
            threshold, trials = _select_threshold(development, gate_config)
            thresholds[method_id] = threshold
            threshold_tuning[method_id] = {
                "provenance": "development sessions only",
                "selected_threshold": threshold,
                "minimum_fresh_acceptance": 0.95,
                "minimum_final_availability": 0.995,
                "trials": trials,
            }

    primary = []
    for method_id in method_ids:
        method_rows = [row for row in raw_primary if row["profile"] == method_id]
        for confidence_mode in CONFIDENCE_MODES:
            threshold = thresholds[method_id] if confidence_mode == "selected" else 0.0
            thresholded = _apply_threshold(method_rows, threshold)
            for row in thresholded:
                row["confidence_mode"] = confidence_mode
            primary.extend(thresholded)
    _write_csv(output_path / "raw_primary_measurements.csv", raw_primary)
    _write_csv(output_path / "primary_measurements.csv", primary)

    e2e_rows = []
    solution_configs = {}
    for method_id in method_ids:
        for confidence_mode in CONFIDENCE_MODES:
            method_rows = [
                row
                for row in primary
                if row["profile"] == method_id
                and row["confidence_mode"] == confidence_mode
            ]
            for fallback_label, fallback_id in FALLBACKS.items():
                solution = "{}+{}+{}".format(
                    method_id, confidence_mode, fallback_label
                )
                solution_configs[solution] = {
                    "candidate": method_id,
                    "confidence_mode": confidence_mode,
                    "fallback": fallback_label,
                }
                for scenario in OUTAGE_SCENARIOS:
                    replayed = _replay_sessions(
                        method_rows,
                        solution,
                        fallback_id,
                        gate_config,
                        scenario,
                    )
                    for row in replayed:
                        row["confidence_mode"] = confidence_mode
                    e2e_rows.extend(replayed)
    _write_csv(output_path / "e2e_rows.csv", e2e_rows)
    aggregate = _aggregate(e2e_rows, development_sessions)
    for row in aggregate:
        row.update(solution_configs[row["solution"]])

    tested_files = [
        Path(__file__).resolve(),
        Path(inspect.getsourcefile(ExperimentalPoseEstimator)).resolve(),
        Path(inspect.getsourcefile(CandidateEngine)).resolve(),
        Path(inspect.getsourcefile(CursorPoseEstimator)).resolve(),
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "cursor_pose_candidate_complete_e2e",
        "objective_order": [
            "final_output_available_rate_desc",
            "worst_e2e_error_asc",
            "p95_e2e_error_asc",
            "primary_measurement_accepted_rate_desc",
            "median_latency_asc",
        ],
        "sessions": list(sessions),
        "development_sessions": list(development_sessions),
        "holdout_sessions": sorted(set(sessions) - set(development_sessions)),
        "candidates": list(method_ids),
        "fallbacks": FALLBACKS,
        "confidence_modes": list(CONFIDENCE_MODES),
        "outage_scenarios": OUTAGE_SCENARIOS,
        "confidence_thresholds": thresholds,
        "aggregate": aggregate,
        "provenance": {
            "run_completed_utc": datetime.now(timezone.utc).isoformat(),
            "repository": _repository_identity(root, tested_files),
            "environment": {
                "python": sys.version,
                "python_executable": sys.executable,
                "platform": platform.platform(),
                "opencv": cv2.__version__,
                "numpy": np.__version__,
            },
            "benchmark_source": _source_identity(run),
            "experimental_estimator": _source_identity(
                ExperimentalPoseEstimator.estimate
            ),
        },
    }
    (output_path / "input_manifest.json").write_text(
        json.dumps(input_manifest, indent=2), encoding="utf-8"
    )
    (output_path / "threshold_tuning.json").write_text(
        json.dumps(threshold_tuning, indent=2), encoding="utf-8"
    )
    (output_path / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _write_plain_report(output_path / "REPORT.txt", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("session_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--skip-video-hash", action="store_true")
    arguments = parser.parse_args()
    result = run(
        arguments.calibration,
        arguments.session_root,
        arguments.output,
        hash_videos=not arguments.skip_video_hash,
    )
    print(result["confidence_thresholds"])


if __name__ == "__main__":
    main()
