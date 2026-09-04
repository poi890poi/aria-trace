"""Build the standard real-time cursor-pose control report.

This is a post-processing tool.  It does not rerun estimators and does not
change production tracking.  It deliberately distinguishes estimator-only
deadline evidence from unmeasured capture-to-publication latency.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


TARGET_HZ = 30.0
MINIMUM_HZ = 15.0
TARGET_DEADLINE_MS = 1000.0 / TARGET_HZ
MINIMUM_DEADLINE_MS = 1000.0 / MINIMUM_HZ

# These are production scheduler settings, not measured throughput.
PRODUCTION_INTERVAL_SECONDS = {
    "realtime_cascade_ambiguous": 0.05,
    "fast_cascade_minimal": 0.10,
    "accurate_vectorized_full": 0.15,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percent(value) -> str:
    return "n/a" if value is None else "{:.1f}%".format(100.0 * float(value))


def _number(value, suffix="") -> str:
    return "n/a" if value is None else "{:.2f}{}".format(float(value), suffix)


def _bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _timing_summary(values_ms) -> dict:
    values = np.asarray(list(values_ms), dtype=np.float64)
    if not len(values):
        return {"sample_count": 0, "mean": None, "median": None, "p95": None, "worst": None}
    return {
        "sample_count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "worst": float(np.max(values)),
    }


def _source_timing(frames_path: Path) -> dict:
    timestamps = []
    with frames_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("stream_id", "main") == "main":
                timestamps.append(int(row["session_time_ns"]))
    timestamps.sort()
    intervals_ms = np.diff(np.asarray(timestamps, dtype=np.int64)) / 1.0e6
    duration_s = (timestamps[-1] - timestamps[0]) / 1.0e9 if len(timestamps) > 1 else 0.0
    supplied_fps = (len(timestamps) - 1) / duration_s if duration_s > 0.0 else 0.0
    return {
        "frame_count": len(timestamps),
        "duration_seconds": duration_s,
        "supplied_fps": supplied_fps,
        "frame_interval_ms": _timing_summary(intervals_ms),
        "source_interval_within_33_3ms_rate": (
            float(np.mean(intervals_ms <= TARGET_DEADLINE_MS)) if len(intervals_ms) else None
        ),
        "source_interval_within_66_7ms_rate": (
            float(np.mean(intervals_ms <= MINIMUM_DEADLINE_MS)) if len(intervals_ms) else None
        ),
        "can_evaluate_30_fps": supplied_fps >= 29.5,
        "can_evaluate_15_fps": supplied_fps >= 15.0,
    }


def _aggregate_lookup(source: dict) -> dict:
    return {
        (
            row["profile"],
            row["confidence_mode"],
            row["fallback_strategy"],
            row["outage_scenario"],
        ): row
        for row in source["aggregate"]
        if row["split"] == "holdout"
    }


def _candidate_summaries(rows, holdout_sessions, aggregate_lookup) -> list[dict]:
    selected = [
        row
        for row in rows
        if row["session"] in holdout_sessions and row["confidence_mode"] == "selected"
    ]
    output = []
    for profile in sorted({row["profile"] for row in selected}):
        profile_rows = [row for row in selected if row["profile"] == profile]
        latency_ms = np.asarray(
            [float(row["primary_latency_ns"]) / 1.0e6 for row in profile_rows],
            dtype=np.float64,
        )
        accepted = np.asarray(
            [_bool(row["natural_primary_accepted"]) for row in profile_rows],
            dtype=bool,
        )
        interval_s = PRODUCTION_INTERVAL_SECONDS.get(profile)
        configured_cap_hz = 1.0 / interval_s if interval_s else None
        chronological = aggregate_lookup[
            (profile, "selected", "reuse_previous_state", "natural")
        ]
        output.append(
            {
                "profile": profile,
                "attempt_count": len(profile_rows),
                "fresh_measurement_accepted_rate": float(np.mean(accepted)),
                "estimator_latency_ms": _timing_summary(latency_ms),
                "estimator_fresh_within_33_3ms_rate": float(
                    np.mean(accepted & (latency_ms <= TARGET_DEADLINE_MS))
                ),
                "estimator_fresh_within_66_7ms_rate": float(
                    np.mean(accepted & (latency_ms <= MINIMUM_DEADLINE_MS))
                ),
                "production_cursor_interval_seconds": interval_s,
                "production_configured_rate_cap_hz": configured_cap_hz,
                "configured_30_fps_possible": (
                    None if configured_cap_hz is None else configured_cap_hz >= TARGET_HZ
                ),
                "configured_15_fps_possible": (
                    None if configured_cap_hz is None else configured_cap_hz >= MINIMUM_HZ
                ),
                "chronological_functional_e2e_error_deg": chronological[
                    "e2e_absolute_error_deg"
                ],
                "chronological_held_rate": chronological["output_provenance_rate"][
                    "held"
                ],
            }
        )
    return output


def _confidence_comparison(aggregate_lookup, profiles) -> list[dict]:
    output = []
    for profile in profiles:
        for mode in ("selected", "accept_all"):
            row = aggregate_lookup[
                (profile, mode, "reuse_previous_state", "natural")
            ]
            output.append(
                {
                    "profile": profile,
                    "confidence_mode": mode,
                    "fresh_measurement_accepted_rate": row[
                        "primary_measurement_accepted_rate"
                    ],
                    "held_rate": row["output_provenance_rate"]["held"],
                    "functional_e2e_error_deg": row["e2e_absolute_error_deg"],
                }
            )
    return output


def _fallback_comparison(aggregate_lookup, profiles) -> list[dict]:
    output = []
    for profile in profiles:
        for fallback in (
            "reuse_previous_state",
            "constant_velocity_last_2_accepted",
            "no_fallback",
        ):
            row = aggregate_lookup[
                (profile, "selected", fallback, "three_frame_burst_every_90")
            ]
            output.append(
                {
                    "profile": profile,
                    "fallback_strategy": fallback,
                    "fresh_measurement_accepted_rate": row[
                        "primary_measurement_accepted_rate"
                    ],
                    "held_rate": row["output_provenance_rate"]["held"],
                    "predicted_rate": row["output_provenance_rate"]["predicted"],
                    "unavailable_rate": row["output_provenance_rate"]["unavailable"],
                    "functional_e2e_error_deg": row["e2e_absolute_error_deg"],
                }
            )
    return output


def _decision(candidate: dict) -> str:
    cap = candidate["production_configured_rate_cap_hz"]
    if cap is not None and cap < MINIMUM_HZ:
        return "REJECT FOR LIVE CONTROL: configured below 15 FPS"
    if cap is not None and cap < TARGET_HZ:
        return "BASELINE ONLY: can target 15 FPS, cannot reach 30 FPS as configured"
    if candidate["estimator_fresh_within_33_3ms_rate"] < 0.95:
        return "RETAIN FOR EVIDENCE: estimator misses the 30 FPS compute budget"
    return "RETAIN FOR EVIDENCE: compute is promising; full pipeline is unmeasured"


def _report_lines(result: dict) -> list[str]:
    lines = [
        "CURSOR POSE REAL-TIME CONTROL BENCHMARK",
        "",
        "PREMISES",
        "",
        "Target: 30 fresh pose fixes per second.",
        "Target deadline: 33.3 ms from frame capture to pose publication.",
        "Minimum: 15 fresh pose fixes per second.",
        "Minimum deadline: 66.7 ms from frame capture to pose publication.",
        "Held and predicted states are continuity outputs, not fresh fixes.",
        "A non-null stale pose is not counted as real-time availability.",
        "",
        "EVIDENCE BOUNDARY",
        "",
        "Measured here: source frame timing, estimator latency, confidence acceptance, and travel-reference error from the source benchmark.",
        "Not measured here: live capture, decode, crop, IPC, scheduling, fusion, rendering, and publication latency.",
        "Therefore estimator deadline rates are necessary evidence, not complete end-to-end proof.",
        "",
        "SOURCE TIMING",
        "",
    ]
    for session, timing in result["source_timing"].items():
        lines.extend(
            [
                session,
                "Supplied  {} FPS".format(_number(timing["supplied_fps"])),
                "Interval median / P95 / worst  {} / {} / {}".format(
                    _number(timing["frame_interval_ms"]["median"], " ms"),
                    _number(timing["frame_interval_ms"]["p95"], " ms"),
                    _number(timing["frame_interval_ms"]["worst"], " ms"),
                ),
                "Frame gaps within 33.3 / 66.7 ms  {} / {}".format(
                    _percent(timing["source_interval_within_33_3ms_rate"]),
                    _percent(timing["source_interval_within_66_7ms_rate"]),
                ),
                "Average-rate eligibility for 30 FPS evaluation  {}".format(
                    "YES" if timing["can_evaluate_30_fps"] else "NO"
                ),
                "Average-rate eligibility for 15 FPS evaluation  {}".format(
                    "YES" if timing["can_evaluate_15_fps"] else "NO"
                ),
                "",
            ]
        )
    lines.extend(["CANDIDATE RESULTS", ""])
    for row in result["candidates"]:
        latency = row["estimator_latency_ms"]
        error = row["chronological_functional_e2e_error_deg"]
        cap = row["production_configured_rate_cap_hz"]
        lines.extend(
            [
                row["profile"],
                "Decision  {}".format(row["decision"]),
                "Fresh accepted  {}".format(_percent(row["fresh_measurement_accepted_rate"])),
                "Fresh within estimator 33.3 ms  {}".format(
                    _percent(row["estimator_fresh_within_33_3ms_rate"])
                ),
                "Fresh within estimator 66.7 ms  {}".format(
                    _percent(row["estimator_fresh_within_66_7ms_rate"])
                ),
                "Estimator latency median / P95 / worst  {} / {} / {}".format(
                    _number(latency["median"], " ms"),
                    _number(latency["p95"], " ms"),
                    _number(latency["worst"], " ms"),
                ),
                "Production configured cap  {}".format(
                    "not integrated" if cap is None else _number(cap, " FPS")
                ),
                "Functional error mean / median / P95 / worst  {} / {} / {} / {}".format(
                    _number(error["mean"], " deg"),
                    _number(error["median"], " deg"),
                    _number(error["p95"], " deg"),
                    _number(error["worst"], " deg"),
                ),
                "Held by natural confidence rejection  {}".format(
                    _percent(row["chronological_held_rate"])
                ),
                "",
            ]
        )
    lines.extend(["CONFIDENCE COMPARISON", ""])
    for row in result["confidence_comparison"]:
        error = row["functional_e2e_error_deg"]
        lines.extend(
            [
                "{}  {}".format(row["profile"], row["confidence_mode"]),
                "Fresh accepted  {}".format(
                    _percent(row["fresh_measurement_accepted_rate"])
                ),
                "Held  {}".format(_percent(row["held_rate"])),
                "Functional error P95 / worst  {} / {}".format(
                    _number(error["p95"], " deg"),
                    _number(error["worst"], " deg"),
                ),
                "",
            ]
        )
    lines.extend(["FALLBACK COMPARISON", ""])
    lines.extend(
        [
            "The same deterministic three-frame outage is injected every 90 frames.",
            "Fallback never changes the fresh-fix rate; it changes only continuity output.",
            "",
        ]
    )
    for row in result["fallback_comparison"]:
        error = row["functional_e2e_error_deg"]
        lines.extend(
            [
                "{}  {}".format(row["profile"], row["fallback_strategy"]),
                "Fresh accepted  {}".format(
                    _percent(row["fresh_measurement_accepted_rate"])
                ),
                "Held / predicted / unavailable  {} / {} / {}".format(
                    _percent(row["held_rate"]),
                    _percent(row["predicted_rate"]),
                    _percent(row["unavailable_rate"]),
                ),
                "Functional error P95 / worst  {} / {}".format(
                    _number(error["p95"], " deg"),
                    _number(error["worst"], " deg"),
                ),
                "",
            ]
        )
    lines.extend(
        [
            "CONFIDENCE AND FALLBACK",
            "",
            "Confidence rejection reduces fresh-fix availability. It must earn its cost by reducing independently measured outliers.",
            "Hold and prediction may maintain a control-state stream, but neither repairs missed visual deadlines.",
            "Fallback accuracy remains reported by the source benchmark; it is not promoted to fresh availability here.",
            "",
            "CURRENT RECOMMENDATIONS",
            "",
            "1. Instrument capture-to-publication timestamps and stage latency in the live tracker.",
            "2. Benchmark the complete latest-frame-wins pipeline at 30 Hz and 15 Hz.",
            "3. Do not select a production estimator from estimator-only timing.",
            "4. Reject fast and accurate production profiles for live control at their current cadence caps.",
            "5. Retain the real-time profile only as the current 15 FPS baseline; it cannot satisfy 30 FPS as configured.",
            "6. Continue angular and symmetric-pixel candidates only with pose-labelled turning evidence and complete-pipeline timing.",
            "",
            "LIMITATIONS",
            "",
            "The travel direction reference is functional end-to-end evidence, not per-frame pixel pose truth.",
            "Runs below 29.5 captured FPS cannot verify a 30 FPS control target.",
            "Offline estimator timing does not include live process and publication overhead.",
            "No production behavior was changed.",
            "",
            "TRACEABILITY",
            "",
            "Source results  {}".format(result["source_results"]["path"]),
            "Source SHA-256  {}".format(result["source_results"]["sha256"]),
            "Primary measurements  {}".format(result["primary_measurements"]["path"]),
            "Primary SHA-256  {}".format(result["primary_measurements"]["sha256"]),
            "Generated machine report  realtime_control_results.json",
            "",
        ]
    )
    return lines


def build(benchmark_path: Path, session_root: Path, output_path: Path) -> dict:
    benchmark_path = Path(benchmark_path).resolve()
    session_root = Path(session_root).resolve()
    output_path = Path(output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    results_path = benchmark_path / "results.json"
    measurements_path = benchmark_path / "primary_measurements.csv"
    source = json.loads(results_path.read_text(encoding="utf-8"))
    with measurements_path.open("r", encoding="utf-8", newline="") as stream:
        measurements = list(csv.DictReader(stream))
    holdout_sessions = set(source["holdout_sessions"])
    aggregate_lookup = _aggregate_lookup(source)
    candidates = _candidate_summaries(
        measurements, holdout_sessions, aggregate_lookup
    )
    for candidate in candidates:
        candidate["decision"] = _decision(candidate)
    result = {
        "schema_version": "1.0",
        "benchmark": "cursor_pose_realtime_control_report",
        "premises": {
            "target_fresh_pose_hz": TARGET_HZ,
            "minimum_fresh_pose_hz": MINIMUM_HZ,
            "target_capture_to_publication_deadline_ms": TARGET_DEADLINE_MS,
            "minimum_capture_to_publication_deadline_ms": MINIMUM_DEADLINE_MS,
            "held_or_predicted_counts_as_fresh": False,
        },
        "measurement_scope": "offline estimator timing only; not live capture-to-publication",
        "source_timing": {
            session: _source_timing(session_root / session / "frames.jsonl")
            for session in source["sessions"]
        },
        "candidates": candidates,
        "confidence_comparison": _confidence_comparison(
            aggregate_lookup,
            (
                "realtime_cascade_ambiguous",
                "angular_projection_ncc_parabolic",
                "symmetric_pixel_fft_ncc_parabolic",
            ),
        ),
        "fallback_comparison": _fallback_comparison(
            aggregate_lookup,
            (
                "realtime_cascade_ambiguous",
                "angular_projection_ncc_parabolic",
                "symmetric_pixel_fft_ncc_parabolic",
            ),
        ),
        "source_results": {"path": str(results_path), "sha256": _sha256(results_path)},
        "primary_measurements": {
            "path": str(measurements_path),
            "sha256": _sha256(measurements_path),
        },
    }
    (output_path / "realtime_control_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output_path / "REPORT.txt").write_text(
        "\n".join(_report_lines(result)), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("session_root", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    result = build(arguments.benchmark, arguments.session_root, arguments.output)
    print(json.dumps(result["premises"], indent=2))


if __name__ == "__main__":
    main()
