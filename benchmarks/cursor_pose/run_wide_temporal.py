"""Benchmark promising cursor-pose cores and causal temporal policies.

Forward-only sessions provide absolute direction evidence.  Long repeated-lap
sessions provide 30 Hz latency, availability, cross-method agreement, and turn
response against timestamp-aligned input/KLT channels.  Cross-method agreement
is explicitly diagnostic and is never called ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from benchmarks.cursor_pose.run_candidate_e2e import _candidate_factory
from benchmarks.temporal_turns import evaluate_reversal_response
from rig_runtime.adapters.filesystem.session import SessionReader


CANDIDATES = (
    "angular_projection_ncc_parabolic",
    "symmetric_pixel_fft_ncc_parabolic",
    "polygon_von_mises_moment",
    "analytic_lm_ambiguous",
)
POLICIES = (
    "raw",
    "physical_gate",
    "confidence_hold",
    "schmitt",
    "ema_085",
    "alpha_beta_085_005",
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _delta(value, reference):
    return (float(value) - float(reference) + 180.0) % 360.0 - 180.0


def _circular_mean(values):
    radians = np.radians(values)
    return math.degrees(math.atan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))) % 360.0


def _dist(values):
    values = np.asarray(values, np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p95": None, "worst": None}
    return {"count": int(len(values)), "mean": float(np.mean(values)), "median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "worst": float(np.max(values))}


def _apply(rows, policy, threshold, turn_rate_limit_deg_s=None):
    output, state, velocity, previous_ns, locked = [], None, 0.0, None, False
    for source in rows:
        row = dict(source)
        measured = row.get("angle_deg")
        accepted = measured is not None
        if policy == "confidence_hold":
            accepted = accepted and float(row["confidence"]) >= threshold
        elif policy == "schmitt":
            accepted = accepted and float(row["confidence"]) >= (max(0.0, threshold - 0.08) if locked else threshold)
            locked = bool(accepted)
        physical_rate = None
        if (
            accepted
            and policy == "physical_gate"
            and state is not None
            and previous_ns is not None
        ):
            dt = (int(row["session_time_ns"]) - previous_ns) / 1e9
            if 0.005 <= dt <= 2.0:
                physical_rate = abs(_delta(float(measured), state)) / dt
                accepted = physical_rate <= float(turn_rate_limit_deg_s)
        if accepted:
            measured = float(measured)
            dt = max(1.0 / 30.0, (int(row["session_time_ns"]) - previous_ns) / 1e9) if previous_ns is not None else 1.0 / 30.0
            if state is None or policy in (
                "raw",
                "physical_gate",
                "confidence_hold",
                "schmitt",
            ):
                state = measured
            elif policy == "ema_085":
                state = (state + 0.85 * _delta(measured, state)) % 360.0
            elif policy == "alpha_beta_085_005":
                prediction = state + velocity * dt
                residual = _delta(measured, prediction)
                state = (prediction + 0.85 * residual) % 360.0
                velocity += 0.05 * residual / dt
            previous_ns = int(row["session_time_ns"])
            provenance = "fresh_filtered" if policy.startswith(("ema", "alpha")) else "fresh"
        else:
            provenance = "held" if state is not None else "unavailable"
        row.update(
            output_angle_deg=state,
            output_provenance=provenance,
            measurement_accepted_by_policy=bool(accepted),
            physical_rate_deg_s=physical_rate,
            physical_rate_limit_deg_s=turn_rate_limit_deg_s,
            final_physical_gate_rejected=bool(
                policy == "physical_gate" and measured is not None and not accepted
            ),
        )
        output.append(row)
    return output


def _turn_signal(rows):
    output, previous = [], None
    for row in rows:
        angle = row["output_angle_deg"]
        value = None
        if angle is not None and previous is not None:
            dt = max((int(row["session_time_ns"]) - previous[0]) / 1e9, 1e-6)
            value = _delta(angle, previous[1]) / dt
        output.append({"session_time_ns": int(row["session_time_ns"]), "value": value, "valid": value is not None})
        if angle is not None:
            previous = (int(row["session_time_ns"]), float(angle))
    return output


def _summary(rows, *, reference_key=None, agreement_key=None):
    count = max(len(rows), 1)
    errors = []
    for row in rows:
        angle = row["output_angle_deg"]
        reference = row.get(reference_key) if reference_key else None
        if reference is None and agreement_key:
            reference = row.get(agreement_key)
        if angle is not None and reference is not None:
            errors.append(abs(_delta(angle, reference)))
    fresh = sum(row["output_provenance"].startswith("fresh") for row in rows)
    available = sum(row["output_angle_deg"] is not None for row in rows)
    physical_rejections = sum(
        bool(row.get("final_physical_gate_rejected")) for row in rows
    )
    return {"frames": len(rows), "fresh_rate": fresh / count, "available_rate": available / count, "final_physical_gate_rejection_rate": physical_rejections / count, "error_deg": _dist(errors), "latency_ms": _dist([row["latency_ms"] for row in rows])}


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _raw_cache_identity(args, reader, session_id):
    session = Path(args.session_root) / session_id
    return {
        "schema_version": "1.0",
        "session_id": session_id,
        "candidates": list(CANDIDATES),
        "calibration_sha256": _sha(args.calibration),
        "frames_sha256": _sha(session / "frames.jsonl"),
        "video_sha256": _sha(reader.video_path("main")),
        "estimator_source_sha256": _sha(
            Path(__file__).with_name("run_candidate_e2e.py")
        ),
    }


def _load_raw_cache(cache_root, identity):
    manifest_path = Path(cache_root) / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity:
            return None
        result = {
            name: _read_jsonl(Path(cache_root) / (name + ".jsonl"))
            for name in CANDIDATES
        }
        lengths = {len(rows) for rows in result.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
            return None
        return result
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _write_raw_cache(cache_root, identity, by_method):
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    for name in CANDIDATES:
        path = cache_root / (name + ".jsonl")
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in by_method[name]:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        temporary.replace(path)
    manifest_path = cache_root / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "identity": identity,
                "row_count": len(by_method[CANDIDATES[0]]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(manifest_path)


def run(args):
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    prior = json.loads(args.prior_results.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    envelope = (
        calibration.get("cursor_temporal_dynamics") or {}
    ).get("recommended_runtime_envelope") or {}
    turn_rate_limit = float(
        envelope.get("calibrated_turn_rate_p99_deg_s") or 1440.0
    )
    thresholds = prior["confidence_thresholds"]
    wide_raw = {}
    raw_cache = {}
    for session_id in ("run_17", "run_18"):
        reader = SessionReader(args.session_root / session_id)
        records = list(reader.frames_by_stream["main"])
        identity = _raw_cache_identity(args, reader, session_id)
        cache_root = output / "raw" / session_id
        by_method = _load_raw_cache(cache_root, identity)
        if by_method is None:
            estimators = {name: _candidate_factory(args.calibration, name) for name in CANDIDATES}
            capture = cv2.VideoCapture(str(reader.video_path("main")))
            by_method = {name: [] for name in CANDIDATES}
            try:
                for ordinal, record in enumerate(records):
                    ok, frame = capture.read()
                    if not ok: break
                    for name, estimator in estimators.items():
                        started = time.perf_counter_ns(); result = estimator.estimate(frame)
                        elapsed = (time.perf_counter_ns() - started) / 1e6
                        angle = result.get("angle_screen_deg") if result.get("detected") else None
                        by_method[name].append({"frame_index": int(record["frame_index"]), "session_time_ns": int(record["session_time_ns"]), "angle_deg": None if angle is None else float(angle), "confidence": float(result.get("confidence") or 0.0), "latency_ms": elapsed})
            finally:
                capture.release()
            for index in range(min(map(len, by_method.values()))):
                for name in CANDIDATES:
                    others = [by_method[other][index]["angle_deg"] for other in CANDIDATES if other != name and by_method[other][index]["angle_deg"] is not None]
                    by_method[name][index]["leave_one_out_consensus_deg"] = _circular_mean(others) if others else None
            _write_raw_cache(cache_root, identity, by_method)
            raw_cache[session_id] = "miss-written"
        else:
            raw_cache[session_id] = "hit"
        wide_raw[session_id] = by_method

    forward_rows = list(csv.DictReader(args.prior_primary.open(encoding="utf-8")))
    results = []
    for name in CANDIDATES:
        raw_forward = []
        for row in forward_rows:
            if row["profile"] != name: continue
            raw_forward.append({"session": row["session"], "session_time_ns": int(row["session_time_ns"]), "angle_deg": float(row["angle_deg"]) if row["angle_deg"] else None, "confidence": float(row["confidence"]), "latency_ms": int(row["primary_latency_ns"]) / 1e6, "reference_angle_deg": float(row["reference_angle_deg"]) if row["reference_angle_deg"] else None})
        for policy in POLICIES:
            forward_filtered = []
            for session in sorted(set(row["session"] for row in raw_forward)):
                forward_filtered.extend(_apply([row for row in raw_forward if row["session"] == session], policy, float(thresholds[name]), turn_rate_limit))
            entry = {"candidate": name, "temporal_policy": policy, "forward_absolute": _summary(forward_filtered, reference_key="reference_angle_deg"), "wide_sessions": {}}
            for session_id in ("run_17", "run_18"):
                filtered = _apply(wide_raw[session_id][name], policy, float(thresholds[name]), turn_rate_limit)
                input_rows = _read_jsonl(args.turn_root / session_id.replace("_", "") / "input_turn_signal.jsonl")
                scene_rows = _read_jsonl(args.turn_root / session_id.replace("_", "") / "scene_klt_turn_signal.jsonl")
                signal = [row for row in _turn_signal(filtered) if row["valid"]]
                entry["wide_sessions"][session_id] = {
                    "agreement_diagnostic": _summary(filtered, agreement_key="leave_one_out_consensus_deg"),
                    "input_turn_response": evaluate_reversal_response(input_rows, signal, alignment_sample_hz=30.0),
                    "scene_turn_response": evaluate_reversal_response(scene_rows, signal, alignment_sample_hz=30.0),
                }
            results.append(entry)
    result = {"schema_version": "1.0", "generated_utc": datetime.now(timezone.utc).isoformat(), "contract": {"target": "30 fresh pose fixes/s; estimator <=33.3ms", "forward_absolute_reference": "forward-only whole-session E2E direction", "wide_accuracy_reference": "leave-one-out cross-method agreement only; not truth", "input_and_klt": "evaluation-only temporal response evidence", "physical_gate": "calibration-derived p99 turning envelope; rejected frames hold and are not fresh", "turn_rate_limit_deg_s": turn_rate_limit}, "results": results, "traceability": {"benchmark_sha256": _sha(__file__), "prior_results_sha256": _sha(args.prior_results), "prior_primary_sha256": _sha(args.prior_primary), "raw_measurement_cache": raw_cache}}
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-root", type=Path, required=True); p.add_argument("--turn-root", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True); p.add_argument("--prior-results", type=Path, required=True)
    p.add_argument("--prior-primary", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__": main()
