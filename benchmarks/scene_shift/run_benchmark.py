"""Benchmark fast whole-scene yaw candidates at recorded 30 FPS cadence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from benchmarks.build_turn_evidence import _mouse_signal
from benchmarks.scene_shift.methods import KltSceneYaw, PhaseSceneYaw, SceneYawResult
from benchmarks.temporal_turns import evaluate_reversal_response
from rig_runtime.adapters.filesystem.session import SessionReader


EXCLUDED_RECTS = (
    (0.0, 0.0, 0.24, 0.30),
    (0.72, 0.0, 1.0, 0.28),
    (0.0, 0.76, 1.0, 1.0),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values, percentile):
    return float(np.percentile(values, percentile)) if values else None


def _method_catalog():
    def klt(width, corners=800, fb=True, essential=True):
        return lambda: KltSceneYaw(
            maximum_width=width,
            max_corners=corners,
            forward_backward=fb,
            essential_gate=essential,
            excluded_rects=EXCLUDED_RECTS,
        )

    def phase(width, signal):
        return lambda: PhaseSceneYaw(
            maximum_width=width,
            signal=signal,
            excluded_rects=EXCLUDED_RECTS,
        )

    return {
        "klt_w960_c800_fb_essential": (klt(960), 1),
        "klt_w640_c800_fb_essential": (klt(640), 1),
        "klt_w480_c800_fb_essential": (klt(480), 1),
        "klt_w320_c800_fb_essential": (klt(320), 1),
        "klt_w240_c800_fb_essential": (klt(240), 1),
        "klt_w480_c400_fb_essential": (klt(480, 400), 1),
        "klt_w480_c200_fb_essential": (klt(480, 200), 1),
        "klt_w480_c100_fb_essential": (klt(480, 100), 1),
        "klt_w480_c800_fb_noessential": (klt(480, 800, True, False), 1),
        "klt_w480_c800_forward_essential": (klt(480, 800, False, True), 1),
        "phase_gray_w480": (phase(480, "gray"), 1),
        "phase_gray_w320": (phase(320, "gray"), 1),
        "phase_gray_w240": (phase(240, "gray"), 1),
        "phase_gradient_w480": (phase(480, "gradient"), 1),
        "phase_gradient_w320": (phase(320, "gradient"), 1),
        "phase_gradient_w240": (phase(240, "gradient"), 1),
        "klt_w480_c400_stride2": (klt(480, 400), 2),
    }


def _git_revision():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def _candidate_summary(rows, reference_rows, input_rows, sample_hz):
    transition_count = max(0, len(rows) - 1)
    ok_indexes = [
        index for index, row in enumerate(rows) if index and row["status"] == "ok"
    ]
    errors = []
    cumulative = 0.0
    cumulative_abs = []
    overlap = 0
    signal = []
    last_ok_time = None
    for index in ok_indexes:
        row = rows[index]
        start = int(row["span_start_index"])
        reference_span = reference_rows[start + 1 : index + 1]
        if not reference_span or any(item["status"] != "ok" for item in reference_span):
            continue
        reference_delta = float(sum(item["delta_deg"] for item in reference_span))
        error = float(row["delta_deg"] - reference_delta)
        errors.append(abs(error))
        cumulative += error
        cumulative_abs.append(abs(cumulative))
        overlap += 1
        elapsed_s = (
            (row["session_time_ns"] - last_ok_time) / 1.0e9
            if last_ok_time is not None
            else None
        )
        signal.append(
            {
                "session_time_ns": row["session_time_ns"],
                "value": (
                    row["delta_deg"] / max(elapsed_s, 1.0e-6)
                    if elapsed_s is not None
                    else None
                ),
            }
        )
        last_ok_time = row["session_time_ns"]
    signal = [row for row in signal if row["value"] is not None]
    elapsed = [float(rows[index]["elapsed_ms"]) for index in ok_indexes]
    fresh_33 = sum(rows[index]["elapsed_ms"] <= 33.3 for index in ok_indexes)
    fresh_66 = sum(rows[index]["elapsed_ms"] <= 66.7 for index in ok_indexes)
    turn = (
        evaluate_reversal_response(input_rows, signal, alignment_sample_hz=sample_hz)
        if len(signal) >= 20 and len(input_rows) >= 20
        else None
    )
    return {
        "frame_transition_count": transition_count,
        "fresh_count": len(ok_indexes),
        "fresh_coverage": len(ok_indexes) / max(1, transition_count),
        "fresh_within_33_3ms_coverage": fresh_33 / max(1, transition_count),
        "fresh_within_66_7ms_coverage": fresh_66 / max(1, transition_count),
        "latency_ms": {
            "median": _percentile(elapsed, 50),
            "p95": _percentile(elapsed, 95),
            "p99": _percentile(elapsed, 99),
            "worst": max(elapsed) if elapsed else None,
        },
        "reference_overlap_count": overlap,
        "absolute_error_deg_per_processed_interval": {
            "mean": float(np.mean(errors)) if errors else None,
            "median": _percentile(errors, 50),
            "p95": _percentile(errors, 95),
            "worst": max(errors) if errors else None,
        },
        "closure_disagreement_deg": abs(cumulative) if errors else None,
        "maximum_cumulative_disagreement_deg": max(cumulative_abs) if cumulative_abs else None,
        "input_direction_lag_evidence": turn,
    }


def _run_session(name, session_path, factories, maximum_frames=None):
    reader = SessionReader(session_path)
    records = list(reader.frames_by_stream.get("main") or ())
    if maximum_frames is not None:
        records = records[: int(maximum_frames)]
    methods = {key: factory() for key, (factory, _) in factories.items()}
    strides = {key: stride for key, (_, stride) in factories.items()}
    rows = {key: [] for key in factories}
    last_processed = {key: None for key in factories}
    capture = cv2.VideoCapture(str(reader.video_path("main")))
    try:
        for index, record in enumerate(records):
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = int(record["session_time_ns"])
            for key, method in methods.items():
                stride = strides[key]
                if index % stride:
                    result = SceneYawResult(0.0, 0.0, 0, 0, "subsampled_hold", 0.0)
                else:
                    result = method.update(frame)
                item = result.as_dict()
                item.update(
                    {
                        "frame_index": int(record["frame_index"]),
                        "session_time_ns": timestamp,
                        "span_start_index": (
                            index if last_processed[key] is None else last_processed[key]
                        ),
                    }
                )
                if index % stride == 0:
                    last_processed[key] = index
                rows[key].append(item)
    finally:
        capture.release()
    actual_count = min(len(value) for value in rows.values())
    for key in rows:
        rows[key] = rows[key][:actual_count]
    sample_times = [int(record["session_time_ns"]) for record in records[:actual_count]]
    input_rows = _mouse_signal(reader.inputs, sample_times)
    reference_key = "klt_w960_c800_fb_essential"
    summaries = {}
    for key in rows:
        summaries[key] = _candidate_summary(
            rows[key], rows[reference_key], input_rows, 30.0
        )
        summaries[key]["parameters"] = methods[key].parameters()
        summaries[key]["frame_stride"] = strides[key]
    return {
        "name": name,
        "path": str(Path(session_path).resolve()),
        "frame_count": actual_count,
        "requested_fps": 30.0,
        "source_hashes": {
            "manifest": _sha256(Path(session_path) / "manifest.json"),
            "frames": _sha256(Path(session_path) / "frames.jsonl"),
            "inputs": _sha256(Path(session_path) / "inputs.jsonl"),
            "video": _sha256(reader.video_path("main")),
        },
        "summaries": summaries,
        "telemetry": rows,
    }


def _aggregate(sessions, candidates):
    def required(value):
        return float("inf") if value is None else float(value)

    output = {}
    for candidate in candidates:
        items = [session["summaries"][candidate] for session in sessions]
        p95_latency = max(required(item["latency_ms"]["p95"]) for item in items)
        p95_error = max(
            required(item["absolute_error_deg_per_processed_interval"]["p95"])
            for item in items
        )
        worst_error = max(
            required(item["absolute_error_deg_per_processed_interval"]["worst"])
            for item in items
        )
        coverage = min(item["fresh_coverage"] for item in items)
        coverage_33 = min(item["fresh_within_33_3ms_coverage"] for item in items)
        output[candidate] = {
            "worst_session_fresh_coverage": coverage,
            "worst_session_fresh_within_33_3ms_coverage": coverage_33,
            "worst_session_latency_p95_ms": p95_latency,
            "worst_session_reference_error_p95_deg": p95_error,
            "worst_reference_error_deg": worst_error,
            "meets_30fps_core": coverage_33 >= 0.95,
            "meets_reference_consistency": p95_error <= 0.10 and worst_error <= 0.50,
            "parameters": items[0]["parameters"],
            "frame_stride": items[0]["frame_stride"],
        }
    return output


def _report(result):
    lines = [
        "WHOLE-SCENE SHIFT BENCHMARK",
        "",
        "Target: fresh 30 FPS control evidence; P95 <= 33.3 ms.",
        "Accuracy is disagreement from accurate KLT, not external truth.",
        "Raw mouse input checks fitted direction and lag only.",
        "",
    ]
    for name, item in sorted(
        result["aggregate"].items(),
        key=lambda pair: (
            not pair[1]["meets_30fps_core"],
            pair[1]["worst_session_reference_error_p95_deg"],
            pair[1]["worst_session_latency_p95_ms"],
        ),
    ):
        lines.extend(
            [
                name,
                "fresh / <=33ms: {:.1%} / {:.1%}".format(
                    item["worst_session_fresh_coverage"],
                    item["worst_session_fresh_within_33_3ms_coverage"],
                ),
                "latency P95: {:.2f} ms".format(item["worst_session_latency_p95_ms"]),
                "reference error P95 / worst: {:.4f} / {:.4f} deg".format(
                    item["worst_session_reference_error_p95_deg"],
                    item["worst_reference_error_deg"],
                ),
                "decision: {}".format(
                    "supported candidate"
                    if item["meets_30fps_core"] and item["meets_reference_consistency"]
                    else "reject: reference disagreement"
                    if not item["meets_reference_consistency"]
                    else "reject: 30 FPS availability"
                ),
                "",
            ]
        )
    return "\n".join(lines)


def run(session_specs, output_path, candidate_names=(), maximum_frames=None):
    catalog = _method_catalog()
    selected = list(candidate_names) or list(catalog)
    reference = "klt_w960_c800_fb_essential"
    if reference not in selected:
        selected.insert(0, reference)
    unknown = sorted(set(selected) - set(catalog))
    if unknown:
        raise ValueError("Unknown candidates: {}".format(", ".join(unknown)))
    factories = {name: catalog[name] for name in selected}
    sessions = [
        _run_session(name, path, factories, maximum_frames=maximum_frames)
        for name, path in session_specs
    ]
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for session in sessions:
        session_path = output_path / session["name"]
        session_path.mkdir(exist_ok=True)
        for candidate, rows in session.pop("telemetry").items():
            _write_jsonl(session_path / (candidate + ".jsonl"), rows)
    result = {
        "schema_version": "1.0",
        "benchmark": "chronological_whole_scene_yaw_30fps",
        "reference": {
            "method": reference,
            "role": "high-cost_cross-method_reference_not_external_truth",
        },
        "evidence_contract": {
            "raw_input": "control_intent_for_direction_and_lag_not_numeric_truth",
            "reference_klt": "observed_camera_motion_comparison_reference",
            "frame_subsampling": "held_frames_are_not_fresh_and_reduce_coverage",
        },
        "git_revision": _git_revision(),
        "implementation_hashes": {
            "runner": _sha256(Path(__file__)),
            "methods": _sha256(Path(__file__).with_name("methods.py")),
        },
        "maximum_frames": maximum_frames,
        "sessions": sessions,
        "aggregate": _aggregate(sessions, selected),
    }
    (output_path / "report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output_path / "REPORT.txt").write_text(_report(result), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--maximum-frames", type=int)
    parser.add_argument("--list-candidates", action="store_true")
    args = parser.parse_args()
    if args.list_candidates:
        print("\n".join(_method_catalog()))
        return
    specs = []
    for value in args.session:
        if "=" not in value:
            parser.error("--session needs NAME=PATH")
        name, path = value.split("=", 1)
        specs.append((name, Path(path)))
    run(specs, args.output, args.candidate, args.maximum_frames)


if __name__ == "__main__":
    main()
