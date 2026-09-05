"""Aggregate causal localization replays into a phone-readable control report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


TARGET_DEADLINE_MS = 1000.0 / 30.0
MINIMUM_DEADLINE_MS = 1000.0 / 15.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values) -> dict:
    rows = np.asarray(list(values), dtype=np.float64)
    if not len(rows):
        return {
            "sample_count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "worst": None,
        }
    return {
        "sample_count": int(len(rows)),
        "mean": float(np.mean(rows)),
        "median": float(np.median(rows)),
        "p95": float(np.percentile(rows, 95)),
        "worst": float(np.max(rows)),
    }


def _percent(value) -> str:
    return "n/a" if value is None else "{:.1f}%".format(100.0 * float(value))


def _number(value, suffix="") -> str:
    return "n/a" if value is None else "{:.2f}{}".format(float(value), suffix)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _candidate_name(root: Path, report_path: Path) -> str:
    relative = report_path.parent.relative_to(root)
    if len(relative.parts) < 2:
        raise ValueError(
            "Expected <root>/<candidate>/<replay>/report.json, got {}".format(
                report_path
            )
        )
    return relative.parts[0]


def _aggregate_candidate(name: str, runs: list[dict]) -> dict:
    rows = [
        row
        for run in runs
        for row in run["rows"]
        if not row.get("initialization_frame", False)
    ]
    fresh = np.asarray([bool(row["measurement_accepted"]) for row in rows])
    candidate = np.asarray(
        [bool(row.get("primary_candidate_produced", True)) for row in rows]
    )
    gate_rejected = np.asarray(
        [bool(row.get("final_gate_rejected", False)) for row in rows]
    )
    latency = np.asarray(
        [float(row["localization_core_elapsed_ms"]) for row in rows]
    )
    serial_latency = np.asarray(
        [
            float(
                row.get(
                    "end_to_end_serial_elapsed_ms",
                    row["localization_core_elapsed_ms"],
                )
            )
            for row in rows
        ]
    )
    fresh_errors = [
        float(row["reference_error_px"])
        for row in rows
        if row.get("reference_error_px") is not None
        and row["measurement_accepted"]
    ]
    served_errors = [
        float(row["reference_error_px"])
        for row in rows
        if row.get("reference_error_px") is not None and row["valid"]
    ]
    init_times = [
        float(run["report"]["initialization"]["elapsed_ms"])
        for run in runs
        if run["report"].get("initialization")
    ]
    per_replay = []
    for run in runs:
        replay_rows = [
            row
            for row in run["rows"]
            if not row.get("initialization_frame", False)
        ]
        replay_fresh = np.asarray(
            [bool(row["measurement_accepted"]) for row in replay_rows]
        )
        replay_serial = np.asarray(
            [
                float(
                    row.get(
                        "end_to_end_serial_elapsed_ms",
                        row["localization_core_elapsed_ms"],
                    )
                )
                for row in replay_rows
            ]
        )
        per_replay.append(
            {
                "session_id": run["report"]["session"]["session_id"],
                "fresh_measurement_accepted_rate": float(np.mean(replay_fresh)),
                "fresh_serial_within_33_3ms_rate": float(
                    np.mean(replay_fresh & (replay_serial <= TARGET_DEADLINE_MS))
                ),
                "serial_decode_to_xy_latency_ms": _summary(replay_serial),
                "reference_role": (
                    (run["report"].get("reference_pose_error") or {}).get("role")
                ),
            }
        )
    result = {
        "candidate": name,
        "replay_count": len(runs),
        "attempt_count": len(rows),
        "primary_candidate_produced_rate": float(np.mean(candidate)),
        "fresh_measurement_accepted_rate": float(np.mean(fresh)),
        "final_gate_rejection_rate": float(np.mean(gate_rejected)),
        "held_output_rate": float(np.mean(~fresh)),
        "fresh_within_33_3ms_rate": float(
            np.mean(fresh & (latency <= TARGET_DEADLINE_MS))
        ),
        "fresh_within_66_7ms_rate": float(
            np.mean(fresh & (latency <= MINIMUM_DEADLINE_MS))
        ),
        "fresh_serial_within_33_3ms_rate": float(
            np.mean(fresh & (serial_latency <= TARGET_DEADLINE_MS))
        ),
        "fresh_serial_within_66_7ms_rate": float(
            np.mean(fresh & (serial_latency <= MINIMUM_DEADLINE_MS))
        ),
        "localization_core_latency_ms": _summary(latency),
        "serial_decode_to_xy_latency_ms": _summary(serial_latency),
        "fresh_reference_error_px": _summary(fresh_errors),
        "served_e2e_reference_error_px": _summary(served_errors),
        "initialization_latency_ms": _summary(init_times),
        "sessions": [run["report"]["session"]["session_id"] for run in runs],
        "parameters": runs[0]["report"]["parameters"],
        "methods": [run["report"]["method_traceability"] for run in runs],
        "per_replay": per_replay,
        "worst_replay_fresh_rate": min(
            item["fresh_measurement_accepted_rate"] for item in per_replay
        ),
        "worst_replay_fresh_serial_within_33_3ms_rate": min(
            item["fresh_serial_within_33_3ms_rate"] for item in per_replay
        ),
    }
    result["meets_30fps_compute_and_95pct_fresh"] = bool(
        result["fresh_within_33_3ms_rate"] >= 0.95
    )
    result["meets_15fps_compute_and_95pct_fresh"] = bool(
        result["fresh_within_66_7ms_rate"] >= 0.95
    )
    result["meets_recorded_video_serial_30fps"] = bool(
        result["worst_replay_fresh_serial_within_33_3ms_rate"] >= 0.95
    )
    return result


def _recommend(candidates: list[dict]) -> tuple[str | None, str]:
    eligible = [
        row
        for row in candidates
        if row["meets_30fps_compute_and_95pct_fresh"]
        and row["meets_recorded_video_serial_30fps"]
        and row["worst_replay_fresh_rate"] >= 0.95
        and row["fresh_reference_error_px"]["sample_count"]
    ]
    if not eligible:
        return None, (
            "No candidate has both at least 95% fresh 33.3 ms completion and "
            "independent reference coverage."
        )
    selected = min(
        eligible,
        key=lambda row: (
            row["served_e2e_reference_error_px"]["worst"],
            row["served_e2e_reference_error_px"]["p95"],
            row["localization_core_latency_ms"]["p95"],
        ),
    )
    return selected["candidate"], (
        "Best worst-case chronological reference error among candidates that "
        "deliver at least 95% fresh fixes inside the 30 FPS compute budget."
    )


def _report_lines(result: dict) -> list[str]:
    lines = [
        "LOCALIZATION REAL-TIME CONTROL BENCHMARK",
        "",
        "PREMISES",
        "",
        "Target: 30 fresh XY fixes per second.",
        "Target compute deadline: 33.3 ms from extracted frame to XY fix.",
        "Recorded-video E2E: serial decode -> mini-map extraction -> XY publication candidate.",
        "Minimum: 15 fresh XY fixes per second; 66.7 ms compute deadline.",
        "Held positions are control continuity, not fresh localization.",
        "Initial absolute localization is measured separately from the high-rate core.",
        "",
        "PIPELINE UNDER TEST",
        "",
        "One initialization fix -> one current-frame local map matcher -> one physical-continuity gate -> publish fresh or hold.",
        "No per-frame descriptor, route, global-search, filter, or prediction fallback is allowed in the two-layer candidates.",
        "A demonstrated route may propose the initial search region; it never supplies the measured XY.",
        "",
        "EVIDENCE BOUNDARY",
        "",
        "Measured: serial recorded-video decode through XY result, fresh acceptance, gate rejection, and chronological map-reference error.",
        "Excluded: camera/GDI capture age, worker scheduling, yaw fusion, overlay rendering, IPC, and consumer publication.",
        "The saved sparse map reference is post-run evidence and never feeds the tested estimator.",
        "",
        "CANDIDATES",
        "",
    ]
    for row in result["candidates"]:
        latency = row["localization_core_latency_ms"]
        serial_latency = row["serial_decode_to_xy_latency_ms"]
        fresh_error = row["fresh_reference_error_px"]
        served_error = row["served_e2e_reference_error_px"]
        lines.extend(
            [
                row["candidate"],
                "Fresh accepted  {}".format(
                    _percent(row["fresh_measurement_accepted_rate"])
                ),
                "Worst replay fresh  {}".format(
                    _percent(row["worst_replay_fresh_rate"])
                ),
                "Fresh within 33.3 / 66.7 ms  {} / {}".format(
                    _percent(row["fresh_within_33_3ms_rate"]),
                    _percent(row["fresh_within_66_7ms_rate"]),
                ),
                "Core latency median / P95 / worst  {} / {} / {}".format(
                    _number(latency["median"], " ms"),
                    _number(latency["p95"], " ms"),
                    _number(latency["worst"], " ms"),
                ),
                "Serial decode-to-XY median / P95 / worst  {} / {} / {}".format(
                    _number(serial_latency["median"], " ms"),
                    _number(serial_latency["p95"], " ms"),
                    _number(serial_latency["worst"], " ms"),
                ),
                "Worst replay fresh inside 33.3 ms  {}".format(
                    _percent(row["worst_replay_fresh_serial_within_33_3ms_rate"])
                ),
                "Layer-1 candidate produced  {}".format(
                    _percent(row["primary_candidate_produced_rate"])
                ),
                "Final-gate rejected  {}".format(
                    _percent(row["final_gate_rejection_rate"])
                ),
                "Fresh reference error mean / median / P95  {} / {} / {}".format(
                    _number(fresh_error["mean"], " px"),
                    _number(fresh_error["median"], " px"),
                    _number(fresh_error["p95"], " px"),
                ),
                "Chronological E2E error mean / median / P95 / worst  {} / {} / {} / {}".format(
                    _number(served_error["mean"], " px"),
                    _number(served_error["median"], " px"),
                    _number(served_error["p95"], " px"),
                    _number(served_error["worst"], " px"),
                ),
                "30 FPS evidence  {}".format(
                    "PASS"
                    if row["meets_30fps_compute_and_95pct_fresh"]
                    else "FAIL"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "CURRENT PENDING RECOMMENDATION",
            "",
            (
                "Candidate  {}".format(result["recommendation"]["candidate"])
                if result["recommendation"]["candidate"]
                else "Candidate  NONE"
            ),
            "Reason  {}".format(result["recommendation"]["reason"]),
            "Production remains unchanged until camera/GDI capture-to-consumer publication is measured in the live tracker.",
            "",
            "TRACEABILITY",
            "",
        ]
    )
    for source in result["source_reports"]:
        lines.append("{}  {}".format(source["sha256"], source["path"]))
    lines.extend(
        [
            "",
            "Machine report  localization_realtime_control_results.json",
            "",
        ]
    )
    return lines


def build(input_root: Path, output_path: Path) -> dict:
    input_root = Path(input_root).resolve()
    output_path = Path(output_path).resolve()
    reports = sorted(input_root.glob("*/*/report.json"))
    if not reports:
        raise ValueError("No <candidate>/<replay>/report.json files under {}".format(input_root))
    grouped: dict[str, list[dict]] = {}
    sources = []
    for report_path in reports:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        telemetry_path = report_path.parent / report["rows_file"]
        rows = _read_jsonl(telemetry_path)
        name = _candidate_name(input_root, report_path)
        grouped.setdefault(name, []).append({"report": report, "rows": rows})
        sources.extend(
            [
                {"path": str(report_path), "sha256": _sha256(report_path)},
                {"path": str(telemetry_path), "sha256": _sha256(telemetry_path)},
            ]
        )
    candidates = [
        _aggregate_candidate(name, grouped[name]) for name in sorted(grouped)
    ]
    candidate, reason = _recommend(candidates)
    result = {
        "schema_version": "1.0",
        "benchmark": "localization_realtime_control",
        "premises": {
            "target_fresh_fix_hz": 30.0,
            "minimum_fresh_fix_hz": 15.0,
            "target_compute_deadline_ms": TARGET_DEADLINE_MS,
            "minimum_compute_deadline_ms": MINIMUM_DEADLINE_MS,
            "held_counts_as_fresh": False,
        },
        "measurement_scope": "serial recorded-video decode through XY result",
        "candidates": candidates,
        "recommendation": {"candidate": candidate, "reason": reason},
        "source_reports": sources,
    }
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "localization_realtime_control_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output_path / "REPORT.txt").write_text(
        "\n".join(_report_lines(result)), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.input_root, args.output), indent=2))


if __name__ == "__main__":
    main()
