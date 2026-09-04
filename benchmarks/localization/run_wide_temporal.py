"""Wide-session local-localization and causal temporal-policy benchmark.

The evaluator supplies one known starting pose, then never feeds another
reference position to the tracker.  This isolates the high-rate continuation
problem from cold-start global localization.  Sparse reference states are used
only after each run to score the output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from poc.benchmark_route_tracer import benchmark_session
from replay.route_tracking import RouteTrackingPackage


CANDIDATES = {
    "gradient8_ccorr": ("gradient", "ccorr_normed", 8.0),
    "gradient12_ccorr": ("gradient", "ccorr_normed", 12.0),
    "gradient18_ccorr": ("gradient", "ccorr_normed", 18.0),
    "intensity12_ccorr": ("intensity", "ccorr_normed", 12.0),
    "canny12_ccorr": ("canny", "ccorr_normed", 12.0),
    "laplacian12_ccorr": ("laplacian", "ccorr_normed", 12.0),
    "gradient12_phase": ("gradient", "phase_correlation", 12.0),
}
TEMPORAL_POLICIES = (
    "raw",
    "hold_below_050",
    "schmitt_052_046",
    "ema_085",
    "alpha_beta_085_005",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ("git", "-C", str(root), *args), check=True, capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _reference_at(package: RouteTrackingPackage, timestamp: int):
    states = package.states
    times = np.asarray([int(row["session_time_ns"]) for row in states])
    index = int(np.searchsorted(times, timestamp))
    if index < len(times) and times[index] == timestamp:
        return np.asarray(states[index]["canonical_xy"], np.float64)
    if index == 0 or index >= len(times):
        return None
    left, right = states[index - 1], states[index]
    gap = int(right["session_time_ns"]) - int(left["session_time_ns"])
    rate = float(package.manifest.get("reference_rate_hz") or 5.0)
    if gap > 1.5e9 / max(rate, 1.0e-6) or left["mode_id"] != right["mode_id"]:
        return None
    fraction = (timestamp - int(left["session_time_ns"])) / max(gap, 1)
    return np.asarray(left["canonical_xy"], np.float64) + fraction * (
        np.asarray(right["canonical_xy"], np.float64)
        - np.asarray(left["canonical_xy"], np.float64)
    )


def _temporal(rows: list[dict], policy: str) -> list[dict]:
    output = []
    state = None
    velocity = np.zeros(2, np.float64)
    previous_ns = None
    locked = False
    for source in rows:
        row = dict(source)
        measurement = (
            np.asarray([row["x"], row["y"]], np.float64)
            if row.get("valid") and row.get("x") is not None
            else None
        )
        accepted = measurement is not None
        if policy == "hold_below_050":
            accepted = accepted and float(row["score"]) >= 0.50
        elif policy == "schmitt_052_046":
            threshold = 0.46 if locked else 0.52
            accepted = accepted and float(row["score"]) >= threshold
            locked = bool(accepted)
        if accepted:
            dt = max(
                1.0 / 30.0,
                (int(row["session_time_ns"]) - previous_ns) / 1.0e9,
            ) if previous_ns is not None else 1.0 / 30.0
            if state is None or policy in ("raw", "hold_below_050", "schmitt_052_046"):
                state = measurement
            elif policy == "ema_085":
                state = 0.85 * measurement + 0.15 * state
            elif policy == "alpha_beta_085_005":
                prediction = state + velocity * dt
                residual = measurement - prediction
                state = prediction + 0.85 * residual
                velocity = velocity + 0.05 * residual / dt
            previous_ns = int(row["session_time_ns"])
            provenance = "fresh_filtered" if policy not in ("raw", "hold_below_050", "schmitt_052_046") else "fresh"
        else:
            provenance = "held" if state is not None else "unavailable"
        row["output_x"] = None if state is None else float(state[0])
        row["output_y"] = None if state is None else float(state[1])
        row["output_provenance"] = provenance
        row["measurement_accepted_by_temporal_policy"] = bool(accepted)
        output.append(row)
    return output


def _distribution(values):
    values = np.asarray(values, np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p95": None, "worst": None}
    return {
        "count": int(len(values)), "mean": float(np.mean(values)),
        "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
        "worst": float(np.max(values)),
    }


def _summarize(rows, reference):
    errors, jumps = [], []
    last = None
    for row in rows:
        if row["output_x"] is None:
            continue
        point = np.asarray([row["output_x"], row["output_y"]], np.float64)
        truth = _reference_at(reference, int(row["session_time_ns"]))
        if truth is not None:
            errors.append(float(np.linalg.norm(point - truth)))
        if last is not None:
            jumps.append(float(np.linalg.norm(point - last)))
        last = point
    count = max(len(rows), 1)
    fresh = sum(row["output_provenance"].startswith("fresh") for row in rows)
    available = sum(row["output_x"] is not None for row in rows)
    deadline = sum(
        row["output_provenance"].startswith("fresh")
        and float(row["localization_core_elapsed_ms"]) <= 1000.0 / 30.0
        for row in rows
    )
    return {
        "frames": len(rows), "fresh_rate": fresh / count,
        "available_rate": available / count, "fresh_within_33_3ms_rate": deadline / count,
        "error_px": _distribution(errors), "adjacent_jump_px": _distribution(jumps),
        "core_latency_ms": _distribution([row["localization_core_elapsed_ms"] for row in rows]),
    }


def run(args):
    root = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    rows = []
    for candidate, (feature, matcher, radius) in CANDIDATES.items():
        for session_id in ("run_17", "run_18"):
            session = args.session_root / session_id
            reference_path = args.reference_root / session_id.replace("_", "")
            raw_dir = output / "raw" / candidate / session_id
            if not (raw_dir / "telemetry.jsonl").exists():
                benchmark_session(
                    session, args.route_package, args.atlas, calibration["config"],
                    calibration, "local_primary_gated", score_min=0.0,
                    local_radius_px=radius, correlation_feature=feature,
                    mode_policy="sticky", initialization="known_start",
                    local_matcher=matcher, reference_package_path=reference_path,
                    output_path=raw_dir,
                )
            telemetry = [json.loads(line) for line in (raw_dir / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()]
            reference = RouteTrackingPackage(reference_path)
            for policy in TEMPORAL_POLICIES:
                filtered = _temporal(telemetry, policy)
                summary = _summarize(filtered, reference)
                rows.append({"candidate": candidate, "session": session_id, "temporal_policy": policy, **summary})
                policy_dir = output / "telemetry" / candidate / policy
                policy_dir.mkdir(parents=True, exist_ok=True)
                with (policy_dir / (session_id + ".jsonl")).open("w", encoding="utf-8") as stream:
                    for row in filtered:
                        stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        flat = []
        for row in rows:
            flat.append({
                "candidate": row["candidate"], "session": row["session"],
                "temporal_policy": row["temporal_policy"], "fresh_rate": row["fresh_rate"],
                "available_rate": row["available_rate"], "fresh_within_33_3ms_rate": row["fresh_within_33_3ms_rate"],
                **{"error_" + key: value for key, value in row["error_px"].items()},
                **{"latency_" + key: value for key, value in row["core_latency_ms"].items()},
                "jump_p95": row["adjacent_jump_px"]["p95"], "jump_worst": row["adjacent_jump_px"]["worst"],
            })
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader(); writer.writerows(flat)
    result = {
        "schema_version": "1.0", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "target": "30 fresh fixes/s within 33.3 ms",
            "initialization": "one evaluator-declared known pose; cold start excluded",
            "future_reference_feeds_tracker": False,
            "accuracy": "sparse post-run map reference, evaluation only",
            "latency_scope": "decode excluded; minimap extraction plus localization core",
        },
        "results": rows,
        "traceability": {
            "revision": _git(root, "rev-parse", "HEAD"),
            "benchmark_source": str(Path(__file__).resolve()),
            "benchmark_sha256": _sha256(Path(__file__)),
            "tracker_source_sha256": _sha256(root / "poc" / "benchmark_route_tracer.py"),
        },
    }
    (output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--route-package", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
