"""Deterministic cursor/no-cursor noise probes with independent raster truth.

Run with python -m benchmarks.cursor_pose.calibration_noise --suite fresh
--output artifacts/cursor-noise-fresh.json. --baseline loads only the estimator
from d869428 via git, without changing the checkout. No capture device required.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from rig_runtime.services.calibration.cursor.center import fit_temporal_cold_circle
from tests.test_cursor_cold_circle import rotating_cursor_frames


def cases(suite):
    if suite == "primary":
        for seed in (271, 409, 557, 811):
            for kind in ("rotating", "static", "empty"):
                for noise in (0, 2, 6, 12):
                    yield dict(seed=seed, pivot=(62.3, 58.7), shape="dot", size=1., dwell=12,
                               angles=[0, 25, 55, 85, 125, 160, 205], kind=kind, noise=noise, impulses=0)
        return
    # 'stress' was opened during development; 'fresh' was fixed only after
    # proposing linear per-frame smoothing and the two measured noise floors.
    sessions = ((19031, (61.1, 66.8), .9, 200), (19087, (67.2, 59.9), 1.2, 75)) if suite == "stress" else (
        (41011, (65.2, 67.1), .85, 180), (41047, (59.1, 62.2), 1.1, 30), (41081, (66.8, 61.3), 1.35, 350))
    if suite == "validation":
        sessions = ((55217, (63.7, 60.4), 1.05, 90), (55313, (60.5, 65.8), .95, 240))
    for seed, pivot, size, dwell in sessions:
        for shape in ("dot", "triangle"):
            for kind in ("rotating", "static", "empty"):
                for noise in (2, 12):
                    for impulses in (0, .003):
                        yield dict(seed=seed, pivot=pivot, shape=shape, size=size, dwell=dwell,
                                   angles=[61, 88, 129, 165, 198, 239, 283] if suite == "stress" else
                                   [53, 72, 106, 149, 188, 229, 267], kind=kind, noise=noise, impulses=impulses)


def run(case, fit):
    angles = [case["angles"][0]] * case["dwell"] + case["angles"][1:]
    frames = rotating_cursor_frames(angles if case["kind"] == "rotating" else [angles[0]] * len(angles),
                                   case["pivot"], case["shape"], case["size"], case["seed"], case["noise"])
    rng = np.random.default_rng(case["seed"] if len(frames) == 18 else case["seed"] + 99)
    if case["kind"] == "empty":
        frames = np.clip(35 + rng.normal(0, case["noise"], frames.shape), 0, 255).astype(np.uint8)
    hot = rng.random(frames.shape[:3]) < case["impulses"]
    frames[hot] = rng.integers(0, 256, (int(hot.sum()), 3), dtype=np.uint8)
    row = dict(case)
    start = time.perf_counter()
    try:
        metrics = fit(frames, [64, 64], 12)["metrics"]
        row.update(status="accepted", metrics=metrics)
        if case["kind"] == "rotating":
            row["error_px"] = float(np.linalg.norm(np.array([metrics["x"], metrics["y"]]) - case["pivot"]))
    except RuntimeError as exc:
        row.update(status="unavailable", reason=str(exc))
    row["fit_ms"] = 1000 * (time.perf_counter() - start)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("primary", "stress", "fresh", "validation"), default="validation")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    fit = fit_temporal_cold_circle
    if args.baseline:
        source = subprocess.check_output(["git", "show", "d869428:rig_runtime/services/calibration/cursor/center.py"], text=True)
        namespace = {}
        exec(compile(source, "baseline_cursor_center.py", "exec"), namespace)
        fit = namespace["fit_temporal_cold_circle"]
    rows = [run(case, fit) for case in cases(args.suite)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for kind in ("rotating", "static", "empty"):
        selected = [row for row in rows if row["kind"] == kind]
        accepted = [row for row in selected if row["status"] == "accepted"]
        print(kind, "accepted", len(accepted), "/", len(selected),
              "max independent error", max((row.get("error_px", 0) for row in accepted), default=0),
              "median/p95 fit ms", np.percentile([row["fit_ms"] for row in selected], [50, 95]))


if __name__ == "__main__":
    main()
