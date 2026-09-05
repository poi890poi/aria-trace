"""Build a narrow-screen report from the wide pose/localization benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _n(value, suffix=""):
    return "n/a" if value is None else "{:.2f}{}".format(float(value), suffix)


def _pct(value):
    return "n/a" if value is None else "{:.1f}%".format(float(value) * 100.0)


def build(pose_path: Path, localization_path: Path, output: Path):
    pose = json.loads(Path(pose_path).read_text(encoding="utf-8"))
    localization = list(csv.DictReader(Path(localization_path).open(encoding="utf-8")))
    pose_lookup = {(r["candidate"], r["temporal_policy"]): r for r in pose["results"]}
    loc_lookup = {(r["candidate"], r["temporal_policy"], r["session"]): r for r in localization}
    lines = [
        "WIDE LIVE-CONTROL BENCHMARK",
        "",
        "DECISION",
        "",
        "Pose: keep angular_projection_ncc_parabolic as the first layer, every frame, without continuous smoothing or a confidence hold.",
        "Pose final layer: reject only a physically impossible angular innovation using the calibrated turning envelope; hold the last state for that rejected frame.",
        "Localization: use gradient CCORR_NORMED in a 12 px local window after one global fix.",
        "Localization final layer: reject only an impossible XY step using the calibrated speed envelope; hold the last state. Do not use the 0.50 score threshold globally.",
        "",
        "The default real-time core and two-layer acceptance policies are landed; full capture-to-publication validation remains pending.",
        "",
        "CONTROL CONTRACT",
        "",
        "Target: 30 fresh pose and XY fixes/s.",
        "Per-stage compute budget: 33.3 ms.",
        "Minimum boundary: 15 fresh fixes/s and 66.7 ms.",
        "Held output is available control state, not a fresh fix.",
        "Cold-start global localization is separate from steady-state tracking.",
        "",
        "POSE CORE COMPARISON",
        "",
    ]
    for name in (
        "angular_projection_ncc_parabolic",
        "symmetric_pixel_fft_ncc_parabolic",
        "polygon_von_mises_moment",
        "analytic_lm_ambiguous",
    ):
        row = pose_lookup[(name, "raw")]
        wide = [row["wide_sessions"][key]["agreement_diagnostic"] for key in ("run_17", "run_18")]
        turn = [row["wide_sessions"][key]["input_turn_response"] for key in ("run_17", "run_18")]
        lines += [
            name,
            "  Forward absolute P95 / worst: {} / {}".format(_n(row["forward_absolute"]["error_deg"]["p95"], " deg"), _n(row["forward_absolute"]["error_deg"]["worst"], " deg")),
            "  Wide fresh rate, worst session: {}".format(_pct(min(x["fresh_rate"] for x in wide))),
            "  Estimator P95, worst session: {}".format(_n(max(x["latency_ms"]["p95"] for x in wide), " ms")),
            "  Leave-one-out agreement P95, worst session: {}".format(_n(max(x["error_deg"]["p95"] for x in wide), " deg")),
            "  Turn onset P95, worst session: {}".format(_n(max(x["onset_lag_ms"]["p95"] for x in turn), " ms")),
            "",
        ]
    lines += ["POSE TEMPORAL COMPARISON", ""]
    for policy in ("raw", "physical_gate", "confidence_hold", "schmitt", "ema_085", "alpha_beta_085_005"):
        row = pose_lookup[("angular_projection_ncc_parabolic", policy)]
        wide = [row["wide_sessions"][key] for key in ("run_17", "run_18")]
        lines += [
            policy,
            "  Fresh, worst session: {}".format(_pct(min(x["agreement_diagnostic"]["fresh_rate"] for x in wide))),
            "  Forward P95 / worst: {} / {}".format(_n(row["forward_absolute"]["error_deg"]["p95"], " deg"), _n(row["forward_absolute"]["error_deg"]["worst"], " deg")),
            "  Final physical-gate rejection, worst session: {}".format(_pct(max(x["agreement_diagnostic"].get("final_physical_gate_rejection_rate", 0.0) for x in wide))),
            "  Wrong-direction persistence P95, worst session: {}".format(_n(max(x["input_turn_response"]["wrong_direction_duration_ms"]["p95"] for x in wide), " ms")),
            "",
        ]
    lines += ["LOCALIZATION CORE COMPARISON", ""]
    for name in ("gradient8_ccorr", "gradient12_ccorr", "gradient18_ccorr", "intensity12_ccorr", "canny12_ccorr", "laplacian12_ccorr", "gradient12_phase"):
        outdoor = loc_lookup[(name, "raw", "run_17")]
        town = loc_lookup[(name, "raw", "run_18")]
        lines += [
            name,
            "  Outdoor fresh / P95 / worst: {} / {} / {}".format(_pct(outdoor["fresh_rate"]), _n(outdoor["error_p95"], " px"), _n(outdoor["error_worst"], " px")),
            "  Town fresh / P95 / worst: {} / {} / {}".format(_pct(town["fresh_rate"]), _n(town["error_p95"], " px"), _n(town["error_worst"], " px")),
            "  Final physical-gate rejection, worst session: {}".format(_pct(max(float(outdoor.get("final_physical_gate_rejection_rate") or 0.0), float(town.get("final_physical_gate_rejection_rate") or 0.0)))),
            "  Core P95, worst session: {}".format(_n(max(float(outdoor["latency_p95"]), float(town["latency_p95"])), " ms")),
            "",
        ]
    lines += ["LOCALIZATION TEMPORAL COMPARISON", ""]
    for policy in ("raw", "hold_below_050", "schmitt_052_046", "ema_085", "alpha_beta_085_005"):
        outdoor = loc_lookup[("gradient12_ccorr", policy, "run_17")]
        town = loc_lookup[("gradient12_ccorr", policy, "run_18")]
        lines += [
            policy,
            "  Outdoor fresh / P95 / worst: {} / {} / {}".format(_pct(outdoor["fresh_rate"]), _n(outdoor["error_p95"], " px"), _n(outdoor["error_worst"], " px")),
            "  Town fresh / P95 / worst: {} / {} / {}".format(_pct(town["fresh_rate"]), _n(town["error_p95"], " px"), _n(town["error_worst"], " px")),
            "",
        ]
    angular = pose_lookup[("angular_projection_ncc_parabolic", "raw")]
    loc17 = loc_lookup[("gradient12_ccorr", "raw", "run_17")]
    loc18 = loc_lookup[("gradient12_ccorr", "raw", "run_18")]
    pose_p95 = max(angular["wide_sessions"][key]["agreement_diagnostic"]["latency_ms"]["p95"] for key in ("run_17", "run_18"))
    loc_p95 = max(float(loc17["latency_p95"]), float(loc18["latency_p95"]))
    lines += [
        "INTERPRETATION",
        "",
        "Angular projection combines 100% wide-session fresh coverage, {} worst-session P95 compute, and approximately 3 deg forward P95 error.".format(_n(pose_p95, " ms")),
        "The calibrated pose gate limit is {} deg/s; raw and gated results show whether this corruption guard rejects any observed frame.".format(_n(pose["contract"].get("turn_rate_limit_deg_s"))),
        "Confidence holding lowers forward P95 slightly but rejects about 10% of wide outdoor frames and does not reduce worst error. It fails the availability objective.",
        "EMA and alpha-beta do not materially improve pose error and increase wrong-direction persistence at sharp reversals. They are rejected for now.",
        "Gradient CCORR is stable in both outdoor and town sessions. Radius 8/12/18 gives the same pose; 12 px keeps balanced search support.",
        "The 0.50 localization score threshold accepts only 63.9% outdoors despite correct tracking. Score is scene-dependent and must not be the main hysteresis gate.",
        "Laplacian and phase correlation remain continuous while being hundreds of pixels wrong outdoors. Continuity alone cannot establish correctness.",
        "EMA improves localization P95 by only about 0.09 px outdoors and 0.05 px in town. Without independent XY turn-lag evidence, that gain does not justify landing smoothing.",
        "Conservative serial sum of pose and localization P95 compute: {}. This is not capture-to-publication latency.".format(_n(pose_p95 + loc_p95, " ms")),
        "",
        "EVIDENCE BOUNDARIES",
        "",
        "Pose absolute accuracy: runs 03, 04, 12, 13 forward-only whole-session motion direction.",
        "Pose temporal evidence: runs 17 and 18 input plus scene KLT; neither is treated as pose truth.",
        "Localization: runs 17 and 18, 10,824 source frames, sparse post-run atlas references used only for scoring.",
        "Localization receives one declared initial pose. Cold-start success and latency are excluded.",
        "Cross-method pose agreement is a diagnostic, not ground truth.",
        "Decode, capture, IPC, scheduling, fusion, rendering, and publication are excluded.",
        "Therefore the compute candidates pass; complete 30 FPS end-to-end control remains unproven.",
        "Recent learned local features such as XFeat and adaptive matchers such as LightGlue are retained as global recovery research, not inserted into this fixed-scale local loop. Their required runtime is not installed in the benchmark environment, and adding it would not address the measured local-continuation failure modes.",
        "",
        "NEGATIVE RESULTS RETAINED",
        "",
        "Pose confidence hold; pose EMA; pose alpha-beta; localization score hold; localization Schmitt threshold; localization EMA; localization alpha-beta; intensity; Canny; Laplacian; phase correlation.",
        "",
        "TRACEABILITY",
        "",
        "Pose machine results: {}".format(Path(pose_path).resolve()),
        "Localization rows: {}".format(Path(localization_path).resolve()),
        "Raw per-frame telemetry is stored beside those results.",
    ]
    output.mkdir(parents=True, exist_ok=True)
    (output / "REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "REPORT.md").write_text("```text\n" + "\n".join(lines) + "\n```\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--localization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); build(args.pose, args.localization, args.output)


if __name__ == "__main__":
    main()
