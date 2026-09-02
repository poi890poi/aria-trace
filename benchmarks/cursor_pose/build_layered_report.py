"""Combine cursor-pose method and chronological E2E results into one report.

The report is deliberately narrow-screen friendly.  The two source result files
remain authoritative and retain the full machine-readable experiment matrix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


METHOD_LABELS = {
    "current_accurate_vectorized_full": "Current accurate",
    "current_realtime_cascade_ambiguous": "Current real-time",
}

PROFILE_ROWS = (
    ("accurate_confidence_hold", "Accurate"),
    ("realtime_confidence_hold", "Real-time"),
    ("fast_confidence_hold", "Fast"),
)
GATE_ROWS = (
    ("realtime_confidence_hold", "Confidence only"),
    ("realtime_strict_hold", "Confidence + temporal gate"),
)
PUBLICATION_ROWS = (
    ("realtime_confidence_hold", "Hold previous state"),
    ("realtime_confidence_predict", "Predict from last two accepted states"),
    ("realtime_confidence_reject", "Publish unavailable"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percent(value) -> str:
    return "n/a" if value is None else "{:.1%}".format(float(value))


def _number(value, suffix="") -> str:
    return "n/a" if value is None else "{:.2f}{}".format(float(value), suffix)


def _method_card(lines, row, decision):
    label = METHOD_LABELS.get(row["method"], row["method"].replace("_", " "))
    lines.extend(
        [
            "### {}".format(label),
            "",
            "- Method ID: `{}`".format(row["method"]),
            "- Family: {}".format(row["group"].replace("_", " ")),
            "- Pose produced: {}".format(_percent(row["pose_production_rate"])),
            "- Error mean / median: {} / {}".format(
                _number(row.get("mean_abs_error_deg"), " deg"),
                _number(row["median_abs_error_deg"], " deg"),
            ),
            "- Error P95: {}".format(_number(row["p95_abs_error_deg"], " deg")),
            "- Latency median / P95: {} / {}".format(
                _number(row["latency"]["median_ms"], " ms"),
                _number(row["latency"]["p95_ms"], " ms"),
            ),
            "- Screen result: {}".format(decision or "UNCLASSIFIED"),
            "",
        ]
    )


def _e2e_card(lines, label, row):
    lines.extend(["### {}".format(label), ""])
    if row is None:
        lines.extend(["Not run.", ""])
        return
    error = row["e2e_absolute_error_deg"]
    lines.extend(
        [
            "- Candidate produced: {}".format(
                _percent(row["primary_candidate_produced_rate"])
            ),
            "- Accepted fresh: {}".format(
                _percent(row["primary_measurement_accepted_rate"])
            ),
            "- Held: {}".format(_percent(row["output_provenance_rate"]["held"])),
            "- Predicted: {}".format(
                _percent(row["output_provenance_rate"]["predicted"])
            ),
            "- Available: {}".format(_percent(row["final_output_available_rate"])),
            "- Error mean / median: {} / {}".format(
                _number(error["mean"], " deg"),
                _number(error["median"], " deg"),
            ),
            "- Error P95 / worst: {} / {}".format(
                _number(error["p95"], " deg"),
                _number(error["worst"], " deg"),
            ),
            "- E2E latency median / P95: {} / {}".format(
                _number(row["e2e_latency_ms"]["median"], " ms"),
                _number(row["e2e_latency_ms"]["p95"], " ms"),
            ),
            "",
        ]
    )


def build(method_results_path: Path, e2e_results_path: Path, output_path: Path) -> dict:
    method_results_path = Path(method_results_path).resolve()
    e2e_results_path = Path(e2e_results_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    method_result = json.loads(method_results_path.read_text(encoding="utf-8"))
    e2e_result = json.loads(e2e_results_path.read_text(encoding="utf-8"))

    method_rows = [
        row for row in method_result["aggregate"] if row["split"] == "holdout"
    ]
    method_rows.sort(
        key=lambda row: (
            row["group"],
            row["p95_abs_error_deg"],
            row["latency"]["median_ms"],
        )
    )
    e2e_lookup = {
        (row["solution"], row["outage_scenario"]): row
        for row in e2e_result["aggregate"]
        if row["split"] == "holdout"
    }

    lines = [
        "# Cursor pose layered benchmark",
        "",
        "## Decision first",
        "",
        "Current supported stack: **real-time estimator + confidence-only gate + "
        "previous-state hold**.",
        "",
        "Experimental estimator methods are screened independently below. A promising "
        "method is not a complete solution until it passes the chronological gate and "
        "publication benchmark.",
        "",
        "## Tested complete combinations",
        "",
        "- Accurate + confidence + hold",
        "- Real-time + confidence + hold",
        "- Fast + confidence + hold",
        "- Real-time + strict temporal gate + hold",
        "- Real-time + confidence + prediction",
        "- Real-time + confidence + unavailable",
        "",
        "This is a one-change matrix, not an unrestricted Cartesian product. It keeps "
        "the cause of every difference identifiable.",
        "",
        "## Layer 1A — experimental estimator methods",
        "",
        "Reference: independent local mini-map travel direction. This is a method "
        "screen, not pixel-level cursor truth.",
        "",
    ]
    for row in method_rows:
        _method_card(lines, row, method_result.get("decisions", {}).get(row["method"]))

    lines.extend(
        [
            "## Layer 1B — production estimator profiles",
            "",
            "Gate is confidence-only. Publication is hold. Outage is natural.",
            "",
        ]
    )
    for solution, label in PROFILE_ROWS:
        _e2e_card(lines, label, e2e_lookup.get((solution, "natural")))

    lines.extend(
        [
            "## Layer 2 — acceptance gate",
            "",
            "Estimator is real-time. Publication is hold. Outage is natural.",
            "",
        ]
    )
    for solution, label in GATE_ROWS:
        _e2e_card(lines, label, e2e_lookup.get((solution, "natural")))

    lines.extend(
        [
            "## Layer 3 — rejected-state publication",
            "",
            "Estimator is real-time. Gate is confidence-only. The same deterministic "
            "three-frame outage is injected every 90 frames.",
            "",
        ]
    )
    for solution, label in PUBLICATION_ROWS:
        _e2e_card(
            lines,
            label,
            e2e_lookup.get((solution, "three_frame_burst_every_90")),
        )

    lines.extend(
        [
            "## How to read the rates",
            "",
            "- Candidate produced: estimator returned a pose / attempted frames.",
            "- Accepted fresh: gate accepted a new pose / attempted frames.",
            "- Held and predicted: published fallback provenance / attempted frames.",
            "- Available: fresh + held + predicted / attempted frames.",
            "- Internal Gaussian or pixel validation use is not an acceptance rate.",
            "",
            "## Accuracy boundary",
            "",
            "The method screen is sampled and therefore has no meaningful worst-case "
            "number. Mean, median, P95, and worst are all shown only for the complete "
            "chronological E2E stacks.",
            "",
            "Travel direction can disagree with correct cursor pose during alignment, "
            "curved travel, or collision. It can compare functional E2E behavior, but "
            "must not train the pose gate by itself.",
            "",
            "## Traceability",
            "",
            "- Method result: `{}`".format(method_results_path),
            "- Method result SHA-256: `{}`".format(_sha256(method_results_path)),
            "- E2E result: `{}`".format(e2e_results_path),
            "- E2E result SHA-256: `{}`".format(_sha256(e2e_results_path)),
            "- Full raw matrices remain in those two source result files.",
            "",
        ]
    )

    combined = {
        "schema_version": "1.0",
        "benchmark": "cursor_pose_layered_comparison",
        "method_results": {
            "path": str(method_results_path),
            "sha256": _sha256(method_results_path),
            "holdout": method_rows,
            "decisions": method_result.get("decisions", {}),
        },
        "e2e_results": {
            "path": str(e2e_results_path),
            "sha256": _sha256(e2e_results_path),
            "holdout": list(e2e_lookup.values()),
        },
        "tested_complete_combinations": [
            "accurate+confidence+hold",
            "realtime+confidence+hold",
            "fast+confidence+hold",
            "realtime+strict_temporal+hold",
            "realtime+confidence+predict",
            "realtime+confidence+unavailable",
        ],
        "supported_stack": "realtime+confidence+hold",
    }
    (output_path / "layered_results.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )
    (output_path / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method_results", type=Path)
    parser.add_argument("e2e_results", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    result = build(arguments.method_results, arguments.e2e_results, arguments.output)
    print(json.dumps(result["tested_complete_combinations"], indent=2))


if __name__ == "__main__":
    main()
