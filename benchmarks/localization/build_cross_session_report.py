"""Compare localization candidates by atlas anchor and cross-family agreement."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from benchmarks.localization.build_realtime_control_report import (
    MINIMUM_DEADLINE_MS,
    TARGET_DEADLINE_MS,
    _number,
    _percent,
    _read_jsonl,
    _sha256,
    _summary,
)


def _family(parameters: dict) -> str:
    matcher = str(parameters.get("local_matcher") or "ccorr_normed")
    feature = str(parameters.get("correlation_feature") or "gradient")
    return "{}:{}".format(matcher, feature)


def _reference_modes(package_path: Path):
    states = _read_jsonl(Path(package_path) / "route_states.jsonl")
    times = np.asarray([int(row["session_time_ns"]) for row in states], np.int64)
    modes = np.asarray([str(row["mode_id"]) for row in states])
    return times, modes


def _nearest_mode(timestamp_ns: int, times, modes) -> str:
    insertion = int(np.searchsorted(times, int(timestamp_ns)))
    choices = [index for index in (insertion - 1, insertion) if 0 <= index < len(times)]
    if not choices:
        return "unknown"
    selected = min(choices, key=lambda index: abs(int(times[index]) - int(timestamp_ns)))
    return str(modes[selected])


def _representative_by_family(candidates: list[dict]) -> dict[str, dict]:
    grouped = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["family"]].append(candidate)
    return {
        family: min(
            rows,
            key=lambda row: (
                float(row["parameters"].get("local_radius_px") or 1.0e9),
                row["name"],
            ),
        )
        for family, rows in grouped.items()
    }


def _load(input_root: Path):
    candidates = []
    source_files = []
    for report_path in sorted(Path(input_root).glob("*/*/report.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        telemetry_path = report_path.parent / report["rows_file"]
        name = report_path.parent.parent.name
        candidates.append(
            {
                "name": name,
                "family": _family(report["parameters"]),
                "parameters": report["parameters"],
                "report": report,
                "rows": _read_jsonl(telemetry_path),
            }
        )
        source_files.extend(
            [
                {"path": str(report_path), "sha256": _sha256(report_path)},
                {"path": str(telemetry_path), "sha256": _sha256(telemetry_path)},
            ]
        )
    if not candidates:
        raise ValueError("No candidate reports found under {}".format(input_root))
    return candidates, source_files


def _annotate(candidates: list[dict]) -> None:
    mode_cache = {}
    by_session = defaultdict(list)
    for candidate in candidates:
        session = str(candidate["report"]["session"]["session_id"])
        by_session[session].append(candidate)
        package = candidate["report"].get("reference_package")
        if package and package not in mode_cache:
            mode_cache[package] = _reference_modes(Path(package))
        if package:
            times, modes = mode_cache[package]
            for row in candidate["rows"]:
                row["evaluation_mode_id"] = _nearest_mode(
                    int(row["session_time_ns"]), times, modes
                )
        else:
            for row in candidate["rows"]:
                row["evaluation_mode_id"] = str(row.get("mode_id") or "unknown")

    for session_candidates in by_session.values():
        representatives = _representative_by_family(session_candidates)
        lookups = {
            row["family"]: {
                int(item["session_time_ns"]): item for item in row["rows"]
            }
            for row in representatives.values()
        }
        for candidate in session_candidates:
            for row in candidate["rows"]:
                row["cross_family_consensus_error_px"] = None
                if not row.get("measurement_accepted"):
                    continue
                points = []
                for family, lookup in lookups.items():
                    if family == candidate["family"]:
                        continue
                    peer = lookup.get(int(row["session_time_ns"]))
                    if peer and peer.get("measurement_accepted"):
                        points.append([float(peer["x"]), float(peer["y"])])
                if len(points) < 2:
                    continue
                consensus = np.median(np.asarray(points, dtype=np.float64), axis=0)
                row["cross_family_consensus_error_px"] = float(
                    np.linalg.norm(
                        np.asarray([float(row["x"]), float(row["y"])]) - consensus
                    )
                )


def _aggregate(candidates: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for candidate in candidates:
        for row in candidate["rows"]:
            if row.get("initialization_frame", False):
                continue
            grouped[(candidate["name"], row["evaluation_mode_id"])].append(row)
    output = []
    for (name, mode), rows in sorted(grouped.items()):
        fresh = np.asarray([bool(row["measurement_accepted"]) for row in rows])
        latency = np.asarray(
            [float(row["localization_core_elapsed_ms"]) for row in rows]
        )
        anchor_errors = [
            float(row["reference_error_px"])
            for row in rows
            if row.get("reference_error_px") is not None and row["measurement_accepted"]
        ]
        consensus_errors = [
            float(row["cross_family_consensus_error_px"])
            for row in rows
            if row.get("cross_family_consensus_error_px") is not None
        ]
        output.append(
            {
                "candidate": name,
                "mode_id": mode,
                "attempt_count": len(rows),
                "fresh_measurement_accepted_rate": float(np.mean(fresh)),
                "fresh_within_33_3ms_rate": float(
                    np.mean(fresh & (latency <= TARGET_DEADLINE_MS))
                ),
                "fresh_within_66_7ms_rate": float(
                    np.mean(fresh & (latency <= MINIMUM_DEADLINE_MS))
                ),
                "localization_core_latency_ms": _summary(latency),
                "offline_anchor_error_px": _summary(anchor_errors),
                "cross_family_consensus_error_px": _summary(consensus_errors),
            }
        )
    return output


def _report_lines(result: dict) -> list[str]:
    lines = [
        "LOCALIZATION CROSS-SESSION CONSISTENCY",
        "",
        "EVIDENCE MEANING",
        "",
        "Offline anchor: sparse 5 Hz feature-plus-correlation localization against the atlas.",
        "Cross-family consensus: median XY from other matcher families on the same frame.",
        "Gradient 12 and gradient 18 are one family and never vote twice.",
        "Neither measure is external ground truth; agreement estimates consistency, not absolute accuracy.",
        "Held positions never count as fresh measurements or consensus votes.",
        "",
    ]
    for mode in ("world", "town", "unknown"):
        rows = [row for row in result["results"] if row["mode_id"] == mode]
        if not rows:
            continue
        lines.extend(["MODE  {}".format(mode.upper()), ""])
        for row in rows:
            anchor = row["offline_anchor_error_px"]
            consensus = row["cross_family_consensus_error_px"]
            latency = row["localization_core_latency_ms"]
            lines.extend(
                [
                    row["candidate"],
                    "Fresh accepted  {}".format(
                        _percent(row["fresh_measurement_accepted_rate"])
                    ),
                    "Fresh within 33.3 ms  {}".format(
                        _percent(row["fresh_within_33_3ms_rate"])
                    ),
                    "Anchor error mean / P95 / worst  {} / {} / {}".format(
                        _number(anchor["mean"], " px"),
                        _number(anchor["p95"], " px"),
                        _number(anchor["worst"], " px"),
                    ),
                    "Cross-family disagreement mean / P95 / worst  {} / {} / {}".format(
                        _number(consensus["mean"], " px"),
                        _number(consensus["p95"], " px"),
                        _number(consensus["worst"], " px"),
                    ),
                    "Core latency P95 / worst  {} / {}".format(
                        _number(latency["p95"], " ms"),
                        _number(latency["worst"], " ms"),
                    ),
                    "",
                ]
            )
    lines.extend(
        [
            "LIMITATION",
            "",
            "The anchor shares the atlas and some image operations with the candidates.",
            "Use these results to rank consistency and find isolated failures, not to certify true map-pixel accuracy.",
            "",
            "Machine report  cross_session_results.json",
            "",
        ]
    )
    return lines


def build(input_root: Path, output_path: Path) -> dict:
    candidates, sources = _load(Path(input_root))
    _annotate(candidates)
    result = {
        "schema_version": "1.0",
        "benchmark": "localization_cross_session_consistency",
        "causal_constraints": {
            "offline_anchor_feeds_candidate": False,
            "cross_method_consensus_feeds_candidate": False,
            "held_output_counts_as_fresh": False,
            "duplicate_method_families_get_multiple_votes": False,
        },
        "results": _aggregate(candidates),
        "source_files": sources,
    }
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "cross_session_results.json").write_text(
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
