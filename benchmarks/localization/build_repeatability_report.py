"""Build an automatic repeated-waypoint localization benchmark report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.localization.repeated_waypoints import (
    discover_repeated_waypoints,
    evaluate_candidate_repeatability,
)


def _read_jsonl(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _identity(path: Path) -> dict:
    data = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(data).hexdigest()}


def _number(value, suffix="") -> str:
    return "n/a" if value is None else "{:.2f}{}".format(float(value), suffix)


def build(
    reference_path: Path,
    candidates: dict[str, Path],
    output_path: Path,
    *,
    reference_kind: str,
    heading_semantics: str,
    **waypoint_parameters,
) -> dict:
    reference_path = Path(reference_path)
    output_path = Path(output_path)
    reference_rows = _read_jsonl(reference_path)
    waypoints = discover_repeated_waypoints(reference_rows, **waypoint_parameters)
    if not waypoints["groups"]:
        raise RuntimeError("No repeated complete-state waypoint groups were found")
    results = []
    sources = [_identity(reference_path)]
    for name, path in sorted(candidates.items()):
        path = Path(path)
        score = evaluate_candidate_repeatability(waypoints, _read_jsonl(path))
        score["candidate"] = name
        results.append(score)
        sources.append(_identity(path))
    result = {
        "schema_version": "1.0",
        "benchmark": "automatic_repeated_complete_state_waypoints",
        "reference": {
            "kind": str(reference_kind),
            "heading_semantics": str(heading_semantics),
            "absolute_accuracy_claim": reference_kind == "external_ground_truth",
            **_identity(reference_path),
        },
        "causal_boundary": {
            "waypoint_groups_are_post_run_evidence_only": True,
            "candidate_receives_lap_identity": False,
            "candidate_receives_reference_state": False,
            "pause_or_boundary_marker_required": False,
            "equal_path_or_timing_required": False,
        },
        "waypoints": waypoints,
        "candidates": results,
        "sources": sources,
        "implementation": {
            "report_builder": _identity(Path(__file__)),
            "waypoint_evaluator": _identity(
                Path(__file__).with_name("repeated_waypoints.py")
            ),
        },
    }
    lines = [
        "AUTOMATIC ROUTE REPEATABILITY BENCHMARK",
        "",
        "No pauses, lap boundaries, equal timing, or identical paths are assumed.",
        "Waypoint identity is map mode + nearby XY + nearby heading.",
        "Human path variation is subtracted before repeatability is scored.",
        "Reference kind: {}".format(reference_kind),
        "Heading semantics: {}".format(heading_semantics),
        "Absolute accuracy claim: {}".format(
            "YES" if result["reference"]["absolute_accuracy_claim"] else "NO"
        ),
        "Repeated waypoint groups: {}".format(waypoints["group_count"]),
        "Grouped visits: {}".format(waypoints["grouped_visit_count"]),
        "",
    ]
    for row in results:
        position = row["reference_position_error_px"]
        repeatability = row["position_repeatability_residual_px"]
        heading = row["heading_repeatability_residual_deg"]
        lines.extend(
            [
                row["candidate"],
                "Available / fresh: {:.1f}% / {:.1f}%".format(
                    100.0 * row["availability_rate"], 100.0 * row["fresh_rate"]
                ),
                "Reference position mean / median / P95: {} / {} / {}".format(
                    _number(position["mean"], " px"),
                    _number(position["median"], " px"),
                    _number(position["p95"], " px"),
                ),
                "Position repeatability median / P95 / max: {} / {} / {}".format(
                    _number(repeatability["median"], " px"),
                    _number(repeatability["p95"], " px"),
                    _number(repeatability["maximum"], " px"),
                ),
                "Heading repeatability median / P95 / max: {} / {} / {}".format(
                    _number(heading["median"], " deg"),
                    _number(heading["p95"], " deg"),
                    _number(heading["maximum"], " deg"),
                ),
                "",
            ]
        )
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "repeatability_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    (output_path / "REPORT.txt").write_text("\n".join(lines), encoding="utf-8")
    return result


def _parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Candidate must be NAME=TELEMETRY.jsonl")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("Candidate must be NAME=TELEMETRY.jsonl")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--reference-kind",
        choices=("external_ground_truth", "cross_method_consensus", "offline_atlas_anchor"),
        required=True,
    )
    parser.add_argument(
        "--heading-semantics",
        choices=("cursor_pose", "trajectory_tangent", "external_body_heading"),
        required=True,
    )
    parser.add_argument("--candidate", action="append", type=_parse_candidate, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spatial-radius-px", type=float, default=5.0)
    parser.add_argument("--heading-radius-deg", type=float, default=45.0)
    parser.add_argument("--waypoint-spacing-px", type=float, default=10.0)
    parser.add_argument("--minimum-recurrence-s", type=float, default=8.0)
    parser.add_argument("--minimum-visits", type=int, default=3)
    args = parser.parse_args()
    build(
        args.reference,
        dict(args.candidate),
        args.output,
        reference_kind=args.reference_kind,
        heading_semantics=args.heading_semantics,
        spatial_radius_px=args.spatial_radius_px,
        heading_radius_deg=args.heading_radius_deg,
        waypoint_spacing_px=args.waypoint_spacing_px,
        minimum_recurrence_s=args.minimum_recurrence_s,
        minimum_visits=args.minimum_visits,
    )


if __name__ == "__main__":
    main()
