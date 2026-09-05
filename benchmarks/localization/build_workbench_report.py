"""Consolidate actual-loop replays, including dropped source-frame denominators."""

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.localization.run_workbench_replay import distribution, read_rows
from benchmarks.localization.reference_cache import identity
from benchmarks.localization.repeated_waypoints import discover_repeated_waypoints, evaluate_candidate_repeatability


def heading_publication_latency(row):
    """Age of the consumed heading at Workbench publication, after engine work."""
    source = (row.get("cursor_pose") or {}).get("session_time_ns")
    published = row.get("control_published_host_time_ns")
    return (published-source)/1e6 if source is not None and published is not None else None


def build(root):
    results = []
    for path in sorted(root.glob("*/run*/report.json")):
        if path.parent.parent.name.startswith("smoke"):
            continue
        report = json.loads(path.read_text())
        rows = read_rows(path.parent / "scored_telemetry.jsonl")
        source = read_rows(path.parent / "source_telemetry.jsonl")
        first = next((r["host_time_ns"] for r in rows if r.get("pose")), None)
        steady = [r for r in rows if first is not None and r["host_time_ns"] >= first]
        source_count = sum(r["host_time_ns"] >= first for r in source) if first else len(source)
        cursor_fresh = [r for r in steady if r.get("cursor_pose_measurement_fresh_accepted")]
        heading_latencies = [heading_publication_latency(r) for r in cursor_fresh]
        heading_latencies = [v for v in heading_latencies if v is not None]
        summary = {k:v for k,v in report.items() if k not in ("implementation", "request", "git_status")}
        summary["cohort"] = path.parent.parent.name
        # The free-roam runtime reports fresh relative motion even before an
        # absolute pose exists. That is not an available fresh XY output.
        summary["unavailable_frames"] = sum(not r.get("pose") for r in rows)
        summary["held_frames"] = sum(bool(r.get("pose")) and not r.get("xy_measurement_fresh_accepted") for r in rows)
        summary["all_source_fresh_rate"] = sum(bool(r.get("pose")) and bool(r.get("xy_measurement_fresh_accepted")) for r in rows)/len(source)
        summary["operating_mode_counts"] = {mode:sum(r.get("mode")==mode for r in rows) for mode in sorted({r.get("mode") for r in rows})}
        summary["steady_source_frames"] = source_count
        summary["steady_source_fresh_xy_rate"] = sum(bool(r.get("xy_measurement_fresh_accepted")) for r in steady)/max(source_count,1)
        for limit, name in ((1000/30,"33ms"),(1000/15,"67ms")):
            summary["steady_source_fresh_xy_within_"+name] = sum(bool(r.get("xy_measurement_fresh_accepted")) and r["capture_to_control_publish_ms"] <= limit for r in steady)/max(source_count,1)
            summary["steady_source_fresh_heading_within_"+name] = sum(v <= limit for v in heading_latencies)/max(source_count,1)
        summary["fresh_heading_capture_to_publish_ms"] = distribution(heading_latencies)
        summary["heading_available_rate"] = sum((r.get("pose") or {}).get("cursor_screen_deg") is not None for r in steady)/max(len(steady),1)
        summary["visual_control"] = {
            "longest_control_lost_s": None,
            "status": "not-measured",
            "reason": "Recorded tracking replay does not close the action/response loop or validate local visual steering, route-stage decisions, path clearance, braking/turn response, or per-frame heading correctness.",
            "proxy_loss_is_control_loss": False,
        }
        summary["error_change_between_reference_samples_px"] = distribution(np.abs(np.diff([r["reference_error_px"] for r in rows if r.get("reference_error_px") is not None])))
        if report["session"] in (17,18):
            reference = read_rows(Path(report["reference"]) / "route_states.jsonl")
            waypoints = discover_repeated_waypoints(reference)
            candidates = [{"session_time_ns":r["session_time_ns"], "frame_index":r["frame_index"],
                           "x":(r.get("pose") or {}).get("x"), "y":(r.get("pose") or {}).get("y"),
                           "mode_id":r.get("active_map_mode_id"), "measurement_accepted":r.get("xy_measurement_fresh_accepted"),
                           "localization_core_elapsed_ms":r["capture_to_control_publish_ms"]} for r in rows]
            repeatability = evaluate_candidate_repeatability(waypoints,candidates)
            summary["repeatability"] = {"group_count":waypoints["group_count"], "visit_count":waypoints["grouped_visit_count"],
                                       "reference_heading_semantics":"trajectory_tangent_for_visit_grouping_only", "timing_boundary":"source_schedule_to_publication", **repeatability}
            (path.parent/"repeatability.json").write_text(json.dumps({"waypoints":waypoints,"result":summary["repeatability"]},indent=2))
        results.append(summary)
    value = {"measurement_boundary":"Original-cadence recorded video through actual Workbench loop; excludes physical capture and browser/display.",
             "implementation":identity(__file__),
             "scoring_reference":"Slow inferred atlas localization, not external ground truth. Sparse gaps and mode changes are not interpolated. Demo-session route-assisted reference also supplies proposals and is not independent.",
             "source_denominator":"All source frames from first available pose, including frames overwritten by latest-frame queue.",
             "results":results}
    (root/"results.json").write_text(json.dumps(value,indent=2))
    lines=["# Rebuilt-atlas production Workbench replay", "", value["measurement_boundary"], "", value["scoring_reference"], "",
           "Fresh XY below is per processed frame after initialization; deadline columns use ALL source frames after initialization, including dropped frames. Errors are canonical map pixels over scored samples only.", "",
           "Longest tracking lost includes acquisition and ends only after 0.5 s of fresh, reference-consistent XY on the correct layer. Its position envelope is derived separately per map scale from OTHER recordings' reference self-consistency and raster resolution, without using tracking outputs. It measures distinguishable localization failure, not whether auto cruise can steer successfully. Unknown reference intervals never confirm recovery. * marks an episode reaching recording end without verified recovery; ? marks unknown time within the longest episode.", "",
           "Longest CONTROL lost: n/a for every replay. These recordings do not evaluate the closed action/response loop, local visual steering, or heading correctness. Neither low proxy loss nor smooth XY establishes a cruising pass.", "",
           "| Cohort | Run | Init s | Fresh XY | Source XY <=33ms | Source XY <=67ms | XY latency P95 ms | Proxy error P95 px | Proxy coverage | Longest tracking lost s | Lost after acquisition s | Unknown s |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def number(v):
        return "n/a" if v is None else f"{v:.2f}"
    def lost(value):
        episodes = value["episodes"]
        longest = max(episodes, key=lambda e:e["seconds"], default={})
        return number(value["longest_lost_s"]) + ("*" if longest.get("unrecovered_at_end") else "") + ("?" if longest.get("unknown_s") else "")
    for r in results:
        loss = r['tracking_loss']
        lines.append(f"| {r['cohort']} | {r['session']} | {number(r['initialization_s'])} | {100*r['steady_fresh_rate']:.1f}% | {100*r['steady_source_fresh_xy_within_33ms']:.1f}% | {100*r['steady_source_fresh_xy_within_67ms']:.1f}% | {number(r['steady_capture_to_publish_ms']['p95'])} | {number(r['reference_error_px']['p95'])} | {100*r['reference_coverage_rate']:.1f}% | {lost(loss)} | {number(loss['longest_post_acquisition_lost_s'])} | {number(loss['unknown_s'])} |")
    lines += ["", "## Reference resolution envelopes", "", "These distinguish errors from reference/raster uncertainty; they are not path widths or learned cruising tolerances. Each evaluated recording is excluded from its calibration. Missing calibration leaves position agreement unverified.", "",
              "| Cohort | Run | World envelope px | Town envelope px |", "|---|---:|---:|---:|"]
    for r in results:
        modes = (r['tracking_loss'].get('calibration') or {}).get('modes', {})
        lines.append(f"| {r['cohort']} | {r['session']} | {number(modes.get('world', {}).get('error_limit_px'))} | {number(modes.get('town', {}).get('error_limit_px'))} |")
    lines += ["", "## Diagnostic loss tolerance sensitivity", "", "Arbitrary fixed pixel limits below are sensitivity checks, not the headline or cruising acceptance requirements. Wrong layer and unavailable XY cause loss at every threshold. Nonfresh XY is retained separately and does not establish position correctness. Zero observed loss with unknown intervals is not a reliability pass.", "",
              "| Cohort | Run | Lost at 5 px s | Lost at 10 px s | Lost at 20 px s | Longest nonfresh XY s |", "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        values = r['tracking_loss_sensitivity']
        lines.append(f"| {r['cohort']} | {r['session']} | {lost(values['5.0'])} | {lost(values['10.0'])} | {lost(values['20.0'])} | {r['longest_nonfresh_xy_s']:.2f} |")
    lines += ["", "## Heading publication", "", "Heading availability and timing are measured; these recordings do not provide external per-frame heading truth.", "",
              "| Cohort | Run | Heading available | Fresh heading / processed | Heading latency P95 ms | Source fresh heading <=33ms | Source fresh heading <=67ms |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['cohort']} | {r['session']} | {100*r['heading_available_rate']:.1f}% | {100*r['cursor_fresh_rate']:.1f}% | {number(r['fresh_heading_capture_to_publish_ms']['p95'])} | {100*r['steady_source_fresh_heading_within_33ms']:.1f}% | {100*r['steady_source_fresh_heading_within_67ms']:.1f}% |")
    lines += ["", "Full per-session initialization, raw telemetry, held/unavailable counts, loss episodes, jumps, error mean/median/P95/worst, decode/scheduling timing, source identities and implementation hashes are in each run directory. Repeated-waypoint evidence for laps is in repeatability.json. Results do not certify a physical live capture or display pass."]
    (root/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("Wrote",len(results),"runs to",root/"REPORT.md")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path)
    build(p.parse_args().root)
