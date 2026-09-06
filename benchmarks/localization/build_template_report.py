"""Summarize known-start template replays without treating priors as measurements."""

import argparse
from collections import Counter
import json
from pathlib import Path

from benchmarks.localization.reference_cache import identity
from benchmarks.localization.run_workbench_replay import distribution, read_rows


def build(root):
    comparisons = []
    checked = {}
    for path in sorted(root.glob("*/run*/report.json")):
        report = json.loads(path.read_text())
        reference = Path(report["reference"])
        marker = json.loads((reference/"cache.json").read_text())
        # Verify actual capture, atlas and calibration inputs as well as cache outputs.
        for item in marker["protocol"]["inputs"]:
            if item["path"] not in checked:
                actual = identity(item["path"])
                if actual["sha256"] != item["sha256"]:
                    raise RuntimeError("Frozen input changed: "+item["path"])
                checked[item["path"]] = actual
        for item in marker["outputs"]:
            if identity(reference/item["name"])["sha256"] != item["sha256"]:
                raise RuntimeError("Frozen reference output changed")
        rows = read_rows(path.parent/"scored_telemetry.jsonl")
        source = read_rows(path.parent/"source_telemetry.jsonl")
        first_ns = source[0]["session_time_ns"]
        fresh = [r for r in rows if r.get("xy_measurement_fresh_accepted")]
        loss = {k:v for k,v in report["tracking_loss"].items() if k != "calibration"}
        joint = [r for r in rows if r.get("xy_measurement_fresh_accepted")
                 and r.get("cursor_pose_measurement_fresh_accepted")
                 and r["capture_to_control_publish_ms"] <= 1000/30]
        first = rows[0]
        result = {"cohort": path.parent.parent.name, "report": identity(path),
                  "source_frames": len(source), "processed_frames": len(rows),
                  "duration_s": report["duration_s"], "mode": report["mode"],
                  "experiment": report["experiment"], "error": report["error"],
                  "first_pose_s": report["initialization_s"],
                  "first_fresh_xy_s": (fresh[0]["session_time_ns"]-first_ns)/1e9 if fresh else None,
                  "first_frame": {k:first.get(k) for k in (
                      "frame_index", "xy_measurement_fresh_accepted", "pose", "reference_error_px",
                      "capture_to_control_publish_ms", "cursor_pose_measurement_fresh_accepted", "known_start")},
                  "output_counts": dict(Counter(r["output_provenance"] for r in rows)),
                  "tracking_loss": loss,
                  "global_fresh_results": sum(bool(r.get("global_fix_fresh")) for r in rows),
                  "frames_with_global_running": sum(bool(r.get("global_localization_running")) for r in rows),
                  "all_source_fresh_xy_rate": len(fresh)/len(source),
                  "all_source_joint_xy_heading_33ms_rate": len(joint)/len(source),
                  "local_template_ms": distribution([(r.get("route_tracking") or {})["elapsed_ms"]
                      for r in rows if r.get("route_tracking_fresh") and "elapsed_ms" in (r.get("route_tracking") or {})]),
                  "reference_error_px": report["reference_error_px"],
                  "output_step_px": report["output_step_px"], "steps_over_8px": report["steps_over_8px"],
                  "publication_ms": report["steady_capture_to_publish_ms"],
                  "engine_ms": report["steady_engine_ms"], "decode_ms": report["decode_ms"],
                  "release_lateness_ms": report["source_release_lateness_ms"]}
        comparisons.append(result)
    result = {"reference_role": "same-atlas SIFT-derived proxy; demonstrated start is explicitly also an inference prior",
              "verified_inputs": list(checked.values()), "replays": comparisons}
    (root/"comparison.json").write_text(json.dumps(result, indent=2))
    lines = ["# Known-start template replay comparison", "",
             "Source-time acquisition is distinct from first-frame publication latency. Loss uses inferred references; unknown intervals remain unknown.", "",
             "| Cohort | First fresh XY s | P95 error px | Longest lost / after acquisition s | Fresh / held / unavailable | Publication P95 ms |",
             "|---|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "—" if value is None else f"{value:.3f}"
    for r in comparisons:
        c, loss = r["output_counts"], r["tracking_loss"]
        lines.append(f"| {r['cohort']} | {fmt(r['first_fresh_xy_s'])} | {fmt(r['reference_error_px']['p95'])} | "
                     f"{fmt(loss['longest_lost_s'])} / {fmt(loss['longest_post_acquisition_lost_s'])} | "
                     f"{c.get('fresh',0)} / {c.get('held',0)} / {c.get('unavailable',0)} | {fmt(r['publication_ms']['p95'])} |")
    (root/"COMPARISON.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    build(parser.parse_args().root)
