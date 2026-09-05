"""Score the preserved CPU XFeat probes and summarize paired Workbench replays."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from benchmarks.localization.reference_cache import identity
from benchmarks.localization.run_workbench_replay import distribution, read_rows


def reference_at(rows, t, max_gap):
    times = np.array([r["session_time_ns"] for r in rows], dtype=np.int64)
    i = int(np.searchsorted(times, t))
    if i < len(rows) and times[i] == t:
        return np.array(rows[i]["canonical_xy"]), rows[i]["mode_id"]
    if 0 < i < len(rows) and times[i]-times[i-1] <= max_gap and rows[i]["mode_id"] == rows[i-1]["mode_id"]:
        f = (t-times[i-1])/(times[i]-times[i-1])
        return np.array(rows[i-1]["canonical_xy"])*(1-f)+np.array(rows[i]["canonical_xy"])*f, rows[i]["mode_id"]
    return None, None


def build(root):
    refs = json.loads(Path("artifacts/poc/workbench-rebuilt-atlas-20260905/references/references.json").read_text())
    probes, replays = [], []
    source_hashes = {}
    cache_ids = {}
    for path in sorted(root.glob("probe-*/probe.json")):
        cohort = path.parent
        metadata = json.loads((cohort/"candidate-source/manifest.json").read_text())
        for original in json.loads(path.read_text())["summaries"]:
            number = original["session"]
            entry = Path(refs[str(number)])
            marker = json.loads((entry/"cache.json").read_text())
            for item in marker["protocol"]["inputs"]:
                if "map_atlases" in item["path"] or "minimap_calibrations" in item["path"]:
                    if identity(item["path"])["sha256"] != item["sha256"]:
                        raise RuntimeError("Atlas/calibration no longer matches frozen reference inputs")
            for item in marker["outputs"]:
                if identity(entry/item["name"])["sha256"] != item["sha256"]:
                    raise RuntimeError("Damaged reference")
            cache_ids[str(number)] = {"path": str(entry), "cache_marker": identity(entry/"cache.json")}
            reference = read_rows(entry/"route_states.jsonl")
            max_gap = 1.5e9/json.loads((entry/"manifest.json").read_text())["reference_rate_hz"]
            rows = read_rows(cohort/f"probe{number:02d}.jsonl")
            errors, wrong_modes, known = [], 0, 0
            enriched = []
            for row in rows:
                key = (number, row["frame_index"])
                digest = row["observation_sha256"]
                if key in source_hashes and source_hashes[key] != digest:
                    raise RuntimeError("Compared observations differ")
                source_hashes[key] = digest
                xy, mode = reference_at(reference, row["session_time_ns"], max_gap)
                if xy is not None:
                    known += 1
                if xy is not None and row["valid"]:
                    err = float(np.linalg.norm(np.array(row["xy"])-xy))
                    errors.append(err)
                    wrong_modes += row["mode"] != mode
                    row = dict(row, reference_error_px=err, reference_mode=mode)
                enriched.append(row)
            (cohort/f"scored_probe{number:02d}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in enriched))
            probes.append(dict(original, cohort=cohort.name, variant=metadata["variant"], feature_scale=metadata["feature_scale"] if "feature_scale" in metadata else 1,
                input_mask=metadata.get("input_mask", "none"), mask_region=metadata.get("mask_region", "both"),
                reference_known_samples=known, reference_accepted_error_px=distribution(errors), wrong_reference_modes=wrong_modes,
                rejections=dict(Counter(reason for r in rows for reason in r["reasons"]))))
    for path in sorted(root.glob("*/run*/report.json")):
        report = json.loads(path.read_text())
        keys = ["session", "mode", "duration_s", "source_frames", "processed_frames", "initialization_s", "held_frames", "unavailable_frames",
                "tracking_loss", "reference_error_px", "steady_fresh_rate", "steady_capture_to_publish_ms", "steady_engine_ms", "error"]
        row = {k: report.get(k) for k in keys}
        row["cohort"] = path.parent.parent.name
        telemetry = read_rows(path.parent/"scored_telemetry.jsonl")
        row["all_processed_publication_ms"] = distribution([r["capture_to_control_publish_ms"] for r in telemetry])
        replays.append(row)
    result = {"reference_role": "frozen slow SIFT/atlas inferred proxy, not external truth", "caches": cache_ids,
              "probes": probes, "replays": replays, "unique_observations_hash_checked": len(source_hashes)}
    (root/"comparison.json").write_text(json.dumps(result, indent=2))
    lines = ["# CPU XFeat measured comparison", "", "Probe timing is the whole unrestricted two-layer query, excluding decode/model setup. Accepted is a pose-gate result, not external accuracy.", "",
             "| Cohort | Run | Accepted | Wall mean / P95 ms | CPU mean ms | Accepted reference P95 px |", "|---|---:|---:|---:|---:|---:|"]
    def fmt(x):
        return "—" if x is None else f"{x:.2f}"
    for row in probes:
        lines.append(f"| {row['cohort']} | {row['session']} | {row['accepted']}/{row['samples']} | {fmt(row['wall_ms']['mean'])} / {fmt(row['wall_ms']['p95'])} | {fmt(row['process_cpu_ms']['mean'])} | {fmt(row['reference_accepted_error_px']['p95'])} |")
    lines += ["", "| E2E cohort | Run | Mode | First pose s | Unavailable / processed | Reference P95 px |", "|---|---:|---|---:|---:|---:|"]
    for row in replays:
        lines.append(f"| {row['cohort']} | {row['session']} | {row['mode']} | {fmt(row['initialization_s'])} | {row['unavailable_frames']}/{row['processed_frames']} | {fmt(row['reference_error_px']['p95'])} |")
    (root/"COMPARISON.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    build(parser.parse_args().root)
