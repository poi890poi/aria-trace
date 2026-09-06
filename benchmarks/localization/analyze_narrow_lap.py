"""Post-run narrow-lap evidence: reference path, native frames, and transitions."""

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2
import numpy as np

from benchmarks.localization.run_workbench_replay import read_rows, distribution
from benchmarks.localization.reference_cache import identity


def analyze(root):
    session = Path("sessions/workbench/recordings-genshin-impact-pc/run_20")
    reference = Path(json.loads((root / "references.json").read_text())["20"])
    refs = read_rows(reference / "route_states.jsonl")
    frames = read_rows(session / "frames.jsonl")
    inputs = read_rows(session / "inputs.jsonl")
    times = np.array([r["session_time_ns"] / 1e9 for r in frames])
    changes = []
    for previous, current in zip(refs, refs[1:]):
        if previous["mode_id"] != current["mode_id"]:
            changes.append({"from": previous["mode_id"], "to": current["mode_id"],
                            "last_old_reference_s": previous["session_time_ns"] / 1e9,
                            "first_new_reference_s": current["session_time_ns"] / 1e9,
                            "new_reference_xy": current["canonical_xy"],
                            "old_scale": previous["map_scale"], "new_scale": current["map_scale"]})
    result = {"reference": identity(reference / "cache.json"), "source": identity(session / "manifest.json"),
              "reference_transition_brackets": changes, "capture_interval_ms": distribution(np.diff(times) * 1000),
              "capture_gaps_over_100ms": int(sum(np.diff(times) > .1)), "replays": []}
    for path in sorted(root.glob("*/run20/report.json")):
        rows = read_rows(path.parent / "scored_telemetry.jsonl")
        report = json.loads(path.read_text())
        events = []
        previous_mode = None
        for r in rows:
            mode = r.get("active_map_mode_id")
            if mode and mode != previous_mode:
                events.append({"time_s": r["session_time_ns"] / 1e9, "from": previous_mode, "to": mode})
                previous_mode = mode
        matches = []
        for i, change in enumerate(changes):
            begin = change["last_old_reference_s"]
            finish = changes[i+1]["last_old_reference_s"] if i+1 < len(changes) else times[-1]
            hit = next((e for e in events if begin <= e["time_s"] < finish and e["to"] == change["to"]), None)
            matches.append({**change, "detected": hit is not None,
                            "detection_s": hit["time_s"] if hit else None,
                            "relative_to_first_new_reference_s": hit["time_s"] - change["first_new_reference_s"] if hit else None})
        wrong_fresh = [r for r in rows if r.get("xy_measurement_fresh_accepted") and r.get("reference_mode") and r.get("active_map_mode_id") != r["reference_mode"]]
        result["replays"].append({"cohort": path.parent.parent.name, "events": events, "transitions": matches,
                                   "wrong_layer_fresh_frames": len(wrong_fresh),
                                   "bad_fresh_over_20px": sum(bool(r.get("xy_measurement_fresh_accepted")) and (r.get("reference_error_px") or 0) > 20 for r in rows),
                                   "loss": {k: v for k, v in report["tracking_loss"].items() if k != "calibration"},
                                   "mode_counts": dict(Counter(r.get("active_map_mode_id") for r in rows))})
    # Input behavior is a timing observation, not position or attention truth.
    windows = []
    for start in range(0, 240, 5):
        records = [r for r in inputs if start <= r["session_time_ns"] / 1e9 < start + 5]
        mouse = [r["payload"] for r in records if r["kind"] == "pc_raw_mouse"]
        keys = Counter(r["payload"].get("key_name") for r in records if r["kind"] == "pc_raw_keyboard" and r["payload"].get("pressed"))
        windows.append({"start_s": start, "mouse_dx_net": sum(r.get("delta_x", 0) for r in mouse),
                        "mouse_dx_absolute": sum(abs(r.get("delta_x", 0)) for r in mouse),
                        "nonzero_mouse_events": sum(bool(r.get("delta_x") or r.get("delta_y")) for r in mouse),
                        "key_down_packets_including_repeat": dict(keys)})
    result["input_windows_5s"] = windows
    quiet = [w for w in windows if not w["nonzero_mouse_events"] and w["key_down_packets_including_repeat"].get("W")]
    result["quiet_mouse_heading_diagnostics"] = []
    for path in sorted(root.glob("*/run20/report.json")):
        rows = read_rows(path.parent / "scored_telemetry.jsonl")
        for window in quiet:
            start = window["start_s"]
            selected_rows = [r for r in rows if start <= r["session_time_ns"] / 1e9 < start + 5 and r.get("cursor_pose_measurement_fresh_accepted")]
            if len(selected_rows) < 2:
                continue
            angles = np.degrees(np.unwrap(np.radians([r["cursor_pose"]["angle_screen_deg"] for r in selected_rows])))
            result["quiet_mouse_heading_diagnostics"].append({"cohort": path.parent.parent.name, "start_s": start,
                "samples": len(angles), "angular_span_deg": float(np.ptp(angles)), "angular_std_deg": float(np.std(angles)),
                "successive_step_deg": distribution(np.abs(np.diff(angles))),
                "role": "input-idle cursor-heading stability, not independent yaw accuracy"})
    visual = root / "visual-review"
    visual.mkdir(exist_ok=True)
    selected = [2, 8, 15, 25, 35, 45, 55, 66, 76, 85, 98, 112, 125, 142, 155, 176, 191, 202, 221, 234]
    cap = cv2.VideoCapture(str(session / "video_main.mkv"))
    thumb_groups = []
    for second in selected:
        index = int(np.argmin(abs(times - second)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Cannot decode native frame")
        cv2.imwrite(str(visual / ("native-%03ds.png" % second)), frame)
        thumb = cv2.resize(frame, (640, 360))
        thumb = cv2.copyMakeBorder(thumb, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        cv2.putText(thumb, "%.3f s / frame %d" % (times[index], index), (10, 21), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
        thumb_groups.append(thumb)
    cap.release()
    for i in range(0, len(thumb_groups), 4):
        cv2.imwrite(str(visual / ("scenes-%d.png" % (i // 4 + 1))), np.vstack([np.hstack(thumb_groups[i:i+2]), np.hstack(thumb_groups[i+2:i+4])]))
    xy = np.asarray([r["canonical_xy"] for r in refs])
    atlas_path = Path("artifacts/workbench/map_atlases/genshin-impact-pc/08b6f2d6-820a-4bfd-875a-6a55d1986a4e/canonical_mosaic.png")
    atlas = cv2.imread(str(atlas_path))
    low = np.floor(xy.min(0) - 20).astype(int); high = np.ceil(xy.max(0) + 20).astype(int)
    zoom = 3
    canvas = cv2.resize(atlas[low[1]:high[1], low[0]:high[0]], None, fx=zoom, fy=zoom)
    points = np.rint((xy-low)*zoom).astype(int)
    for i in range(1, len(points)):
        if refs[i]["session_time_ns"] - refs[i-1]["session_time_ns"] > 600_000_000:
            continue
        hue = int(150 * refs[i]["session_time_ns"] / 240e9)
        color = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])
        cv2.line(canvas, points[i-1], points[i], color, 2, cv2.LINE_AA)
    for second in range(0, 240, 10):
        i = min(range(len(refs)), key=lambda i: abs(refs[i]["session_time_ns"] / 1e9 - second))
        cv2.circle(canvas, points[i], 3, (255, 255, 255), -1)
        cv2.putText(canvas, str(second), points[i]+[5,-5], cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1, cv2.LINE_AA)
    cv2.imwrite(str(visual / "inferred-path.png"), canvas)
    (root / "lap-analysis.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("input_windows_5s", "reference", "source")}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("root", type=Path)
    analyze(p.parse_args().root)
