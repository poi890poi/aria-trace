"""Replay original-cadence video through the actual Workbench publication loop.

Only the capture adapter is replaced. This measures recorded-source E2E, not
physical GDI/camera capture or browser/display latency. No scoring references
enter free-roam tracking. Route-assisted runs may use only the declared demo.
"""

import argparse
import json
import platform
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from aria_trace.apps.workbench.application import AcquisitionWorkbench
from rig_runtime.domain.packets import FramePacket
from rig_runtime.adapters.filesystem.profiles import ProfileCatalog
from benchmarks.localization.reference_cache import ensure_reference, identity


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class RecordedSource:
    def __init__(self, session, max_seconds=None):
        self.session = Path(session)
        self.frames = read_rows(self.session / "frames.jsonl")
        if max_seconds is not None:
            self.frames = [r for r in self.frames if r["session_time_ns"] <= max_seconds * 1e9]
        self.stop_event = threading.Event()
        self.finished = threading.Event()
        self.index = 0
        self.rows = []

    def start(self):
        self.capture = cv2.VideoCapture(str(self.session / "video_main.mkv"))
        if not self.capture.isOpened():
            raise RuntimeError("Replay video cannot be opened")
        self.origin = time.perf_counter_ns() + 100_000_000

    def read(self):
        if self.index >= len(self.frames) or self.stop_event.is_set():
            self.finished.set()
            self.stop_event.wait(0.005)
            return None
        frame = self.frames[self.index]
        scheduled = self.origin + frame["session_time_ns"]
        if self.stop_event.wait(max(0.0, (scheduled - time.perf_counter_ns()) / 1e9)):
            return None
        started = time.perf_counter_ns()
        ok, image = self.capture.read()
        decoded = time.perf_counter_ns()
        if not ok:
            raise RuntimeError("Unexpected video EOF at frame {}".format(self.index))
        self.rows.append({"frame_index": frame["frame_index"], "session_time_ns": frame["session_time_ns"],
                          "host_time_ns": scheduled, "decode_ms": (decoded-started)/1e6,
                          "release_lateness_ms": (started-scheduled)/1e6})
        self.index += 1
        return FramePacket("main", image, scheduled, decoded)

    def stop(self):
        self.stop_event.set()


def distribution(values):
    a = np.asarray(values, dtype=float)
    return {"count": len(a), **({k: None for k in ("mean", "median", "p95", "worst")} if not len(a) else
            {"mean": float(a.mean()), "median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "worst": float(a.max())})}


def score(rows, source, reference):
    source_by_time = {r["host_time_ns"]: r for r in source.rows}
    refs = read_rows(reference / "route_states.jsonl")
    times = np.array([r["session_time_ns"] for r in refs], dtype=np.int64)
    manifest = json.loads((reference / "manifest.json").read_text())
    max_gap = 1.5e9 / manifest["reference_rate_hz"]
    errors, steps, losses, enriched = [], [], [], []
    previous = None
    loss = None
    first_pose = next((r["host_time_ns"] for r in rows if r.get("pose")), None)
    mode_matches = []
    for row in rows:
        row = dict(row)
        record = source_by_time[row["host_time_ns"]]
        row.update(record)
        pose = row.get("pose")
        fresh = bool(row.get("xy_measurement_fresh_accepted"))
        row["output_provenance"] = "fresh" if fresh else "held" if pose else "unavailable"
        if first_pose is not None and row["host_time_ns"] >= first_pose:
            if not fresh and loss is None:
                loss = row["session_time_ns"]
            elif fresh and loss is not None:
                losses.append({"start_ns": loss, "end_ns": row["session_time_ns"], "seconds": (row["session_time_ns"]-loss)/1e9})
                loss = None
        t = row["session_time_ns"]
        i = int(np.searchsorted(times, t))
        target = None
        mode = None
        if i < len(times) and times[i] == t:
            target, mode = np.array(refs[i]["canonical_xy"]), refs[i]["mode_id"]
        elif 0 < i < len(times) and times[i]-times[i-1] <= max_gap and refs[i]["mode_id"] == refs[i-1]["mode_id"]:
            fraction = (t-times[i-1])/(times[i]-times[i-1])
            target = np.array(refs[i-1]["canonical_xy"]) * (1-fraction) + np.array(refs[i]["canonical_xy"]) * fraction
            mode = refs[i]["mode_id"]
        if pose:
            xy = np.array([pose["x"], pose["y"]])
            if previous is not None:
                step = float(np.linalg.norm(xy-previous))
                steps.append(step)
                row["output_step_px"] = step
            previous = xy
            if target is not None:
                error = float(np.linalg.norm(xy-target))
                errors.append(error)
                row["reference_error_px"] = error
                row["reference_mode"] = mode
                mode_matches.append(row.get("active_map_mode_id") == mode)
        enriched.append(row)
    if loss is not None:
        end = source.frames[-1]["session_time_ns"]
        losses.append({"start_ns": loss, "end_ns": end, "seconds": (end-loss)/1e9})
    steady = [r for r in enriched if first_pose is not None and r["host_time_ns"] >= first_pose]
    duration = (source.frames[-1]["session_time_ns"] - source.frames[0]["session_time_ns"])/1e9
    fresh = sum(r["output_provenance"] == "fresh" for r in steady)
    return enriched, {
        "source_frames": len(source.rows), "processed_frames": len(rows), "steady_frames": len(steady),
        "duration_s": duration, "processed_fps": len(rows)/duration,
        "initialization_s": (first_pose-source.origin-source.frames[0]["session_time_ns"])/1e9 if first_pose else None,
        "steady_fresh_rate": fresh/max(len(steady), 1),
        "all_source_fresh_rate": sum(bool(r.get("xy_measurement_fresh_accepted")) for r in rows)/len(source.rows),
        "held_frames": sum(r["output_provenance"] == "held" for r in enriched),
        "unavailable_frames": sum(r["output_provenance"] == "unavailable" for r in enriched),
        "loss_episodes": losses, "longest_loss_s": max((r["seconds"] for r in losses), default=0),
        "reference_error_px": distribution(errors), "reference_coverage_rate": len(errors)/max(len(rows),1),
        "reference_mode_agreement": float(np.mean(mode_matches)) if mode_matches else None,
        "output_step_px": distribution(steps), "steps_over_8px": sum(v > 8 for v in steps),
        "steady_capture_to_publish_ms": distribution([r["capture_to_control_publish_ms"] for r in steady]),
        "steady_engine_ms": distribution([r["update_elapsed_ms"] for r in steady]),
        "steady_fresh_within_33ms_rate": sum(bool(r.get("xy_measurement_fresh_accepted")) and r["capture_to_control_publish_ms"] <= 1000/30 for r in steady)/max(len(steady),1),
        "steady_fresh_within_67ms_rate": sum(bool(r.get("xy_measurement_fresh_accepted")) and r["capture_to_control_publish_ms"] <= 1000/15 for r in steady)/max(len(steady),1),
        "cursor_fresh_rate": sum(bool(r.get("cursor_pose_measurement_fresh_accepted")) for r in steady)/max(len(steady),1),
        "decode_ms": distribution([r["decode_ms"] for r in source.rows]),
        "source_release_lateness_ms": distribution([r["release_lateness_ms"] for r in source.rows]),
    }


def run(args):
    root = Path.cwd()
    artifacts = root / "artifacts/workbench"
    game = "genshin-impact-pc"
    atlas = artifacts / "map_atlases" / game / args.atlas
    calibration = artifacts / "minimap_calibrations" / game / args.calibration
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    profiles = ProfileCatalog()
    mini_config = profiles.game(game)["minimap_calibration"]
    references = {}
    runs = list(dict.fromkeys(([11] if args.mode == "route-assisted" else []) + args.runs))
    for number in runs:
        session = root / "sessions/workbench/recordings-genshin-impact-pc" / f"run_{number:02d}"
        entry, hit = ensure_reference(root, session, atlas, calibration, mini_config, args.cache, rate=5.0 if number == 11 else 2.0)
        references[number] = entry
        print("REFERENCE", number, "HIT" if hit else "BUILT", entry, flush=True)
    (output / "references.json").write_text(json.dumps({str(k): str(v) for k,v in references.items()}, indent=2))
    if args.references_only:
        return
    for number in args.runs:
        target = output / f"run{number:02d}"
        if (target / "report.json").exists():
            raise RuntimeError("Refusing to overwrite completed run " + str(target))
        source = RecordedSource(root / "sessions/workbench/recordings-genshin-impact-pc" / f"run_{number:02d}", args.max_seconds)
        state = AcquisitionWorkbench(output / "empty_sessions", output, profiles=profiles)
        state._minimap_calibration_root = lambda game: artifacts / "minimap_calibrations" / game
        state._scene_yaw_root = lambda game: artifacts / "scene_yaw_calibrations" / game
        state._map_atlas_root = lambda game: artifacts / "map_atlases" / game
        state._route_tracking_root = lambda game: args.cache.resolve()
        state._live_tracking_root = lambda game: target
        state.sources.capture_sources = lambda frame, inputs: (source, None)
        request = {"game_profile_id": game, "minimap_calibration_id": args.calibration,
                   "scene_yaw_calibration_id": args.scene_yaw, "map_atlas_id": args.atlas,
                   "tracking_profile": "real-time", "tracking_mode": args.mode,
                   "route_package_id": references[11].name if args.mode == "route-assisted" else None,
                   "record_route_video": args.record_video, "auto_stop_at_route_finish": False,
                   "frame_source": {"adapter": "windows_window", "window_title": "RECORDED SOURCE", "fps": 30}}
        print("START", number, args.mode, flush=True)
        try:
            state.start_live_tracker(request)
            while not source.finished.wait(0.25):
                if state._live_tracker.get("error"):
                    raise RuntimeError(state._live_tracker["error"])
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                latest = state._live_tracker.get("latest") or {}
                if latest.get("host_time_ns") == source.rows[-1]["host_time_ns"]:
                    break
                time.sleep(0.02)
            state.stop_live_tracker()
            state._live_tracker["thread"].join(timeout=30)
            if state._live_tracker["thread"].is_alive():
                raise RuntimeError("Workbench did not finish")
            tracking_id = state._live_tracker["tracking_id"]
            rows = read_rows(target / tracking_id / "telemetry.jsonl")
            enriched, summary = score(rows, source, references[number])
            summary.update({"session": number, "mode": args.mode, "request": request,
                            "reference": str(references[number]), "reference_role": "slow-inferred-atlas-reference-not-external-truth",
                            "evidence": str(target/tracking_id), "status": state._live_tracker["status"],
                            "error": state._live_tracker.get("error"), "capture_dropped": state._live_tracker.get("capture_dropped_before_processing"),
                            "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                            "git_status": subprocess.check_output(["git", "status", "--short"], text=True).strip(),
                            "python": platform.python_version(), "opencv": cv2.__version__, "opencv_threads": cv2.getNumThreads(),
                            "implementation": [identity(p) for directory in ("aria_trace", "rig_runtime", "replay", "benchmarks/localization") for p in sorted((root/directory).rglob("*.py"))],
                            "measurement_boundary": "recorded source schedule through actual Workbench publication; excludes physical capture and browser/display",
                            "record_video": args.record_video})
            (target / "scored_telemetry.jsonl").write_text("".join(json.dumps(r)+"\n" for r in enriched))
            (target / "source_telemetry.jsonl").write_text("".join(json.dumps(r)+"\n" for r in source.rows))
            (target / "report.json").write_text(json.dumps(summary, indent=2))
            print("DONE", number, json.dumps({k:v for k,v in summary.items() if k in ("steady_fresh_rate", "initialization_s", "reference_error_px", "steady_capture_to_publish_ms", "error")}), flush=True)
        finally:
            state.close()
            if hasattr(source, "capture"):
                source.capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=int, default=[11, 14, 15, 16, 17, 18])
    parser.add_argument("--atlas", default="08b6f2d6-820a-4bfd-875a-6a55d1986a4e")
    parser.add_argument("--calibration", default="segments-df624035-833-bd07601f-708")
    parser.add_argument("--scene-yaw", default="01dbaa74-8e00-4763-a215-9ea37e18b1b2")
    parser.add_argument("--mode", choices=["free-roam", "route-assisted"], default="free-roam")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("artifacts/benchmark_cache/atlas_references"))
    parser.add_argument("--references-only", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
