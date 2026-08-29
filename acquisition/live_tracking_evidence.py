"""Asynchronous, crash-tolerant evidence capture for live localization runs."""

import json
import math
import queue
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


SCHEMA_VERSION = "1.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_value(value), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _encode_jpeg(image, quality=88):
    if image is None:
        return None
    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    )
    return encoded.tobytes() if ok else None


def _write_image(path: Path, image) -> bool:
    if image is None:
        return False
    return bool(cv2.imwrite(str(path), image))


class LiveTrackingEvidenceRecorder:
    """Persist telemetry and visual evidence without blocking the tracker loop."""

    def __init__(
        self,
        output_path: Path,
        metadata: dict,
        frame_sample_interval_s: float = 0.25,
        incident_pre_s: float = 2.0,
        incident_post_s: float = 2.0,
        jump_threshold_map_px: float = 8.0,
        queue_capacity: int = 256,
    ) -> None:
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        (self.output_path / "global_fixes").mkdir(exist_ok=True)
        (self.output_path / "incidents").mkdir(exist_ok=True)
        self.frame_sample_interval_ns = int(frame_sample_interval_s * 1.0e9)
        self.incident_pre_ns = int(incident_pre_s * 1.0e9)
        self.incident_post_ns = int(incident_post_s * 1.0e9)
        self.jump_threshold_map_px = float(jump_threshold_map_px)
        self._queue = queue.Queue(maxsize=max(8, int(queue_capacity)))
        self._lock = threading.Lock()
        self._last_frame_sample_ns = None
        self._dropped_records = 0
        self._error = None
        self._closed = False
        self._counts = {
            "telemetry_rows": 0,
            "global_fixes": 0,
            "jump_incidents": 0,
            "sampled_frames": 0,
        }
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "started_utc": _utc_now(),
            "finished_utc": None,
            "metadata": _json_value(metadata),
            "files": {
                "telemetry": "telemetry.jsonl",
                "global_fixes": "global_fixes.jsonl",
                "global_fix_evidence": "global_fixes/",
                "jump_incidents": "incidents/",
            },
            "capture_policy": {
                "frame_sample_interval_s": frame_sample_interval_s,
                "incident_pre_s": incident_pre_s,
                "incident_post_s": incident_post_s,
                "jump_threshold_map_px": jump_threshold_map_px,
                "queue_capacity": queue_capacity,
            },
            "counts": dict(self._counts),
            "dropped_records": 0,
            "error": None,
        }
        _atomic_json(self.output_path / "live_tracking.json", self.manifest)
        self._thread = threading.Thread(
            target=self._work,
            name="aria-live-evidence",
            daemon=True,
        )
        self._thread.start()

    @property
    def summary(self) -> dict:
        with self._lock:
            return {
                "status": "failed" if self._error else (
                    "complete" if self._closed else "running"
                ),
                "counts": dict(self._counts),
                "dropped_records": self._dropped_records,
                "error": self._error,
            }

    @staticmethod
    def _telemetry(state: dict) -> dict:
        row = {
            key: value
            for key, value in state.items()
            if key not in ("trail", "global_fix")
        }
        row["global_fix"] = (
            state.get("global_fix") if state.get("global_fix_fresh") else None
        )
        return _json_value(row)

    def record(self, frame, minimap, state: dict, diagnostics=None) -> bool:
        if self._closed:
            return False
        timestamp_ns = int(state.get("host_time_ns") or 0)
        sample_frame = (
            self._last_frame_sample_ns is None
            or timestamp_ns - self._last_frame_sample_ns
            >= self.frame_sample_interval_ns
        )
        if sample_frame:
            self._last_frame_sample_ns = timestamp_ns
        global_fresh = bool(state.get("global_fix_fresh"))
        item = {
            "kind": "record",
            "telemetry": self._telemetry(state),
            "frame": frame.copy() if sample_frame or global_fresh else None,
            "minimap": minimap.copy()
            if minimap is not None and (sample_frame or global_fresh)
            else None,
            "diagnostics": diagnostics if global_fresh else None,
            "sample_frame": sample_frame,
        }
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            with self._lock:
                self._dropped_records += 1
            return False

    def close(self, status="stopped", error=None, processed_frames=None) -> dict:
        if self._closed:
            return self.summary
        self._closed = True
        self._queue.put({"kind": "close"})
        self._thread.join(timeout=15.0)
        with self._lock:
            if self._thread.is_alive() and self._error is None:
                self._error = "Evidence worker did not stop within 15 seconds"
            final_status = "failed" if self._error else str(status)
            self.manifest.update(
                {
                    "status": final_status,
                    "finished_utc": _utc_now(),
                    "counts": dict(self._counts),
                    "dropped_records": self._dropped_records,
                    "error": self._error or error,
                    "processed_frames": processed_frames,
                }
            )
        _atomic_json(self.output_path / "live_tracking.json", self.manifest)
        return self.summary

    def _write_global_fix(self, index, item, global_stream) -> None:
        telemetry = item["telemetry"]
        fix = telemetry.get("global_fix") or {}
        fix_root = self.output_path / "global_fixes" / "fix_{:06d}".format(index)
        fix_root.mkdir(parents=True, exist_ok=True)
        diagnostics = item.get("diagnostics") or {}
        evidence = []
        images = {
            "source_frame.jpg": item.get("frame"),
            "minimap.png": item.get("minimap"),
            "observation.png": diagnostics.get("observation"),
            "observation_mask.png": diagnostics.get("mask"),
            "transformed_gradient.png": diagnostics.get("transformed_gradient"),
            "search_region.png": diagnostics.get("search_region"),
            "correlation_heatmap.png": diagnostics.get("correlation_heatmap"),
            "candidate_overlay.png": diagnostics.get("candidate_overlay"),
            "map_overlay.png": diagnostics.get("map_overlay"),
        }
        for name, image in images.items():
            if _write_image(fix_root / name, image):
                evidence.append(name)
        record = {
            "fix_index": index,
            "sequence": telemetry.get("sequence"),
            "host_time_ns": telemetry.get("host_time_ns"),
            "pose": telemetry.get("pose"),
            "global_fix": fix,
            "evidence": evidence,
            "artifact_relative_path": "global_fixes/fix_{:06d}".format(index),
        }
        _atomic_json(fix_root / "global_fix.json", record)
        global_stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        global_stream.flush()

    def _work(self) -> None:
        ring = deque()
        active_incidents = []
        previous_pose = None
        telemetry_path = self.output_path / "telemetry.jsonl"
        global_path = self.output_path / "global_fixes.jsonl"
        try:
            with telemetry_path.open("a", encoding="utf-8") as telemetry_stream, \
                    global_path.open("a", encoding="utf-8") as global_stream:
                while True:
                    item = self._queue.get()
                    if item.get("kind") == "close":
                        break
                    telemetry = item["telemetry"]
                    telemetry_stream.write(
                        json.dumps(telemetry, separators=(",", ":")) + "\n"
                    )
                    self._counts["telemetry_rows"] += 1
                    if self._counts["telemetry_rows"] % 30 == 0:
                        telemetry_stream.flush()

                    timestamp_ns = int(telemetry.get("host_time_ns") or 0)
                    sample = None
                    if item.get("sample_frame") and item.get("frame") is not None:
                        sample = {
                            "sequence": int(telemetry.get("sequence") or 0),
                            "host_time_ns": timestamp_ns,
                            "frame_jpeg": _encode_jpeg(item["frame"]),
                            "minimap_jpeg": _encode_jpeg(item.get("minimap")),
                        }
                        ring.append(sample)
                        self._counts["sampled_frames"] += 1
                    while ring and timestamp_ns - ring[0]["host_time_ns"] > self.incident_pre_ns:
                        ring.popleft()

                    if telemetry.get("global_fix_fresh"):
                        self._counts["global_fixes"] += 1
                        self._write_global_fix(
                            self._counts["global_fixes"], item, global_stream
                        )

                    pose = telemetry.get("pose")
                    pose_delta = 0.0
                    if pose and previous_pose:
                        pose_delta = math.hypot(
                            float(pose["x"]) - float(previous_pose["x"]),
                            float(pose["y"]) - float(previous_pose["y"]),
                        )
                    fix = telemetry.get("global_fix") or {}
                    fusion = fix.get("fusion") or {}
                    applied = float(fusion.get("applied_position_change_map_px") or 0.0)
                    jump = bool(
                        telemetry.get("global_fix_fresh")
                        and fusion.get("accepted")
                        and max(applied, pose_delta) >= self.jump_threshold_map_px
                    )
                    if jump:
                        self._counts["jump_incidents"] += 1
                        incident_index = self._counts["jump_incidents"]
                        incident_root = self.output_path / "incidents" / "jump_{:06d}".format(incident_index)
                        incident_root.mkdir(parents=True, exist_ok=True)
                        _atomic_json(
                            incident_root / "incident.json",
                            {
                                "incident_index": incident_index,
                                "sequence": telemetry.get("sequence"),
                                "host_time_ns": timestamp_ns,
                                "pose_delta_map_px": pose_delta,
                                "applied_position_change_map_px": applied,
                                "global_fix": fix,
                                "pre_s": self.incident_pre_ns / 1.0e9,
                                "post_s": self.incident_post_ns / 1.0e9,
                            },
                        )
                        for buffered in ring:
                            self._write_incident_sample(incident_root, buffered)
                        active_incidents.append(
                            (incident_root, timestamp_ns + self.incident_post_ns)
                        )
                    if sample is not None:
                        remaining = []
                        for incident_root, end_ns in active_incidents:
                            self._write_incident_sample(incident_root, sample)
                            if timestamp_ns < end_ns:
                                remaining.append((incident_root, end_ns))
                        active_incidents = remaining
                    if pose:
                        previous_pose = dict(pose)
                telemetry_stream.flush()
                global_stream.flush()
        except Exception as exc:
            with self._lock:
                self._error = "{}: {}".format(type(exc).__name__, exc)

    @staticmethod
    def _write_incident_sample(root: Path, sample: dict) -> None:
        stem = "frame_{:08d}_{:019d}".format(
            int(sample["sequence"]), int(sample["host_time_ns"])
        )
        frame = sample.get("frame_jpeg")
        minimap = sample.get("minimap_jpeg")
        if frame is not None:
            (root / (stem + ".jpg")).write_bytes(frame)
        if minimap is not None:
            (root / (stem + "_minimap.jpg")).write_bytes(minimap)
