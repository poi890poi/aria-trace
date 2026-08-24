"""Versioned session writer and reader."""

import json
import os
import re
import sqlite3
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from . import SCHEMA_VERSION
from .models import FramePacket, InputPacket
from .video import create_video_sink
from .annotations import AnnotationStore


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


INPUT_ERROR_KINDS = {
    "pc_input_error",
    "pc_xinput_error",
    "pc_raw_input_error",
}


def input_capture_health(manifest: dict) -> dict:
    """Summarize whether a configured workbench input stream has evidence."""
    context = manifest.get("context") or {}
    adapter = context.get("input_adapter")
    required = bool(adapter and adapter != "none")
    counts = manifest.get("input_counts") or {}
    control_events = sum(
        int(count)
        for key, count in counts.items()
        if str(key).rsplit(":", 1)[-1] not in INPUT_ERROR_KINDS
    )
    error_events = sum(
        int(count)
        for key, count in counts.items()
        if str(key).rsplit(":", 1)[-1] in INPUT_ERROR_KINDS
    )
    return {
        "adapter": adapter or "unknown",
        "required": required,
        "control_events": control_events,
        "error_events": error_events,
        "healthy": not required or control_events > 0,
    }


class SessionWriter:
    def __init__(
        self,
        path: Path,
        frame_sources,
        input_sources,
        video_encoding: str = "h264",
        video_fps: float = 30.0,
        video_crf: int = 20,
        video_preset: str = "veryfast",
        ffmpeg: Optional[Path] = None,
        frame_processors=(),
        session_context: Optional[dict] = None,
    ) -> None:
        self.path = Path(path)
        if self.path.exists() and any(self.path.iterdir()):
            raise RuntimeError("Session directory is not empty: {}".format(self.path))
        self.path.mkdir(parents=True, exist_ok=True)
        self.video_encoding = video_encoding
        self.video_fps = video_fps
        self.video_crf = video_crf
        self.video_preset = video_preset
        self.ffmpeg = ffmpeg
        self.frame_processors = list(frame_processors)
        self.session_id = str(uuid.uuid4())
        self.session_context = dict(session_context or {})
        self.origin_ns = time.perf_counter_ns()
        self.frame_counts = Counter()
        self.input_counts = Counter()
        self.drop_counts = Counter()
        self.input_drop_counts = Counter()
        self._video_sinks = {}
        self._video_shapes = {}
        self._frames_file = (self.path / "frames.jsonl").open("w", encoding="utf-8", buffering=1)
        self._inputs_file = (self.path / "inputs.jsonl").open("w", encoding="utf-8", buffering=1)
        (self.path / "annotations.jsonl").open("a", encoding="utf-8").close()
        self._closed = False
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "status": "recording",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "pc_monotonic_origin_ns": self.origin_ns,
            "context": self.session_context,
            "frame_sources": [source.describe() for source in frame_sources],
            "input_sources": [source.describe() for source in input_sources],
            "video_storage": {
                "encoding": video_encoding,
                "container_fps": video_fps,
                "crf": video_crf if video_encoding == "h264" else None,
                "preset": video_preset if video_encoding == "h264" else None,
                "timing_authority": "frames.jsonl",
            },
            "online_frame_artifacts": [processor.describe() for processor in self.frame_processors],
        }
        _write_json_atomic(self.path / "manifest.json", self.manifest)
        for processor in self.frame_processors:
            processor.start(self.path, self.session_id, self.origin_ns)

    def _video_sink(self, packet: FramePacket):
        stream_id = packet.stream_id
        height, width = packet.image.shape[:2]
        shape = (width, height)
        if stream_id in self._video_sinks:
            if self._video_shapes[stream_id] != shape:
                raise RuntimeError("Frame size changed for stream {}".format(stream_id))
            return self._video_sinks[stream_id]
        path_without_suffix = self.path / "video_{}".format(_safe_id(stream_id))
        # The container rate is for ordinary playback only. Exact acquisition
        # timing is retained in frames.jsonl.
        sink = create_video_sink(
            path_without_suffix,
            shape,
            self.video_encoding,
            self.video_fps,
            self.ffmpeg,
            self.video_crf,
            self.video_preset,
        )
        self._video_sinks[stream_id] = sink
        self._video_shapes[stream_id] = shape
        return sink

    def write_frame(self, packet: FramePacket) -> None:
        sink = self._video_sink(packet)
        video_index = self.frame_counts[packet.stream_id]
        for processor in self.frame_processors:
            processor.process(
                packet,
                int(video_index),
                packet.host_capture_time_ns - self.origin_ns,
            )
        sink.write(packet.image)
        height, width = packet.image.shape[:2]
        record = {
            "stream_id": packet.stream_id,
            "frame_index": int(video_index),
            "source_time_ns": packet.source_time_ns,
            "host_capture_time_ns": packet.host_capture_time_ns,
            "host_receive_time_ns": packet.host_receive_time_ns,
            "session_time_ns": packet.host_capture_time_ns - self.origin_ns,
            "width": width,
            "height": height,
            "dropped_before": packet.dropped_before,
            "metadata": packet.metadata,
        }
        self._frames_file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.frame_counts[packet.stream_id] += 1
        self.drop_counts[packet.stream_id] += packet.dropped_before

    def write_input(self, packet: InputPacket) -> None:
        record = {
            "source_id": packet.source_id,
            "kind": packet.kind,
            "source_time_ns": packet.source_time_ns,
            "host_time_ns": packet.host_time_ns,
            "session_time_ns": packet.host_time_ns - self.origin_ns,
            "payload": packet.payload,
        }
        self._inputs_file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.input_counts[(packet.source_id, packet.kind)] += 1

    def record_frame_drops(self, stream_id: str, count: int) -> None:
        self.drop_counts[stream_id] += count

    def record_input_drops(self, source_id: str, count: int) -> None:
        self.input_drop_counts[source_id] += count

    def close(self, status: str = "complete", error: Optional[str] = None) -> None:
        if self._closed:
            return
        self._closed = True
        close_errors = []
        for sink in self._video_sinks.values():
            try:
                sink.close()
            except Exception as exc:
                close_errors.append(str(exc))
        for processor in self.frame_processors:
            try:
                processor.close(status=status if not close_errors else "incomplete")
            except Exception as exc:
                close_errors.append(str(exc))
        self._frames_file.close()
        self._inputs_file.close()
        end_ns = time.perf_counter_ns()
        self.manifest.update(
            {
                "status": status,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "duration_ns": end_ns - self.origin_ns,
                "frame_counts": dict(self.frame_counts),
                "dropped_frames": dict(self.drop_counts),
                "dropped_inputs": dict(self.input_drop_counts),
                "input_counts": {
                    "{}:{}".format(source, kind): count
                    for (source, kind), count in self.input_counts.items()
                },
                "videos": {
                    stream_id: sink.path.name
                    for stream_id, sink in self._video_sinks.items()
                },
                "video_streams": {
                    stream_id: sink.describe()
                    for stream_id, sink in self._video_sinks.items()
                },
                "online_frame_artifacts": [
                    processor.describe() for processor in self.frame_processors
                ],
            }
        )
        if error:
            self.manifest["error"] = error
        if close_errors:
            self.manifest["status"] = "incomplete"
            self.manifest["video_close_errors"] = close_errors
        _write_json_atomic(self.path / "manifest.json", self.manifest)
        if close_errors:
            raise RuntimeError("; ".join(close_errors))


def read_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError("Invalid JSON at {}:{}: {}".format(path, line_number, exc))
    return records


class SessionReader:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError("Unsupported session schema: {}".format(self.manifest.get("schema_version")))
        self.frames = list(read_jsonl(self.path / "frames.jsonl"))
        self.inputs = list(read_jsonl(self.path / "inputs.jsonl"))
        self.frames_by_stream = defaultdict(list)
        for frame in self.frames:
            self.frames_by_stream[frame["stream_id"]].append(frame)
        for frames in self.frames_by_stream.values():
            frames.sort(key=lambda item: item["frame_index"])
        self.inputs.sort(key=lambda item: item["session_time_ns"])

    def video_path(self, stream_id: str) -> Path:
        filename = self.manifest.get("videos", {}).get(stream_id)
        if not filename:
            filename = "video_{}.avi".format(_safe_id(stream_id))
        return self.path / filename

    def nearby_inputs(self, session_time_ns: int, radius_ns: int = 100_000_000) -> List[dict]:
        lower = session_time_ns - radius_ns
        upper = session_time_ns + radius_ns
        return [event for event in self.inputs if lower <= event["session_time_ns"] <= upper]

    def online_features_for_frame(self, stream_id: str, frame_index: int) -> List[dict]:
        observations = []
        for artifact in self.manifest.get("online_frame_artifacts", []):
            if artifact.get("type") != "OnlineSiftRecorder":
                continue
            database = self.path / artifact["path"]
            if not database.exists():
                continue
            connection = sqlite3.connect(str(database))
            try:
                row = connection.execute(
                    """
                    SELECT keypoint_count, frame_sha256, descriptors_stored_dtype,
                           lossless_png IS NOT NULL
                    FROM observations WHERE stream_id = ? AND frame_index = ?
                    """,
                    (stream_id, frame_index),
                ).fetchone()
            finally:
                connection.close()
            if row:
                observations.append(
                    {
                        "artifact": artifact["path"],
                        "extractor": "opencv_sift",
                        "keypoint_count": row[0],
                        "raw_frame_sha256": row[1],
                        "descriptor_storage_dtype": row[2],
                        "has_lossless_frame": bool(row[3]),
                    }
                )
        return observations

    def summary(self) -> dict:
        streams = {}
        for stream_id, frames in self.frames_by_stream.items():
            capture_times = [frame["host_capture_time_ns"] for frame in frames]
            intervals_ms = np.diff(capture_times) / 1.0e6 if len(capture_times) > 1 else np.array([])
            streams[stream_id] = {
                "frames": len(frames),
                "width": frames[0]["width"] if frames else None,
                "height": frames[0]["height"] if frames else None,
                "duration_s": (
                    (capture_times[-1] - capture_times[0]) / 1.0e9 if len(capture_times) > 1 else 0.0
                ),
                "interval_ms_median": float(np.median(intervals_ms)) if len(intervals_ms) else None,
                "interval_ms_p95": float(np.percentile(intervals_ms, 95)) if len(intervals_ms) else None,
                "dropped_frames": sum(int(frame.get("dropped_before", 0)) for frame in frames),
            }
        input_counts = Counter((event["source_id"], event["kind"]) for event in self.inputs)
        annotations = AnnotationStore(self.path).list()
        file_sizes = {
            str(path.relative_to(self.path)): path.stat().st_size
            for path in self.path.rglob("*")
            if path.is_file()
        }
        video_names = set(self.manifest.get("videos", {}).values())
        video_bytes = sum(size for name, size in file_sizes.items() if name in video_names)
        total_bytes = sum(file_sizes.values())
        observed_span_s = max(
            [stream["duration_s"] for stream in streams.values()] or [0.0]
        )
        rate_span_s = observed_span_s or self.manifest.get("duration_ns", 0) / 1.0e9
        return {
            "path": str(self.path.resolve()),
            "status": self.manifest.get("status"),
            "schema_version": self.manifest.get("schema_version"),
            "session_id": self.manifest.get("session_id"),
            "context": self.manifest.get("context", {}),
            "duration_s": self.manifest.get("duration_ns", 0) / 1.0e9,
            "streams": streams,
            "input_events": len(self.inputs),
            "input_counts": {
                "{}:{}".format(source, kind): count
                for (source, kind), count in input_counts.items()
            },
            "online_frame_artifacts": self.manifest.get("online_frame_artifacts", []),
            "annotation_count": len(annotations),
            "annotation_counts": dict(Counter(item["kind"] for item in annotations)),
            "storage": {
                "total_bytes": total_bytes,
                "video_bytes": video_bytes,
                "sidecar_and_other_bytes": total_bytes - video_bytes,
                "estimated_gib_per_hour": (
                    total_bytes / rate_span_s * 3600.0 / (1024.0 ** 3)
                    if rate_span_s > 0 else None
                ),
                "rate_basis_seconds": rate_span_s,
                "files": file_sizes,
            },
        }
