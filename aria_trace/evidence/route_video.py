"""Asynchronous game-frame video evidence for live route tracing."""

import copy
import json
import math
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path

from rig_runtime.adapters.filesystem.video import create_video_sink


SCHEMA_VERSION = "1.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


class RouteTraceVideoRecorder:
    """Encode source frames plus a supplied overlay without blocking capture.

    ``submit`` is deliberately nonblocking. If encoding cannot keep up, the
    oldest queued image is discarded so neither acquisition nor tracking waits
    for video evidence. The fixed-rate output timeline is reconstructed from
    capture timestamps and repeats the last encoded frame across short gaps.
    """

    def __init__(
        self,
        output_path: Path,
        renderer,
        *,
        fps: float = 30.0,
        encoding: str = "h264",
        ffmpeg=None,
        crf: int = 20,
        preset: str = "veryfast",
        queue_capacity: int = 24,
    ) -> None:
        if fps <= 0.0:
            raise ValueError("Route video FPS must be positive")
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.renderer = renderer
        self.fps = float(fps)
        self.encoding = str(encoding)
        self.ffmpeg = Path(ffmpeg) if ffmpeg else None
        self.crf = int(crf)
        self.preset = str(preset)
        self._queue = queue.Queue(maxsize=max(2, int(queue_capacity)))
        self._state_lock = threading.Lock()
        self._latest_state = {}
        self._summary_lock = threading.Lock()
        self._closed = False
        self._status = "running"
        self._error = None
        self._source_frames = 0
        self._written_frames = 0
        self._dropped_frames = 0
        self._repeated_frames = 0
        self._video_file = None
        self._started_utc = _utc_now()
        self._manifest_path = self.output_path / "route_trace_video.json"
        self._write_manifest("running")
        self._thread = threading.Thread(
            target=self._work,
            name="aria-route-video",
            daemon=True,
        )
        self._thread.start()

    @property
    def summary(self) -> dict:
        with self._summary_lock:
            return {
                "status": "failed" if self._error else self._status,
                "video_file": self._video_file,
                "manifest_file": self._manifest_path.name,
                "source_frames": self._source_frames,
                "written_frames": self._written_frames,
                "dropped_frames": self._dropped_frames,
                "repeated_frames": self._repeated_frames,
                "error": self._error,
            }

    def update_state(self, state: dict) -> None:
        if self._closed:
            return
        with self._state_lock:
            self._latest_state = copy.deepcopy(state or {})

    def submit(self, image, host_capture_time_ns: int) -> bool:
        if self._closed or image is None:
            return False
        with self._state_lock:
            state = self._latest_state
        item = (
            image.copy(),
            int(host_capture_time_ns),
            state,
        )
        with self._summary_lock:
            self._source_frames += 1
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            with self._summary_lock:
                self._dropped_frames += 1
            try:
                self._queue.put_nowait(item)
                return True
            except queue.Full:
                with self._summary_lock:
                    self._dropped_frames += 1
                return False

    def close(self, status="complete", error=None) -> dict:
        if self._closed:
            return self.summary
        self._closed = True
        while True:
            try:
                self._queue.put_nowait(None)
                break
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                with self._summary_lock:
                    self._dropped_frames += 1
        self._thread.join(timeout=65.0)
        with self._summary_lock:
            if self._thread.is_alive() and self._error is None:
                self._error = "Video worker did not stop within 65 seconds"
            final_status = "failed" if self._error else str(status)
            self._status = final_status
        self._write_manifest(final_status, error=error)
        return self.summary

    def _write_manifest(self, status: str, error=None, sink=None) -> None:
        summary = self.summary
        _atomic_json(
            self._manifest_path,
            {
                "schema_version": SCHEMA_VERSION,
                "recording_type": "route_trace_game_with_overlay",
                "status": status,
                "started_utc": self._started_utc,
                "finished_utc": _utc_now() if status != "running" else None,
                "video_file": summary["video_file"],
                "fps": self.fps,
                "encoding": self.encoding,
                "sink": sink,
                "source_frames": summary["source_frames"],
                "written_frames": summary["written_frames"],
                "dropped_frames": summary["dropped_frames"],
                "repeated_frames": summary["repeated_frames"],
                "error": summary["error"] or error,
                "composition": {
                    "base": "selected_game_frame_source",
                    "overlay": "route_guide_and_compact_tracking_panel",
                    "desktop_or_workbench_included": False,
                },
            },
        )

    def _work(self) -> None:
        sink = None
        first_timestamp_ns = None
        last_frame = None
        sink_description = None
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                image, timestamp_ns, state = item
                if sink is not None:
                    target_index = max(
                        0,
                        int(math.floor(
                            (timestamp_ns - first_timestamp_ns)
                            * self.fps
                            / 1.0e9
                        )),
                    )
                    with self._summary_lock:
                        written = self._written_frames
                    if target_index < written:
                        continue
                composed = self.renderer(image, state)
                height, width = composed.shape[:2]
                if self.encoding == "h264" and (width % 2 or height % 2):
                    composed = composed[: height - (height % 2), : width - (width % 2)]
                    height, width = composed.shape[:2]
                if sink is None:
                    sink = create_video_sink(
                        self.output_path / "route_trace_game_overlay",
                        (width, height),
                        self.encoding,
                        self.fps,
                        self.ffmpeg,
                        self.crf,
                        self.preset,
                    )
                    with self._summary_lock:
                        self._video_file = sink.path.name
                    sink_description = sink.describe()
                    first_timestamp_ns = timestamp_ns
                target_index = max(
                    0,
                    int(math.floor(
                        (timestamp_ns - first_timestamp_ns) * self.fps / 1.0e9
                    )),
                )
                with self._summary_lock:
                    written = self._written_frames
                if target_index < written:
                    continue
                while last_frame is not None and written < target_index:
                    sink.write(last_frame)
                    written += 1
                    with self._summary_lock:
                        self._written_frames = written
                        self._repeated_frames += 1
                sink.write(composed)
                written += 1
                with self._summary_lock:
                    self._written_frames = written
                last_frame = composed
        except Exception as exc:
            with self._summary_lock:
                self._error = "{}: {}".format(type(exc).__name__, exc)
        finally:
            if sink is not None:
                try:
                    sink.close()
                except Exception as exc:
                    with self._summary_lock:
                        if self._error is None:
                            self._error = "{}: {}".format(type(exc).__name__, exc)
            self._write_manifest(
                "failed" if self._error else "finalizing",
                sink=sink_description,
            )
