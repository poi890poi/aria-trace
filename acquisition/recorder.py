"""Concurrent acquisition orchestrator."""

import queue
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

from .models import FramePacket, InputPacket
from .session import SessionWriter


class AcquisitionRecorder:
    def __init__(
        self,
        output: Path,
        frame_sources: Iterable[object],
        input_sources: Iterable[object] = (),
        queue_size: int = 4096,
        video_encoding: str = "h264",
        video_fps: float = 30.0,
        video_crf: int = 20,
        video_preset: str = "veryfast",
        ffmpeg: Optional[Path] = None,
        frame_processors=(),
        session_context=None,
    ) -> None:
        self.output = Path(output)
        self.frame_sources = list(frame_sources)
        self.input_sources = list(input_sources)
        self.queue_size = queue_size
        self.video_encoding = video_encoding
        self.video_fps = video_fps
        self.video_crf = video_crf
        self.video_preset = video_preset
        self.ffmpeg = ffmpeg
        self.frame_processors = list(frame_processors)
        self.session_context = dict(session_context or {})

    def run(
        self,
        duration_s: Optional[float] = None,
        external_stop: Optional[threading.Event] = None,
        started_event: Optional[threading.Event] = None,
    ) -> dict:
        if not self.frame_sources:
            raise ValueError("At least one frame source is required")
        event_queue = queue.Queue(maxsize=self.queue_size)
        stop_event = threading.Event()
        workers = []
        finished_streams = set()
        writer = SessionWriter(
            self.output,
            self.frame_sources,
            self.input_sources,
            video_encoding=self.video_encoding,
            video_fps=self.video_fps,
            video_crf=self.video_crf,
            video_preset=self.video_preset,
            ffmpeg=self.ffmpeg,
            frame_processors=self.frame_processors,
            session_context=self.session_context,
        )

        def emit_input(packet: InputPacket) -> None:
            try:
                event_queue.put_nowait(("input", packet))
            except queue.Full:
                writer.record_input_drops(packet.source_id, 1)

        def capture(source) -> None:
            pending_drops = 0
            try:
                while not stop_event.is_set():
                    packet = source.read()
                    if packet is None:
                        break
                    packet.dropped_before += pending_drops
                    try:
                        event_queue.put_nowait(("frame", packet))
                        pending_drops = 0
                    except queue.Full:
                        pending_drops += 1
            except Exception as exc:
                event_queue.put(("error", (source.stream_id, exc)))
            finally:
                if pending_drops:
                    writer.record_frame_drops(source.stream_id, pending_drops)
                event_queue.put(("finished", source.stream_id))

        status = "complete"
        error_text = None
        try:
            for source in self.frame_sources:
                source.start()
            for source in self.input_sources:
                source.start(emit_input)
            # Some source descriptions gain device clock/calibration details at start.
            writer.manifest["frame_sources"] = [source.describe() for source in self.frame_sources]
            writer.manifest["input_sources"] = [source.describe() for source in self.input_sources]
            for source in self.frame_sources:
                worker = threading.Thread(
                    target=capture,
                    args=(source,),
                    name="capture-{}".format(source.stream_id),
                    daemon=True,
                )
                workers.append(worker)
                worker.start()

            if started_event is not None:
                started_event.set()
            start_ns = time.perf_counter_ns()
            while True:
                if external_stop is not None and external_stop.is_set():
                    break
                if duration_s is not None and (time.perf_counter_ns() - start_ns) / 1.0e9 >= duration_s:
                    break
                if len(finished_streams) == len(self.frame_sources):
                    break
                try:
                    kind, value = event_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if kind == "frame":
                    writer.write_frame(value)
                elif kind == "input":
                    writer.write_input(value)
                elif kind == "finished":
                    finished_streams.add(value)
                elif kind == "error":
                    stream_id, exc = value
                    raise RuntimeError("Frame source {} failed: {}".format(stream_id, exc))
        except KeyboardInterrupt:
            status = "interrupted"
        except Exception as exc:
            status = "incomplete"
            error_text = "{}: {}".format(type(exc).__name__, exc)
            raise
        finally:
            if started_event is not None:
                started_event.set()
            stop_event.set()
            for source in self.input_sources:
                try:
                    source.stop()
                except Exception:
                    pass
            # Let capture loops observe stop_event and leave read() themselves.
            # Releasing an OpenCV capture from another thread while read() is in
            # FFmpeg can abort the process. Only force-stop sources that remain
            # blocked after a grace period.
            for worker in workers:
                worker.join(timeout=1)
            for source, worker in zip(self.frame_sources, workers):
                if worker.is_alive():
                    try:
                        source.stop()
                    except Exception:
                        pass
            for worker in workers:
                worker.join(timeout=2)
            for source, worker in zip(self.frame_sources, workers):
                if not worker.is_alive():
                    try:
                        source.stop()
                    except Exception:
                        pass
            while True:
                try:
                    kind, value = event_queue.get_nowait()
                except queue.Empty:
                    break
                if kind == "frame":
                    writer.write_frame(value)
                elif kind == "input":
                    writer.write_input(value)
            writer.close(status=status, error=error_text)
        return writer.manifest
