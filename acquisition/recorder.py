"""Concurrent acquisition orchestrator."""

import queue
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

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
        video_stream_options=None,
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
        self.video_stream_options = dict(video_stream_options or {})

    def run(
        self,
        duration_s: Optional[float] = None,
        external_stop: Optional[threading.Event] = None,
        started_event: Optional[threading.Event] = None,
        on_sources_started: Optional[Callable[[], None]] = None,
        start_on_input: bool = False,
        start_after_delay_s: Optional[float] = None,
        input_start_delay_s: float = 0.0,
        input_start_predicate: Optional[Callable[[InputPacket], bool]] = None,
        on_input_eligible: Optional[Callable[[], None]] = None,
        on_recording_started: Optional[
            Callable[[Optional[InputPacket]], None]
        ] = None,
        on_input_recorded: Optional[Callable[[InputPacket], None]] = None,
    ) -> dict:
        if not self.frame_sources:
            raise ValueError("At least one frame source is required")
        if input_start_delay_s < 0.0:
            raise ValueError("Input start delay cannot be negative")
        if start_after_delay_s is not None and start_after_delay_s < 0.0:
            raise ValueError("Automatic start delay cannot be negative")
        if start_on_input and start_after_delay_s is not None:
            raise ValueError("Choose either input-triggered or timer-triggered start")
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
            video_stream_options=self.video_stream_options,
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
        deferred_start = start_on_input or start_after_delay_s is not None
        recording_started = not deferred_start
        # Source startup may include Android server launch and clock sampling.
        # Begin a bounded immediate take only after those sources are ready.
        start_ns = None
        first_input_kind = None
        input_eligible_ns = None
        input_eligible_announced = False

        def handle_event(kind, value) -> None:
            nonlocal recording_started, start_ns, first_input_kind
            if kind == "finished":
                finished_streams.add(value)
                return
            if kind == "error":
                stream_id, exc = value
                raise RuntimeError("Frame source {} failed: {}".format(stream_id, exc))
            if not recording_started:
                if (
                    input_eligible_ns is not None
                    and kind == "input"
                    and value.host_time_ns < input_eligible_ns
                ):
                    return
                qualifies = (
                    kind == "input"
                    and (
                        input_start_predicate(value)
                        if input_start_predicate is not None
                        else True
                    )
                )
                if not qualifies:
                    return
                writer.rebase_origin(value.host_time_ns)
                recording_started = True
                start_ns = value.host_time_ns
                first_input_kind = value.kind
                if on_recording_started is not None:
                    on_recording_started(value)
            if kind == "frame":
                if value.host_capture_time_ns < writer.origin_ns:
                    return
                writer.write_frame(value)
            elif kind == "input":
                if value.host_time_ns < writer.origin_ns:
                    return
                writer.write_input(value)
                if on_input_recorded is not None:
                    on_input_recorded(value)

        try:
            for source in self.frame_sources:
                source.start()
            if recording_started and not self.frame_processors:
                writer.rebase_origin(time.perf_counter_ns())
            for source in self.input_sources:
                source.start(emit_input)
            if recording_started:
                start_ns = time.perf_counter_ns()
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
            if on_sources_started is not None:
                on_sources_started()
            if deferred_start:
                input_eligible_ns = time.perf_counter_ns() + int(
                    (
                        input_start_delay_s
                        if start_on_input
                        else float(start_after_delay_s)
                    )
                    * 1.0e9
                )
            while True:
                if (
                    deferred_start
                    and not input_eligible_announced
                    and time.perf_counter_ns() >= input_eligible_ns
                ):
                    input_eligible_announced = True
                    if on_input_eligible is not None:
                        on_input_eligible()
                    if not start_on_input:
                        writer.rebase_origin(input_eligible_ns)
                        recording_started = True
                        start_ns = input_eligible_ns
                        if on_recording_started is not None:
                            on_recording_started(None)
                if external_stop is not None and external_stop.is_set():
                    break
                if (
                    recording_started
                    and duration_s is not None
                    and (time.perf_counter_ns() - start_ns) / 1.0e9 >= duration_s
                ):
                    break
                if len(finished_streams) == len(self.frame_sources):
                    break
                try:
                    kind, value = event_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                handle_event(kind, value)
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
                if recording_started and kind in ("frame", "input"):
                    handle_event(kind, value)
            # Input sources can only report final receive/filter counters after
            # their worker threads stop. Keep those diagnostics in the durable
            # manifest so an empty stream is explainable instead of mysterious.
            writer.manifest["input_sources"] = [
                source.describe() for source in self.input_sources
            ]
            writer.manifest["frame_sources"] = [
                source.describe() for source in self.frame_sources
            ]
            if deferred_start and not recording_started and status == "complete":
                status = "incomplete"
                error_text = (
                    "No qualifying input was received"
                    if start_on_input
                    else "Automatic recording start was not reached"
                )
            writer.manifest["recording_start"] = {
                "policy": (
                    "first_qualifying_input"
                    if start_on_input
                    else "settled_timer"
                    if start_after_delay_s is not None
                    else "immediate"
                ),
                "started": recording_started,
                "first_input_kind": first_input_kind,
                "input_settle_delay_s": (
                    float(input_start_delay_s) if start_on_input else 0.0
                ),
                "automatic_start_delay_s": (
                    float(start_after_delay_s)
                    if start_after_delay_s is not None
                    else 0.0
                ),
            }
            writer.close(status=status, error=error_text)
        return writer.manifest
