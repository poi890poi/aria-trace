"""Replaceable frame and input sources for acquisition sessions."""

import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import cv2
import numpy as np

from .models import FramePacket, InputPacket


class FrameSource:
    stream_id = "frames"

    def start(self) -> None:
        raise NotImplementedError

    def read(self) -> Optional[FramePacket]:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def describe(self) -> Dict[str, object]:
        return {"type": type(self).__name__, "stream_id": self.stream_id}


class InputSource:
    source_id = "input"

    def start(self, emit: Callable[[InputPacket], None]) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def describe(self) -> Dict[str, object]:
        return {"type": type(self).__name__, "source_id": self.source_id}


class VideoFileFrameSource(FrameSource):
    def __init__(self, path: Path, stream_id: str = "video", realtime: bool = True) -> None:
        self.path = Path(path)
        self.stream_id = stream_id
        self.realtime = realtime
        self._capture = None
        self._fps = 0.0
        self._index = 0
        self._start_ns = 0

    def start(self) -> None:
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise RuntimeError("Cannot open video: {}".format(self.path))
        self._fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 30.0)
        if self._fps <= 0.0:
            self._fps = 30.0
        self._index = 0
        self._start_ns = time.perf_counter_ns()

    def read(self) -> Optional[FramePacket]:
        if self._capture is None:
            raise RuntimeError("VideoFileFrameSource is not started")
        target_ns = self._start_ns + int(self._index * 1.0e9 / self._fps)
        if self.realtime:
            remaining = target_ns - time.perf_counter_ns()
            if remaining > 0:
                time.sleep(remaining / 1.0e9)
        capture_time = time.perf_counter_ns()
        ok, image = self._capture.read()
        receive_time = time.perf_counter_ns()
        if not ok:
            return None
        source_time = int(self._index * 1.0e9 / self._fps)
        packet = FramePacket(
            self.stream_id,
            image,
            capture_time,
            receive_time,
            source_time_ns=source_time,
            metadata={"source_frame_index": self._index},
        )
        self._index += 1
        return packet

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def describe(self) -> Dict[str, object]:
        result = super().describe()
        result.update({"path": str(self.path.resolve()), "realtime": self.realtime})
        return result


class OpenCvCameraFrameSource(FrameSource):
    def __init__(
        self,
        device: int = 0,
        stream_id: str = "uvc",
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
    ) -> None:
        self.device = device
        self.stream_id = stream_id
        self.width = width
        self.height = height
        self.fps = fps
        self._capture = None

    def start(self) -> None:
        self._capture = cv2.VideoCapture(self.device, cv2.CAP_DSHOW)
        if self.width:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            self._capture.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._capture.isOpened():
            raise RuntimeError("Cannot open camera device {}".format(self.device))

    def read(self) -> Optional[FramePacket]:
        if self._capture is None:
            raise RuntimeError("OpenCvCameraFrameSource is not started")
        capture_time = time.perf_counter_ns()
        ok, image = self._capture.read()
        receive_time = time.perf_counter_ns()
        if not ok:
            return None
        return FramePacket(self.stream_id, image, capture_time, receive_time)

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def describe(self) -> Dict[str, object]:
        result = super().describe()
        result.update(
            {"device": self.device, "requested_width": self.width, "requested_height": self.height, "requested_fps": self.fps}
        )
        return result


class AdbScreenshotFrameSource(FrameSource):
    def __init__(
        self,
        adb: Path,
        serial: Optional[str] = None,
        stream_id: str = "adb",
        fps: float = 2.0,
    ) -> None:
        self.adb = Path(adb)
        self.serial = serial
        self.stream_id = stream_id
        self.fps = fps
        self._running = False
        self._next_frame_ns = 0

    def _base_command(self):
        command = [str(self.adb)]
        if self.serial:
            command += ["-s", self.serial]
        return command

    def start(self) -> None:
        if not self.adb.exists():
            raise RuntimeError("ADB executable does not exist: {}".format(self.adb))
        self._running = True
        self._next_frame_ns = time.perf_counter_ns()

    def read(self) -> Optional[FramePacket]:
        if not self._running:
            return None
        remaining = self._next_frame_ns - time.perf_counter_ns()
        if remaining > 0:
            time.sleep(remaining / 1.0e9)
        self._next_frame_ns += int(1.0e9 / self.fps)
        capture_time = time.perf_counter_ns()
        data = subprocess.check_output(
            self._base_command() + ["exec-out", "screencap", "-p"], timeout=10
        )
        receive_time = time.perf_counter_ns()
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("ADB returned an invalid screenshot")
        return FramePacket(self.stream_id, image, capture_time, receive_time)

    def stop(self) -> None:
        self._running = False

    def describe(self) -> Dict[str, object]:
        result = super().describe()
        result.update({"adb": str(self.adb.resolve()), "serial": self.serial, "requested_fps": self.fps})
        return result


GETEVENT_PATTERN = re.compile(
    r"^\[\s*(?P<seconds>\d+\.\d+)\]\s+(?P<device>\S+):\s+"
    r"(?P<event_type>\S+)\s+(?P<code>\S+)\s+(?P<value>\S+)"
)


def parse_getevent_line(line: str):
    match = GETEVENT_PATTERN.match(line.strip())
    if not match:
        return None
    fields = match.groupdict()
    fields["device_time_ns"] = int(float(fields.pop("seconds")) * 1.0e9)
    return fields


def estimate_clock_offset(samples):
    """Choose the device-to-host offset from the lowest-RTT clock sample."""
    if not samples:
        raise ValueError("At least one clock sample is required")
    best = min(samples, key=lambda item: item[1] - item[0])
    host_before_ns, host_after_ns, device_ns = best
    host_midpoint_ns = (host_before_ns + host_after_ns) // 2
    return host_midpoint_ns - device_ns, host_after_ns - host_before_ns


class AdbGetEventSource(InputSource):
    def __init__(self, adb: Path, serial: Optional[str] = None, source_id: str = "getevent") -> None:
        self.adb = Path(adb)
        self.serial = serial
        self.source_id = source_id
        self._process = None
        self._thread = None
        self._running = False
        self.clock_offset_ns = None
        self.clock_rtt_ns = None
        self.clock_status = "not-calibrated"

    def _command(self):
        command = [str(self.adb)]
        if self.serial:
            command += ["-s", self.serial]
        return command + ["shell", "getevent", "-lt"]

    def start(self, emit: Callable[[InputPacket], None]) -> None:
        if not self.adb.exists():
            raise RuntimeError("ADB executable does not exist: {}".format(self.adb))
        samples = []
        try:
            base = [str(self.adb)]
            if self.serial:
                base += ["-s", self.serial]
            for _ in range(5):
                before = time.perf_counter_ns()
                output = subprocess.check_output(
                    base + ["shell", "cat", "/proc/uptime"],
                    timeout=3,
                    universal_newlines=True,
                )
                after = time.perf_counter_ns()
                device_ns = int(float(output.split()[0]) * 1.0e9)
                samples.append((before, after, device_ns))
            self.clock_offset_ns, self.clock_rtt_ns = estimate_clock_offset(samples)
            self.clock_status = "mapped-from-proc-uptime"
        except Exception:
            self.clock_offset_ns = None
            self.clock_rtt_ns = None
            self.clock_status = "host-receive-time-fallback"
        self._process = subprocess.Popen(
            self._command(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )
        self._running = True

        def read_output():
            assert self._process is not None and self._process.stdout is not None
            for line in self._process.stdout:
                if not self._running:
                    break
                host_time = time.perf_counter_ns()
                parsed = parse_getevent_line(line)
                if parsed is None:
                    emit(InputPacket(self.source_id, "raw_getevent", host_time, {"raw": line.rstrip()}))
                else:
                    source_time = parsed.pop("device_time_ns")
                    parsed["host_receive_time_ns"] = host_time
                    mapped_time = (
                        source_time + self.clock_offset_ns
                        if self.clock_offset_ns is not None
                        else host_time
                    )
                    emit(InputPacket(self.source_id, "linux_input", mapped_time, parsed, source_time))

        self._thread = threading.Thread(target=read_output, name="getevent-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def describe(self) -> Dict[str, object]:
        result = super().describe()
        result.update(
            {
                "adb": str(self.adb.resolve()),
                "serial": self.serial,
                "clock_status": self.clock_status,
                "device_to_pc_offset_ns": self.clock_offset_ns,
                "clock_sample_rtt_ns": self.clock_rtt_ns,
            }
        )
        return result


class SyntheticInputSource(InputSource):
    """Test source that emits gamepad-like axes without hardware."""

    def __init__(self, source_id: str = "synthetic-gamepad", hz: float = 20.0) -> None:
        self.source_id = source_id
        self.hz = hz
        self._running = False
        self._thread = None

    def start(self, emit: Callable[[InputPacket], None]) -> None:
        self._running = True

        def run():
            sequence = 0
            while self._running:
                now = time.perf_counter_ns()
                emit(
                    InputPacket(
                        self.source_id,
                        "gamepad_state",
                        now,
                        {
                            "axes": {"left_x": 0.0, "left_y": -1.0, "right_x": 0.15, "right_y": 0.0},
                            "buttons": {},
                            "synthetic_sequence": sequence,
                        },
                    )
                )
                sequence += 1
                time.sleep(1.0 / self.hz)

        self._thread = threading.Thread(target=run, name="synthetic-input", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    def describe(self) -> Dict[str, object]:
        result = super().describe()
        result["hz"] = self.hz
        return result
