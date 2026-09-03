"""Continuous Android display capture with synchronized multi-ROI fan-out.

The phone-side component is the pinned scrcpy server.  It emits H.264 packets
with device presentation timestamps.  One decoded full-screen frame is cropped
and resized into any number of host streams, so all derived streams retain the
same source timestamp instead of running independent Android encoders.
"""

import collections
import os
import queue
import re
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import cv2
import numpy as np

from rig_runtime.adapters.android.spaces import image_space_from_surface
from rig_runtime.domain.packets import FramePacket
from rig_runtime.adapters.sources import AdbClockMapper, FrameSource
from rig_runtime.adapters.filesystem.video import find_ffmpeg
from rig_runtime.adapters.runtime_tools import find_release_tool


SCRCPY_SERVER_VERSION = "4.1"
_SESSION_PACKET_FLAG = 1 << 63
_CONFIG_PACKET_FLAG = 1 << 62
_PTS_MASK = (1 << 61) - 1


@dataclass(frozen=True)
class AndroidRoiSpec:
    stream_id: str
    x: int
    y: int
    width: int
    height: int
    output_width: Optional[int] = None
    output_height: Optional[int] = None
    crf: Optional[int] = None


def parse_android_roi(value: str) -> AndroidRoiSpec:
    """Parse ID=x,y,w,h[,output_w,output_h[,crf]].

    A width or height of zero extends the ROI to the corresponding frame edge.
    """
    if "=" not in value:
        raise ValueError("Android ROI must use ID=x,y,w,h[,output_w,output_h[,crf]]")
    stream_id, raw_values = value.split("=", 1)
    stream_id = stream_id.strip()
    if not stream_id or not re.match(r"^[A-Za-z0-9_.-]+$", stream_id):
        raise ValueError("Android ROI stream ID may contain only letters, numbers, ., _, and -")
    parts = [part.strip() for part in raw_values.split(",")]
    if len(parts) not in (4, 6, 7):
        raise ValueError("Android ROI needs 4, 6, or 7 numeric values")
    try:
        numbers = [int(part) for part in parts]
    except ValueError:
        raise ValueError("Android ROI values must be integers")
    x, y, width, height = numbers[:4]
    if min(x, y, width, height) < 0:
        raise ValueError("Android ROI coordinates and dimensions cannot be negative")
    output_width = output_height = crf = None
    if len(numbers) >= 6:
        output_width, output_height = numbers[4:6]
        if output_width <= 0 or output_height <= 0:
            raise ValueError("Android ROI output dimensions must be positive")
    if len(numbers) == 7:
        crf = numbers[6]
        if crf < 0 or crf > 51:
            raise ValueError("Android ROI CRF must be between 0 and 51")
    return AndroidRoiSpec(
        stream_id, x, y, width, height, output_width, output_height, crf
    )


def find_scrcpy_server(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        candidate = Path(explicit)
        if candidate.is_file():
            return candidate
        raise RuntimeError("scrcpy-server was not found at {}".format(candidate))
    configured = os.environ.get("IRIS_SCRCPY_SERVER")
    if configured:
        candidate = Path(configured)
        if candidate.is_file():
            return candidate
    packaged = find_release_tool("third_party/scrcpy/scrcpy-server")
    if packaged is not None:
        return packaged
    executable = shutil.which("scrcpy")
    if executable:
        sibling = Path(executable).resolve().parent / "scrcpy-server"
        if sibling.is_file():
            return sibling
    bundled = Path(".tools") / "scrcpy-win64-v{}".format(SCRCPY_SERVER_VERSION) / "scrcpy-server"
    if bundled.is_file():
        return bundled
    raise RuntimeError(
        "scrcpy-server v{} is required; pass --scrcpy-server PATH".format(
            SCRCPY_SERVER_VERSION
        )
    )


def _read_exact(stream, count: int):
    chunks = []
    remaining = count
    while remaining:
        chunk = stream.recv(remaining) if hasattr(stream, "recv") else stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class ScrcpyCaptureHub:
    """Decode one scrcpy display stream and publish frames to ROI subscribers."""

    def __init__(
        self,
        adb: Path,
        scrcpy_server: Path,
        serial: Optional[str] = None,
        ffmpeg: Optional[Path] = None,
        clock: Optional[AdbClockMapper] = None,
        bit_rate: int = 16_000_000,
        max_fps: float = 60.0,
        subscriber_queue_size: int = 16,
    ) -> None:
        self.adb = Path(adb)
        self.scrcpy_server = Path(scrcpy_server)
        self.serial = serial
        self.ffmpeg = find_ffmpeg(ffmpeg)
        self.clock = clock or AdbClockMapper(self.adb, serial)
        self.bit_rate = int(bit_rate)
        self.max_fps = float(max_fps)
        self.subscriber_queue_size = int(subscriber_queue_size)
        self._subscribers = {}
        self._subscriber_drops = collections.Counter()
        self._lock = threading.Lock()
        self._start_refs = 0
        self._running = False
        self._socket = None
        self._server_process = None
        self._decoder = None
        self._packet_thread = None
        self._decode_thread = None
        self._server_log_thread = None
        self._server_log = collections.deque(maxlen=100)
        self._pts_queue = queue.Queue(maxsize=256)
        self._port = None
        self._scid = None
        self._frame_size = None
        self._device_name = None
        self._sequence = 0

    def _adb(self):
        command = [str(self.adb)]
        if self.serial:
            command += ["-s", self.serial]
        return command

    def register(self, stream_id: str):
        with self._lock:
            if stream_id in self._subscribers:
                raise ValueError("Duplicate Android stream ID: {}".format(stream_id))
            stream_queue = queue.Queue(maxsize=self.subscriber_queue_size)
            self._subscribers[stream_id] = stream_queue
            return stream_queue

    def start(self) -> None:
        with self._lock:
            self._start_refs += 1
            if self._running:
                return
            self._running = True
        try:
            self._start_capture()
        except Exception:
            with self._lock:
                self._running = False
                self._start_refs = max(0, self._start_refs - 1)
            self._cleanup()
            raise

    def _start_capture(self) -> None:
        if not self.adb.is_file():
            raise RuntimeError("ADB executable does not exist: {}".format(self.adb))
        if not self.scrcpy_server.is_file():
            raise RuntimeError("scrcpy-server does not exist: {}".format(self.scrcpy_server))
        self.clock.calibrate()
        remote_server = "/data/local/tmp/iris-scrcpy-server-v{}.jar".format(
            SCRCPY_SERVER_VERSION
        )
        subprocess.check_call(
            self._adb() + ["push", str(self.scrcpy_server), remote_server],
            stdout=subprocess.DEVNULL,
            timeout=30,
        )
        self._scid = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF
        socket_name = "localabstract:scrcpy_{:08x}".format(self._scid)
        output = subprocess.check_output(
            self._adb() + ["forward", "tcp:0", socket_name],
            timeout=10,
            universal_newlines=True,
        )
        self._port = int(output.strip())
        command = self._adb() + [
            "shell",
            "CLASSPATH={}".format(remote_server),
            "app_process",
            "/",
            "com.genymobile.scrcpy.Server",
            SCRCPY_SERVER_VERSION,
            "scid={:08x}".format(self._scid),
            "log_level=info",
            "audio=false",
            "control=false",
            "tunnel_forward=true",
            "video_codec=h264",
            "video_bit_rate={}".format(self.bit_rate),
            "max_fps={}".format(self.max_fps),
            "capture_orientation=@",
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._server_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            creationflags=creationflags,
        )

        def collect_server_log():
            if self._server_process is None or self._server_process.stdout is None:
                return
            for line in self._server_process.stdout:
                self._server_log.append(line.rstrip())

        self._server_log_thread = threading.Thread(
            target=collect_server_log, name="scrcpy-server-log", daemon=True
        )
        self._server_log_thread.start()
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if self._server_process.poll() is not None:
                raise RuntimeError(
                    "scrcpy server exited before connection: {}".format(
                        " | ".join(self._server_log)
                    )
                )
            try:
                candidate = socket.create_connection(("127.0.0.1", self._port), timeout=0.5)
                # With adb forward, the TCP connect itself may succeed before
                # the device localabstract socket exists. scrcpy's dummy byte
                # distinguishes a real server connection from that false start.
                if _read_exact(candidate, 1) != b"\x00":
                    candidate.close()
                    time.sleep(0.05)
                    continue
                candidate.settimeout(None)
                self._socket = candidate
                break
            except OSError:
                time.sleep(0.05)
        if self._socket is None:
            raise RuntimeError("Timed out connecting to the scrcpy server")
        device_meta = _read_exact(self._socket, 64)
        if device_meta is None:
            raise RuntimeError("scrcpy server disconnected before device metadata")
        self._device_name = device_meta.split(b"\x00", 1)[0].decode("utf-8", "replace")
        self._packet_thread = threading.Thread(
            target=self._read_packets, name="scrcpy-video-packets", daemon=True
        )
        self._packet_thread.start()

    def _start_decoder(self, width: int, height: int) -> None:
        self._frame_size = (width, height)
        command = [
            str(self.ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-flags",
            "low_delay",
            "-probesize",
            "32768",
            "-analyzeduration",
            "0",
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-an",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._decoder = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        self._decode_thread = threading.Thread(
            target=self._read_decoded_frames, name="scrcpy-video-decode", daemon=True
        )
        self._decode_thread.start()

    def _read_packets(self) -> None:
        try:
            codec = _read_exact(self._socket, 4)
            if codec != b"h264":
                raise RuntimeError("Expected scrcpy H.264 stream, got {!r}".format(codec))
            while self._running:
                header = _read_exact(self._socket, 12)
                if header is None:
                    break
                value, packet_size = struct.unpack(">QI", header)
                if value & _SESSION_PACKET_FLAG:
                    width = struct.unpack(">I", header[4:8])[0]
                    height = packet_size
                    if self._decoder is not None:
                        raise RuntimeError("Android display size changed during locked capture")
                    if width <= 0 or height <= 0:
                        raise RuntimeError("Invalid scrcpy video size {}x{}".format(width, height))
                    self._start_decoder(width, height)
                    continue
                payload = _read_exact(self._socket, packet_size)
                if payload is None:
                    break
                if self._decoder is None or self._decoder.stdin is None:
                    raise RuntimeError("scrcpy sent media before video dimensions")
                if not value & _CONFIG_PACKET_FLAG:
                    pts_ns = (value & _PTS_MASK) * 1000
                    self._pts_queue.put(pts_ns, timeout=2)
                self._decoder.stdin.write(payload)
                self._decoder.stdin.flush()
            if self._running:
                raise RuntimeError("scrcpy video stream ended unexpectedly")
        except Exception as exc:
            if self._running:
                self._publish_error(exc)
        finally:
            if self._decoder is not None and self._decoder.stdin is not None:
                try:
                    self._decoder.stdin.close()
                except OSError:
                    pass

    def _read_decoded_frames(self) -> None:
        width, height = self._frame_size
        byte_count = width * height * 3
        try:
            while self._running:
                raw = _read_exact(self._decoder.stdout, byte_count)
                if raw is None:
                    break
                pts_ns = self._pts_queue.get(timeout=2)
                receive_ns = time.perf_counter_ns()
                capture_ns = self.clock.to_host_time_ns(pts_ns, receive_ns)
                image = memoryview(raw)
                frame = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 3)).copy()
                self._publish_frame(frame, pts_ns, capture_ns, receive_ns)
        except Exception as exc:
            if self._running:
                self._publish_error(exc)

    def _put_latest(self, stream_id: str, item) -> None:
        stream_queue = self._subscribers[stream_id]
        try:
            stream_queue.put_nowait(item)
        except queue.Full:
            try:
                stream_queue.get_nowait()
                self._subscriber_drops[stream_id] += 1
            except queue.Empty:
                pass
            stream_queue.put_nowait(item)

    def _publish_frame(self, image, pts_ns: int, capture_ns: int, receive_ns: int) -> None:
        self._sequence += 1
        item = ("frame", self._sequence, image, pts_ns, capture_ns, receive_ns)
        for stream_id in list(self._subscribers):
            self._put_latest(stream_id, item)

    def _publish_error(self, exc: Exception) -> None:
        item = ("error", exc)
        for stream_id in list(self._subscribers):
            self._put_latest(stream_id, item)

    def _publish_end(self) -> None:
        for stream_id, stream_queue in list(self._subscribers.items()):
            while True:
                try:
                    stream_queue.get_nowait()
                except queue.Empty:
                    break
            stream_queue.put_nowait(("end",))

    def take_drops(self, stream_id: str) -> int:
        count = int(self._subscriber_drops[stream_id])
        self._subscriber_drops[stream_id] = 0
        return count

    def stop(self) -> None:
        with self._lock:
            if self._start_refs:
                self._start_refs -= 1
            if self._start_refs or not self._running:
                return
            self._running = False
        self._publish_end()
        self._cleanup()

    def _cleanup(self) -> None:
        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._socket.close()
            self._socket = None
        if self._decoder is not None:
            if self._decoder.stdin and not self._decoder.stdin.closed:
                self._decoder.stdin.close()
            try:
                self._decoder.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._decoder.kill()
                self._decoder.wait()
            self._decoder = None
        if self._server_process is not None:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
                self._server_process.wait()
            self._server_process = None
        if self._port is not None:
            subprocess.run(
                self._adb() + ["forward", "--remove", "tcp:{}".format(self._port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            self._port = None
        for thread in (self._packet_thread, self._decode_thread, self._server_log_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2)

    def describe(self) -> Dict[str, object]:
        result = {
            "type": type(self).__name__,
            "transport": "scrcpy-server-over-adb",
            "scrcpy_server_version": SCRCPY_SERVER_VERSION,
            "scrcpy_server": str(self.scrcpy_server.resolve()),
            "adb": str(self.adb.resolve()),
            "serial": self.serial,
            "device_name": self._device_name,
            "capture_codec": "h264",
            "capture_bit_rate": self.bit_rate,
            "requested_max_fps": self.max_fps,
            "decoded_size": list(self._frame_size) if self._frame_size else None,
            "timestamp_source": "scrcpy MediaCodec presentation timestamp",
            "server_log_tail": list(self._server_log),
        }
        result.update(self.clock.describe())
        return result


class AndroidRoiFrameSource(FrameSource):
    def __init__(
        self,
        hub: ScrcpyCaptureHub,
        spec: AndroidRoiSpec,
        image_space_context: Optional[Mapping[str, object]] = None,
    ) -> None:
        self.hub = hub
        self.spec = spec
        self.image_space_context = (
            dict(image_space_context) if image_space_context is not None else None
        )
        self.stream_id = spec.stream_id
        self._queue = hub.register(self.stream_id)
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._started = True
            self.hub.start()

    def read(self) -> Optional[FramePacket]:
        if not self._started:
            return None
        item = self._queue.get()
        if item[0] == "end":
            return None
        if item[0] == "error":
            raise RuntimeError("Android capture failed: {}".format(item[1]))
        _, sequence, image, source_ns, capture_ns, receive_ns = item
        frame_height, frame_width = image.shape[:2]
        width = self.spec.width or frame_width - self.spec.x
        height = self.spec.height or frame_height - self.spec.y
        right = self.spec.x + width
        bottom = self.spec.y + height
        if width <= 0 or height <= 0 or right > frame_width or bottom > frame_height:
            raise RuntimeError(
                "ROI {} [{},{},{},{}] exceeds Android frame {}x{}".format(
                    self.stream_id,
                    self.spec.x,
                    self.spec.y,
                    width,
                    height,
                    frame_width,
                    frame_height,
                )
            )
        cropped = image[self.spec.y:bottom, self.spec.x:right]
        if self.spec.output_width is not None:
            interpolation = (
                cv2.INTER_AREA
                if self.spec.output_width < width or self.spec.output_height < height
                else cv2.INTER_LINEAR
            )
            cropped = cv2.resize(
                cropped,
                (self.spec.output_width, self.spec.output_height),
                interpolation=interpolation,
            )
        else:
            cropped = cropped.copy()
        clock = getattr(self.hub, "clock", None)
        timestamp_mapping = (
            dict(clock.describe())
            if clock is not None and callable(getattr(clock, "describe", None))
            else {}
        )
        packet_metadata = {
            "source": "android_scrcpy",
            "source_sequence": sequence,
            "source_size": [frame_width, frame_height],
            "roi_xywh": [self.spec.x, self.spec.y, width, height],
            "target_size": [cropped.shape[1], cropped.shape[0]],
            "timestamp_timebase": "scrcpy_media_pts_mapped_to_host",
            "timestamp_mapping": timestamp_mapping,
        }
        if self.image_space_context is not None:
            packet_metadata["image_space"] = image_space_from_surface(
                self.image_space_context,
                source_size_px=[frame_width, frame_height],
                roi_xywh=[self.spec.x, self.spec.y, width, height],
                stored_size_px=[cropped.shape[1], cropped.shape[0]],
            )
        return FramePacket(
            self.stream_id,
            cropped,
            capture_ns,
            receive_ns,
            source_time_ns=source_ns,
            metadata=packet_metadata,
            dropped_before=self.hub.take_drops(self.stream_id),
        )

    def stop(self) -> None:
        if self._started:
            self._started = False
            self.hub.stop()

    def describe(self) -> Dict[str, object]:
        result = {
            "type": type(self).__name__,
            "stream_id": self.stream_id,
            "roi_xywh": [self.spec.x, self.spec.y, self.spec.width, self.spec.height],
            "target_size": (
                [self.spec.output_width, self.spec.output_height]
                if self.spec.output_width is not None
                else None
            ),
            "video_crf": self.spec.crf,
            "shared_capture": self.hub.describe(),
        }
        if self.image_space_context is not None:
            result["image_space_contract"] = {
                "canonical_space_id": "android_phone_natural_display_pixels",
                "canonical_size_px": list(
                    self.image_space_context["natural_size_px"]
                ),
                "surface_quarter_turns_clockwise_from_canonical": int(
                    self.image_space_context[
                        "quarter_turns_clockwise_from_natural"
                    ]
                ),
                "per_frame_metadata": "frames.jsonl#metadata.image_space",
            }
        return result


def push_session_archive_to_device(
    adb: Path,
    serial: Optional[str],
    session_path: Path,
    remote_root: str,
) -> str:
    """Zip a finalized session, push it to the phone, and return its device path."""
    if not re.match(r"^/(?:sdcard|storage/emulated/0)(?:/[A-Za-z0-9._-]+)*$", remote_root):
        raise ValueError("Phone save directory must be a safe path under /sdcard or /storage/emulated/0")
    session_path = Path(session_path).resolve()
    remote_path = remote_root.rstrip("/") + "/{}.zip".format(session_path.name)
    command = [str(adb)]
    if serial:
        command += ["-s", serial]
    subprocess.check_call(command + ["shell", "mkdir", "-p", remote_root], timeout=10)
    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary) / session_path.name
        archive = Path(shutil.make_archive(str(base), "zip", str(session_path)))
        subprocess.check_call(command + ["push", str(archive), remote_path], timeout=300)
    return remote_path
