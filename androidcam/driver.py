"""Camera-style Python interface for an Android camera streamed by scrcpy."""

from __future__ import annotations

import copy
import os
import queue
import shutil
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from aria_trace.adapters.android.capture import (
    SCRCPY_SERVER_VERSION,
    ScrcpyCaptureHub,
    _read_exact,
)


@dataclass(frozen=True)
class AndroidCameraFrame:
    """One decoded BGR camera frame with device and host timestamps."""

    image: np.ndarray
    capture_time_ns: int
    receive_time_ns: int
    source_time_ns: int
    sequence: int
    metadata: Mapping[str, Any]


def _workspace_root() -> Path:
    return Path(__file__).resolve().parent.parent


def find_adb(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        candidate = Path(explicit)
        if candidate.is_file():
            return candidate.resolve()
        raise RuntimeError("ADB executable does not exist: {}".format(candidate))
    executable = shutil.which("adb")
    if executable:
        return Path(executable).resolve()
    candidates = (
        _workspace_root() / ".tools" / "scrcpy-win64-v4.1" / "adb.exe",
        Path("E:/Android/Sdk/platform-tools/adb.exe"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError("ADB was not found; pass adb=Path(...) or add it to PATH")


def find_server(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        candidate = Path(explicit)
        if candidate.is_file():
            return candidate.resolve()
        raise RuntimeError("scrcpy-server does not exist: {}".format(candidate))
    candidate = (
        _workspace_root()
        / ".tools"
        / "scrcpy-win64-v{}".format(SCRCPY_SERVER_VERSION)
        / "scrcpy-server"
    )
    if candidate.is_file():
        return candidate.resolve()
    executable = shutil.which("scrcpy")
    if executable:
        sibling = Path(executable).resolve().parent / "scrcpy-server"
        if sibling.is_file():
            return sibling
    raise RuntimeError(
        "scrcpy-server v{} was not found; pass scrcpy_server=Path(...)".format(
            SCRCPY_SERVER_VERSION
        )
    )


def select_serial(adb: Path, requested: Optional[str]) -> str:
    output = subprocess.check_output(
        [str(adb), "devices", "-l"], timeout=10, universal_newlines=True
    )
    devices = []
    states = {}
    for line in output.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            states[parts[0]] = parts[1]
            if parts[1] == "device":
                devices.append(parts[0])
    if requested:
        state = states.get(str(requested))
        if state != "device":
            raise RuntimeError(
                "Android camera device {} is not ready in adb devices -l (state={!r})"
                .format(requested, state)
            )
        return str(requested)
    if len(devices) == 1:
        return devices[0]
    if not devices:
        raise RuntimeError("No authorized Android device is available over ADB")
    raise RuntimeError(
        "Multiple Android devices are connected; select the camera source with "
        "--serial ({})".format(", ".join(devices))
    )


class ScrcpyCameraHub(ScrcpyCaptureHub):
    """The verified scrcpy decoder with camera-specific server options."""

    def __init__(
        self,
        *args: Any,
        camera_id: Optional[str] = None,
        camera_facing: str = "front",
        camera_width_px: int = 1280,
        camera_height_px: int = 720,
        camera_fps: int = 30,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, max_fps=float(camera_fps), **kwargs)
        if camera_id is None and camera_facing not in ("front", "back", "external"):
            raise ValueError("camera_facing must be front, back, or external")
        if min(camera_width_px, camera_height_px, camera_fps) <= 0:
            raise ValueError("camera width, height, and FPS must be positive")
        self.camera_id = None if camera_id is None else str(camera_id)
        self.camera_facing = str(camera_facing)
        self.camera_width_px = int(camera_width_px)
        self.camera_height_px = int(camera_height_px)
        self.camera_fps = int(camera_fps)

    def _server_command(self, remote_server: str) -> list:
        selection = (
            "camera_id={}".format(self.camera_id)
            if self.camera_id is not None
            else "camera_facing={}".format(self.camera_facing)
        )
        return self._adb() + [
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
            "video_source=camera",
            selection,
            "camera_size={}x{}".format(
                self.camera_width_px, self.camera_height_px
            ),
            "camera_fps={}".format(self.camera_fps),
            "video_codec=h264",
            "video_bit_rate={}".format(self.bit_rate),
            "capture_orientation=@",
        ]

    def _start_capture(self) -> None:
        if not self.adb.is_file():
            raise RuntimeError("ADB executable does not exist: {}".format(self.adb))
        if not self.scrcpy_server.is_file():
            raise RuntimeError(
                "scrcpy-server does not exist: {}".format(self.scrcpy_server)
            )
        self.clock.calibrate()
        remote_server = "/data/local/tmp/aria-trace-scrcpy-server-v{}.jar".format(
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
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._server_process = subprocess.Popen(
            self._server_command(remote_server),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            creationflags=creationflags,
        )

        def collect_server_log() -> None:
            if self._server_process is None or self._server_process.stdout is None:
                return
            for line in self._server_process.stdout:
                self._server_log.append(line.rstrip())

        self._server_log_thread = threading.Thread(
            target=collect_server_log, name="scrcpy-camera-server-log", daemon=True
        )
        self._server_log_thread.start()
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if self._server_process.poll() is not None:
                raise RuntimeError(
                    "scrcpy camera server exited before connection: {}".format(
                        " | ".join(self._server_log)
                    )
                )
            try:
                candidate = socket.create_connection(
                    ("127.0.0.1", self._port), timeout=0.5
                )
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
            raise RuntimeError("Timed out connecting to the scrcpy camera server")
        device_meta = _read_exact(self._socket, 64)
        if device_meta is None:
            raise RuntimeError(
                "scrcpy camera server disconnected before device metadata"
            )
        self._device_name = device_meta.split(b"\x00", 1)[0].decode(
            "utf-8", "replace"
        )
        self._packet_thread = threading.Thread(
            target=self._read_packets,
            name="scrcpy-camera-video-packets",
            daemon=True,
        )
        self._packet_thread.start()

    def describe(self) -> Dict[str, object]:
        result = super().describe()
        result.update(
            {
                "video_source": "camera",
                "camera_id": self.camera_id,
                "camera_facing": (
                    None if self.camera_id is not None else self.camera_facing
                ),
                "requested_camera_size_px": [
                    self.camera_width_px,
                    self.camera_height_px,
                ],
                "requested_camera_fps": self.camera_fps,
                "capture_orientation": "locked_initial",
                "screen_power_policy": "scrcpy_default; demo sleeps devices after close",
            }
        )
        return result


class AndroidCamera:
    """Blocking camera interface backed by Android Camera2 and scrcpy H.264."""

    def __init__(
        self,
        serial: Optional[str] = None,
        *,
        camera_id: Optional[str] = None,
        camera_facing: str = "front",
        width_px: int = 1280,
        height_px: int = 720,
        fps: int = 30,
        bit_rate: int = 12_000_000,
        adb: Optional[Path] = None,
        scrcpy_server: Optional[Path] = None,
        ffmpeg: Optional[Path] = None,
        read_timeout_seconds: float = 10.0,
    ) -> None:
        if read_timeout_seconds <= 0:
            raise ValueError("read_timeout_seconds must be positive")
        self.adb = find_adb(adb)
        self.scrcpy_server = find_server(scrcpy_server)
        self.serial = serial
        self.camera_id = camera_id
        self.camera_facing = camera_facing
        self.width_px = int(width_px)
        self.height_px = int(height_px)
        self.fps = int(fps)
        self.bit_rate = int(bit_rate)
        self.ffmpeg = ffmpeg
        self.read_timeout_seconds = float(read_timeout_seconds)
        self._hub: Optional[ScrcpyCameraHub] = None
        self._queue = None
        self._first_frame: Optional[AndroidCameraFrame] = None
        self._effective: Dict[str, Any] = {}

    @property
    def is_open(self) -> bool:
        return self._hub is not None

    def isOpened(self) -> bool:
        return self.is_open

    @property
    def effective_configuration(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._effective)

    def open(self) -> "AndroidCamera":
        if self.is_open:
            return self
        self.serial = select_serial(self.adb, self.serial)
        hub = ScrcpyCameraHub(
            adb=self.adb,
            scrcpy_server=self.scrcpy_server,
            serial=self.serial,
            ffmpeg=self.ffmpeg,
            bit_rate=self.bit_rate,
            subscriber_queue_size=1,
            camera_id=self.camera_id,
            camera_facing=self.camera_facing,
            camera_width_px=self.width_px,
            camera_height_px=self.height_px,
            camera_fps=self.fps,
        )
        self._queue = hub.register("camera")
        self._hub = hub
        try:
            hub.start()
            self._first_frame = self._next_frame()
            self._effective = {
                "driver": "android_scrcpy_camera",
                "serial": self.serial,
                "requested": {
                    "camera_id": self.camera_id,
                    "camera_facing": self.camera_facing,
                    "width_px": self.width_px,
                    "height_px": self.height_px,
                    "fps": self.fps,
                    "bit_rate": self.bit_rate,
                },
                "effective": {
                    "width_px": int(self._first_frame.image.shape[1]),
                    "height_px": int(self._first_frame.image.shape[0]),
                    "color_order": "BGR",
                },
                "transport": hub.describe(),
                "controls": {
                    "host_camera_controls": "not_exposed_by_scrcpy_camera_transport",
                    "autofocus": "Camera2 TEMPLATE_RECORD device default",
                },
            }
        except Exception:
            self.close()
            raise
        return self

    def _next_frame(self) -> AndroidCameraFrame:
        if self._hub is None or self._queue is None:
            raise RuntimeError("Android camera is not open")
        try:
            item = self._queue.get(timeout=self.read_timeout_seconds)
        except queue.Empty:
            raise RuntimeError(
                "Timed out waiting {:.1f}s for an Android camera frame".format(
                    self.read_timeout_seconds
                )
            )
        if item[0] == "error":
            raise RuntimeError("Android camera capture failed: {}".format(item[1]))
        if item[0] == "end":
            raise RuntimeError("Android camera stream ended")
        _, sequence, image, source_ns, capture_ns, receive_ns = item
        metadata = {
            "schema_version": "1.0",
            "driver": "android_scrcpy_camera",
            "serial": self.serial,
            "sequence": int(sequence),
            "source_time_ns": int(source_ns),
            "capture_time_ns": int(capture_ns),
            "receive_time_ns": int(receive_ns),
            "timestamp_timebase": "Android CLOCK_MONOTONIC mapped to host",
            "transport_latency_ms": max(0.0, (receive_ns - capture_ns) / 1.0e6),
            "dropped_before": self._hub.take_drops("camera"),
            "image_space": {
                "schema_version": "1.0",
                "space_id": "android_camera_scrcpy_bgr_pixels",
                "stored_size_px": [int(image.shape[1]), int(image.shape[0])],
                "orientation": "scrcpy_capture_orientation_locked_initial",
                "mirroring": "as_delivered_by_android_camera_transport",
                "color_order": "BGR",
                "operation": "Camera2_to_MediaCodec_H264_to_FFmpeg_BGR",
            },
        }
        return AndroidCameraFrame(
            image=image,
            capture_time_ns=int(capture_ns),
            receive_time_ns=int(receive_ns),
            source_time_ns=int(source_ns),
            sequence=int(sequence),
            metadata=metadata,
        )

    def read_sample(self) -> AndroidCameraFrame:
        if self._first_frame is not None:
            frame, self._first_frame = self._first_frame, None
            return frame
        return self._next_frame()

    def get_frame(self) -> np.ndarray:
        return self.read_sample().image

    def get_frame_with_metadata(self) -> Tuple[np.ndarray, Mapping[str, Any]]:
        sample = self.read_sample()
        return sample.image, sample.metadata

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        try:
            return True, self.get_frame()
        except RuntimeError:
            return False, None

    def controls(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._effective.get("controls", {}))

    def set_control(self, _name: str, _value: Any) -> bool:
        return False

    def close(self) -> None:
        hub, self._hub = self._hub, None
        self._queue = None
        self._first_frame = None
        self._effective = {}
        if hub is not None:
            hub.stop()

    release = close

    def __enter__(self) -> "AndroidCamera":
        return self.open()

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def open_camera(serial: Optional[str] = None, **kwargs: Any) -> AndroidCamera:
    return AndroidCamera(serial, **kwargs).open()


Camera = AndroidCamera
