"""Reusable touch control over a pinned scrcpy server control socket.

This module owns only Android control transport.  It does not capture frames,
record sessions, or know why a gesture is being issued.
"""

from __future__ import annotations

import collections
import os
import socket
import struct
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Sequence


CONTROL_MESSAGE_INJECT_TOUCH_EVENT = 2
MOTION_ACTIONS = {"DOWN": 0, "UP": 1, "MOVE": 2, "CANCEL": 3}
GENERIC_FINGER_POINTER_ID = (1 << 64) - 2


def serialize_touch_event(
    action: str,
    point_xy: Sequence[int],
    screen_size_px: Sequence[int],
    *,
    pointer_id: int = GENERIC_FINGER_POINTER_ID,
    pressure: Optional[float] = None,
    action_button: int = 0,
    buttons: int = 0,
) -> bytes:
    """Serialize one scrcpy 4.1 ``INJECT_TOUCH_EVENT`` control message."""

    normalized_action = str(action).upper()
    if normalized_action not in MOTION_ACTIONS:
        raise ValueError("Touch action must be DOWN, MOVE, UP, or CANCEL")
    if len(point_xy) != 2 or len(screen_size_px) != 2:
        raise ValueError("Touch point and screen size must each contain two values")
    x, y = map(int, point_xy)
    width, height = map(int, screen_size_px)
    if width <= 0 or height <= 0 or width > 65535 or height > 65535:
        raise ValueError("scrcpy touch screen dimensions must be in 1..65535")
    if x < 0 or y < 0 or x >= width or y >= height:
        raise ValueError(
            "Touch point {},{} is outside {}x{}".format(x, y, width, height)
        )
    if pressure is None:
        pressure = 0.0 if normalized_action in ("UP", "CANCEL") else 1.0
    pressure = float(pressure)
    if pressure < 0.0 or pressure > 1.0:
        raise ValueError("Touch pressure must be in 0..1")
    pressure_fixed = min(0xFFFF, int(pressure * 65536.0))
    return struct.pack(
        ">BBQiiHHHII",
        CONTROL_MESSAGE_INJECT_TOUCH_EVENT,
        MOTION_ACTIONS[normalized_action],
        int(pointer_id) & ((1 << 64) - 1),
        x,
        y,
        width,
        height,
        pressure_fixed,
        int(action_button),
        int(buttons),
    )


def _read_exact(stream: socket.socket, count: int) -> Optional[bytes]:
    chunks = []
    remaining = count
    while remaining:
        chunk = stream.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class ScrcpyTouchController:
    """Persistent, control-only scrcpy session with an OpenCV-like lifecycle."""

    def __init__(
        self,
        adb: Path,
        scrcpy_server: Path,
        serial: str,
        screen_size_px: Sequence[int],
        *,
        server_version: str = "4.1",
        connect_timeout_seconds: float = 10.0,
    ) -> None:
        self.adb = Path(adb)
        self.scrcpy_server = Path(scrcpy_server)
        self.serial = str(serial)
        self.screen_size_px = tuple(map(int, screen_size_px))
        self.server_version = str(server_version)
        self.connect_timeout_seconds = float(connect_timeout_seconds)
        self._socket = None
        self._server_process = None
        self._server_log_thread = None
        self._server_log = collections.deque(maxlen=100)
        self._port = None
        self._scid = None
        self._device_name = None
        self._send_lock = threading.Lock()
        self._events_sent = 0
        self._last_send_time_ns = None

    def _adb(self):
        return [str(self.adb), "-s", self.serial]

    def open(self) -> "ScrcpyTouchController":
        if self._socket is not None:
            return self
        if not self.adb.is_file():
            raise RuntimeError("ADB executable does not exist: {}".format(self.adb))
        if not self.scrcpy_server.is_file():
            raise RuntimeError(
                "scrcpy server does not exist: {}".format(self.scrcpy_server)
            )
        width, height = self.screen_size_px
        if width <= 0 or height <= 0 or width > 65535 or height > 65535:
            raise ValueError("Invalid Android control screen size")

        remote_server = "/data/local/tmp/iris-scrcpy-control-v{}.jar".format(
            self.server_version
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
            self.server_version,
            "scid={:08x}".format(self._scid),
            "log_level=info",
            "video=false",
            "audio=false",
            "control=true",
            "clipboard_autosync=false",
            "power_on=false",
            "tunnel_forward=true",
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
            target=collect_server_log,
            name="scrcpy-control-server-log",
            daemon=True,
        )
        self._server_log_thread.start()
        try:
            deadline = time.time() + self.connect_timeout_seconds
            while time.time() < deadline:
                if self._server_process.poll() is not None:
                    raise RuntimeError(
                        "scrcpy control server exited before connection: {}".format(
                            " | ".join(self._server_log)
                        )
                    )
                candidate = None
                try:
                    candidate = socket.create_connection(
                        ("127.0.0.1", self._port), timeout=0.5
                    )
                    if _read_exact(candidate, 1) != b"\x00":
                        candidate.close()
                        time.sleep(0.05)
                        continue
                    device_meta = _read_exact(candidate, 64)
                    if device_meta is None:
                        candidate.close()
                        time.sleep(0.05)
                        continue
                    candidate.settimeout(None)
                    self._socket = candidate
                    self._device_name = device_meta.split(b"\x00", 1)[0].decode(
                        "utf-8", "replace"
                    )
                    break
                except OSError:
                    if candidate is not None:
                        candidate.close()
                    time.sleep(0.05)
            if self._socket is None:
                raise RuntimeError("Timed out connecting to scrcpy control server")
        except Exception:
            self.close()
            raise
        return self

    def is_opened(self) -> bool:
        return self._socket is not None

    def inject_touch(self, action: str, point_xy: Sequence[int]) -> int:
        message = serialize_touch_event(action, point_xy, self.screen_size_px)
        with self._send_lock:
            if self._socket is None:
                raise RuntimeError("scrcpy touch controller is not open")
            self._socket.sendall(message)
            sent_ns = time.perf_counter_ns()
            self._events_sent += 1
            self._last_send_time_ns = sent_ns
            return sent_ns

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._socket.close()
            self._socket = None
        if self._server_process is not None:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
                self._server_process.wait()
            self._server_process = None
        if self._server_log_thread is not None:
            self._server_log_thread.join(timeout=1)
            self._server_log_thread = None
        if self._port is not None:
            try:
                subprocess.check_call(
                    self._adb() + ["forward", "--remove", "tcp:{}".format(self._port)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
            except Exception:
                pass
            self._port = None

    def describe(self) -> Dict[str, object]:
        return {
            "type": type(self).__name__,
            "transport": "scrcpy_control_socket",
            "protocol_version": self.server_version,
            "serial": self.serial,
            "screen_size_px": list(self.screen_size_px),
            "device_name": self._device_name,
            "events_sent": int(self._events_sent),
            "last_send_time_ns": self._last_send_time_ns,
            "server_log_tail": list(self._server_log),
        }

    def __enter__(self) -> "ScrcpyTouchController":
        return self.open()

    def __exit__(self, *_exc) -> None:
        self.close()
