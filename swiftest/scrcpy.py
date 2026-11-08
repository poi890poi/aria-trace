"""Read the raw H.264 display stream from the pinned scrcpy server using PyAV."""

import os
from pathlib import Path
import secrets
import socket
import subprocess
import threading
import time


class Scrcpy:
    mode = "scrcpy"
    frame_driven = True
    reason = None
    repeat = False

    def __init__(self, phone, server=None):
        try:
            import av
        except ImportError as exc:
            raise RuntimeError(
                "Install swiftest[scrcpy] for streaming."
            ) from exc
        supplied = server or os.environ.get("SCRCPY_SERVER")
        if not supplied or not Path(supplied).is_file():
            raise ValueError(
                "Supply the scrcpy 4.1 server with --scrcpy-server or SCRCPY_SERVER."
            )
        self.phone = phone
        self.bytes = 0
        self.setup_bytes = Path(supplied).stat().st_size
        self.condition = threading.Condition()
        self.sequence = 0
        self.frame = self.error = self.process = self.sock = self.port = None
        self.closed = False
        scid = format(secrets.randbelow(2**31), "08x")
        self.remote = f"/data/local/tmp/swiftest-scrcpy-{scid}.jar"
        try:
            phone.run("push", str(supplied), self.remote)
            self.port = int(
                phone.run("forward", "tcp:0", "localabstract:scrcpy_" + scid)
            )
            command = [
                phone.adb,
                "-s",
                phone.serial,
                "shell",
                "CLASSPATH=" + self.remote,
                "app_process",
                "/",
                "com.genymobile.scrcpy.Server",
                "4.1",
                "scid=" + scid,
                "tunnel_forward=true",
                "audio=false",
                "control=false",
                "cleanup=false",
                "raw_stream=true",
                "video_codec=h264",
                "max_fps=12",
                "video_bit_rate=2000000",
                "video_codec_options=repeat-previous-frame-after:long=150000",
            ]
            self.process = subprocess.Popen(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        "scrcpy server stopped; verify the 4.1 server and device support."
                    )
                sock = socket.create_connection(("127.0.0.1", self.port), timeout=1)
                sock.settimeout(1)
                try:
                    first = sock.recv(65536)
                except socket.timeout:
                    first = b""
                if first:
                    self.sock = sock
                    break
                sock.close()
                time.sleep(0.08)
            if self.sock is None:
                raise RuntimeError("scrcpy did not produce a video stream.")
            self.sock.settimeout(8)
            self.reader = threading.Thread(
                target=self._decode, args=(av, first), daemon=True
            )
            self.reader.start()
        except BaseException:
            self.close()
            raise

    def _decode(self, av, data):
        try:
            codec = av.CodecContext.create("h264", "r")
            codec.thread_count = 1
            while data and not self.closed:
                self.bytes += len(data)
                for packet in codec.parse(data):
                    for frame in codec.decode(packet):
                        image = frame.to_ndarray(format="bgr24")
                        with self.condition:
                            self.frame = image
                            self.sequence += 1
                            self.condition.notify_all()
                data = self.sock.recv(65536)
            if not self.closed:
                raise RuntimeError("scrcpy disconnected.")
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def read(self):
        with self.condition:
            sequence = self.sequence
            if not self.condition.wait_for(
                lambda: self.sequence > sequence or self.error or self.closed, timeout=8
            ):
                raise RuntimeError("Timed out waiting for a fresh scrcpy frame.")
            if self.error or self.closed:
                raise RuntimeError(f"scrcpy unavailable: {self.error}")
            return self.frame

    def close(self):
        self.closed = True
        with self.condition:
            self.condition.notify_all()
        if self.sock:
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.sock.close()
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if hasattr(self, "reader"):
            self.reader.join(2)
        try:
            if self.port:
                self.phone.run("forward", "--remove", f"tcp:{self.port}")
        finally:
            self.phone.shell("rm", "-f", self.remote)
