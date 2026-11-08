"""Prepare a small Android capture helper locally. Fall back to PNG on unsupported devices."""

import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
import queue
import base64
from pathlib import Path


def build_capture():
    source = Path(__file__).parent / "assets/IntentScreenshot.java"
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    cache = Path(tempfile.gettempdir()) / "swiftest-build" / fingerprint
    cache.mkdir(parents=True, exist_ok=True)
    jar = cache / "snapshot.jar"
    if jar.is_file():
        return jar
    homes = [
        os.environ.get("ANDROID_HOME"),
        os.environ.get("ANDROID_SDK_ROOT"),
        Path.home() / "Library/Android/sdk",
        Path.home() / "Android/Sdk",
        Path(os.environ.get("LOCALAPPDATA", Path.home())) / "Android/Sdk",
    ]
    sdk = next(
        (Path(p) for p in homes if p and (Path(p) / "build-tools").is_dir()), None
    )
    if sdk is None:
        raise RuntimeError("Android SDK unavailable")
    platforms = sorted(sdk.glob("platforms/android-*/android.jar"), reverse=True)
    compilers = sorted(sdk.glob("build-tools/*/lib/d8.jar"), reverse=True)
    if (
        not platforms
        or not compilers
        or not shutil.which("javac")
        or not shutil.which("java")
    ):
        raise RuntimeError("Android SDK or JDK unavailable")
    with tempfile.TemporaryDirectory(dir=str(cache)) as build:
        for command in (
            [
                "javac",
                "-source",
                "8",
                "-target",
                "8",
                "-classpath",
                str(platforms[0]),
                "-d",
                build,
                str(source),
            ],
            [
                "java",
                "-cp",
                str(compilers[0]),
                "com.android.tools.r8.D8",
                "--min-api",
                "26",
                "--lib",
                str(platforms[0]),
                "--output",
                str(Path(build) / "snapshot.jar"),
                str(Path(build) / "IntentScreenshot.class"),
            ],
        ):
            subprocess.run(
                command,
                check=True,
                timeout=90,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        os.replace(str(Path(build) / "snapshot.jar"), str(jar))
    return jar


class Capture:
    def __init__(self, phone, mode="auto"):
        self.phone = phone
        self.mode = "png"
        self.reason = None
        self.remote = None
        self.bytes = 0
        self.setup_bytes = 0
        self.process = None
        self.responses = queue.Queue()
        self.repeat = False
        self.previous = None
        self.prefer_direct = mode in ("auto", "direct", "bitmap")
        self.prefer_bitmap = mode in ("auto", "bitmap")
        self.acquisition = None
        if mode != "png":
            try:
                jar = build_capture()
                self.remote = "/data/local/tmp/aria-intent-{}-{}.jar".format(
                    jar.parent.name, os.getpid()
                )
                self.phone.run("push", str(jar), self.remote)
                self.setup_bytes = jar.stat().st_size
                self.process = subprocess.Popen(
                    [
                        phone.adb,
                        "-s",
                        phone.serial,
                        "shell",
                        "-T",
                        "env",
                        "CLASSPATH=" + self.remote,
                        "app_process",
                        "/system/bin",
                        "IntentScreenshot",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
                threading.Thread(target=self._receive, daemon=True).start()
                self.mode = "jpeg"
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                self.reason = str(error)

    def _receive(self):
        try:
            stream = self.process.stdout
            while True:
                line = stream.readline(32 * 1024 * 1024)
                if not line or not line.endswith(b"\n"):
                    raise RuntimeError("Capture helper closed the stream")
                acquisition = None
                payload = line.rstrip()
                if payload.startswith((b"D:", b"S:")):
                    acquisition = "direct" if payload[:1] == b"D" else "screencap"
                    payload = payload[2:]
                self.responses.put((base64.b64decode(payload, validate=True), len(line), acquisition))
        except (OSError, RuntimeError, ValueError) as error:
            self.responses.put(error)

    def read(self):
        self.repeat = False
        if self.mode == "jpeg":
            try:
                self.process.stdin.write(b"R" if self.prefer_bitmap else b"D" if self.prefer_direct else b"C")
                self.process.stdin.flush()
                data = self.responses.get(timeout=15)
                if isinstance(data, Exception):
                    raise data
                data, response_bytes, acquisition = data
                self.acquisition = acquisition or "screencap"
                if acquisition == "direct" and self.prefer_bitmap:
                    self.acquisition = "direct-bitmap"
                if self.prefer_direct and acquisition != "direct":
                    self.reason = "Direct capture unavailable; using screencap."
                self.bytes += response_bytes
                if not data and self.previous is not None:
                    self.repeat = True
                    return self.previous
                if not (data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")):
                    raise RuntimeError("Invalid on-device JPEG")
                self.previous = data
                return data
            except (
                OSError,
                RuntimeError,
                ValueError,
                subprocess.SubprocessError,
                queue.Empty,
            ) as error:
                self.reason = str(error)
                self.mode = "png"
        data = self.phone.run("exec-out", "screencap", "-p")
        self.acquisition = "screencap-png"
        self.bytes += len(data)
        return data

    def close(self):
        if self.process:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process.stdout.close()
        if self.remote:
            self.phone.shell("rm", "-f", self.remote)
