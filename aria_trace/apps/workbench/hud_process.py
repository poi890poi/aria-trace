"""Subprocess manager for the Windows acquisition HUD."""

import os
import queue
import subprocess
import sys
import threading
import time

from .hud import WDA_EXCLUDEFROMCAPTURE


class WorkbenchHudProcess:
    def __init__(self, state_url: str) -> None:
        self.state_url = str(state_url)
        self.process = None
        self.display_affinity = None
        self._lines = queue.Queue()

    def start(self) -> None:
        if os.name != "nt":
            raise RuntimeError("The in-game HUD is available only on Windows")
        if self.process is not None:
            if self.process.poll() is None:
                return
            self.stop()
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "aria_trace.apps.workbench.hud",
                "--state-url",
                self.state_url,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        def read_output() -> None:
            if self.process is None or self.process.stdout is None:
                return
            for line in self.process.stdout:
                self._lines.put(line.rstrip())

        threading.Thread(
            target=read_output,
            name="acquisition-hud-output",
            daemon=True,
        ).start()
        deadline = time.monotonic() + 8.0
        output = []
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                break
            try:
                line = self._lines.get(timeout=0.1)
            except queue.Empty:
                continue
            output.append(line)
            if line.startswith("ARIATRACE_HUD_READY "):
                self.display_affinity = int(line.rsplit("=", 1)[1])
                if self.display_affinity != WDA_EXCLUDEFROMCAPTURE:
                    self.stop()
                    raise RuntimeError(
                        "HUD capture exclusion was not applied: {}".format(
                            self.display_affinity
                        )
                    )
                return
        self.stop()
        raise RuntimeError(
            "HUD process did not become ready{}".format(
                ": " + " | ".join(output) if output else ""
            )
        )

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
        if process.stdout is not None:
            process.stdout.close()
