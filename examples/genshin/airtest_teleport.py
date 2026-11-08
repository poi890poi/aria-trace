"""Airtest implementation of launch -> map -> waypoint -> teleport -> arrival.

Python only. Install airtest==1.4.3, then pass --serial, --assets and --output.
Assets are local screen crops described by assets/manifest.json. They are not
distributed with this example. See benchmarks/android/README.md.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from airtest.core.api import init_device, start_app, touch
from airtest.core.cv import Template
from airtest.core.settings import Settings

PACKAGE = "com.miHoYo.GenshinImpact"


class Teleport:
    def __init__(self, serial, assets, output, timeout=240):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.manifest = json.loads((Path(assets) / "manifest.json").read_text())
        self.templates = {
            name: Template(str(Path(assets) / spec["file"]), threshold=spec["threshold"])
            for name, spec in self.manifest["templates"].items()
        }
        # Exact-resolution UI text crops: feature matching is neither needed nor
        # appropriate for these small controls. Keep Airtest's ordinary input delay.
        Settings.CVSTRATEGY = ["tpl"]
        self.device = init_device("Android", serial, adb_path=shutil.which("adb"),
                                  cap_method="JAVACAP", touch_method="MAXTOUCH")
        self.deadline = time.monotonic() + timeout
        self.events = []
        self.frame = None
        self.captures = 0
        self.capture_s = self.match_s = 0.0
        self.started = time.perf_counter()

    def event(self, name, **values):
        row = dict(event=name, seconds=time.perf_counter() - self.started, **values)
        self.events.append(row)
        print(json.dumps(row), flush=True)

    def snapshot(self):
        t = time.perf_counter()
        image = self.device.snapshot()
        if image is None:
            raise RuntimeError("Airtest returned no frame")
        width, height = self.manifest["resolution"]
        # Javacap on this Samsung returns natural portrait pixels even while
        # the game is landscape-left. Explicitly normalize this measured layout.
        if image.shape[:2] == (width, height):
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if image.shape[:2] != (height, width):
            raise RuntimeError("Capture resolution differs from the recorded assets")
        self.capture_s += time.perf_counter() - t
        self.captures += 1
        self.frame = image
        return image

    def match(self, name):
        h, w = self.frame.shape[:2]
        l, t, r, b = self.manifest["templates"][name]["region"]
        x, y, right, bottom = round(l*w), round(t*h), round(r*w), round(b*h)
        began = time.perf_counter()
        point = self.templates[name].match_in(self.frame[y:bottom, x:right])
        self.match_s += time.perf_counter() - began
        return (point[0] + x, point[1] + y) if point else None

    def world(self):
        return self.match("uid") and self.match("level")

    def save(self, name):
        cv2.imwrite(str(self.output / (name + ".png")), self.frame)

    def tap(self, point, reason):
        # Match TestKit's focus guard. Never tap a login/system prompt.
        if self.device.get_top_activity()[0] != PACKAGE:
            raise RuntimeError("Genshin lost focus")
        touch(tuple(int(round(v)) for v in point))
        self.event("tap", reason=reason, point=point)

    def wait_for(self, predicate, name, interval=0.5):
        while time.monotonic() < self.deadline:
            self.snapshot()
            result = predicate()
            if result:
                self.event(name)
                return result
            time.sleep(interval)
        raise TimeoutError("Timed out waiting for " + name)

    def run(self):
        start_app(PACKAGE)
        self.event("launch_dispatched")
        entry_sent = set()

        def ready():
            nonlocal entry_sent
            if self.world():
                return True
            for name in ("entry", "door"):
                if name not in entry_sent and (button := self.match(name)):
                    self.save(name)
                    self.tap(button, name)
                    entry_sent.add(name)
                    break
            return False

        self.wait_for(ready, "world_ready")
        self.save("before")
        w, h = self.manifest["resolution"]
        self.tap((0.085*(w-1), 0.145*(h-1)), "minimap")
        self.wait_for(lambda: self.match("map"), "map_ready")
        self.save("map")
        x, y = self.manifest["waypoint"]
        self.tap((x*(w-1), y*(h-1)), "waypoint")
        button = self.wait_for(lambda: self.match("waypoint_title") and self.match("teleport"),
                               "waypoint_selected")
        self.save("selected")
        self.tap(button, "teleport")

        def loading():
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            neutral = hsv[:, :, 1] < 35
            return float(np.mean(neutral & (hsv[:, :, 2] > 220))) > .80 or \
                float(np.mean(hsv[:, :, 2] < 35)) > .90

        self.wait_for(loading, "loading", interval=.25)
        self.save("loading")
        consecutive = 0

        def arrived():
            nonlocal consecutive
            consecutive = consecutive + 1 if self.world() else 0
            return consecutive >= 2

        self.wait_for(arrived, "arrival", interval=.5)
        self.save("arrival")
        return {"status": "teleport_completed", "captures": self.captures,
                "capture_seconds": self.capture_s, "match_seconds": self.match_s,
                "backend": "JAVACAP", "touch_backend": "MAXTOUCH",
                "events": self.events}

    def close(self):
        self.device.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=240)
    args = parser.parse_args()
    logging.getLogger("airtest").setLevel(logging.WARNING)
    args.output.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    runner = None
    result = {"status": "failed"}
    try:
        runner = Teleport(args.serial, args.assets, args.output, args.timeout)
        result = runner.run()
    except Exception as error:
        result["error"] = repr(error)
        if runner:
            result["events"] = runner.events
            if runner.frame is not None:
                runner.save("failure")
        raise
    finally:
        result["task_seconds"] = time.perf_counter() - began
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        if runner:
            runner.close()


if __name__ == "__main__":
    main()
