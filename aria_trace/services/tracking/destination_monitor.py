"""Bounded asynchronous destination observation for the normal workbench runtime."""
import base64
import json
import os
import struct
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .destination import MAX_AGE_S, checked_id, load_package, read_json

REPO = Path(__file__).resolve().parents[3]


def runtime_configuration():
    root = Path(os.environ.get("ARIA_TRACE_FEATURE_ROOT", str(REPO / ".tools"))).resolve()
    configured = os.environ.get("ARIA_TRACE_FEATURE_PYTHON")
    python = Path(configured) if configured else root / "feature-gpu/Scripts/python.exe"
    if not python.is_file():
        raise ValueError("Destination observation needs the feature Python runtime. Set ARIA_TRACE_FEATURE_PYTHON; see docs/destination-observer.md.")
    for path in [root / "lightglue-upstream/lightglue/aliked.py", root / "model-hub/checkpoints/aliked-n16rot.pth"]:
        if not path.is_file():
            raise ValueError("Destination feature dependencies are missing; see docs/destination-observer.md")
    return python.resolve(), root


def catalog(root):
    result = {}
    for path in sorted(Path(root).glob("*/*/destination.json")):
        try:
            meta = read_json(path)
            if meta.get("schema") != "aria-destination-1" or meta.get("game_profile_id") != path.parent.parent.name or meta.get("destination_id") != path.parent.name:
                continue
            checked_id(meta["destination_id"]); checked_id(meta["game_profile_id"])
            result.setdefault(meta["game_profile_id"], []).append({k: meta.get(k) for k in ["destination_id", "label", "view_count", "source_description", "image_size_wh"]})
        except (OSError, ValueError, TypeError):
            continue
    return result


class DestinationMonitor:
    """One in-flight worker request plus one replaceable pending frame; no queue."""
    def __init__(self, package, output, command=None, clock=None, startup_timeout_s=60.):
        self.meta, _ = load_package(package)
        self.output = Path(output); self.output.mkdir(parents=True, exist_ok=True)
        if command is None:
            python, root = runtime_configuration()
            command = [str(python), "-u", "-m", "aria_trace.services.tracking.destination_worker", "--package", str(Path(package).resolve()), "--runtime-root", str(root)]
        self.command = command
        self.clock = clock or time.perf_counter_ns
        self.condition = threading.Condition(); self.pending = None; self.stopped = False
        self.process = None; self.latest = None; self.image = None
        self.status = "starting"; self.error = None; self.geometry_error = None; self.device = None
        self.submitted = 0; self.replaced = 0; self.processed = 0
        self.closed_ns = None
        self.first_fresh_ns = None; self.last_fresh_ns = None; self.loss_start_ns = None
        self.longest_unavailable_s = 0.; self.loss_episodes = 0
        self.deadline = time.monotonic() + startup_timeout_s
        self.watch_stop = threading.Event()
        self.watchdog = threading.Thread(target=self._watch, name="destination-watchdog", daemon=True)
        self.thread = threading.Thread(target=self._work, name="destination-observer", daemon=True)
        self.thread.start()
        self.watchdog.start()

    @staticmethod
    def _terminate(process):
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except OSError:
                pass  # Process may exit between poll and terminate.

    def _watch(self):
        while not self.watch_stop.wait(.1):
            with self.condition:
                overdue = self.deadline is not None and time.monotonic() > self.deadline
                if not overdue or self.stopped:
                    continue
                self.status = "failed"
                self.error = "Destination worker timed out; see worker.log. Normal tracking continues."
                process = self.process
            self._terminate(process)
            return

    def submit(self, image, capture_ns):
        with self.condition:
            if self.stopped or self.status == "failed":
                return
        h, w = image.shape[:2]
        if abs(w / h - 640 / 360) > .02:
            with self.condition:
                self.geometry_error = "Destination package expects a 16:9 scene; capture geometry differs"
                self.latest = None; self.image = None; self.pending = None
            return
        # Capture packets may be reused by their source: own a small immutable copy.
        gray = cv2.resize(image, (640, 360), interpolation=cv2.INTER_AREA)
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        with self.condition:
            if self.stopped or self.status == "failed":
                return
            self.submitted += 1
            self.geometry_error = None
            if self.pending is not None:
                self.replaced += 1
            self.pending = (np.ascontiguousarray(gray), int(capture_ns))
            self.condition.notify_all()

    def _accept(self, row):
        received_ns = self.clock(); row = dict(row)
        image = row.pop("image_jpeg", None)
        row["received_ns"] = received_ns
        row["capture_to_result_ms"] = max(0., (received_ns - row["capture_ns"]) / 1e6)
        row["fresh_at_publication"] = row.get("aim") is not None and row["capture_to_result_ms"] <= MAX_AGE_S * 1000
        with self.condition:
            if self.stopped or self.status == "failed":
                return None
            self.processed += 1; self.latest = row; self.image = image
            self.error = None
            if row["fresh_at_publication"]:
                if self.first_fresh_ns is None:
                    self.first_fresh_ns = row["capture_ns"]
                if self.last_fresh_ns is not None and self.loss_start_ns is None and row["capture_ns"] - self.last_fresh_ns > int(MAX_AGE_S * 1e9):
                    self.loss_start_ns = self.last_fresh_ns + int(MAX_AGE_S * 1e9)
                if self.loss_start_ns is not None:
                    self.longest_unavailable_s = max(self.longest_unavailable_s, (row["capture_ns"] - self.loss_start_ns) / 1e9)
                    self.loss_episodes += 1; self.loss_start_ns = None
                self.last_fresh_ns = row["capture_ns"]
            elif self.first_fresh_ns is not None and self.loss_start_ns is None:
                self.loss_start_ns = min(row["capture_ns"], self.last_fresh_ns + int(MAX_AGE_S * 1e9))
        return row

    def snapshot(self):
        now = self.closed_ns or self.clock()
        with self.condition:
            result = dict(self.latest or {})
            age = None if self.latest is None else max(0., (now - self.latest["capture_ns"]) / 1e9)
            result.update(destination_id=self.meta["destination_id"], label=self.meta["label"], device=self.device, worker_status=self.status, error=self.error or self.geometry_error, submitted_frames=self.submitted, replaced_frames=self.replaced, processed_frames=self.processed, observation_age_ms=None if age is None else age * 1000, held_as_fresh=0)
            if self.latest is None or age > MAX_AGE_S or self.status in ("failed", "stopped") or self.error or self.geometry_error:
                result.update(state="unavailable", kind="stale" if self.latest else self.status, aim=None, bearing=None, points=[])
            start = self.loss_start_ns
            if start is None and self.last_fresh_ns is not None and now - self.last_fresh_ns > int(MAX_AGE_S * 1e9):
                start = self.last_fresh_ns + int(MAX_AGE_S * 1e9)
            current = 0. if start is None else max(0., (now - start) / 1e9)
            result.update(current_unavailable_s=current, longest_unavailable_s=max(self.longest_unavailable_s, current), loss_episodes=self.loss_episodes + int(start is not None))
            return result

    def overlay(self):
        with self.condition:
            encoded = self.image
            row = self.snapshot()
        image = None if encoded is None else cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            image = np.zeros((360, 640, 3), np.uint8)
        if row.get("aim") is not None:
            for point in row.get("points", []):
                cv2.circle(image, tuple(np.rint(point).astype(int)), 2, (90, 210, 90), 1)
            cv2.drawMarker(image, tuple(np.rint(row["aim"]).astype(int)), (0, 220, 255), cv2.MARKER_CROSS, 20, 2)
        banner = "{} | {} | age {} ms".format(self.meta["label"][:35], row.get("state", self.status), "-" if row.get("observation_age_ms") is None else int(row["observation_age_ms"]))
        cv2.rectangle(image, (0, 0), (640, 28), (25, 25, 25), -1)
        cv2.putText(image, banner, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, .47, (240, 240, 240), 1)
        ok, data = cv2.imencode(".jpg", image)
        if not ok:
            raise RuntimeError("Could not render destination observation")
        return data.tobytes()

    def _work(self):
        try:
            with (self.output / "worker.log").open("w", encoding="utf-8") as errors, (self.output / "observations.jsonl").open("w", encoding="utf-8") as records:
                # The workbench's Python 3.7 PYTHONPATH includes legacy binary
                # wheels. The separate feature interpreter must use its own.
                env = os.environ.copy()
                env["PYTHONPATH"] = str(REPO)
                process = subprocess.Popen(self.command, cwd=str(REPO), env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                with self.condition:
                    self.process = process
                    if self.stopped or self.status == "failed":
                        self._terminate(process)
                ready = json.loads(process.stdout.readline())
                if not ready.get("ready"):
                    raise RuntimeError(ready.get("error") or "Destination worker did not initialize")
                with self.condition:
                    if self.stopped or self.status == "failed":
                        return
                    self.status = "running"; self.device = ready.get("device")
                    self.deadline = None
                while True:
                    with self.condition:
                        self.condition.wait_for(lambda: self.stopped or self.pending is not None)
                        if self.stopped:
                            break
                        image, capture_ns = self.pending; self.pending = None
                        self.deadline = time.monotonic() + 10.
                    process.stdin.write(struct.pack("!Q", capture_ns) + image.tobytes()); process.stdin.flush()
                    line = process.stdout.readline(2 * 1024 * 1024)
                    if not line:
                        raise RuntimeError("Destination worker exited; see worker.log")
                    row = json.loads(line)
                    if row.get("error"):
                        raise RuntimeError(row["error"])
                    if row.get("capture_ns") != capture_ns:
                        raise RuntimeError("Destination worker returned a mismatched frame")
                    record = self._accept(row)
                    if record is None:
                        break
                    with self.condition:
                        self.deadline = None
                    records.write(json.dumps(record, allow_nan=False) + "\n"); records.flush()
        except Exception as exc:
            with self.condition:
                if not self.stopped and self.status != "failed":
                    self.status = "failed"; self.error = "{}: {}".format(type(exc).__name__, exc)
        finally:
            process = self.process
            if process is not None:
                self._terminate(process)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=3)
                for pipe in (process.stdin, process.stdout):
                    if pipe:
                        pipe.close()
            with self.condition:
                if self.status != "failed":
                    self.status = "stopped"
                self.closed_ns = self.closed_ns or self.clock()
            self.watch_stop.set()
            (self.output / "destination_observer.json").write_text(json.dumps(self.snapshot(), indent=2, allow_nan=False), encoding="utf-8")

    def close(self):
        with self.condition:
            self.stopped = True; self.pending = None; self.condition.notify_all()
            self.closed_ns = self.closed_ns or self.clock()
            process = self.process
        self._terminate(process)
        self.thread.join(timeout=5)
        self.watch_stop.set(); self.watchdog.join(timeout=1)
        return self.snapshot()
