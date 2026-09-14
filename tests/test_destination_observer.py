import hashlib
import json
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from aria_trace.services.tracking.destination import (
    DEFAULT_MASK, MODEL, MODEL_SHA256, SCHEMA, DestinationObserver, load_package,
)
from aria_trace.services.tracking.destination_monitor import DestinationMonitor, REPO
from aria_trace.services.tracking.destination_packages import import_bank


def package(root):
    root.mkdir(parents=True, exist_ok=True)
    points = np.float32([[200, 60], [240, 70], [280, 80], [205, 110], [245, 120], [285, 130], [320, 65], [340, 110]])
    descriptors = np.eye(8, 128, dtype=np.float32)
    np.savez_compressed(str(root / "bank.npz"), vp=points, vd=descriptors, vo=np.ones(8, bool))
    meta = dict(schema=SCHEMA, model=MODEL, model_sha256=MODEL_SHA256, destination_id="gate", game_profile_id="game", label="Gate", image_size_wh=[640, 360], mask=DEFAULT_MASK, views=[dict(key="v", aim=[265, 95], width=40)], view_count=1, bank_sha256=hashlib.sha256((root / "bank.npz").read_bytes()).hexdigest())
    (root / "destination.json").write_text(json.dumps(meta), encoding="utf-8")
    return root, points, descriptors


FAKE_WORKER = '''import sys,json,struct,time
print(json.dumps(dict(ready=True,device="fixture")),flush=True)
while True:
    header=sys.stdin.buffer.read(8)
    if not header:break
    image=sys.stdin.buffer.read(640*360)
    time.sleep(.15)
    print(json.dumps(dict(capture_ns=struct.unpack("!Q",header)[0],state="tracking",kind="anchor",aim=[320,100],bearing=0.,points=[],image_jpeg=None,commands_issued=0)),flush=True)
'''


class DestinationObserverTests(unittest.TestCase):
    def test_import_publishes_complete_package_without_private_staging_acl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _ = package(Path(tmp) / "source")
            views = Path(tmp) / "views.json"
            views.write_text(json.dumps({"target": load_package(root)[0]["views"]}))
            destination = import_bank(root / "bank.npz", views, "target", Path(tmp) / "installed", "game", "gate", "Gate", "test")
            load_package(destination, "game")
            self.assertEqual(list(destination.parent.glob(".destination-import-*")), [])
            with self.assertRaises(ValueError):
                import_bank(root / "bank.npz", views, "target", Path(tmp) / "installed", "game", "gate", "Gate", "test")
            if sys.platform == "win32":
                import ctypes
                from ctypes import wintypes
                api = ctypes.windll.advapi32
                required = wintypes.DWORD()
                api.GetFileSecurityW(str(destination), 4, None, 0, ctypes.byref(required))
                descriptor = ctypes.create_string_buffer(required.value)
                self.assertTrue(api.GetFileSecurityW(str(destination), 4, descriptor, required.value, ctypes.byref(required)))
                control = wintypes.WORD(); revision = wintypes.DWORD()
                self.assertTrue(api.GetSecurityDescriptorControl(descriptor, ctypes.byref(control), ctypes.byref(revision)))
                self.assertFalse(control.value & 0x1000, "Published package must inherit its parent DACL")

    def test_package_rejects_wrong_game_tampering_and_invalid_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _ = package(Path(tmp))
            load_package(root, "game")
            with self.assertRaises(ValueError):
                load_package(root, "other")
            meta = json.loads((root / "destination.json").read_text())
            meta["mask"]["include"] = [0, 0, 10000, 360]
            (root / "destination.json").write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                load_package(root)
            meta["mask"] = DEFAULT_MASK
            meta["bank_sha256"] = "wrong"
            (root / "destination.json").write_text(json.dumps(meta))
            with self.assertRaises(ValueError):
                load_package(root)

    def test_physical_goal_transfers_and_unavailable_does_not_hold_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, points, descriptors = package(Path(tmp))
            observer = DestinationObserver(*load_package(root))
            image = np.zeros((360, 640), np.uint8)
            result = observer.step(image, (points + [12, -5], descriptors), 0.)
            np.testing.assert_allclose(result["aim"], [277, 90], atol=.001)
            self.assertEqual(result["commands_issued"], 0)
            empty = (np.empty((0, 2), np.float32), np.empty((0, 128), np.float32))
            lost = observer.step(image, empty, .21)
            self.assertIsNone(lost["aim"])
            self.assertEqual(lost["state"], "unavailable")
            self.assertEqual(lost["travel_relation"], "unknown")

    def test_long_frame_gap_cannot_bridge_with_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, points, descriptors = package(Path(tmp))
            observer = DestinationObserver(*load_package(root))
            image = np.random.RandomState(0).randint(0, 255, (360, 640), dtype=np.uint8)
            observer.step(image, (points, descriptors), 1.)
            self.assertIsNone(observer.step(image, None, 2.)["aim"])


class DestinationMonitorTests(unittest.TestCase):
    def wait_for(self, predicate, timeout=4):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if predicate():
                return
            time.sleep(.01)
        self.fail("Timed out waiting for destination fixture")

    def test_worker_does_not_inherit_legacy_binary_package_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _ = package(Path(tmp) / "package")
            worker = Path(tmp) / "worker.py"
            worker.write_text("import os,json\nprint(json.dumps(dict(ready=True,device=os.environ.get('PYTHONPATH'))),flush=True)\nimport sys\nsys.stdin.buffer.read()")
            with patch.dict('os.environ', {'PYTHONPATH': 'legacy-python-binaries'}):
                monitor = DestinationMonitor(root, Path(tmp) / "output", command=[sys.executable, "-u", str(worker)])
                try:
                    self.wait_for(lambda: monitor.status == "running")
                    self.assertEqual(monitor.snapshot()["device"], str(REPO))
                finally:
                    monitor.close()

    def test_slow_worker_replaces_pending_frames_expires_and_closes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _ = package(Path(tmp) / "package")
            worker = Path(tmp) / "worker.py"; worker.write_text(FAKE_WORKER)
            monitor = DestinationMonitor(root, Path(tmp) / "output", command=[sys.executable, "-u", str(worker)])
            try:
                self.wait_for(lambda: monitor.status == "running")
                image = np.zeros((360, 640, 3), np.uint8)
                monitor.submit(image, time.perf_counter_ns())
                self.wait_for(lambda: monitor.pending is None)
                for _ in range(20):
                    monitor.submit(image, time.perf_counter_ns())
                self.wait_for(lambda: monitor.processed >= 2)
                self.assertGreaterEqual(monitor.replaced, 18)
                self.assertLessEqual(monitor.processed, 3)
                self.assertIsNotNone(monitor.snapshot()["aim"])
                time.sleep(.55)
                stale = monitor.snapshot()
                self.assertIsNone(stale["aim"])
                self.assertEqual(stale["state"], "unavailable")
                self.assertGreater(stale["longest_unavailable_s"], 0)
                monitor.submit(np.zeros((480, 640, 3), np.uint8), time.perf_counter_ns())
                self.assertIn("geometry", monitor.snapshot()["error"])
                self.assertIsNone(monitor.snapshot()["aim"])
            finally:
                monitor.close()
            self.assertFalse(monitor.thread.is_alive())
            self.assertIsNotNone(monitor.process.poll())
            self.assertTrue((Path(tmp) / "output/observations.jsonl").is_file())
            stopped = monitor.snapshot()["longest_unavailable_s"]
            time.sleep(.05)
            self.assertEqual(stopped, monitor.snapshot()["longest_unavailable_s"])

    def test_worker_failure_is_explicit_and_restart_has_no_old_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _ = package(Path(tmp) / "package")
            for index in range(2):
                monitor = DestinationMonitor(root, Path(tmp) / str(index), command=[sys.executable, "-c", "print('{\"error\":\"model unavailable\"}')"])
                self.wait_for(lambda: monitor.status == "failed")
                result = monitor.snapshot()
                self.assertIn("model unavailable", result["error"])
                self.assertIsNone(result["aim"])
                self.assertEqual(result["processed_frames"], 0)
                late = monitor._accept(dict(capture_ns=time.perf_counter_ns(), aim=[320, 100]))
                self.assertIsNone(late)
                self.assertIn("model unavailable", monitor.snapshot()["error"])
                self.assertEqual(monitor.snapshot()["processed_frames"], 0)
                monitor.close()


class DestinationTimeoutTests(unittest.TestCase):
    def test_hung_startup_times_out_and_releases_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _, _ = package(Path(tmp) / "package")
            monitor = DestinationMonitor(root, Path(tmp) / "output", command=[sys.executable, "-c", "import time; time.sleep(30)"], startup_timeout_s=.2)
            try:
                end = time.monotonic() + 4
                while monitor.status != "failed" and time.monotonic() < end:
                    time.sleep(.01)
                self.assertIn("timed out", monitor.snapshot()["error"])
            finally:
                monitor.close()
            self.assertFalse(monitor.thread.is_alive())
            self.assertFalse(monitor.watchdog.is_alive())


if __name__ == "__main__":
    unittest.main()
