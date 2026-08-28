import threading
import time
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from acquisition.live_tracker import (
    GlobalFix,
    GlobalMapLocalizer,
    TwoRateRealtimeTracker,
    render_map_overlay,
)
from acquisition.models import FramePacket
from acquisition.profiles import ProfileCatalog
from acquisition.workbench import AcquisitionWorkbench


class GlobalMapLocalizerTests(unittest.TestCase):
    def test_localizes_exact_observed_patch_on_mosaic(self):
        random = np.random.RandomState(42)
        mosaic = random.randint(0, 256, (180, 220, 3), dtype=np.uint8)
        center_x, center_y = 137, 104
        observation = mosaic[74:134, 107:167].copy()
        mask = np.zeros((60, 60), np.uint8)
        cv2.circle(mask, (30, 30), 27, 255, -1)
        localizer = GlobalMapLocalizer(
            mosaic,
            scales=(1.0,),
            coarse_angle_step_deg=30.0,
            refine_angle_step_deg=2.0,
        )

        fix = localizer.localize(observation, mask, yaw_prior_deg=0.0)

        self.assertAlmostEqual(fix.x, center_x, delta=2.0)
        self.assertAlmostEqual(fix.y, center_y, delta=2.0)
        self.assertAlmostEqual(fix.yaw_deg, 0.0, delta=2.1)
        self.assertGreater(fix.score, 0.90)


class BlockingLocalizer:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def localize(self, observation, mask, yaw_prior_deg=None):
        self.started.set()
        if not self.release.wait(2.0):
            raise RuntimeError("test did not release global localization")
        return GlobalFix(80.0, 70.0, 15.0, 1.0, 0.91, 0.12, 123.0)


class TwoRateTrackerTests(unittest.TestCase):
    @staticmethod
    def _tracker(localizer):
        mosaic = np.full((160, 180, 3), 40, np.uint8)
        minimap_config = {"crop_xywh": [0, 0, 80, 80]}
        minimap_calibration = {
            "outer_boundary": {"center_x": 40, "center_y": 40, "radius": 34}
        }
        scene_yaw = {
            "focal_ratio": 0.9,
            "camera_matrix": [[72, 0, 40], [0, 72, 40], [0, 0, 1]],
            "config": {"excluded_rects": []},
        }
        return TwoRateRealtimeTracker(
            mosaic,
            minimap_config,
            minimap_calibration,
            scene_yaw,
            global_interval_s=10.0,
            localizer=localizer,
        )

    def test_global_search_does_not_block_high_rate_update(self):
        localizer = BlockingLocalizer()
        tracker = self._tracker(localizer)
        frame = np.zeros((100, 100, 3), np.uint8)
        try:
            first = tracker.update(frame, 1)
            self.assertTrue(localizer.started.wait(0.5))
            self.assertTrue(first["global_localization_running"])
            self.assertIsNone(first["pose"])

            second = tracker.update(frame, 2)
            self.assertTrue(second["global_localization_running"])
            self.assertIsNone(second["pose"])

            localizer.release.set()
            result = second
            for timestamp in range(3, 50):
                result = tracker.update(frame, timestamp)
                if result["global_fix_fresh"]:
                    break
                time.sleep(0.005)
            self.assertTrue(result["global_fix_fresh"])
            self.assertEqual(result["global_fix"]["decision"], "initialized")
            self.assertAlmostEqual(result["pose"]["x"], 80.0)
            self.assertAlmostEqual(result["pose"]["y"], 70.0)
        finally:
            localizer.release.set()
            tracker.close()

    def test_renders_pose_and_quality_overlay(self):
        mosaic = np.full((420, 620, 3), (35, 45, 55), np.uint8)
        state = {
            "mode": "TRACK",
            "pose": {"x": 310.0, "y": 210.0, "yaw_deg": 30.0},
            "trail": [[300.0, 205.0], [305.0, 208.0]],
            "global_fix": {"score": 0.9, "margin": 0.1, "elapsed_ms": 120},
            "local_motion": {"response": 0.8},
            "scene_yaw": {"confidence": 0.7},
            "update_elapsed_ms": 6.0,
            "position_sigma_map_px": 3.0,
            "yaw_sigma_deg": 2.0,
        }
        image = render_map_overlay(mosaic, state)
        self.assertEqual(image.shape, (360, 520, 3))
        self.assertGreater(int(np.max(image[:, :, 1])), 200)


class FakeLiveFrameSource:
    def __init__(self):
        self.stopped = threading.Event()
        self.sequence = 0

    def start(self):
        return None

    def read(self):
        if self.stopped.wait(0.005):
            return None
        self.sequence += 1
        now = time.perf_counter_ns()
        return FramePacket(
            "main", np.full((240, 260, 3), 70, np.uint8), now, now
        )

    def stop(self):
        self.stopped.set()


class FakeLiveEngine:
    def __init__(self, *args, **kwargs):
        self.sequence = 0
        self.closed = False

    def update(self, frame, host_time_ns):
        self.sequence += 1
        return {
            "sequence": self.sequence,
            "mode": "TRACK",
            "pose": {"x": 45.0, "y": 55.0, "yaw_deg": 10.0},
            "trail": [[45.0, 55.0]],
            "global_fix": {
                "score": 0.9,
                "margin": 0.1,
                "elapsed_ms": 80.0,
            },
            "local_motion": {"response": 0.8},
            "scene_yaw": {"confidence": 0.7},
            "update_elapsed_ms": 4.0,
            "position_sigma_map_px": 3.0,
            "yaw_sigma_deg": 2.0,
        }

    def close(self):
        self.closed = True


class WorkbenchLiveTrackerTests(unittest.TestCase):
    def test_tracker_lifecycle_is_serializable_and_recording_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_root = root / "artifacts"
            game_id = "genshin-impact-pc"
            calibration_root = artifact_root / "minimap_calibrations" / game_id / "mini-a"
            scene_root = artifact_root / "scene_yaw_calibrations" / game_id / "yaw-a"
            stitch_root = artifact_root / "map_stitches" / game_id / "map-a"
            for path in (calibration_root, scene_root, stitch_root):
                path.mkdir(parents=True)
            (calibration_root / "calibration.json").write_text(
                json.dumps({"outer_boundary": {"center_x": 111, "center_y": 83, "radius": 68}}),
                encoding="utf-8",
            )
            (scene_root / "scene_yaw_calibration.json").write_text(
                json.dumps({"focal_ratio": 0.9, "camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}),
                encoding="utf-8",
            )
            (stitch_root / "map_stitch.json").write_text(
                json.dumps({"evidence": [{"name": "mosaic.png"}]}),
                encoding="utf-8",
            )
            cv2.imwrite(
                str(stitch_root / "mosaic.png"),
                np.full((180, 220, 3), 40, np.uint8),
            )
            state = AcquisitionWorkbench(
                root / "sessions", artifact_root, profiles=ProfileCatalog()
            )
            source = FakeLiveFrameSource()
            state.sources.capture_sources = lambda frame, inputs: (source, None)
            try:
                with patch(
                    "acquisition.workbench.TwoRateRealtimeTracker", FakeLiveEngine
                ):
                    state.start_live_tracker(
                        {
                            "game_profile_id": game_id,
                            "minimap_calibration_id": "mini-a",
                            "scene_yaw_calibration_id": "yaw-a",
                            "map_stitch_id": "map-a",
                            "frame_source": {
                                "adapter": "windows_window",
                                "window_title": "Genshin Impact",
                            },
                        }
                    )
                    for _ in range(100):
                        descriptor = state.descriptor()
                        if (descriptor.get("live_tracker") or {}).get("latest"):
                            break
                        time.sleep(0.005)
                    runtime = descriptor["live_tracker"]
                    self.assertEqual(runtime["status"], "running")
                    self.assertEqual(runtime["latest"]["mode"], "TRACK")
                    json.dumps(descriptor)
                    overlay = cv2.imdecode(
                        np.frombuffer(state.live_tracker_overlay_image(), np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    self.assertEqual(overlay.shape, (360, 520, 3))
                    with self.assertRaisesRegex(RuntimeError, "Stop live tracking"):
                        state.start_session({"game_profile_id": game_id})
                    state.stop_live_tracker()
                    for _ in range(100):
                        if state.descriptor()["live_tracker"]["status"] == "stopped":
                            break
                        time.sleep(0.005)
                    self.assertEqual(
                        state.descriptor()["live_tracker"]["status"], "stopped"
                    )
            finally:
                source.stop()
                state.close()


if __name__ == "__main__":
    unittest.main()
