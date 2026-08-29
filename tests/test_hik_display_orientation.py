import tempfile
import unittest
from unittest import mock

import numpy as np

from acquisition.rig_calibration.geometry import CharucoLayout
from acquisition.rig_calibration.hik.display import AdbDisplayTarget
from acquisition.rig_calibration.hik.workflow import HikRigCalibrationSession


class OrientationChangingPhone:
    def __init__(self, orientations):
        self.orientations = list(orientations)
        self.commands = []
        self.viewer_activity = None

    def display_orientation_quarter_turns(self):
        if len(self.orientations) > 1:
            return self.orientations.pop(0)
        return self.orientations[0]

    def ensure_display_on(self, timeout_seconds):
        return {"state": "ON", "elapsed_ms": 0.0, "probes": 1}

    def shell(self, *args):
        self.commands.append(tuple(map(str, args)))
        return ""

    def sleeper(self, _seconds):
        return None


class HikDisplayOrientationTests(unittest.TestCase):
    def _target(self, phone):
        target = AdbDisplayTarget(
            phone,
            component="com.example/.Display",
            settle_seconds=0.1,
            presentation_timeout_seconds=1.0,
            minimum_ui_settle_seconds=0.0,
            stable_probe_count=1,
        )
        target._layout = CharucoLayout((100, 200), 5, 10, (0, 0))
        target._temporary = tempfile.TemporaryDirectory()
        target._canonical_orientation_quarter_turns = 0
        return target

    def test_target_is_reissued_for_orientation_selected_by_viewer(self):
        phone = OrientationChangingPhone([0, 1, 1, 1, 1])
        target = self._target(phone)
        canonical = np.zeros((200, 100, 3), np.uint8)
        canonical[10:50, 5:30] = 255
        launched = []

        def launch(image, _revision, _attempt):
            launched.append(image.copy())

        with mock.patch.object(target, "_launch_target", side_effect=launch), mock.patch.object(
            target, "_capture_screenshot", side_effect=lambda _revision: launched[-1].copy()
        ):
            target._show(canonical, "image", "target", "orientation-test")

        self.assertEqual([image.shape[:2] for image in launched], [(200, 100), (100, 200)])
        self.assertTrue(np.array_equal(launched[-1], np.rot90(canonical, k=3)))
        self.assertIn(("input", "tap", "100", "50"), phone.commands)
        viewer = target.telemetry()["viewer"]
        self.assertEqual(viewer["canonical_orientation_quarter_turns"], 0)
        self.assertEqual(viewer["display_orientation_quarter_turns"], 1)
        self.assertEqual(viewer["canonical_to_display_rotation_quarter_turns"], 1)
        self.assertEqual(viewer["canonical_target_size_px"], [100, 200])
        self.assertEqual(viewer["logical_target_size_px"], [200, 100])
        target.stop()

    def test_portrait_target_keeps_existing_raster_and_launches_once(self):
        phone = OrientationChangingPhone([0, 0, 0, 0])
        target = self._target(phone)
        canonical = np.zeros((200, 100, 3), np.uint8)
        canonical[10:50, 5:30] = 255
        launched = []

        def launch(image, _revision, _attempt):
            launched.append(image.copy())

        with mock.patch.object(target, "_launch_target", side_effect=launch), mock.patch.object(
            target, "_capture_screenshot", side_effect=lambda _revision: launched[-1].copy()
        ):
            target._show(canonical, "image", "target", "portrait-test")

        self.assertEqual(len(launched), 1)
        self.assertTrue(np.array_equal(launched[0], canonical))
        self.assertIn(("input", "tap", "50", "100"), phone.commands)
        target.stop()

    def test_workflow_accepts_rotated_logical_canvas_for_canonical_target(self):
        telemetry = {
            "canvas_width": 2400,
            "canvas_height": 1080,
            "canonical_target_size_px": [1080, 2400],
            "logical_target_size_px": [2400, 1080],
        }
        self.assertTrue(
            HikRigCalibrationSession._target_canvas_matches(telemetry, [1080, 2400])
        )
        self.assertFalse(
            HikRigCalibrationSession._target_canvas_matches(telemetry, [2400, 1080])
        )


if __name__ == "__main__":
    unittest.main()
