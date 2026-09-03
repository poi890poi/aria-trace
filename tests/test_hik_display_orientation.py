import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from acquisition.rig_calibration.geometry import CharucoLayout
from acquisition.rig_calibration.hik.display import AdbDisplayTarget
from acquisition.rig_calibration.hik.workflow import HikRigCalibrationSession


class OrientationChangingPhone:
    def __init__(self, orientations):
        self.orientations = list(orientations)
        self.commands = []
        self.runs = []
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

    def run(self, *args):
        values = tuple(map(str, args))
        self.runs.append(values)
        if values and values[0] == "pull":
            cv2.imwrite(str(Path(values[2])), np.zeros((8, 8, 3), np.uint8))
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

    def test_start_preserves_configured_atlas_orientation_for_every_display_rotation(self):
        canonical = np.zeros((200, 100, 3), np.uint8)
        canonical[10:50, 5:30] = 255
        for quarter_turns in range(4):
            with self.subTest(quarter_turns=quarter_turns):
                phone = OrientationChangingPhone([quarter_turns] * 4)
                target = AdbDisplayTarget(
                    phone,
                    component="com.example/.Display",
                    settle_seconds=0.1,
                    presentation_timeout_seconds=1.0,
                    minimum_ui_settle_seconds=0.0,
                    stable_probe_count=1,
                )
                target.configure_canonical_orientation(0)
                launched = []

                def launch(image, _revision, _attempt):
                    launched.append(image.copy())

                with mock.patch(
                    "rig_runtime.adapters.android.hik_display.generate_charuco_target",
                    return_value=canonical,
                ), mock.patch.object(
                    target, "_launch_target", side_effect=launch
                ), mock.patch.object(
                    target,
                    "_capture_screenshot",
                    side_effect=lambda _revision: launched[-1].copy(),
                ):
                    target.start(CharucoLayout((100, 200), 5, 10, (0, 0)))

                expected = np.rot90(canonical, k=-quarter_turns)
                self.assertEqual(len(launched), 1)
                self.assertTrue(np.array_equal(launched[0], expected))
                viewer = target.telemetry()["viewer"]
                self.assertEqual(viewer["canonical_orientation_quarter_turns"], 0)
                self.assertEqual(
                    viewer["display_orientation_quarter_turns"], quarter_turns
                )
                self.assertEqual(
                    viewer["canonical_to_display_rotation_quarter_turns"],
                    quarter_turns,
                )
                target.stop()

    def test_independent_presenters_never_reuse_gallery_or_probe_paths(self):
        gallery_uris = []
        probe_paths = []
        for _ in range(2):
            phone = OrientationChangingPhone([0, 0, 0, 0])
            target = self._target(phone)
            image = np.zeros((200, 100, 3), np.uint8)

            target._launch_target(image, revision=1, orientation_attempt=1)
            target._capture_screenshot(revision=1)

            gallery_uris.append(
                next(
                    value
                    for command in phone.commands
                    for value in command
                    if value.startswith("file:///sdcard/Download/")
                )
            )
            probe_paths.append(
                next(
                    command[-1]
                    for command in phone.commands
                    if command and command[0] == "screencap"
                )
            )
            force_stop_index = next(
                index
                for index, command in enumerate(phone.commands)
                if command == ("am", "force-stop", "com.example")
            )
            launch_index = next(
                index
                for index, command in enumerate(phone.commands)
                if command[:2] == ("am", "start")
            )
            self.assertLess(force_stop_index, launch_index)
            target.stop()

        self.assertNotEqual(gallery_uris[0], gallery_uris[1])
        self.assertNotEqual(probe_paths[0], probe_paths[1])
        self.assertIn("iris_calibration_target_", gallery_uris[0])
        self.assertIn("iris_display_probe_", probe_paths[0])

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

    def test_screenshot_probe_failure_is_diagnostic_by_default(self):
        phone = OrientationChangingPhone([0, 0, 0, 0])
        target = self._target(phone)
        canonical = np.zeros((200, 100, 3), np.uint8)
        with mock.patch.object(target, "_launch_target"), mock.patch.object(
            target, "_capture_screenshot", side_effect=RuntimeError("probe unavailable")
        ):
            target._show(canonical, "image", "target", "non-gating-test")
        viewer = target.telemetry()["viewer"]
        self.assertFalse(viewer["screenshot_presentation_verified"])
        self.assertIn("probe unavailable", viewer["screenshot_verification_warning"])
        self.assertIsNone(target.last_screenshot)
        target.stop()

    def test_probe_failure_still_waits_for_post_change_ui_quiet_period(self):
        phone = OrientationChangingPhone([0, 0, 0, 0])
        waits = []
        phone.sleeper = waits.append
        target = AdbDisplayTarget(
            phone,
            component="com.example/.Display",
            settle_seconds=0.1,
            presentation_timeout_seconds=1.0,
            minimum_ui_settle_seconds=2.0,
            stable_probe_count=1,
        )
        target._layout = CharucoLayout((100, 200), 5, 10, (0, 0))
        target._temporary = tempfile.TemporaryDirectory()
        target._canonical_orientation_quarter_turns = 0
        canonical = np.zeros((200, 100, 3), np.uint8)
        with mock.patch.object(target, "_launch_target"), mock.patch.object(
            target,
            "_capture_screenshot",
            side_effect=RuntimeError("probe unavailable"),
        ):
            target._show(canonical, "image", "target", "quiet-period-test")
        viewer = target.telemetry()["viewer"]
        self.assertEqual(0.1, waits[0])
        self.assertGreater(waits[-1], 1.9)
        self.assertEqual(2.0, viewer["minimum_ui_settle_seconds"])
        self.assertGreater(viewer["additional_ui_settle_seconds"], 1.9)
        target.stop()

    def test_strict_screenshot_probe_remains_available_for_harnesses(self):
        phone = OrientationChangingPhone([0, 0, 0, 0])
        target = self._target(phone)
        target.strict_screenshot_verification = True
        canonical = np.zeros((200, 100, 3), np.uint8)
        with mock.patch.object(target, "_launch_target"), mock.patch.object(
            target, "_capture_screenshot", side_effect=RuntimeError("probe unavailable")
        ), self.assertRaisesRegex(RuntimeError, "probe unavailable"):
            target._show(canonical, "image", "target", "strict-test")
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
