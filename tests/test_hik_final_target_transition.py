import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from rig_runtime.adapters.android.phone import PhoneMetrics
from rig_runtime.services.calibration.rig.geometry import generate_charuco_target, estimate_screen_geometry
from rig_runtime.workflows.hik_rig_calibration import HikCalibrationOptions, HikRigCalibrationSession, screen_filling_charuco_layout
from tests.test_hik_rig_calibration import rig_frame_sample


class HikFinalTargetTransitionTests(unittest.TestCase):
    def make_session(self, directory, geometry_frames=2):
        size = (320, 640)
        layout = screen_filling_charuco_layout(size)
        atlas = generate_charuco_target(layout)
        if atlas.ndim == 2:
            atlas = cv2.cvtColor(atlas, cv2.COLOR_GRAY2BGR)
        target = mock.Mock(last_target=atlas, last_screenshot=None)
        target.present_charuco.return_value = mock.Mock(revision=37)
        target.telemetry.return_value = {"acknowledgements": [{
            "revision": 37, "painted": True, "fullscreen": True,
            "canvas_width": size[0], "canvas_height": size[1],
        }]}
        session = HikRigCalibrationSession(
            HikCalibrationOptions("fake", "phone", Path(directory) / "rig", headless=True, geometry_frames=geometry_frames, operation_timeout_seconds=2),
            camera=mock.Mock(), phone=mock.Mock(), target=target, progress=mock.Mock(),
        )
        session.phone_metrics = PhoneMetrics("phone", "Example", "Phone", "14", list(size), 420, 60.)
        session.charuco_layout = layout
        points = np.asarray([[8, 8], [312, 8], [8, 632], [312, 632]], np.float64)
        session.geometry = estimate_screen_geometry(points, points, size, size)
        session.camera_metadata = {"width_px": 320, "height_px": 640}
        session._preview_update = mock.Mock()
        return session, atlas

    def test_acknowledged_atlas_can_arrive_after_previous_white_patch_backlog(self):
        with tempfile.TemporaryDirectory() as directory:
            session, atlas = self.make_session(directory, geometry_frames=12)
            white = np.full_like(atlas, 230)
            gray = np.full_like(atlas, 120)
            images = [white] * 8 + [gray] * 8 + [atlas] * 12
            session.camera.read.side_effect = [rig_frame_sample(image, index + 1) for index, image in enumerate(images)]
            with mock.patch("rig_runtime.workflows.hik_rig_calibration.time.monotonic", side_effect=itertools.count(0, .05)):
                session.verify_final_imaging()
            self.assertEqual(28, session.camera.read.call_count)
            self.assertEqual(12, session.cv_verification["successful_frames"])
            self.assertEqual(12, session.cv_verification["attempted_frames"])
            np.testing.assert_array_equal(atlas, session.final_verification_sample.image)
            self.assertEqual(16, session.cv_verification["target_acquisition"]["discarded_frames"])
            session.camera.set_manual_imaging.assert_not_called()

    def test_acknowledgement_alone_cannot_pass_a_patch_as_an_atlas(self):
        with tempfile.TemporaryDirectory() as directory:
            session, atlas = self.make_session(directory)
            white = np.full_like(atlas, 230)
            session.camera.read.side_effect = lambda: rig_frame_sample(white)
            with mock.patch("rig_runtime.workflows.hik_rig_calibration.time.monotonic", side_effect=itertools.count(0, .25)):
                with self.assertRaisesRegex(RuntimeError, "phone acknowledged.*37.*no camera frame") as raised:
                    session.verify_final_imaging()
            self.assertIsNone(session.final_verification_sample)
            self.assertNotIn("manual imaging", str(raised.exception))
            self.assertLess(session.camera.read.call_count, 12)
            self.assertEqual("not_observed", session.cv_verification["target_acquisition"]["status"])
            evidence = session._write_failure_evidence(raised.exception)
            report = json.loads((evidence / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual("not_observed", report["final_imaging_verification"]["target_acquisition"]["status"])
            np.testing.assert_array_equal(white, cv2.imread(str(evidence / "raw-hik-frame.png")))

    def test_immediately_visible_atlas_needs_no_fixed_discard_or_extra_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            session, atlas = self.make_session(directory)
            session.camera.read.side_effect = [rig_frame_sample(atlas, index) for index in (1, 2)]
            session.verify_final_imaging()
            self.assertEqual(2, session.camera.read.call_count)
            self.assertEqual(2, session.cv_verification["successful_frames"])

    def test_detection_loss_after_acquisition_still_counts_against_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            session, atlas = self.make_session(directory)
            white = np.full_like(atlas, 230)
            session.camera.read.side_effect = [rig_frame_sample(atlas, 1), rig_frame_sample(white, 2)]
            session.verify_final_imaging()
            self.assertEqual(2, session.camera.read.call_count)
            self.assertEqual(2, session.cv_verification["attempted_frames"])
            self.assertEqual(1, session.cv_verification["successful_frames"])
            self.assertEqual(.5, session.cv_verification["detection_rate"])
            self.assertIsNone(session.cv_verification["target_acquisition"]["last_detection_error"])
            np.testing.assert_array_equal(atlas, session.final_verification_sample.image)

    def test_camera_error_while_waiting_is_not_retried_as_a_target_delay(self):
        with tempfile.TemporaryDirectory() as directory:
            session, atlas = self.make_session(directory)
            failure = OSError("Camera disconnected")
            session.camera.read.side_effect = [rig_frame_sample(np.full_like(atlas, 230)), failure]
            with self.assertRaises(OSError) as raised:
                session.verify_final_imaging()
            self.assertIs(failure, raised.exception)
            self.assertEqual(2, session.camera.read.call_count)

if __name__ == "__main__":
    unittest.main()
