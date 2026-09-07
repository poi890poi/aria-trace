import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from rig_runtime.apps.hik_rig_calibration import parser
from rig_runtime.services.calibration.rig import fixed_plane_distortion as fitting
from rig_runtime.services.calibration.rig.contracts import FrameSample
from rig_runtime.services.calibration.rig.distortion import combined_output_to_raw_maps, undistort_pixel_points
from rig_runtime.services.calibration.rig.geometry import transform_points
from rig_runtime.workflows.hik_rig_calibration import HikCalibrationOptions, HikRigCalibrationSession
from tests.test_rig_distortion import synthetic_views


def observations(distortion=(-0.24, 0.08, 0.001, -0.001, 0), noise=0.03, pose=0, seed=817):
    cameras, screens = synthetic_views(distortion)
    rng = np.random.default_rng(seed)
    return ([cameras[pose].astype(float) + rng.normal(0, noise, cameras[pose].shape) for _ in range(8)],
            [screens[pose].astype(float).copy() for _ in range(8)])


class FixedPlaneDistortionTests(unittest.TestCase):
    def test_fixed_phone_poses_improve_without_any_between_frame_pose_change(self):
        for pose in (0, 2, 4):
            with self.subTest(pose=pose):
                camera, screen = observations(pose=pose)
                result = fitting.fit_fixed_plane_distortion(camera, screen, [1280, 960])
                self.assertTrue(result["accepted"], result.get("reason"))
                self.assertGreater(result["holdout"]["relative_p95_improvement"], 0.5)
                self.assertEqual("normalization_gauge_not_physical_intrinsics", result["camera_matrix_role"])

    def test_distortion_free_noisy_phone_does_not_enable_unneeded_correction(self):
        camera, screen = observations((0, 0, 0, 0, 0))
        result = fitting.fit_fixed_plane_distortion(camera, screen, [1280, 960])
        self.assertFalse(result["accepted"])
        self.assertNotIn("distortion_coefficients", result)

    def test_new_spatial_samples_improve_against_independent_projection_ground_truth(self):
        camera, screen = observations()
        result = fitting.fit_fixed_plane_distortion(camera, screen, [1280, 960])
        self.assertTrue(result["accepted"])
        # This dense query grid was never presented to the estimator or its gate.
        x, y = np.meshgrid(np.linspace(7, 233, 23), np.linspace(9, 391, 31))
        query = np.c_[x.ravel(), y.ravel()]
        objects = np.c_[query, np.zeros(len(query))]
        raw, _ = cv2.projectPoints(objects, np.array([0.02, -0.10, 0.01]), np.array([-120., -180., 850.]),
                                   np.array([[900., 0, 640.], [0, 910., 480.], [0, 0, 1.]]),
                                   np.array([-0.24, 0.08, 0.001, -0.001, 0.]))
        raw = raw.reshape(-1, 2)
        baseline, _ = cv2.findHomography(camera[0], screen[0], 0)
        corrected, _ = cv2.findHomography(undistort_pixel_points(camera[0], result), screen[0], 0)
        original_error = np.linalg.norm(transform_points(raw, baseline) - query, axis=1)
        corrected_error = np.linalg.norm(transform_points(undistort_pixel_points(raw, result), corrected) - query, axis=1)
        self.assertLess(np.percentile(corrected_error, 95), np.percentile(original_error, 95) * 0.2)

    def test_holdout_identities_are_excluded_even_with_reordered_detections(self):
        camera, screen = observations()
        for index in range(len(camera)):
            order = np.random.default_rng(index).permutation(len(camera[index]))
            camera[index], screen[index] = camera[index][order], screen[index][order]
        with patch.object(fitting, "_fit", wraps=fitting._fit) as fit:
            result = fitting.fit_fixed_plane_distortion(camera, screen, [1280, 960])
        fit.assert_called_once()
        training = {tuple(point) for point in fit.call_args.args[1]}
        held_out = {tuple(point) for point in result["holdout_display_points_xy"]}
        self.assertFalse(training & held_out)
        self.assertEqual(result["candidate"]["distortion_coefficients"], result["distortion_coefficients"])

    def test_serialized_automatic_model_drives_runtime_remap_on_unseen_pixels(self):
        camera, screen = observations()
        model = json.loads(json.dumps(fitting.fit_fixed_plane_distortion(camera, screen, [1280, 960])))
        self.assertTrue(model["accepted"])
        homography, _ = cv2.findHomography(undistort_pixel_points(camera[0], model), screen[0], 0)
        map_x, map_y = combined_output_to_raw_maps(homography, [240, 400], model)
        x, y = np.meshgrid(np.arange(7, 233, 17), np.arange(9, 391, 19))
        query = np.c_[x.ravel(), y.ravel()]
        # Independently generated physical projection, never supplied to fitting.
        raw, _ = cv2.projectPoints(np.c_[query, np.zeros(len(query))].astype(float),
                                   np.array([0.02, -0.10, 0.01]), np.array([-120., -180., 850.]),
                                   np.array([[900., 0, 640.], [0, 910., 480.], [0, 0, 1.]]),
                                   np.array([-0.24, 0.08, 0.001, -0.001, 0.]))
        actual = np.c_[map_x[y.ravel(), x.ravel()], map_y[y.ravel(), x.ravel()]]
        self.assertLess(np.percentile(np.linalg.norm(actual - raw.reshape(-1, 2), axis=1), 95), 0.15)

    def test_moving_target_and_sparse_evidence_are_explained(self):
        camera, screen = observations()
        camera[-1] += [8, 0]
        result = fitting.fit_fixed_plane_distortion(camera, screen, [1280, 960])
        self.assertFalse(result["accepted"])
        self.assertIn("moved", result["reason"])
        result = fitting.fit_fixed_plane_distortion([item[:20] for item in camera],
                                                    [item[:20] for item in screen], [1280, 960])
        self.assertFalse(result["accepted"])
        self.assertIn("36", result["reason"])

    def test_later_geometry_must_still_match_the_calibrated_placement(self):
        camera, screen = observations()
        result = fitting.fit_fixed_plane_distortion(camera, screen, [1280, 960])
        self.assertTrue(fitting.fixed_plane_pose_matches(result, camera[-1], screen[-1]))
        self.assertFalse(fitting.fixed_plane_pose_matches(result, camera[-1] + [8, 0], screen[-1]))

    def test_folding_candidate_is_rejected(self):
        camera, screen = observations()
        candidate = {"camera_matrix_3x3": [[1280, 0, 640], [0, 1280, 480], [0, 0, 1]],
                     "distortion_coefficients": [-10, 0, 0, 0, 0]}
        with patch.object(fitting, "_fit", return_value=candidate):
            result = fitting.fit_fixed_plane_distortion(camera, screen, [1280, 960])
        self.assertFalse(result["accepted"])
        self.assertIn("unstable", result["reason"])


class AutomaticCollectionTests(unittest.TestCase):
    def session(self, **options):
        session = HikRigCalibrationSession(
            HikCalibrationOptions("camera", "phone", Path("unused"), headless=True, **options),
            camera=Mock(), phone=Mock(), target=Mock(), progress=Mock())
        session.charuco_layout = Mock()
        session.camera_metadata = {"width_px": 1280, "height_px": 960}
        return session

    def test_default_headless_collection_never_requests_operator_input(self):
        self.assertEqual("auto", parser().parse_args([]).distortion_correction)
        camera, screen = observations()
        session = self.session()
        frames = [FrameSample(np.zeros((20, 20, 3), np.uint8), index + 1) for index in range(8)]
        detected = [{"corner_count": len(c), "camera_points_xy": c, "screen_points_xy": s}
                    for c, s in zip(camera, screen)]
        with patch.object(session, "_wait_painted"), patch.object(session, "_read_camera", side_effect=frames), \
                patch.object(session, "_detect_charuco", side_effect=detected), \
                patch.object(session, "_preview_update", side_effect=AssertionError("manual preview")), \
                patch("builtins.input", side_effect=AssertionError("manual input")):
            session.calibrate_lens_distortion()
        self.assertTrue(session.lens_model["accepted"])
        self.assertFalse(session.lens_model["collection"]["operator_input_required"])

    def test_unusable_frames_stop_after_bounded_collection_with_reason(self):
        session = self.session()
        frames = [FrameSample(np.zeros((20, 20, 3), np.uint8), index + 1) for index in range(24)]
        with patch.object(session, "_wait_painted"), patch.object(session, "_read_camera", side_effect=frames) as read, \
                patch.object(session, "_detect_charuco", side_effect=RuntimeError("no board")):
            session.calibrate_lens_distortion()
        self.assertEqual(24, read.call_count)
        self.assertFalse(session.lens_model["accepted"])
        self.assertIn("four", session.lens_model["reason"])

    def test_explicit_off_still_skips_collection(self):
        session = self.session(distortion_correction="off")
        session.calibrate_lens_distortion()
        session.target.present_charuco.assert_not_called()
        self.assertEqual("disabled", session.lens_model["reason"])

    def test_duplicate_frames_cannot_supply_independent_evidence(self):
        session = self.session()
        camera, screen = observations()
        sample = FrameSample(np.zeros((20, 20, 3), np.uint8), 1)
        detected = {"corner_count": len(camera[0]), "camera_points_xy": camera[0], "screen_points_xy": screen[0]}
        with patch.object(session, "_wait_painted"), patch.object(session, "_read_camera", return_value=sample) as read, \
                patch.object(session, "_detect_charuco", return_value=detected) as detect:
            session.calibrate_lens_distortion()
        self.assertEqual(24, read.call_count)
        self.assertEqual(1, detect.call_count)
        self.assertFalse(session.lens_model["accepted"])

    def test_camera_failure_is_not_misreported_as_optional_fit_rejection(self):
        session = self.session()
        with patch.object(session, "_wait_painted"), \
                patch.object(session, "_read_camera", side_effect=RuntimeError("camera disconnected")):
            with self.assertRaisesRegex(RuntimeError, "camera disconnected"):
                session.calibrate_lens_distortion()

    def test_geometry_discards_corrected_samples_if_phone_moves_after_fitting(self):
        from rig_runtime.adapters.android.phone import PhoneMetrics
        from rig_runtime.workflows.hik_rig_calibration import screen_filling_charuco_layout

        session = self.session(geometry_frames=2)
        camera, screen = observations()
        session.lens_model = fitting.fit_fixed_plane_distortion(camera, screen, [1280, 960])
        session.phone_metrics = PhoneMetrics("phone", "Example", "Phone", "14", [240, 400], 420, 60.0)
        session.charuco_layout = screen_filling_charuco_layout((240, 400))
        # The first sample has more corners, so retaining it would select the
        # stale corrected coordinate space after the second sample disables it.
        moved = camera[-1][:-1] + [8, 0]
        detections = [
            {"camera_points_xy": camera[-1].copy(), "screen_points_xy": screen[-1], "corner_count": len(camera[-1])},
            {"camera_points_xy": moved.copy(), "screen_points_xy": screen[-1][:-1], "corner_count": len(moved)},
        ]
        sample = FrameSample(np.zeros((20, 20, 3), np.uint8), 1)
        with patch.object(session, "_wait_painted"), patch.object(session, "_preview_update"), \
                patch.object(session, "_read_camera", return_value=sample), \
                patch.object(session, "_detect_charuco", side_effect=detections):
            session.calibrate_geometry()
        self.assertFalse(session.lens_model["accepted"])
        self.assertIn("placement changed", session.lens_model["reason"])
        np.testing.assert_array_equal(moved, session.correspondences["camera_points_xy"])
        self.assertNotIn("raw_camera_points_xy", session.correspondences)

    def test_repositioning_and_interactive_recalibration_refresh_automatic_distortion(self):
        session = HikRigCalibrationSession(HikCalibrationOptions("camera", "phone", Path("unused")),
                                            camera=Mock(), phone=Mock(), target=Mock(), progress=Mock())
        events = []
        names = ("open", "wait_for_positioning_confirmation", "calibrate_lens_distortion", "calibrate_geometry",
                 "calibrate_black_level", "calibrate_once_auto_imaging", "calibrate_exposure",
                 "calibrate_white_balance", "verify_final_imaging", "close")
        from contextlib import ExitStack
        with ExitStack() as stack:
            for name in names:
                stack.enter_context(patch.object(session, name, side_effect=lambda name=name: events.append(name) or True))
            stack.enter_context(patch.object(session, "focus_loop", side_effect=["recalibrate", "quit"]))
            session.run()
        self.assertEqual(2, events.count("calibrate_lens_distortion"))
        self.assertLess(events.index("wait_for_positioning_confirmation"), events.index("calibrate_lens_distortion"))
        for index, event in enumerate(events):
            if event == "calibrate_geometry":
                self.assertEqual("calibrate_lens_distortion", events[index - 1])
