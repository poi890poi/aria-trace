import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from acquisition.rig_calibration.app.device_adapters import CameraConfiguration
from acquisition.rig_calibration.contracts import FrameSample
from acquisition.rig_calibration.hik.algorithms import (
    BlackLevelObservation,
    ExposureObservation,
    camera_visible_screen_region,
    choose_exposure,
    choose_black_level,
    charuco_orientation_evidence,
    compose_hardware_roi_homography,
    detect_focus_pose_frame,
    estimate_focus_target_pose,
    manual_white_balance_ratios,
    refresh_quantized_exposure_us,
    temporal_black_statistics,
    temporal_white_statistics,
)
import rig_runtime.adapters.hik.compat as hikcam
import rig_runtime.adapters.android.display as android_display
from acquisition.rig_calibration.hik.driver import (
    HikMvsCameraAdapter,
    MvsPythonBackend,
    RectifiedHikCamera,
    create_camera_adapter,
)
from acquisition.rig_calibration.hik.display import AdbDisplayTarget
from rig_runtime.adapters.android.display import (
    LocalPhoneTargetServer,
    NativeImmersivePhoneTarget,
)
from acquisition.rig_calibration.hik.patterns import (
    camera_white_mask,
    focus_edge_regions,
    focus_frame_rect,
    focus_pattern,
    white_patch,
)
from acquisition.rig_calibration.hik.phone import (
    AdbPhoneSession,
    _subprocess_runner,
    connected_adb_devices,
    resolve_adb_executable,
)
from acquisition.rig_calibration.hik.phone import PhoneMetrics
from acquisition.rig_calibration.hik.stream import PhoneDisplayPowerSession
from acquisition.rig_calibration.hik.workflow import (
    HikCalibrationOptions,
    HikRigCalibrationSession,
    android_display_scale_diagnostic,
    cross_source_alignment_evidence,
    is_mtk_phone_platform,
    resolve_panel_scale_mode,
    screen_filling_charuco_layout,
)
from acquisition.rig_calibration.geometry import (
    CharucoLayout,
    charuco_board_metric_to_panel_pixels,
    estimate_screen_geometry,
)
from rig_runtime.evidence.rig_alignment import cross_source_alignment_warning
from rig_runtime.services.calibration.rig.hik.panel_axis import (
    panel_axis_correction_matrix,
)


def rig_frame_sample(image, time_ns=1, roi_xywh=None):
    height, width = image.shape[:2]
    roi = list(roi_xywh or [0, 0, width, height])
    parent_size = [max(width, roi[0] + roi[2]), max(height, roi[1] + roi[3])]
    return FrameSample(
        image,
        time_ns,
        receive_time_ns=time_ns,
        source_id="fake",
        metadata={
            "image_space": {
                "space_id": "hik_camera_acquisition_pixels",
                "stored_size_px": [width, height],
                "parent_space_id": "hik_full_sensor_camera_pixels",
                "parent_size_px": parent_size,
                "roi_in_parent_xywh": roi,
                "local_to_parent_3x3": [
                    [1.0, 0.0, float(roi[0])],
                    [0.0, 1.0, float(roi[1])],
                    [0.0, 0.0, 1.0],
                ],
                "orientation": "hik_camera_native",
                "color_order": "BGR",
            }
        },
    )


class HikAlgorithmTests(unittest.TestCase):
    def test_headless_run_includes_standard_panel_axis_measurement(self):
        options = HikCalibrationOptions(
            "fake", "phone", Path("unused"), headless=True, save_without_prompt=True
        )
        session = HikRigCalibrationSession(
            options, camera=mock.Mock(), phone=mock.Mock(), target=mock.Mock()
        )
        stage_names = [
            "open",
            "calibrate_lens_distortion",
            "wait_for_positioning_confirmation",
            "calibrate_geometry",
            "calibrate_black_level",
            "calibrate_once_auto_imaging",
            "calibrate_exposure",
            "calibrate_white_balance",
            "verify_final_imaging",
            "calibrate_panel_axis",
            "save",
            "close",
        ]
        patches = [mock.patch.object(session, name) for name in stage_names]
        mocked = {
            name: patch.start() for name, patch in zip(stage_names, patches)
        }
        try:
            mocked["wait_for_positioning_confirmation"].return_value = True
            mocked["save"].return_value = Path("saved")
            result = session.run()
        finally:
            for patch in reversed(patches):
                patch.stop()
        self.assertEqual(Path("saved"), result)
        mocked["calibrate_panel_axis"].assert_called_once_with()
        mocked["close"].assert_called_once_with()

    def test_panel_axis_presentation_failure_is_non_gating(self):
        target = mock.Mock()
        target.present_image.side_effect = RuntimeError("presenter unavailable")
        session = HikRigCalibrationSession(
            HikCalibrationOptions("fake", "phone", Path("unused")),
            camera=mock.Mock(),
            phone=mock.Mock(),
            target=target,
            progress=lambda _message: None,
        )
        session.visible_region = {"safe_xywh": [10, 10, 100, 100]}
        session.phone_metrics = PhoneMetrics(
            "phone", "Example", "Phone", "14", [120, 120], 420, 60.0
        )

        result = session.calibrate_panel_axis()

        self.assertEqual("unavailable", result["status"])
        self.assertFalse(result["applied"])
        self.assertTrue(result["non_gating"])
        self.assertIn("target_presentation_failed", result["reason"])

    def test_charuco_board_metric_uses_anisotropic_native_surface_scale(self):
        layout = CharucoLayout((1000, 2000), 10, 20, (0, 0))
        points, metadata = charuco_board_metric_to_panel_pixels(
            [[1.0, 1.0], [9.0, 19.0]], layout, [1000, 2100]
        )
        self.assertTrue(
            np.allclose(points, [[100.0, 105.0], [900.0, 1995.0]])
        )
        self.assertEqual(metadata["panel_px_per_square_xy"], [100.0, 105.0])
        self.assertFalse(metadata["adb_physical_dpi_used"])

    def test_session_charuco_detection_folds_native_surface_scale_into_destinations(self):
        session = HikRigCalibrationSession(
            HikCalibrationOptions("fake", "phone", Path("unused")),
            camera=mock.Mock(), phone=mock.Mock(), target=mock.Mock(),
        )
        session.charuco_layout = CharucoLayout((1000, 2000), 10, 20, (0, 0))
        session.effective_panel_size_px = [1000, 2100]
        session.panel_scale_measurement = {"resolved_mode": "hik_charuco"}
        detected = {
            "board_points_square_xy": np.asarray([[1.0, 1.0], [9.0, 19.0]]),
            "screen_points_xy": np.asarray([[100.0, 100.0], [900.0, 1900.0]]),
            "corner_count": 2,
        }
        with mock.patch(
            "rig_runtime.workflows.hik_rig_calibration.detect_charuco_correspondences",
            return_value=detected,
        ):
            result = session._detect_charuco(np.zeros((8, 8, 3), np.uint8))
        self.assertTrue(
            np.allclose(
                result["screen_points_xy"],
                [[100.0, 105.0], [900.0, 1995.0]],
            )
        )
        self.assertEqual(result["target_raster_to_panel_scale_xy"], [1.0, 1.05])

    def test_display_scale_anomaly_is_diagnostic_and_mtk_auto_uses_charuco(self):
        metrics = PhoneMetrics(
            "phone",
            "Example",
            "MTK phone",
            "14",
            [1080, 2400],
            420,
            120.0,
            active_app_size_px=[1080, 2300],
            physical_dpi_xy=[409.0, 430.0],
            hardware_platform="mt6893",
        )
        diagnostic = android_display_scale_diagnostic(metrics)
        self.assertFalse(diagnostic["active_app_matches_screen"])
        self.assertFalse(diagnostic["pitch_isotropic_within_tolerance"])
        self.assertTrue(diagnostic["non_gating"])
        self.assertTrue(is_mtk_phone_platform(metrics))
        self.assertEqual(resolve_panel_scale_mode("auto", metrics), "hik_charuco")
        self.assertEqual(resolve_panel_scale_mode("adb", metrics), "adb")

    def test_non_mtk_auto_keeps_compatibility_raster(self):
        metrics = PhoneMetrics(
            "phone", "Example", "Phone", "14", [1080, 2400], 420, 120.0,
            hardware_platform="qcom",
        )
        self.assertEqual(resolve_panel_scale_mode("auto", metrics), "adb")

    def test_native_exact_pixel_presenter_is_default_and_fallbacks_are_explicit(self):
        owned = HikRigCalibrationSession(
            HikCalibrationOptions("fake", "phone", Path("unused")),
            camera=mock.Mock(),
            phone=mock.Mock(),
        )
        self.assertIsInstance(owned.target, NativeImmersivePhoneTarget)
        self.assertEqual(owned.target.bind_host, "127.0.0.1")
        self.assertEqual(owned.target.port, 0)

        browser = HikRigCalibrationSession(
            HikCalibrationOptions(
                "fake", "phone", Path("unused"), target_presenter="owned_http"
            ),
            camera=mock.Mock(),
            phone=mock.Mock(),
        )
        self.assertIsInstance(browser.target, LocalPhoneTargetServer)
        self.assertNotIsInstance(browser.target, NativeImmersivePhoneTarget)

        legacy = HikRigCalibrationSession(
            HikCalibrationOptions(
                "fake",
                "phone",
                Path("unused"),
                target_presenter="legacy_gallery",
            ),
            camera=mock.Mock(),
            phone=mock.Mock(),
        )
        self.assertIsInstance(legacy.target, AdbDisplayTarget)

    def test_native_presenter_resolves_explicit_apk(self):
        with tempfile.TemporaryDirectory() as directory:
            apk = Path(directory) / "target.apk"
            apk.write_bytes(b"apk")
            target = NativeImmersivePhoneTarget(apk_path=apk)
            self.assertEqual(target.resolved_apk_path(), apk.resolve())

    def test_native_presenter_finds_release_apk_from_python_module_ancestor(self):
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory) / "IRIS-Windows-x64"
            module = (
                release
                / "python"
                / "aria_trace"
                / "adapters"
                / "android"
                / "display.py"
            )
            apk = release / "phone-target" / "iris-phone-target.apk"
            apk.parent.mkdir(parents=True)
            apk.write_bytes(b"apk")
            unrelated_executable = Path(directory) / "Python312" / "python.exe"
            unrelated_cwd = Path(directory) / "third-party-app"
            unrelated_cwd.mkdir()
            with mock.patch.object(android_display, "__file__", str(module)), mock.patch.object(
                sys, "executable", str(unrelated_executable)
            ), mock.patch("pathlib.Path.cwd", return_value=unrelated_cwd):
                self.assertEqual(
                    NativeImmersivePhoneTarget().resolved_apk_path(),
                    apk.resolve(),
                )

    def test_gui_positioning_waits_for_explicit_space_signal(self):
        options = HikCalibrationOptions(
            "fake", "phone", Path("unused"), headless=False
        )
        sample = rig_frame_sample(np.zeros((48, 64, 3), np.uint8))
        camera = mock.Mock()
        camera.read.return_value = sample
        target = mock.Mock()
        target.present_charuco.return_value = mock.Mock(revision=1)
        session = HikRigCalibrationSession(
            options,
            camera=camera,
            target=target,
            progress=lambda _message: None,
        )
        session.charuco_layout = screen_filling_charuco_layout((64, 48))
        session._wait_painted = mock.Mock()
        observed_prompts = []

        def preview_update(*_args, **_kwargs):
            observed_prompts.append(session._preview_settings.get("operator_prompt"))
            return None if len(observed_prompts) == 1 else 32

        with mock.patch(
            "rig_runtime.workflows.hik_rig_calibration.detect_charuco_correspondences",
            return_value={"corner_count": 12},
        ), mock.patch.object(
            session, "_preview_update", side_effect=preview_update
        ):
            self.assertTrue(session.wait_for_positioning_confirmation())
        self.assertEqual(2, camera.read.call_count)
        self.assertEqual(2, len(observed_prompts))
        self.assertIn("POSITION RIG", observed_prompts[0])
        self.assertIn("ENTER / SPACE = START", observed_prompts[0])
        self.assertNotIn("operator_prompt", session._preview_settings)

    def test_focus_save_gate_requires_consecutive_displaced_frames(self):
        options = HikCalibrationOptions(
            "fake",
            "phone",
            Path("unused"),
            save_max_displacement_px=12.0,
            save_movement_consecutive_frames=3,
        )
        session = HikRigCalibrationSession(
            options, camera=mock.Mock(), target=mock.Mock()
        )
        self.assertFalse(session._update_focus_save_gate(13.0, 12.0))
        self.assertFalse(session._update_focus_save_gate(13.0, 12.0))
        self.assertFalse(session._update_focus_save_gate(4.0, 12.0))
        self.assertFalse(session._update_focus_save_gate(13.0, 12.0))
        self.assertFalse(session._update_focus_save_gate(13.0, 12.0))
        self.assertTrue(session._update_focus_save_gate(13.0, 12.0))

    def test_final_verification_sample_is_not_replaced_by_later_benchmark_frame(self):
        options = HikCalibrationOptions(
            "fake",
            "phone",
            Path("unused"),
            headless=True,
            geometry_frames=2,
        )
        white = rig_frame_sample(np.full((48, 64, 3), 240, np.uint8), 1)
        gray = rig_frame_sample(np.full((48, 64, 3), 120, np.uint8), 2)
        black = rig_frame_sample(np.zeros((48, 64, 3), np.uint8), 3)
        camera = mock.Mock()
        camera.read.side_effect = [white, gray, black]
        target = mock.Mock()
        target.present_charuco.return_value = mock.Mock(revision=1)
        session = HikRigCalibrationSession(
            options,
            camera=camera,
            target=target,
            progress=lambda _message: None,
        )
        session.phone_metrics = PhoneMetrics(
            "phone", "Example", "Phone", "14", [64, 48], 420, 60.0
        )
        session.charuco_layout = screen_filling_charuco_layout((64, 48))
        points = np.asarray([[8, 8], [56, 8], [8, 40], [56, 40]], np.float64)
        session.geometry = estimate_screen_geometry(points, points, (64, 48), (64, 48))
        session.camera_metadata = {"width_px": 64, "height_px": 48}
        session._wait_painted = mock.Mock()
        session._preview_update = mock.Mock()
        with mock.patch(
            "rig_runtime.workflows.hik_rig_calibration.detect_charuco_correspondences",
            return_value={
                "camera_points_xy": points,
                "screen_points_xy": points,
                "corner_count": len(points),
            },
        ):
            session.verify_final_imaging()
        session._read_camera()
        self.assertEqual(0.0, float(np.mean(session.last_sample.image)))
        self.assertEqual(
            240.0, float(np.mean(session.final_verification_sample.image))
        )

    def test_headless_auto_uses_reduced_final_benchmark_without_display_cycles(self):
        options = HikCalibrationOptions(
            "fake",
            "phone",
            Path("unused"),
            headless=True,
            final_benchmark_mode="auto",
        )
        camera = mock.Mock()
        camera.align_roi.side_effect = lambda roi: list(map(int, roi))
        camera.set_roi.side_effect = lambda roi: list(map(int, roi))
        target = mock.Mock()
        session = HikRigCalibrationSession(
            options,
            camera=camera,
            target=target,
            progress=lambda _message: None,
        )
        session.geometry = mock.Mock(inverse_matrix_3x3=np.eye(3))
        session.visible_region = {"xywh": [0, 0, 8, 8]}
        session.white_mask = np.full((8, 8), 255, np.uint8)
        session.camera_metadata = {"width_px": 8, "height_px": 8}
        session.camera_controls = {}
        session.lens_model = {}
        session._read_camera = mock.Mock(
            side_effect=[
                rig_frame_sample(np.zeros((8, 8, 3), np.uint8), index + 1)
                for index in range(6)
            ]
        )
        session.benchmark_final_stream()
        self.assertEqual(6, session._read_camera.call_count)
        self.assertEqual("reduced", session.transport_benchmark["benchmark_mode"])
        self.assertEqual(6, session.transport_benchmark["sample_count"])
        self.assertEqual(
            "skipped_in_reduced_headless_mode",
            session.latency_benchmark["status"],
        )
        target.present_signal.assert_not_called()

    def test_latency_rejects_cross_clock_negative_or_unbounded_values(self):
        elapsed = HikRigCalibrationSession._same_clock_elapsed_ms
        self.assertEqual(elapsed(1_000_000_000, 1_075_000_000, 1000.0), 75.0)
        self.assertIsNone(elapsed(1_000_000_000, 900_000_000, 1000.0))
        self.assertIsNone(elapsed(1_000_000_000, 3_000_000_000, 1000.0))

    def test_windows_publish_falls_back_to_copy_after_locked_directory_rename(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / ".calibration.tmp"
            output = Path(directory) / "calibration"
            temporary.mkdir()
            (temporary / "hik_camera_calibration.json").write_text(
                "{}", encoding="utf-8"
            )
            with mock.patch(
                "rig_runtime.workflows.hik_rig_calibration.os.replace",
                side_effect=PermissionError("directory locked"),
            ), mock.patch(
                "rig_runtime.workflows.hik_rig_calibration.time.sleep"
            ):
                method = HikRigCalibrationSession._publish_calibration_directory(
                    temporary, output
                )
            self.assertEqual(method, "copy_fallback_after_windows_lock")
            self.assertTrue((output / "hik_camera_calibration.json").is_file())
            self.assertFalse(temporary.exists())

    def test_failed_publish_retains_completed_temporary_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory) / ".calibration.tmp"
            output = Path(directory) / "calibration"
            temporary.mkdir()
            (temporary / "hik_camera_calibration.json").write_text(
                "{}", encoding="utf-8"
            )
            with mock.patch(
                "rig_runtime.workflows.hik_rig_calibration.os.replace",
                side_effect=PermissionError("directory locked"),
            ), mock.patch(
                "rig_runtime.workflows.hik_rig_calibration.shutil.copytree",
                side_effect=PermissionError("file locked"),
            ), mock.patch(
                "rig_runtime.workflows.hik_rig_calibration.time.sleep"
            ):
                with self.assertRaisesRegex(PermissionError, "temporary bundle was retained"):
                    HikRigCalibrationSession._publish_calibration_directory(
                        temporary, output
                    )
            self.assertTrue((temporary / "hik_camera_calibration.json").is_file())
            self.assertFalse(output.exists())

    def test_cross_source_check_scores_already_aligned_images_high(self):
        image = np.zeros((120, 160, 3), np.uint8)
        image[10:55, 20:75] = 255
        image[65:105, 90:145] = 180
        cv2.line(image, (5, 115), (155, 5), (90, 90, 90), 3)
        mask = np.full(image.shape[:2], 255, np.uint8)
        metrics, evidence = cross_source_alignment_evidence(image, image.copy(), mask)
        self.assertGreater(metrics["confidence"], 0.99)
        self.assertGreater(metrics["edge_overlap"], 0.99)
        self.assertIn("edge_overlay_adb_red_hik_cyan.png", evidence)

    def test_cross_source_check_does_not_fit_away_a_bad_shift(self):
        image = np.zeros((120, 160, 3), np.uint8)
        image[10:55, 20:75] = 255
        image[65:105, 90:145] = 180
        shifted = np.roll(image, 24, axis=1)
        mask = np.full(image.shape[:2], 255, np.uint8)
        aligned, _ = cross_source_alignment_evidence(image, image, mask)
        displaced, _ = cross_source_alignment_evidence(image, shifted, mask)
        self.assertLess(displaced["confidence"], aligned["confidence"] - 0.25)
        residual = displaced["residual_translation"]
        self.assertEqual("measured", residual["status"])
        self.assertAlmostEqual(24.0, residual["hik_offset_xy_px_from_adb"][0], delta=1.0)
        self.assertAlmostEqual(0.0, residual["hik_offset_xy_px_from_adb"][1], delta=1.0)
        self.assertIn("displaced", cross_source_alignment_warning(displaced))

    def test_cross_source_check_rectifies_saved_roi_and_writes_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            options = HikCalibrationOptions(
                "fake", "phone", output / "unused", headless=True, settle_frames=1
            )
            screenshot = np.zeros((80, 100, 3), np.uint8)
            screenshot[20:35, 10:30] = 255
            screenshot[35:50, 30:50] = 160
            camera_crop = screenshot[20:50, 10:50].copy()
            adapter = FakeRectifiedAdapter(camera_crop)
            target = mock.Mock()
            target.last_screenshot = screenshot
            target.present_charuco.return_value = mock.Mock(revision=1)
            session = HikRigCalibrationSession(
                options, camera=adapter, target=target, progress=lambda _message: None
            )
            session._opened = True
            session.hardware_roi = [10, 20, 40, 30]
            session._wait_painted = lambda *_args, **_kwargs: None
            yy, xx = np.mgrid[20:50, 10:50]
            maps = (xx.astype(np.float32), yy.astype(np.float32))
            mask = np.full((30, 40), 255, np.uint8)
            result = session._save_cross_source_check(
                output, maps, mask, {"xywh": [10, 20, 40, 30]}
            )
            self.assertEqual("measured", result["status"])
            self.assertGreater(result["confidence"], 0.99)
            self.assertTrue(
                (output / "cross_source_check" / "edge_overlay_adb_red_hik_cyan.png")
                .is_file()
            )
            self.assertTrue(
                (
                    output
                    / "cross_source_check"
                    / "full_camera_and_projected_phone_review.png"
                ).is_file()
            )
            self.assertEqual(9, len(result["media"]))
            full_adb = next(
                row
                for row in result["media"]
                if row["file"] == "adb_full_screenshot.png"
            )
            self.assertEqual(
                "android_logical_display_pixels", full_adb["space"]["id"]
            )

    def test_complete_focus_view_preserves_entire_frame_inside_pane(self):
        image = np.zeros((10, 20, 3), np.uint8)
        image[0, 0] = (1, 2, 255)
        image[-1, -1] = (255, 2, 1)
        fitted = HikRigCalibrationSession._fit_complete_view(image, 100, 100)
        self.assertEqual(fitted.shape, (100, 100, 3))
        self.assertEqual(tuple(fitted[25, 0]), (1, 2, 255))
        self.assertEqual(tuple(fitted[74, 99]), (255, 2, 1))

    def test_data_matrix_early_rejection_waits_until_target_is_impossible(self):
        cannot_qualify = HikRigCalibrationSession._data_matrix_cannot_qualify
        self.assertFalse(cannot_qualify(38, 40, 40, 0.95))
        self.assertFalse(cannot_qualify(0, 2, 40, 0.95))
        self.assertTrue(cannot_qualify(0, 3, 40, 0.95))

    def test_data_matrix_requires_at_least_twenty_patterns(self):
        with self.assertRaisesRegex(ValueError, "at least 20"):
            HikCalibrationOptions("fake", "phone", Path("unused"), data_matrix_trials_per_size=19)
        self.assertEqual(
            HikCalibrationOptions("fake", "phone", Path("unused")).data_matrix_trials_per_size,
            40,
        )

    def test_data_matrix_capture_drains_target_change_backlog(self):
        session = HikRigCalibrationSession(
            HikCalibrationOptions("fake", "phone", Path("unused")),
            camera=mock.Mock(),
            progress=lambda _message: None,
        )
        session.camera.read.side_effect = [
            rig_frame_sample(np.full((2, 2, 3), index, np.uint8), index)
            for index in range(8)
        ]
        expected = np.full((2, 2, 3), 99, np.uint8)
        session._capture_settled = lambda: expected
        session._preview_update = lambda *_args, **_kwargs: None
        self.assertIs(session._capture_data_matrix_frame(), expected)
        self.assertEqual(session.camera.read.call_count, 8)

    def test_data_matrix_crop_tracks_rotation_and_keeps_quiet_margin(self):
        unrotated = HikRigCalibrationSession._data_matrix_screen_crop(
            [40, 45, 20, 10], [10, 10, 80, 80], 0.0, 2, [100, 100]
        )
        rotated = HikRigCalibrationSession._data_matrix_screen_crop(
            [40, 45, 20, 10], [10, 10, 80, 80], 90.0, 2, [100, 100]
        )
        self.assertEqual(unrotated, [36, 41, 28, 18])
        self.assertEqual(rotated, [40, 36, 19, 28])

    def test_data_matrix_batch_packs_multiple_exact_payloads_on_one_screen(self):
        session = HikRigCalibrationSession(
            HikCalibrationOptions("fake", "phone", Path("unused")),
            progress=lambda _message: None,
        )

        def render(screen_size, cell, payload, _module, **kwargs):
            screen_width, screen_height = screen_size
            x, y, width, height = cell
            image = np.full((screen_height, screen_width, 3), 127, np.uint8)
            image[y : y + height, x : x + width] = 255
            symbol = [x + (width - 8) // 2, y + (height - 8) // 2, 8, 8]
            sx, sy, sw, sh = symbol
            image[sy : sy + sh, sx : sx + sw] = 0
            return mock.Mock(
                image=image,
                payload=payload,
                symbol_rect_screen_xywh=symbol,
                trial_id=kwargs["trial_id"],
            )

        specifications = [
            {
                "trial_index": index,
                "angle_deg": (0.0, 15.0, -15.0, 30.0)[index % 4],
                "color_bgr": (1.0, 1.0, 1.0),
                "intensity": 1.0,
            }
            for index in range(8)
        ]
        with mock.patch(
            "rig_runtime.workflows.hik_rig_calibration.render_data_matrix_target",
            side_effect=render,
        ):
            batch = session._compose_data_matrix_batch(
                [100, 180], [10, 10, 80, 160], 2, specifications
            )
        self.assertIsNotNone(batch)
        self.assertEqual(len(batch["items"]), 8)
        self.assertEqual(len({item["payload"] for item in batch["items"]}), 8)

    def test_focus_metric_formatter_accepts_partial_esfr_results(self):
        self.assertEqual(HikRigCalibrationSession._format_metric(None), "n/a")
        self.assertEqual(HikRigCalibrationSession._format_metric(float("nan")), "n/a")
        self.assertEqual(HikRigCalibrationSession._format_metric(12.345, 2), "12.35")

    def test_camera_panel_wraps_long_hints_to_measured_width(self):
        lines = HikRigCalibrationSession._wrap_panel_lines(
            [
                "Physical MTF50 and MTF10 use Android-reported panel pitch and must remain visible",
                "android_active_display_mode_xdpi_ydpi",
            ],
            180,
            0.55,
        )
        self.assertGreater(len(lines), 1)
        self.assertTrue(
            all(
                cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
                <= 180
                for line in lines
            )
        )

    def test_camera_panel_uses_same_static_font_scale_at_different_heights(self):
        scales = []

        def record(*args, **_kwargs):
            scales.append(float(args[4]))

        with mock.patch.object(cv2, "putText", side_effect=record):
            HikRigCalibrationSession._draw_text_panel(
                np.zeros((180, 400, 3), np.uint8), 0, 400, ["one", "two", "three"]
            )
            HikRigCalibrationSession._draw_text_panel(
                np.zeros((800, 400, 3), np.uint8), 0, 400, ["one", "two", "three"]
            )
        self.assertTrue(scales)
        self.assertEqual({0.50}, set(scales))

    def test_operator_prompt_is_high_contrast_and_names_controls(self):
        canvas = np.full((360, 640, 3), 24, np.uint8)
        drawn_text = []
        original_put_text = cv2.putText

        def record(image, text, *args, **kwargs):
            drawn_text.append(str(text))
            return original_put_text(image, text, *args, **kwargs)

        with mock.patch.object(cv2, "putText", side_effect=record):
            height = HikRigCalibrationSession._draw_operator_prompt(
                canvas,
                340,
                300,
                (
                    "POSITION RIG",
                    "Click this preview window, then:",
                    "ENTER / SPACE = START",
                    "Q / ESC = CANCEL",
                ),
            )
        self.assertGreater(height, 0)
        self.assertIn("POSITION RIG", drawn_text)
        self.assertIn("ENTER / SPACE = START", drawn_text)
        self.assertIn("Q / ESC = CANCEL", drawn_text)
        self.assertFalse(np.all(canvas[:height, 340:] == 24))

    def test_plugin_factory_is_lazy_and_does_not_require_vendor_sdk(self):
        adapter = create_camera_adapter()
        self.assertEqual(adapter.devices(probe=False), ())
        self.assertIsNone(adapter._backend)

    def test_mvs_runtime_can_be_found_without_inherited_path(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = Path(directory)
            with mock.patch.dict("os.environ", {"MVS_RUNTIME_PATH": str(runtime), "PATH": ""}):
                directories = MvsPythonBackend._runtime_directories(None)
            self.assertIn(runtime, directories)

    def test_mvs_python_wrapper_accepts_discovered_install_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            wrapper = Path(directory)
            (wrapper / "MvCameraControl_class.py").write_text("", encoding="utf-8")
            self.assertEqual(
                MvsPythonBackend._python_wrapper_directories(str(wrapper))[0],
                wrapper.resolve(),
            )

    def test_charuco_probes_app_up_relative_to_camera_up(self):
        identity = charuco_orientation_evidence(np.eye(3), [100, 100])
        self.assertAlmostEqual(identity["camera_up_to_app_up_clockwise_degrees"], 0.0)
        clockwise = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float64)
        rotated = charuco_orientation_evidence(clockwise, [100, 100])
        self.assertAlmostEqual(rotated["camera_up_to_app_up_clockwise_degrees"], 90.0)

    def test_android_display_screenshot_probes_viewer_rotation(self):
        target = np.zeros((80, 120, 3), np.uint8)
        target[5:25, 10:45] = (255, 255, 255)
        target[45:70, 80:110] = (128, 128, 128)
        screenshot = cv2.rotate(target, cv2.ROTATE_90_CLOCKWISE)
        match = AdbDisplayTarget._rotation_match(target, screenshot)
        self.assertEqual(match["viewer_rotation_quarter_turns"], 1)
        self.assertGreater(match["correlation"], 0.999)
        self.assertEqual(match["matching_pixel_fraction"], 1.0)

    def test_android_display_match_exposes_transient_system_bar_pixels(self):
        target = np.full((200, 100, 3), 255, np.uint8)
        target[20:180, 10:90] = 0
        screenshot = target.copy()
        screenshot[:12] = 0
        match = AdbDisplayTarget._rotation_match(target, screenshot)
        self.assertLess(match["matching_pixel_fraction"], 0.999)

    def test_stable_correlated_target_is_not_rejected_by_fixed_system_pixels(self):
        target = AdbDisplayTarget(
            mock.Mock(),
            minimum_screenshot_correlation=0.98,
            minimum_matching_pixel_fraction=0.995,
            minimum_stable_frame_fraction=0.9995,
            minimum_ui_settle_seconds=1.0,
        )
        evidence = {
            "correlation": 0.9844,
            "matching_pixel_fraction": 0.926878,
        }
        self.assertTrue(target._presentation_is_stable(evidence, 1.0, 1.0))
        self.assertFalse(target._presentation_is_stable(evidence, 0.99, 1.0))
        evidence["correlation"] = 0.97
        self.assertFalse(target._presentation_is_stable(evidence, 1.0, 1.0))

    def test_phone_charuco_layout_fills_portrait_and_landscape(self):
        portrait = screen_filling_charuco_layout((1080, 2400))
        landscape = screen_filling_charuco_layout((2400, 1080))
        self.assertEqual((portrait.squares_x, portrait.squares_y), (9, 20))
        self.assertEqual(portrait.board_size_px, (1080, 2400))
        self.assertEqual((landscape.squares_x, landscape.squares_y), (20, 9))
        self.assertEqual(landscape.board_size_px, (2400, 1080))

    def test_rig_open_locks_rotation_zero_before_assigning_charuco_space(self):
        initial = PhoneMetrics(
            "phone", "Example", "Phone", "14", [2400, 1080], 420, 60.0,
            orientation_quarter_turns=1,
            natural_screen_size_px=[1080, 2400],
        )
        canonical = PhoneMetrics(
            "phone", "Example", "Phone", "14", [1080, 2400], 420, 60.0,
            orientation_quarter_turns=0,
            natural_screen_size_px=[1080, 2400],
        )
        phone = mock.Mock()
        phone.metrics.side_effect = [initial, canonical]
        phone.display_brightness_state.return_value = {
            "brightness_value": 255,
            "declared_maximum": 255,
        }
        target = mock.Mock()
        target.telemetry.return_value = {
            "viewer": {
                "canvas_width": 1080,
                "canvas_height": 2400,
                "canonical_target_size_px": [1080, 2400],
                "logical_target_size_px": [1080, 2400],
                "fullscreen": True,
            }
        }
        camera = mock.Mock()
        camera.open.return_value = {
            "width_px": 1440,
            "height_px": 1080,
            "fps": 30.0,
        }
        camera.reset_full_sensor_roi.return_value = [0, 0, 1440, 1080]
        camera.controls.return_value = {}
        session = HikRigCalibrationSession(
            HikCalibrationOptions("camera", "phone", Path("unused")),
            camera=camera,
            phone=phone,
            target=target,
            progress=lambda _message: None,
        )
        session._wait_auto_imaging = lambda: {"status": "settled"}

        session.open()

        target.configure_canonical_orientation.assert_called_once_with(0)
        phone.wake_and_hold_display.assert_called_once_with(0)
        self.assertEqual(session.charuco_layout.screen_size_px, (1080, 2400))
        self.assertEqual(session.phone_metrics.orientation_quarter_turns, 0)

    def test_native_canonical_gate_waits_for_paint_and_stable_adb_rotation(self):
        options = HikCalibrationOptions("camera", "phone", Path("unused"))
        phone = mock.Mock()
        phone.display_orientation_quarter_turns.side_effect = [1, 0, 0]
        canonical = PhoneMetrics(
            "phone",
            "Example",
            "Phone",
            "14",
            [1080, 2400],
            420,
            60.0,
            orientation_quarter_turns=0,
            natural_screen_size_px=[1080, 2400],
        )
        phone.metrics.return_value = canonical
        target = NativeImmersivePhoneTarget()
        target.telemetry = mock.Mock(
            return_value={
                "browser": {
                    "native_surface": True,
                    "canonical_orientation_ready": True,
                    "display_rotation": 0,
                    "canvas_width": 1080,
                    "canvas_height": 2400,
                },
                "acknowledgements": [
                    {
                        "painted": True,
                        "native_surface": True,
                        "target_contract_version": 2,
                        "canonical_orientation_ready": True,
                        "display_rotation": 0,
                        "canvas_width": 1080,
                        "canvas_height": 2400,
                    }
                ],
            }
        )
        session = HikRigCalibrationSession(
            options,
            camera=mock.Mock(),
            phone=phone,
            target=target,
            progress=lambda _message: None,
        )

        with mock.patch(
            "rig_runtime.workflows.hik_rig_calibration.time.sleep"
        ):
            observed = session._wait_native_canonical_surface(timeout_seconds=1.0)

        self.assertIs(observed, canonical)
        self.assertEqual(phone.display_orientation_quarter_turns.call_count, 3)
        phone.metrics.assert_called_once_with(None)

    def test_native_canonical_gate_rejects_unpainted_rotation_zero_surface(self):
        options = HikCalibrationOptions("camera", "phone", Path("unused"))
        phone = mock.Mock()
        phone.display_orientation_quarter_turns.return_value = 0
        target = NativeImmersivePhoneTarget()
        target.telemetry = mock.Mock(
            return_value={
                "browser": {
                    "native_surface": True,
                    "canonical_orientation_ready": True,
                    "display_rotation": 0,
                    "canvas_width": 1080,
                    "canvas_height": 2400,
                },
                "acknowledgements": [],
            }
        )
        session = HikRigCalibrationSession(
            options,
            camera=mock.Mock(),
            phone=phone,
            target=target,
            progress=lambda _message: None,
        )

        with self.assertRaisesRegex(RuntimeError, "painted contract-v2 target"):
            session._wait_native_canonical_surface(timeout_seconds=0.02)

        phone.metrics.assert_not_called()

    def test_refresh_quantization_and_exposure_selection(self):
        self.assertAlmostEqual(refresh_quantized_exposure_us(60.0, 2), 8333.333333, places=3)
        self.assertAlmostEqual(
            refresh_quantized_exposure_us(120.0, 0.5), 16666.666667, places=3
        )
        self.assertAlmostEqual(
            refresh_quantized_exposure_us(120.0, 1.0 / 3.0), 25000.0, places=3
        )
        with self.assertRaises(ValueError):
            refresh_quantized_exposure_us(120.0, 0.75)
        rows = [
            ExposureObservation(2, 8333.3, 0.0, (228.0, 229.0, 230.0), (0.0, 0.0, 0.0)),
            ExposureObservation(1, 16666.7, 0.0, (228.0, 229.0, 230.0), (0.0, 0.0, 0.0)),
            ExposureObservation(2, 8333.3, 3.0, (229.0, 229.0, 229.0), (0.0, 0.0, 0.0)),
        ]
        selected = choose_exposure(rows)
        self.assertEqual(selected.shutter_refresh_multiplier, 2)
        self.assertEqual(selected.exposure_refresh_periods, 0.5)
        self.assertEqual(selected.gain, 0.0)

    def test_default_hik_auto_limits_one_refresh_period_and_twelve_db(self):
        with tempfile.TemporaryDirectory() as directory:
            options = HikCalibrationOptions(
                "fake", "phone", Path(directory) / "output"
            )
        self.assertEqual(options.maximum_exposure_periods, 1)
        self.assertEqual(options.maximum_auto_gain_db, 12.0)
        self.assertEqual(options.visible_screen_margin_px, 8)
        self.assertAlmostEqual(
            refresh_quantized_exposure_us(
                120.0, 1.0 / options.maximum_exposure_periods
            ),
            8333.333333,
            places=3,
        )

    def test_exposure_lock_keeps_completed_hik_one_shot_result(self):
        class Camera:
            def __init__(self):
                self.manual_calls = []

            def set_white_balance(self, red, green, blue):
                return {
                    "ratio_red": red,
                    "ratio_green": green,
                    "ratio_blue": blue,
                }

            def set_manual_imaging(self, exposure_us, gain):
                self.manual_calls.append((float(exposure_us), float(gain)))
                return {
                    "exposure_us": float(exposure_us),
                    "gain": float(gain),
                    "fps": 30.0,
                }

            def read(self):
                return rig_frame_sample(np.full((16, 16, 3), 180, np.uint8))

        class Target:
            def present_image(self, _image, _label):
                return object()

        with tempfile.TemporaryDirectory() as directory:
            camera = Camera()
            session = HikRigCalibrationSession(
                HikCalibrationOptions(
                    "fake", "phone", Path(directory) / "output"
                ),
                camera=camera,
                phone=object(),
                target=Target(),
                progress=lambda _message: None,
            )
            session.phone_metrics = mock.Mock(
                refresh_hz=60.0,
                screen_size_px=(32, 32),
            )
            session.visible_region = {
                "xywh": [0, 0, 16, 16],
                "safe_xywh": [0, 0, 16, 16],
            }
            session.white_mask = np.full((16, 16), 255, np.uint8)
            session.auto_imaging_seed = {
                "exposure_us": 16667.0,
                "gain": 12.0,
                "white_balance": {
                    "ratio_red": 1500,
                    "ratio_green": 1024,
                    "ratio_blue": 1900,
                },
            }
            session._wait_painted = lambda _presentation: None
            session._capture_settled = lambda _mask: np.full(
                (16, 16, 3), 180, np.uint8
            )
            session.calibrate_exposure()

        self.assertEqual(camera.manual_calls, [(16667.0, 12.0)])
        self.assertEqual(session.exposure.exposure_us, 16667.0)
        self.assertEqual(session.exposure.gain, 12.0)
        self.assertEqual(len(session.exposure_observations), 1)
        self.assertEqual(
            session.auto_imaging_seed["locked_manual_readback"]["source"],
            "completed_hik_one_shot_auto",
        )

    def test_clipped_candidate_is_rejected(self):
        safe = ExposureObservation(1, 1000.0, 1.0, (223.0, 224.0, 225.0), (0.0, 0.0, 0.0))
        clipped = ExposureObservation(1, 1000.0, 0.0, (229.0, 229.0, 229.0), (0.06, 0.0, 0.0))
        self.assertIs(choose_exposure([clipped, safe]), safe)
        self.assertIs(choose_exposure([clipped]), clipped)

    def test_exposure_selection_uses_post_wb_reference_not_unbalanced_average(self):
        # Reduced from the 20260829-010352 hardware evidence. Both rows have
        # nearly identical pre-WB averages, but residual WB raises B/R to the
        # brightest channel. The two-period row is safe, quieter, and needs
        # almost no gain, so it must not lose on an average-rounding artifact.
        two_periods = ExposureObservation(
            0.5,
            16667.0,
            0.0718,
            (218.63, 226.30, 219.16),
            (0.0130, 0.0434, 0.0388),
            (2.97, 2.42, 2.85),
        )
        one_period = ExposureObservation(
            1.0,
            8333.0,
            6.103,
            (218.14, 226.98, 218.97),
            (0.0128, 0.0500, 0.0398),
            (3.95, 3.27, 3.76),
        )
        self.assertLess(two_periods.brightness, 224.4)
        self.assertGreater(two_periods.white_balance_reference_brightness, 224.4)
        self.assertIs(choose_exposure([one_period, two_periods]), two_periods)

    def test_residual_wb_clipping_falls_back_to_camera_one_shot_without_failure(self):
        class Camera:
            def __init__(self):
                self.ratios = (1000, 1000, 1000)

            def set_manual_imaging(self, exposure_us, gain):
                return {"exposure_us": exposure_us, "gain": gain, "fps": 30.0}

            def set_white_balance(self, red, green, blue):
                self.ratios = (red, green, blue)
                return {
                    "ratio_red": red,
                    "ratio_green": green,
                    "ratio_blue": blue,
                }

            def read(self):
                if self.ratios == (1000, 1000, 1000):
                    image = np.zeros((16, 16, 3), np.uint8)
                    image[:] = (220, 226, 220)
                else:
                    image = np.full((16, 16, 3), 230, np.uint8)
                    image[:, :8, 1] = 255
                return rig_frame_sample(image)

        with tempfile.TemporaryDirectory() as directory:
            camera = Camera()
            options = HikCalibrationOptions(
                "fake", "phone", Path(directory) / "output"
            )
            messages = []
            session = HikRigCalibrationSession(
                options, camera=camera, progress=messages.append
            )
            session.exposure = ExposureObservation(
                0.5,
                16667.0,
                0.0718,
                (218.7, 226.1, 218.7),
                (0.013, 0.042, 0.036),
            )
            session.white_mask = np.full((16, 16), 255, np.uint8)
            session.auto_imaging_seed = {
                "white_balance": {
                    "ratio_red": 1000,
                    "ratio_green": 1000,
                    "ratio_blue": 1000,
                }
            }
            quarter_image = np.zeros((16, 16, 3), np.uint8)
            quarter_image[:] = (100, 120, 100)
            session._capture_settled = lambda _mask: quarter_image.copy()
            session.calibrate_white_balance()
        self.assertEqual(session.white_balance["method"], "hik_one_shot_awb_fallback")
        self.assertEqual(camera.ratios, (1000, 1000, 1000))
        self.assertEqual(len(session.white_balance_attempts), 2)
        self.assertGreater(
            session.white_balance_attempts[0]["statistics"][
                "maximum_clipped_fraction"
            ],
            0.05,
        )
        self.assertTrue(any("kept HIK one-shot WB" in value for value in messages))

    def test_temporal_statistics_measure_noise_and_black_crushing(self):
        mask = np.full((8, 8), 255, np.uint8)
        first = np.full((8, 8, 3), 10, np.uint8)
        second = np.full((8, 8, 3), 12, np.uint8)
        white = temporal_white_statistics([first, second], mask)
        black = temporal_black_statistics([first, second], mask)
        self.assertEqual(white["mean_bgr"], [11.0, 11.0, 11.0])
        self.assertGreater(white["temporal_noise_bgr"][0], 1.4)
        self.assertEqual(black["zero_fraction_bgr"], [0.0, 0.0, 0.0])

    def test_black_level_chooses_lowest_non_crushing_candidate(self):
        crushed = BlackLevelObservation(0, (0, 0, 0), (0.50, 0.50, 0.50), (0.1, 0.1, 0.1))
        low = BlackLevelObservation(120, (5, 5, 5), (0.01, 0.01, 0.01), (0.4, 0.4, 0.4))
        quiet_high = BlackLevelObservation(240, (10, 10, 10), (0, 0, 0), (0.1, 0.1, 0.1))
        self.assertIs(choose_black_level([crushed, quiet_high, low]), low)

    def test_white_balance_uses_bgr_channel_evidence(self):
        image = np.zeros((80, 80, 3), np.uint8)
        image[:] = (100, 150, 200)
        mask = np.full((80, 80), 255, np.uint8)
        result = manual_white_balance_ratios(image, mask, blur_sigma=3.0)
        self.assertEqual(result["ratio_red"], 1000)
        self.assertEqual(result["ratio_green"], 1333)
        self.assertEqual(result["ratio_blue"], 2000)

    def test_visible_region_covers_skewed_footprint_and_keeps_safe_target(self):
        polygon = np.asarray([[25, 10], [190, 35], [160, 190], [5, 150]], np.float32)
        result = camera_visible_screen_region(polygon, (200, 200), margin_px=3)
        x, y, width, height = result["xywh"]
        self.assertLessEqual(x, int(np.floor(np.min(polygon[:, 0]))))
        self.assertLessEqual(y, int(np.floor(np.min(polygon[:, 1]))))
        self.assertGreaterEqual(x + width - 1, int(np.ceil(np.max(polygon[:, 0]))))
        self.assertGreaterEqual(y + height - 1, int(np.ceil(np.max(polygon[:, 1]))))
        safe_x, safe_y, safe_width, safe_height = result["safe_xywh"]
        corners = [
            (safe_x, safe_y),
            (safe_x + safe_width - 1, safe_y),
            (safe_x + safe_width - 1, safe_y + safe_height - 1),
            (safe_x, safe_y + safe_height - 1),
        ]
        contour = polygon.reshape((-1, 1, 2))
        self.assertTrue(all(cv2.pointPolygonTest(contour, point, False) >= 0 for point in corners))
        self.assertGreater(width * height, safe_width * safe_height)

    def test_visible_region_margin_expands_coverage_outward_only(self):
        polygon = np.asarray([[25, 20], [150, 20], [150, 160], [25, 160]], np.float32)
        baseline = camera_visible_screen_region(
            polygon, (200, 200), margin_px=3, coverage_margin_px=0
        )
        expanded = camera_visible_screen_region(
            polygon, (200, 200), margin_px=3, coverage_margin_px=8
        )
        self.assertEqual([25, 20, 126, 141], baseline["xywh"])
        self.assertEqual([17, 12, 142, 157], expanded["xywh"])
        self.assertEqual(baseline["safe_xywh"], expanded["safe_xywh"])
        self.assertEqual(8, expanded["coverage_margin_px"])

    def test_pattern_and_projected_mask_share_declared_region(self):
        pattern = white_patch((120, 200), (20, 30, 60, 80))
        self.assertEqual(tuple(pattern[40, 30]), (255, 255, 255))
        self.assertEqual(tuple(pattern[10, 10]), (0, 0, 0))
        mask = camera_white_mask((120, 200), (120, 200), (20, 30, 60, 80), np.eye(3), inset_screen_px=2)
        self.assertEqual(int(mask[32, 22]), 255)
        self.assertEqual(int(mask[30, 20]), 0)

    def test_focus_chart_has_four_standard_field_and_direction_edges(self):
        region = (20, 30, 600, 800)
        edges = focus_edge_regions(region)
        self.assertEqual(len(edges), 4)
        self.assertEqual(
            {edge["angle_deg"] for edge in edges}, {5.0, 85.0, 95.0, -5.0}
        )
        chart = focus_pattern((700, 900), region)
        self.assertEqual(chart.shape, (900, 700, 3))
        self.assertGreater(len(np.unique(chart.reshape(-1, 3), axis=0)), 2)
        frame_x, frame_y, frame_width, frame_height = focus_frame_rect(region)
        self.assertGreaterEqual(frame_x - region[0], int(region[2] * 0.18))
        self.assertGreaterEqual(frame_y - region[1], int(region[3] * 0.18))
        self.assertLessEqual(frame_width, int(region[2] * 0.63))
        self.assertLessEqual(frame_height, int(region[3] * 0.63))

    def test_focus_chart_pose_frame_is_detected_by_primitive_contour(self):
        region = (20, 30, 600, 800)
        chart = focus_pattern((700, 900), region)
        x, y, width, height = focus_frame_rect(region)
        expected = np.asarray(
            [[x, y], [x + width - 1, y], [x + width - 1, y + height - 1], [x, y + height - 1]],
            np.float64,
        )
        detected = detect_focus_pose_frame(chart, expected)
        self.assertLess(
            float(np.max(np.linalg.norm(np.asarray(detected["camera_quad_xy"]) - expected, axis=1))),
            2.0,
        )

    def test_known_focus_rectangle_recovers_camera_only_pitch_yaw_and_distance(self):
        camera_size = (1440, 1080)
        focal = 1100.0
        camera_matrix = np.asarray(
            [[focal, 0, 719.5], [0, focal, 539.5], [0, 0, 1]], np.float64
        )
        target_size = (40.0, 55.0)
        objects = np.asarray(
            [[-20, -27.5, 0], [20, -27.5, 0], [20, 27.5, 0], [-20, 27.5, 0]],
            np.float64,
        )
        rotation_vector = np.radians(np.asarray([-7.0, 11.0, 2.0], np.float64))
        translation = np.asarray([3.0, -2.0, 280.0], np.float64)
        projected, _ = cv2.projectPoints(
            objects, rotation_vector, translation, camera_matrix, np.zeros(5)
        )
        pose = estimate_focus_target_pose(
            projected.reshape((4, 2)), target_size, camera_size, focal_length_px=focal
        )
        rotation, _ = cv2.Rodrigues(rotation_vector)
        normal = rotation[:, 2]
        expected_distance = abs(float(np.dot(normal, translation)))
        expected_yaw = np.degrees(np.arctan2(normal[0], normal[2]))
        expected_pitch = np.degrees(
            np.arctan2(-normal[1], np.sqrt(normal[0] ** 2 + normal[2] ** 2))
        )
        top_midpoint = np.mean(projected.reshape((4, 2))[:2], axis=0)
        bottom_midpoint = np.mean(projected.reshape((4, 2))[2:], axis=0)
        app_up = top_midpoint - bottom_midpoint
        expected_rotation = np.degrees(np.arctan2(app_up[0], -app_up[1]))
        self.assertAlmostEqual(
            pose["lens_to_panel_distance_mm"], expected_distance, delta=1.0
        )
        self.assertAlmostEqual(pose["pitch_deg"], expected_pitch, delta=0.2)
        self.assertAlmostEqual(pose["yaw_deg"], expected_yaw, delta=0.2)
        self.assertAlmostEqual(
            pose["phone_rotation_clockwise_from_camera_up_deg"],
            expected_rotation,
            delta=0.2,
        )
        inferred = estimate_focus_target_pose(
            projected.reshape((4, 2)), target_size, camera_size
        )
        self.assertAlmostEqual(inferred["focal_length_px"], focal, delta=2.0)
        self.assertAlmostEqual(inferred["pitch_deg"], expected_pitch, delta=0.3)
        self.assertAlmostEqual(inferred["yaw_deg"], expected_yaw, delta=0.3)

    def test_hardware_roi_homography_accounts_for_crop_origin(self):
        full = np.asarray([[2.0, 0.0, 5.0], [0.0, 3.0, 7.0], [0.0, 0.0, 1.0]])
        crop = compose_hardware_roi_homography(full, [10, 20, 100, 80])
        point = crop.dot(np.asarray([0.0, 0.0, 1.0]))
        self.assertTrue(np.allclose(point[:2], [25.0, 67.0]))

    def test_roi_alignment_handles_dynamic_zero_offset_maximum(self):
        adapter = HikMvsCameraAdapter(backend=object())
        adapter.controls = lambda: {
            "offset_x": {"minimum": 0, "maximum": 0, "increment": 2},
            "offset_y": {"minimum": 0, "maximum": 0, "increment": 2},
            "width": {"minimum": 64, "maximum": 1920, "increment": 4},
            "height": {"minimum": 64, "maximum": 1080, "increment": 4},
        }
        self.assertEqual(adapter.align_roi([101, 51, 1001, 701]), [100, 50, 1004, 704])

    def test_camera_rejects_exposure_readback_that_breaks_refresh_quantization(self):
        class Backend:
            def __init__(self):
                self.values = {"AcquisitionFrameRate": 30.0, "ExposureTime": 0.0, "Gain": 0.0}

            def set_enum(self, _node, _value):
                pass

            def float_range(self, node):
                return {"minimum": 1.0, "maximum": 120.0, "current": self.values[node]}

            def set_float(self, node, value):
                self.values[node] = float(value) + (100.0 if node == "ExposureTime" else 0.0)

            def get_float(self, node):
                return self.values[node]

            def close(self):
                pass

        adapter = HikMvsCameraAdapter(backend=Backend())
        adapter.configuration = CameraConfiguration("fake", 100, 100, 30.0, "hik_mvs")
        with self.assertRaisesRegex(RuntimeError, "refresh quantization"):
            adapter.set_manual_imaging(16666.6667, 0.0)

    def test_hik_one_shot_auto_uses_camera_state_machines_and_reads_modes(self):
        class Backend:
            def __init__(self):
                self.enums = {
                    "ExposureAuto": 0,
                    "GainAuto": 0,
                    "BalanceWhiteAuto": 0,
                }

            def set_enum(self, node, value):
                self.enums[node] = MvsPythonBackend.ENUM_VALUES[node][value]

            def get_enum(self, node):
                return self.enums[node]

            def get_float(self, node):
                return {"ExposureTime": 5840.0, "Gain": 7.5}[node]

        adapter = HikMvsCameraAdapter(backend=Backend())
        started = adapter.set_once_auto_imaging()
        self.assertEqual(started["modes"]["ExposureAuto"], "Once")
        self.assertEqual(started["modes"]["GainAuto"], "Once")
        self.assertEqual(started["modes"]["BalanceWhiteAuto"], "Once")

    def test_hik_one_shot_auto_limits_cap_exposure_and_gain(self):
        class Backend:
            def __init__(self):
                self.integers = {"AutoExposureTimeUpperLimit": 5840}
                self.floats = {"AutoGainUpperLimit": 8.0}

            def int_range(self, _node):
                return {"minimum": 15, "maximum": 9999813, "increment": 1}

            def set_int(self, node, value):
                self.integers[node] = int(value)

            def get_int(self, node):
                return self.integers[node]

            def float_range(self, _node):
                return {"minimum": 0.0, "maximum": 16.9806995, "increment": 0.01}

            def set_float(self, node, value):
                self.floats[node] = float(value)

            def get_float(self, node):
                return self.floats[node]

        adapter = HikMvsCameraAdapter(backend=Backend())
        result = adapter.configure_once_auto_limits(16666.667, 12.0)
        self.assertEqual(result["exposure_upper_us"], 16667)
        self.assertAlmostEqual(result["gain_upper"], 12.0)

    def test_hik_auto_function_aoi_uses_separate_intensity_and_wb_regions(self):
        class Backend:
            def __init__(self):
                self.selector = 0
                self.rows = {
                    0: {"Width": 1440, "Height": 1080, "OffsetX": 0, "OffsetY": 0},
                    1: {"Width": 1440, "Height": 1080, "OffsetX": 0, "OffsetY": 0},
                }

            def set_enum(self, node, value):
                self.selector = MvsPythonBackend.ENUM_VALUES[node][value]

            @staticmethod
            def _short(node):
                return node.replace("AutoFunctionAOI", "")

            def set_int(self, node, value):
                self.rows[self.selector][self._short(node)] = int(value)

            def get_int(self, node):
                return self.rows[self.selector][self._short(node)]

            def int_range(self, node):
                short = self._short(node)
                maximum = {
                    "Width": 1440,
                    "Height": 1080,
                    "OffsetX": 1440 - self.rows[self.selector]["Width"],
                    "OffsetY": 1080 - self.rows[self.selector]["Height"],
                }[short]
                minimum = 32 if short == "Width" else 8 if short == "Height" else 0
                return {"minimum": minimum, "maximum": maximum, "increment": 4}

            def set_bool(self, _node, _value):
                pass

            def get_bool(self, node):
                return (
                    node == "AutoFunctionAOIUsageIntensity" and self.selector == 0
                ) or (
                    node == "AutoFunctionAOIUsageWhiteBalance" and self.selector == 1
                )

        adapter = HikMvsCameraAdapter(backend=Backend())
        result = adapter.configure_auto_function_roi([101, 53, 803, 707])
        self.assertEqual(result["selectors"]["AOI1"]["xywh"], [100, 52, 800, 704])
        self.assertEqual(result["selectors"]["AOI2"]["xywh"], [100, 52, 800, 704])
        self.assertTrue(result["selectors"]["AOI1"]["usage_intensity"])
        self.assertTrue(result["selectors"]["AOI2"]["usage_white_balance"])


class FakeAdbRunner:
    def __init__(self):
        self.commands = []
        self.settings = {
            ("system", "screen_off_timeout"): "30000",
            ("system", "screen_brightness_mode"): "1",
            ("system", "screen_brightness"): "94",
            ("global", "stay_on_while_plugged_in"): None,
            ("global", "policy_control"): "immersive.navigation=demo",
            ("system", "accelerometer_rotation"): "1",
            ("system", "user_rotation"): "0",
        }
        self.display_on = False

    def __call__(self, command, _timeout):
        args = list(command[3:])
        self.commands.append(args)
        if args[:2] == ["shell", "wm"] and args[2:] == ["size"]:
            return "Physical size: 1080x2400"
        if args[:2] == ["shell", "wm"] and args[2:] == ["density"]:
            return "Physical density: 420"
        if args[:3] == ["shell", "dumpsys", "display"]:
            return "mOverrideDisplayInfo=DisplayInfo{state %s} activeMode refreshRate=120.0" % (
                "ON" if self.display_on else "OFF"
            )
        if args[:4] == ["shell", "input", "keyevent", "KEYCODE_WAKEUP"]:
            self.display_on = True
        if args[:4] == ["shell", "input", "keyevent", "KEYCODE_SLEEP"]:
            self.display_on = False
        if args[:3] == ["shell", "getprop", "ro.product.manufacturer"]:
            return "Example"
        if args[:3] == ["shell", "getprop", "ro.product.model"]:
            return "Phone"
        if args[:3] == ["shell", "getprop", "ro.build.version.release"]:
            return "14"
        if args[:4] == ["shell", "cmd", "package", "resolve-activity"]:
            return "com.android.chrome/com.google.android.apps.chrome.IntentDispatcher"
        if args[:4] == ["shell", "settings", "get", "system"] or args[:4] == ["shell", "settings", "get", "global"]:
            return self.settings.get((args[3], args[4])) or "null"
        if args[:4] == ["shell", "settings", "put", "system"] or args[:4] == ["shell", "settings", "put", "global"]:
            self.settings[(args[3], args[4])] = args[5]
        if args[:4] == ["shell", "settings", "delete", "system"] or args[:4] == ["shell", "settings", "delete", "global"]:
            self.settings[(args[3], args[4])] = None
        return ""


class HikPhoneTests(unittest.TestCase):
    def test_native_target_launch_uses_explicit_activity_without_browser_gesture(self):
        class NativeRunner(FakeAdbRunner):
            def __call__(self, command, timeout):
                args = list(command[3:])
                if args[:4] == ["shell", "pm", "list", "packages"]:
                    self.commands.append(args)
                    return "package:io.iris.phonetarget"
                if args[:3] == ["shell", "dumpsys", "package"]:
                    self.commands.append(args)
                    return "versionCode=2 minSdk=23 targetSdk=35"
                return super().__call__(command, timeout)

        runner = NativeRunner()
        phone = AdbPhoneSession(
            "SERIAL-1", adb_executable="adb-test", runner=runner,
            sleeper=lambda _seconds: None,
        )
        phone.wake_and_hold_native_target(8765, [1080, 2400])
        launches = [
            command for command in runner.commands
            if command[:3] == ["shell", "am", "start"]
        ]
        self.assertEqual(len(launches), 1)
        self.assertTrue(
            any(
                "io.iris.phonetarget.PhoneTargetActivity" in item
                for item in launches[0]
            )
        )
        self.assertNotIn("input", [item for row in launches for item in row])
        self.assertEqual(
            phone.viewer_activity,
            "io.iris.phonetarget/io.iris.phonetarget.PhoneTargetActivity",
        )
        phone.cleanup(turn_display_off=True)

    def test_native_target_absence_triggers_apk_install_before_launch(self):
        class NativeInstallRunner(FakeAdbRunner):
            def __init__(self):
                super().__init__()
                self.installed = False
                self.install_timeout = None

            def __call__(self, command, timeout):
                args = list(command[3:])
                if args[:3] == ["shell", "pm", "path"]:
                    raise AssertionError("package absence must not be probed with pm path")
                if args[:4] == ["shell", "pm", "list", "packages"]:
                    self.commands.append(args)
                    return (
                        "package:io.iris.phonetarget"
                        if self.installed
                        else ""
                    )
                if args[:2] == ["install", "-r"]:
                    self.commands.append(args)
                    self.install_timeout = timeout
                    self.installed = True
                    return "Success"
                if args[:3] == ["shell", "dumpsys", "package"]:
                    self.commands.append(args)
                    return (
                        "versionCode=2 minSdk=23 targetSdk=35"
                        if self.installed
                        else ""
                    )
                return super().__call__(command, timeout)

        with tempfile.TemporaryDirectory() as directory:
            apk = Path(directory) / "explicit-phone-target.apk"
            apk.write_bytes(b"apk")
            runner = NativeInstallRunner()
            phone = AdbPhoneSession(
                "SERIAL-1",
                adb_executable="adb-test",
                runner=runner,
                sleeper=lambda _seconds: None,
            )
            phone.wake_and_hold_native_target(
                8765, [1080, 2400], apk_path=apk
            )

        installs = [
            command
            for command in runner.commands
            if command[:2] == ["install", "-r"]
        ]
        launches = [
            command for command in runner.commands
            if command[:3] == ["shell", "am", "start"]
        ]
        self.assertEqual(len(installs), 1)
        self.assertEqual(len(launches), 1)
        self.assertGreaterEqual(runner.install_timeout, 120.0)

    def test_native_target_upgrades_installed_contract_v1_before_launch(self):
        class UpgradeRunner(FakeAdbRunner):
            def __init__(self):
                super().__init__()
                self.version_code = 1

            def __call__(self, command, timeout):
                args = list(command[3:])
                if args[:4] == ["shell", "pm", "list", "packages"]:
                    self.commands.append(args)
                    return "package:io.iris.phonetarget"
                if args[:3] == ["shell", "dumpsys", "package"]:
                    self.commands.append(args)
                    return "versionCode={} minSdk=23 targetSdk=35".format(
                        self.version_code
                    )
                if args[:2] == ["install", "-r"]:
                    self.commands.append(args)
                    self.version_code = 2
                    return "Success"
                return super().__call__(command, timeout)

        with tempfile.TemporaryDirectory() as directory:
            apk = Path(directory) / "iris-phone-target.apk"
            apk.write_bytes(b"apk-v2")
            runner = UpgradeRunner()
            phone = AdbPhoneSession(
                "SERIAL-1",
                adb_executable="adb-test",
                runner=runner,
                sleeper=lambda _seconds: None,
            )
            phone.wake_and_hold_native_target(
                8765, [1080, 2400], apk_path=apk
            )

        self.assertEqual(runner.version_code, 2)
        self.assertEqual(
            len([row for row in runner.commands if row[:2] == ["install", "-r"]]),
            1,
        )

    def test_native_target_accepts_package_committed_after_adb_transport_error(self):
        class CommittedInstallRunner(FakeAdbRunner):
            def __init__(self):
                super().__init__()
                self.installed = False

            def __call__(self, command, timeout):
                args = list(command[3:])
                if args[:4] == ["shell", "pm", "list", "packages"]:
                    self.commands.append(args)
                    return (
                        "package:io.iris.phonetarget"
                        if self.installed
                        else ""
                    )
                if args[:2] == ["install", "-r"]:
                    self.commands.append(args)
                    self.installed = True
                    raise RuntimeError("ADB transport ended after package commit")
                if args[:3] == ["shell", "dumpsys", "package"]:
                    self.commands.append(args)
                    return (
                        "versionCode=2 minSdk=23 targetSdk=35"
                        if self.installed
                        else ""
                    )
                return super().__call__(command, timeout)

        with tempfile.TemporaryDirectory() as directory:
            apk = Path(directory) / "iris-phone-target.apk"
            apk.write_bytes(b"apk")
            runner = CommittedInstallRunner()
            phone = AdbPhoneSession(
                "SERIAL-1",
                adb_executable="adb-test",
                runner=runner,
                sleeper=lambda _seconds: None,
            )
            phone.wake_and_hold_native_target(
                8765, [1080, 2400], apk_path=apk
            )

        self.assertTrue(runner.installed)
        self.assertTrue(
            any(
                command[:3] == ["shell", "am", "start"]
                for command in runner.commands
            )
        )

    def test_subprocess_runner_decodes_adb_output_as_utf8_without_locale_dependency(self):
        completed = mock.Mock(returncode=0, stdout="display � text", stderr="")
        with mock.patch("subprocess.run", return_value=completed) as run:
            self.assertEqual(_subprocess_runner(["adb", "devices"], 1.0), "display � text")
        self.assertEqual(run.call_args[1]["encoding"], "utf-8")
        self.assertEqual(run.call_args[1]["errors"], "replace")

    def test_subprocess_runner_restarts_adb_and_retries_once_after_timeout(self):
        timeout = subprocess.TimeoutExpired(["adb", "devices"], 1.0)
        success = mock.Mock(returncode=0, stdout="devices", stderr="")
        recovery = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch(
            "subprocess.run",
            side_effect=[timeout, recovery, recovery, success],
        ) as run:
            self.assertEqual(_subprocess_runner(["adb", "devices"], 1.0), "devices")
        self.assertEqual(
            [call[0][0] for call in run.call_args_list],
            [
                ["adb", "devices"],
                ["adb", "kill-server"],
                ["adb", "start-server"],
                ["adb", "devices"],
            ],
        )

    def test_subprocess_runner_force_kills_adb_when_kill_server_hangs(self):
        command_timeout = subprocess.TimeoutExpired(["adb", "devices"], 1.0)
        kill_timeout = subprocess.TimeoutExpired(["adb", "kill-server"], 5.0)
        success = mock.Mock(returncode=0, stdout="devices", stderr="")
        recovery = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch(
            "rig_runtime.adapters.android.phone.os.name", "nt"
        ), mock.patch(
            "subprocess.run",
            side_effect=[command_timeout, kill_timeout, recovery, recovery, success],
        ) as run:
            self.assertEqual(_subprocess_runner(["adb", "devices"], 1.0), "devices")
        self.assertEqual(
            [call[0][0] for call in run.call_args_list],
            [
                ["adb", "devices"],
                ["adb", "kill-server"],
                ["taskkill.exe", "/F", "/IM", "adb.exe", "/T"],
                ["adb", "start-server"],
                ["adb", "devices"],
            ],
        )

    def test_subprocess_runner_force_kills_and_retries_when_start_server_hangs(self):
        command_timeout = subprocess.TimeoutExpired(["adb", "devices"], 1.0)
        start_timeout = subprocess.TimeoutExpired(["adb", "start-server"], 5.0)
        success = mock.Mock(returncode=0, stdout="devices", stderr="")
        recovery = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch(
            "rig_runtime.adapters.android.phone.os.name", "nt"
        ), mock.patch(
            "subprocess.run",
            side_effect=[
                command_timeout,
                recovery,
                start_timeout,
                recovery,
                recovery,
                success,
            ],
        ) as run:
            self.assertEqual(_subprocess_runner(["adb", "devices"], 1.0), "devices")
        self.assertEqual(
            [call[0][0] for call in run.call_args_list],
            [
                ["adb", "devices"],
                ["adb", "kill-server"],
                ["adb", "start-server"],
                ["taskkill.exe", "/F", "/IM", "adb.exe", "/T"],
                ["adb", "start-server"],
                ["adb", "devices"],
            ],
        )

    def test_subprocess_runner_reports_force_kill_diagnostic_if_restart_fails(self):
        command_timeout = subprocess.TimeoutExpired(["adb", "devices"], 1.0)
        kill_timeout = subprocess.TimeoutExpired(["adb", "kill-server"], 5.0)
        taskkill_failure = mock.Mock(
            returncode=5, stdout="", stderr="Access is denied"
        )
        start_failure = mock.Mock(
            returncode=1, stdout="", stderr="cannot start daemon"
        )
        with mock.patch(
            "rig_runtime.adapters.android.phone.os.name", "nt"
        ), mock.patch(
            "subprocess.run",
            side_effect=[
                command_timeout,
                kill_timeout,
                taskkill_failure,
                start_failure,
            ],
        ) as run:
            with self.assertRaisesRegex(
                RuntimeError,
                "cannot start daemon.*taskkill exited 5: Access is denied",
            ):
                _subprocess_runner(["adb", "devices"], 1.0)
        self.assertEqual(run.call_count, 4)

    def test_subprocess_runner_does_not_restart_for_non_timeout_failure(self):
        failure = mock.Mock(returncode=1, stdout="", stderr="unauthorized")
        with mock.patch("subprocess.run", return_value=failure) as run:
            with self.assertRaisesRegex(RuntimeError, "unauthorized"):
                _subprocess_runner(["adb", "devices"], 1.0)
        self.assertEqual(run.call_count, 1)

    def test_subprocess_runner_stops_after_one_recovery_retry(self):
        first = subprocess.TimeoutExpired(["adb", "devices"], 1.0)
        second = subprocess.TimeoutExpired(["adb", "devices"], 1.0)
        recovery = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch(
            "subprocess.run",
            side_effect=[first, recovery, recovery, second],
        ) as run:
            with self.assertRaisesRegex(RuntimeError, "one server restart"):
                _subprocess_runner(["adb", "devices"], 1.0)
        self.assertEqual(run.call_count, 4)

    def test_adb_discovery_and_connected_device_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            adb = Path(directory) / "adb.exe"
            adb.write_bytes(b"")
            self.assertEqual(resolve_adb_executable(str(adb)), str(adb.resolve()))
            output = "List of devices attached\nPHONE-1\tdevice\nPHONE-2\toffline\n"
            with mock.patch(
                "rig_runtime.adapters.android.phone._subprocess_runner",
                return_value=output,
            ):
                self.assertEqual(connected_adb_devices(str(adb)), ["PHONE-1"])

    def test_metrics_wake_restore_and_sleep_are_scoped_to_serial(self):
        runner = FakeAdbRunner()
        phone = AdbPhoneSession("SERIAL-1", adb_executable="adb-test", runner=runner, sleeper=lambda _seconds: None)
        metrics = phone.metrics()
        self.assertEqual(metrics.screen_size_px, [1080, 2400])
        self.assertEqual(metrics.refresh_hz, 120.0)
        phone.wake_and_hold(8765, metrics.screen_size_px)
        self.assertEqual(phone.display_brightness_state()["mode"], "manual")
        self.assertEqual(phone.display_brightness_state()["brightness_value"], 255)
        phone.request_fullscreen(metrics.screen_size_px)
        phone.cleanup(turn_display_off=True)
        self.assertEqual(runner.settings[("system", "screen_off_timeout")], "30000")
        self.assertEqual(runner.settings[("system", "screen_brightness_mode")], "1")
        self.assertEqual(runner.settings[("system", "screen_brightness")], "94")
        self.assertIsNone(runner.settings[("global", "stay_on_while_plugged_in")])
        self.assertEqual(runner.settings[("global", "policy_control")], "immersive.navigation=demo")
        self.assertEqual(runner.settings[("system", "accelerometer_rotation")], "1")
        self.assertEqual(runner.settings[("system", "user_rotation")], "0")
        flattened = [item for command in runner.commands for item in command]
        self.assertIn("KEYCODE_WAKEUP", flattened)
        self.assertIn("KEYCODE_SLEEP", flattened)
        self.assertGreaterEqual(flattened.count("tap"), 2)
        force_stop = [
            index
            for index, command in enumerate(runner.commands)
            if command == ["shell", "am", "force-stop", "com.android.chrome"]
        ]
        launches = [
            index
            for index, command in enumerate(runner.commands)
            if command[:3] == ["shell", "am", "start"]
        ]
        self.assertEqual(1, len(force_stop))
        self.assertEqual(1, len(launches))
        self.assertLess(force_stop[0], launches[0])
        self.assertFalse(runner.display_on)

    def test_physical_display_is_verified_after_wakeup(self):
        runner = FakeAdbRunner()
        phone = AdbPhoneSession(
            "SERIAL-1", runner=runner, sleeper=lambda _seconds: None
        )
        phone.wake_and_hold_display()
        evidence = phone.ensure_display_on()
        self.assertEqual(evidence["state"], "ON")
        self.assertFalse(evidence["gating"])
        self.assertEqual(
            evidence["visual_verification"], "required_from_adb_and_camera"
        )
        self.assertTrue(runner.display_on)
        phone.cleanup(turn_display_off=True)

    def test_samsung_stale_override_does_not_override_awake_default_display(self):
        class SamsungDisplayRunner(FakeAdbRunner):
            def __call__(self, command, timeout):
                args = list(command[3:])
                if args[:3] == ["shell", "dumpsys", "display"]:
                    self.commands.append(args)
                    return "\n".join(
                        [
                            "Display State=ON",
                            'mBaseDisplayInfo=DisplayInfo{displayId 0, state ON, type INTERNAL}',
                            'mOverrideDisplayInfo=DisplayInfo{displayId 0, state OFF, type INTERNAL}',
                            "mScreenState=ON",
                            "mActualState=ON",
                        ]
                    )
                return super().__call__(command, timeout)

        phone = AdbPhoneSession(
            "SERIAL-1", runner=SamsungDisplayRunner(), sleeper=lambda _seconds: None
        )
        self.assertEqual(phone.display_state(), "ON")
        evidence = phone.ensure_display_on(timeout_seconds=0.5)
        self.assertTrue(evidence["android_reported_on"])

    def test_android_off_report_defers_to_adb_and_camera_visual_checks(self):
        class AlwaysOffRunner(FakeAdbRunner):
            def __call__(self, command, timeout):
                args = list(command[3:])
                if args[:3] == ["shell", "dumpsys", "display"]:
                    self.commands.append(args)
                    return "Display State=OFF\nmActualState=OFF"
                return super().__call__(command, timeout)

        phone = AdbPhoneSession(
            "SERIAL-1", runner=AlwaysOffRunner(), sleeper=lambda _seconds: None
        )
        evidence = phone.ensure_display_on(timeout_seconds=0.5)
        self.assertEqual(evidence["state"], "OFF")
        self.assertFalse(evidence["android_reported_on"])
        self.assertFalse(evidence["gating"])
        self.assertEqual(
            evidence["visual_verification"], "required_from_adb_and_camera"
        )

    def test_metrics_report_current_display_orientation_and_app_viewport(self):
        class LandscapeRunner(FakeAdbRunner):
            def __call__(self, command, timeout):
                args = list(command[3:])
                if args[:3] == ["shell", "dumpsys", "input"]:
                    self.commands.append(args)
                    return "SurfaceOrientation: 1"
                if args[:3] == ["shell", "dumpsys", "display"]:
                    self.commands.append(args)
                    return (
                        "mOverrideDisplayInfo=DisplayInfo{app 2400 x 1000, "
                        "state ON, density 480 (409.432 x 406.4) dpi} "
                        "mActiveSfDisplayMode=DisplayMode{xDpi=409.432, yDpi=406.4, "
                        "refreshRate=120.0}"
                    )
                return super().__call__(command, timeout)

        metrics = AdbPhoneSession(
            "SERIAL-1", runner=LandscapeRunner(), sleeper=lambda _seconds: None
        ).metrics()
        self.assertEqual(metrics.screen_size_px, [2400, 1080])
        self.assertEqual(metrics.orientation_quarter_turns, 1)
        self.assertEqual(metrics.active_app_size_px, [2400, 1000])
        self.assertEqual(metrics.display_state, "ON")
        self.assertEqual(metrics.physical_dpi_xy, [406.4, 409.432])
        self.assertAlmostEqual(
            metrics.to_dict()["physical_pixel_pitch_mm_xy"][0], 25.4 / 406.4
        )

    def test_display_manager_rotation_overrides_stale_input_orientation(self):
        class GameRotationRunner(FakeAdbRunner):
            def __call__(self, command, timeout):
                args = list(command[3:])
                if args[:3] == ["shell", "dumpsys", "input"]:
                    self.commands.append(args)
                    return "SurfaceOrientation: 0"
                if args[:3] == ["shell", "dumpsys", "display"]:
                    self.commands.append(args)
                    return (
                        "mCurrentOrientation=0 "
                        "mOverrideDisplayInfo=DisplayInfo{real 2400 x 1080, "
                        "rotation 1, app 2400 x 1080, state ON} "
                        "mActiveSfDisplayMode=DisplayMode{refreshRate=120.0}"
                    )
                return super().__call__(command, timeout)

        metrics = AdbPhoneSession(
            "SERIAL-1", runner=GameRotationRunner(), sleeper=lambda _seconds: None
        ).metrics()
        self.assertEqual([2400, 1080], metrics.screen_size_px)
        self.assertEqual(1, metrics.orientation_quarter_turns)


class FakeRectifiedAdapter:
    def __init__(self, image):
        self.image = image
        self.closed = False

    def open(self, configuration: CameraConfiguration):
        return {"width_px": configuration.width_px, "height_px": configuration.height_px}

    def set_manual_imaging(self, exposure_us, gain):
        self.imaging = (exposure_us, gain)

    def set_white_balance(self, red, green, blue):
        self.wb = (red, green, blue)

    def set_roi(self, roi):
        self.roi = list(roi)
        return list(roi)

    def read(self):
        return rig_frame_sample(self.image.copy(), roi_xywh=getattr(self, "roi", None))

    def close(self):
        self.closed = True

    def align_roi(self, roi):
        return list(map(int, roi))


class HikRectifiedStreamTests(unittest.TestCase):
    def test_demo_display_session_uses_only_power_key_events(self):
        class FakePowerPhone:
            def __init__(self):
                self.commands = []

            def shell(self, *args):
                self.commands.append(args)
                return ""

        phone = FakePowerPhone()
        session = PhoneDisplayPowerSession("phone", phone=phone)
        session.open()
        session.close()
        self.assertEqual(
            phone.commands,
            [
                ("input", "keyevent", "KEYCODE_WAKEUP"),
                ("input", "keyevent", "KEYCODE_SLEEP"),
            ],
        )

    def test_demo_display_session_does_not_query_display_state(self):
        phone = mock.Mock()
        session = PhoneDisplayPowerSession("phone", phone=phone)
        with session:
            pass
        self.assertEqual(
            phone.shell.call_args_list,
            [
                mock.call("input", "keyevent", "KEYCODE_WAKEUP"),
                mock.call("input", "keyevent", "KEYCODE_SLEEP"),
            ],
        )
        phone.display_state.assert_not_called()

    def test_demo_display_power_errors_never_gate_camera_startup(self):
        phone = mock.Mock()
        phone.shell.side_effect = RuntimeError("ADB unavailable")
        session = PhoneDisplayPowerSession("phone", phone=phone)
        session.open()
        self.assertIn("ADB unavailable", session.last_error)
        session.close()
        self.assertFalse(session._opened)

    @staticmethod
    def _fake_data_matrix_batch(_screen, _region, _module, specifications):
        items = []
        for specification in list(specifications)[:4]:
            trial = int(specification["trial_index"])
            items.append(
                {
                    **dict(specification),
                    "payload": "A{}".format(trial),
                    "trial_id": "dm-{}".format(trial),
                    "cell_rect_screen_xywh": [10, 10, 20, 20],
                    "decode_rect_screen_xywh": [10, 10, 20, 20],
                }
            )
        return {
            "image": np.zeros((100, 100, 3), np.uint8),
            "items": items,
            "columns": 2,
            "rows": 2,
        }

    def test_data_matrix_high_failure_rate_batches_and_skips_after_three_patterns(self):
        with tempfile.TemporaryDirectory() as directory:
            options = HikCalibrationOptions(
                "fake",
                "phone",
                Path(directory) / "output",
                data_matrix_trials_per_size=40,
            )
            target = mock.Mock()
            target.present_image.return_value = mock.Mock()
            session = HikRigCalibrationSession(
                options,
                camera=mock.Mock(),
                phone=mock.Mock(),
                target=target,
                progress=lambda _message: None,
            )
            session.phone_metrics = PhoneMetrics(
                "phone", "Example", "Phone", "14", [100, 100], 420, 60.0
            )
            session.visible_region = {
                "xywh": [10, 10, 80, 80],
                "safe_xywh": [10, 10, 80, 80],
            }
            session.geometry = mock.Mock(
                matrix_3x3=np.eye(3).tolist(),
                inverse_matrix_3x3=np.eye(3).tolist(),
            )
            session._wait_painted = lambda _shown: None
            session._capture_data_matrix_frame = lambda: np.zeros((100, 100, 3), np.uint8)
            session._compose_data_matrix_batch = self._fake_data_matrix_batch
            failed_grade = {
                "grade": 0.0,
                "grade_letter": "F",
                "exact_payload_decoded": False,
            }
            with mock.patch(
                "rig_runtime.workflows.hik_rig_calibration.grade_data_matrix_decode",
                return_value=failed_grade,
            ):
                result = session.grade_data_matrix()
            evidence = Path(result["failure_evidence_directory"])
            self.assertTrue(evidence.is_dir())
            index = json.loads((evidence / "index.json").read_text("utf-8"))
            self.assertEqual(index["failure_count"], result["failure_evidence_count"])
            self.assertGreater(index["failure_count"], 0)
            first = index["failures"][0]
            annotated = evidence / first["files"]["annotated_camera_frame"]
            raw_crop = evidence / first["files"]["raw_camera_crop"]
            decoder_crop = evidence / first["files"]["rectified_decoder_crop"]
            self.assertTrue(annotated.is_file())
            self.assertTrue(raw_crop.is_file())
            self.assertTrue(decoder_crop.is_file())
            marked = cv2.imread(str(annotated))
            self.assertGreater(int(np.max(marked[:, :, 2])), 0)
        self.assertGreater(len(result["per_size"]), 1)
        self.assertTrue(all(row["early_rejected"] for row in result["per_size"]))
        self.assertTrue(all(row["trial_count"] == 3 for row in result["per_size"]))
        self.assertTrue(all(row["presentation_count"] == 1 for row in result["per_size"]))

    def test_data_matrix_reports_plain_decode_success_and_batches_qualified_size(self):
        with tempfile.TemporaryDirectory() as directory:
            options = HikCalibrationOptions(
                "fake", "phone", Path(directory) / "output"
            )
            target = mock.Mock()
            target.present_image.return_value = mock.Mock()
            session = HikRigCalibrationSession(
                options,
                camera=mock.Mock(),
                phone=mock.Mock(),
                target=target,
                progress=lambda _message: None,
            )
            session.phone_metrics = PhoneMetrics(
                "phone", "Example", "Phone", "14", [100, 100], 420, 60.0
            )
            session.visible_region = {
                "xywh": [10, 10, 80, 80],
                "safe_xywh": [10, 10, 80, 80],
            }
            session.geometry = mock.Mock(
                matrix_3x3=np.eye(3).tolist(),
                inverse_matrix_3x3=np.eye(3).tolist(),
            )
            session._wait_painted = lambda _shown: None
            session._capture_data_matrix_frame = lambda: np.zeros(
                (100, 100, 3), np.uint8
            )
            session._compose_data_matrix_batch = self._fake_data_matrix_batch
            decoded = {
                "grade": 4.0,
                "reference_decode_succeeded": True,
                "exact_payload_decoded": True,
            }
            with mock.patch(
                "rig_runtime.workflows.hik_rig_calibration.grade_data_matrix_decode",
                return_value=decoded,
            ):
                result = session.grade_data_matrix()
        size = result["per_size"][0]
        self.assertEqual(result["measurement"], "exact_payload_decode_success")
        self.assertEqual(result["required_decode_success_rate"], 0.95)
        self.assertEqual(size["decode_success_count"], 40)
        self.assertEqual(size["decode_success_rate"], 1.0)
        self.assertEqual(size["presentation_count"], 10)
        self.assertNotIn("grade_4_A_rate", size)
        self.assertNotIn("grade", size["trials"][0])
        self.assertNotIn("grade_letter", size["trials"][0])

    def test_optional_data_matrix_decoder_failure_returns_unavailable_result(self):
        with tempfile.TemporaryDirectory() as directory:
            options = HikCalibrationOptions(
                "fake",
                "phone",
                Path(directory) / "output",
                data_matrix_trials_per_size=20,
            )
            target = mock.Mock()
            target.present_image.return_value = mock.Mock()
            session = HikRigCalibrationSession(
                options,
                camera=mock.Mock(),
                phone=mock.Mock(),
                target=target,
                progress=lambda _message: None,
            )
            session.phone_metrics = PhoneMetrics(
                "phone", "Example", "Phone", "14", [100, 100], 420, 60.0
            )
            session.visible_region = {
                "xywh": [10, 10, 80, 80],
                "safe_xywh": [10, 10, 80, 80],
            }
            session.geometry = mock.Mock(
                matrix_3x3=np.eye(3).tolist(),
                inverse_matrix_3x3=np.eye(3).tolist(),
            )
            session._wait_painted = lambda _shown: None
            session._capture_data_matrix_frame = lambda: np.zeros((100, 100, 3), np.uint8)
            session._compose_data_matrix_batch = self._fake_data_matrix_batch
            with mock.patch(
                "rig_runtime.workflows.hik_rig_calibration.grade_data_matrix_decode",
                side_effect=RuntimeError("decoder unavailable"),
            ):
                result = session.grade_data_matrix()
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["failed_trial_index"], 0)
        self.assertIn("decoder unavailable", result["error"])
        self.assertFalse(session._preview_disabled)
        self.assertIn("Data Matrix decode test", session._preview_stage)

    def test_live_preview_fits_usable_desktop_and_keeps_settings_beside_image(self):
        with tempfile.TemporaryDirectory() as directory:
            options = HikCalibrationOptions(
                "fake", "phone", Path(directory) / "output", camera_width_px=1440,
                camera_height_px=1080
            )
            adapter = FakeRectifiedAdapter(np.zeros((1080, 1440, 3), np.uint8))
            session = HikRigCalibrationSession(options, camera=adapter, progress=lambda _message: None)
            session.camera_metadata = {
                "model": "MV-CS016-10UC",
                "serial": "DA9066154",
                "fps": 30.0,
            }
            session._desktop_work_area = lambda: [1280, 700]
            shown = []
            with mock.patch.object(cv2, "namedWindow"), mock.patch.object(
                cv2, "resizeWindow"
            ), mock.patch.object(cv2, "imshow", side_effect=lambda _name, image: shown.append(image)), mock.patch.object(
                cv2, "waitKey", return_value=-1
            ):
                session._set_preview_stage(
                    "Auto exposure/gain bootstrap",
                    exposure_mode="camera auto",
                    exposure_us=8333.0,
                    gain=12.0,
                )
                session._preview_update(adapter.image)
            self.assertEqual(len(shown), 1)
            self.assertLessEqual(shown[0].shape[0], int(700 * 0.82))
            self.assertLessEqual(shown[0].shape[1], int(1280 * 0.88))
            self.assertTrue(session._preview_created)

    def test_uvc_like_reader_applies_crop_origin_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hik_camera_calibration.json"
            config = {
                "camera": {
                    "device_id": "fake",
                    "full_sensor_mode": {"width_px": 8, "height_px": 8, "fps": 30.0},
                    "hardware_roi_xywh": [2, 3, 4, 4],
                },
                "imaging": {
                    "exposure_us": 1000.0,
                    "gain": 0.0,
                    "white_balance": {"ratio_red": 1000, "ratio_green": 1000, "ratio_blue": 1000},
                },
                "normalization": {
                    "full_sensor_camera_to_output_3x3": [[1, 0, -2], [0, 1, -3], [0, 0, 1]],
                    "output_size_px": [4, 4],
                },
            }
            path.write_text(json.dumps(config), encoding="utf-8")
            image = np.zeros((4, 4, 3), np.uint8)
            image[1, 1] = (3, 4, 5)
            adapter = FakeRectifiedAdapter(image)
            camera = RectifiedHikCamera(path, adapter=adapter).open()
            ok, frame = camera.read()
            self.assertTrue(ok)
            self.assertEqual(tuple(frame[1, 1]), (3, 4, 5))
            camera.release()
            self.assertTrue(adapter.closed)

    def test_minimum_latency_reader_returns_hardware_roi_without_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hik_camera_calibration.json"
            config = {
                "camera": {
                    "device_id": "fake",
                    "full_sensor_mode": {
                        "width_px": 8,
                        "height_px": 8,
                        "fps": 30.0,
                    },
                    "hardware_roi_xywh": [2, 3, 4, 4],
                },
                "imaging": {
                    "exposure_us": 1000.0,
                    "gain": 0.0,
                    "white_balance": {
                        "ratio_red": 1000,
                        "ratio_green": 1000,
                        "ratio_blue": 1000,
                    },
                },
                "normalization": {
                    "full_sensor_camera_to_output_3x3": np.eye(3).tolist(),
                    "output_size_px": [3, 3],
                    "orientation": {
                        "panel_up_reference_when_rectification_disabled": {
                            "source": "broad_orthogonal_panel_edges",
                            "space_id": "hik_full_sensor_camera_pixels",
                            "panel_up_unit_vector_xy": [0.01, -0.99995],
                            "camera_up_to_panel_up_clockwise_degrees": 0.573,
                        }
                    },
                },
            }
            path.write_text(json.dumps(config), encoding="utf-8")
            image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape((4, 4, 3))
            adapter = FakeRectifiedAdapter(image)
            with mock.patch.object(
                cv2, "remap", side_effect=AssertionError("remap must not run")
            ), mock.patch.object(
                cv2,
                "warpPerspective",
                side_effect=AssertionError("warp must not run"),
            ):
                camera = RectifiedHikCamera(
                    path, adapter=adapter, rectify=False
                ).open()
                ok, frame = camera.read()
                sample = camera.read_sample()
            self.assertTrue(ok)
            np.testing.assert_array_equal(frame, image)
            self.assertFalse(sample.metadata["rectified"])
            self.assertTrue(sample.metadata["hardware_roi_output"])
            self.assertTrue(sample.source_id.endswith(":hardware-roi"))
            self.assertEqual(
                [0.01, -0.99995],
                sample.metadata["image_space"]["panel_up_reference"][
                    "panel_up_unit_vector_xy"
                ],
            )
            camera.release()

    def test_production_reader_trusts_effective_roi_and_saved_orientation(self):
        class AlignedByCameraAdapter(FakeRectifiedAdapter):
            def set_roi(self, roi):
                self.roi = list(roi)
                return [1, 3, 4, 4]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hik_camera_calibration.json"
            path.write_text(
                json.dumps(
                    {
                        "camera": {
                            "device_id": "fake",
                            "full_sensor_mode": {
                                "width_px": 8,
                                "height_px": 8,
                                "fps": 30.0,
                            },
                            "hardware_roi_xywh": [2, 3, 4, 4],
                        },
                        "imaging": {
                            "exposure_us": 1000.0,
                            "gain": 0.0,
                            "white_balance": {
                                "ratio_red": 1000,
                                "ratio_green": 1000,
                                "ratio_blue": 1000,
                            },
                        },
                        "normalization": {
                            "full_sensor_camera_to_output_3x3": np.eye(3).tolist(),
                            "output_size_px": [4, 4],
                            "orientation": {"adapter_output_up": "unknown"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            adapter = AlignedByCameraAdapter(np.zeros((4, 4, 3), np.uint8))
            camera = RectifiedHikCamera(path, adapter=adapter).open()
            self.assertTrue(camera.isOpened())
            camera.release()

    def test_explicit_save_writes_reloadable_warning_bundle_and_dense_map(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-calibration"
            options = HikCalibrationOptions("fake", "phone", output, camera_width_px=100, camera_height_px=100)
            adapter = FakeRectifiedAdapter(np.zeros((100, 100, 3), np.uint8))
            session = HikRigCalibrationSession(options, camera=adapter, progress=lambda _message: None)
            points = np.asarray([[0, 0], [99, 0], [99, 99], [0, 99]], np.float64)
            session.phone_metrics = PhoneMetrics("phone", "Example", "Phone", "14", [100, 100], 420, 60.0)
            session.charuco_layout = screen_filling_charuco_layout((100, 100))
            session.camera_metadata = {"width_px": 100, "height_px": 100, "fps": 30.0}
            session.camera_controls = {"gain": {"minimum": 0.0, "maximum": 24.0}}
            session.correspondences = {"camera_points_xy": points, "screen_points_xy": points}
            session.geometry = estimate_screen_geometry(points, points, (100, 100), (100, 100))
            session.visible_region = camera_visible_screen_region(
                session.geometry.viewport_polygon_screen_xy, (100, 100), margin_px=8
            )
            session.orientation_evidence = charuco_orientation_evidence(
                session.geometry.inverse_matrix_3x3, [50, 50]
            )
            session.exposure = ExposureObservation(1, 16666.7, 0.0, (229, 229, 229), (0, 0, 0))
            session.black_level = 240
            session.white_balance = {"ratio_red": 1000, "ratio_green": 1000, "ratio_blue": 1000}
            session.panel_axis_measurement = {
                "schema_version": "1.0",
                "status": "accepted",
                "applied": True,
                "non_gating": True,
                "residual_clockwise_degrees": 1.5,
                "correction_counterclockwise_degrees": 1.5,
                "temporal_p95_deviation_degrees": 0.04,
                "panel_up_unit_vector_full_sensor_camera_xy": [0.02, -0.9998],
                "camera_up_to_panel_up_clockwise_degrees": 1.15,
                "representative_lines": [],
            }
            session.panel_axis_sample = rig_frame_sample(
                np.full((100, 100, 3), 48, np.uint8)
            )
            session.panel_axis_evidence_image = np.full(
                (
                    session.visible_region["xywh"][3],
                    session.visible_region["xywh"][2],
                    3,
                ),
                72,
                np.uint8,
            )
            saved = session.save()
            self.assertEqual(saved, output.resolve())
            self.assertTrue((saved / "calibration.yaml").is_file())
            self.assertTrue((saved / "rectification_maps.npz").is_file())
            self.assertTrue((saved / "hik_camera_calibration.yaml").is_file())
            config = json.loads((saved / "hik_camera_calibration.json").read_text(encoding="utf-8"))
            self.assertEqual(config["normalization"]["output_size_px"], session.visible_region["xywh"][2:])
            self.assertEqual(
                config["normalization"]["orientation"]["source"],
                "charuco_correspondences",
            )
            self.assertEqual(
                config["normalization"]["orientation"]["adapter_output_up"], "app_up"
            )
            axis = config["normalization"]["orientation"]["panel_axis_alignment"]
            self.assertTrue(axis["applied"])
            self.assertEqual(
                0,
                axis["rectification_composition"][
                    "runtime_resampling_passes_added"
                ],
            )
            self.assertEqual(
                [0.02, -0.9998],
                config["normalization"]["orientation"][
                    "panel_up_reference_when_rectification_disabled"
                ]["panel_up_unit_vector_xy"],
            )
            base = np.asarray(
                axis["rectification_composition"][
                    "base_full_sensor_camera_to_output_3x3"
                ]
            )
            expected = panel_axis_correction_matrix(
                config["normalization"]["output_size_px"], 1.5
            ).dot(base)
            expected /= expected[2, 2]
            np.testing.assert_allclose(
                expected,
                config["normalization"]["full_sensor_camera_to_output_3x3"],
            )
            refined_screen = np.asarray(
                config["geometry"]["full_sensor_camera_to_screen_3x3"]
            )
            self.assertFalse(np.allclose(refined_screen, np.eye(3)))
            np.testing.assert_allclose(
                np.linalg.inv(refined_screen),
                config["geometry"]["screen_to_full_sensor_camera_3x3"],
            )
            np.testing.assert_allclose(
                refined_screen,
                config["coordinate_spaces"]["conversions"][
                    "full_sensor_image_to_phone_display_3x3"
                ],
            )
            self.assertEqual(config["camera"]["controls"]["gain"]["maximum"], 24.0)
            self.assertEqual(config["imaging"]["black_level"], 240)
            self.assertEqual(
                "valid_screen_mask.png",
                config["normalization"]["valid_mask_file"],
            )
            self.assertTrue((saved / "valid_screen_mask.png").is_file())
            self.assertTrue(config["results"]["cross_source_check"]["non_gating"])
            self.assertEqual(
                config["results"]["cross_source_check"]["status"], "unavailable"
            )
            self.assertTrue(
                (saved / "cross_source_check" / "cross_source_check.json").is_file()
            )
            self.assertTrue(
                (saved / "cross_source_check" / "cross_source_check.yaml").is_file()
            )
            self.assertEqual(
                {
                    "valid_screen_mask.png",
                    "panel_axis_raw_hik.png",
                    "panel_axis_rectified_evidence.png",
                },
                {row["file"] for row in config["media"]},
            )
            media_by_file = {row["file"]: row for row in config["media"]}
            self.assertEqual(
                "hik_rig_rectified_visible_phone_pixels",
                media_by_file["valid_screen_mask.png"]["space"]["id"],
            )


class FakeFacadeAdapter:
    def __init__(self):
        self.exposure = 1000.0
        self.gain = 2.0
        self.wb = {}

    def set_manual_imaging(self, exposure_us, gain):
        self.exposure = float(exposure_us)
        self.gain = float(gain)
        return {"exposure_us": self.exposure, "gain": self.gain, "fps": 30.0}

    def set_white_balance(self, red, green, blue):
        self.wb = {"ratio_red": red, "ratio_green": green, "ratio_blue": blue}
        return dict(self.wb)

    def set_control(self, name, value):
        self.control = (name, value)
        return True


class FakeFacadeReader:
    instances = []

    def __init__(self, path):
        self.path = Path(path)
        self.adapter = FakeFacadeAdapter()
        self.opened = False
        self.released = False
        self.__class__.instances.append(self)

    def open(self):
        self.opened = True
        return self

    def read(self):
        image = np.zeros((2, 3, 3), np.uint8)
        image[:] = (10, 20, 30)
        return True, image

    def release(self):
        self.released = True


class HikCompatibleFacadeTests(unittest.TestCase):
    def _config_path(self, directory):
        path = Path(directory) / "hik_camera_calibration.json"
        value = {
            "camera": {
                "device_id": "CAMERA-1",
                "full_sensor_mode": {"width_px": 8, "height_px": 8, "fps": 30.0},
                "hardware_roi_xywh": [0, 0, 8, 8],
            },
            "imaging": {
                "exposure_us": 1000.0,
                "gain": 2.0,
                "black_level": 240,
                "white_balance": {"ratio_red": 1000, "ratio_green": 1000, "ratio_blue": 1000},
            },
            "normalization": {
                "full_sensor_camera_to_output_3x3": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "output_size_px": [3, 2],
            },
        }
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_hikcam_alias_is_lazy_and_returns_rgb_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._config_path(directory)
            FakeFacadeReader.instances = []
            camera = hikcam.HikCamera(
                config={
                    "diagnostic_calibration_override": path,
                    "reader_factory": FakeFacadeReader,
                }
            )
            self.assertIs(hikcam.Camera, hikcam.HikCamera)
            self.assertEqual(FakeFacadeReader.instances, [])
            self.assertEqual(camera.ip, "CAMERA-1")
            self.assertEqual(camera.get_shape(), (2, 3, 3))
            with camera:
                frame = camera.get_frame()
                self.assertEqual(tuple(frame[0, 0]), (30, 20, 10))
                self.assertEqual(camera["Width"], 3)
                with self.assertRaisesRegex(RuntimeError, "locked"):
                    camera["ExposureTime"] = 2500.0
                with self.assertRaisesRegex(RuntimeError, "locked"):
                    camera["Gain"] = 4.0
                self.assertEqual(camera.get_exposure(), 1000.0)
                self.assertEqual(camera.get_gain(), 2.0)
                self.assertEqual(camera["BlackLevel"], 240)
                with self.assertRaisesRegex(RuntimeError, "locked"):
                    camera["BlackLevel"] = 120
                camera["BalanceRatioSelector"] = "Blue"
                with self.assertRaisesRegex(RuntimeError, "locked"):
                    camera["BalanceRatio"] = 1200
                self.assertEqual(camera["BalanceRatio"], 1000)
                camera["AcquisitionFrameRate"] = 25.0
                self.assertEqual(camera["AcquisitionFrameRate"], 25.0)
                camera["TriggerMode"] = "Off"
                with self.assertRaisesRegex(RuntimeError, "requires"):
                    camera["ExposureAuto"] = "Continuous"
                camera.set_bgr()
                self.assertEqual(tuple(camera.get_frame()[0, 0]), (10, 20, 30))
                with self.assertRaisesRegex(RuntimeError, "immutable"):
                    camera["Width"] = 4
            self.assertTrue(FakeFacadeReader.instances[0].released)

    def test_hikcam_rectify_false_selects_zero_transform_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._config_path(directory)
            reader = FakeFacadeReader(path)
            with mock.patch.object(
                hikcam, "RectifiedHikCamera", return_value=reader
            ) as factory:
                camera = hikcam.HikCamera(
                    config={
                        "diagnostic_calibration_override": path,
                        "rectify": False,
                    }
                )
                self.assertEqual(camera.get_shape(), (8, 8, 3))
                camera.open()
            factory.assert_called_once_with(
                path.resolve(),
                rectify=False,
                output_quarter_turns_clockwise=0,
            )
            camera.close()

    def test_legacy_implicit_path_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._config_path(directory)
            with self.assertRaisesRegex(ValueError, "obsolete"):
                hikcam.HikCamera(
                    str(path), config={"reader_factory": FakeFacadeReader}
                )
            with self.assertRaisesRegex(ValueError, "obsolete"):
                hikcam.HikCamera(
                    config={"calibration": path, "reader_factory": FakeFacadeReader}
                )

    def test_explicitly_named_diagnostic_override_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._config_path(directory)
            camera = hikcam.HikCamera(
                config={
                    "diagnostic_calibration_override": path,
                    "reader_factory": FakeFacadeReader,
                }
            )
            self.assertEqual(camera.ip, "CAMERA-1")

    def test_explicit_camera_identifier_must_match_saved_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._config_path(directory)
            with self.assertRaisesRegex(ValueError, "does not match"):
                hikcam.HikCamera(
                    "OTHER-CAMERA",
                    config={
                        "diagnostic_calibration_override": path,
                        "reader_factory": FakeFacadeReader,
                    },
                )

    def test_runtime_orientation_correction_reopens_for_dense_map_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._config_path(directory)
            FakeFacadeReader.instances = []
            camera = hikcam.HikCamera(
                config={
                    "diagnostic_calibration_override": path,
                    "reader_factory": FakeFacadeReader,
                    "color_order": "BGR",
                }
            )
            camera.open()
            with mock.patch(
                "rig_runtime.services.calibration.rig.cross_source."
                "match_game_camera_orientation",
                return_value=(
                    {
                        "status": "selected",
                        "selected_camera_adapter_image_quarter_turns_clockwise_from_calibration_display": 1,
                        "selected_camera_adapter_image_degrees_clockwise_from_calibration_display": 90,
                        "selected_confidence": 0.9,
                        "confidence_margin": 0.4,
                        "preferred_confidence": 0.5,
                        "preferred_margin": 0.08,
                    },
                    {},
                ),
            ):
                result = camera.correct_game_orientation(
                    np.zeros((2, 3, 3), np.uint8),
                    np.zeros((2, 3, 3), np.uint8),
                )
            self.assertTrue(result["applied"])
            self.assertTrue(result["adapter_reopened"])
            self.assertEqual(1, camera._game_upright_turns)
            self.assertEqual(2, len(FakeFacadeReader.instances))
            self.assertTrue(FakeFacadeReader.instances[0].released)
            self.assertTrue(FakeFacadeReader.instances[1].opened)
            camera.close()

    def test_runtime_orientation_correction_rejects_ambiguous_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._config_path(directory)
            FakeFacadeReader.instances = []
            camera = hikcam.HikCamera(
                config={
                    "diagnostic_calibration_override": path,
                    "reader_factory": FakeFacadeReader,
                    "color_order": "BGR",
                }
            )
            camera.open()
            with mock.patch(
                "rig_runtime.services.calibration.rig.cross_source."
                "match_game_camera_orientation",
                return_value=(
                    {
                        "status": "selected_low_confidence",
                        "selected_camera_adapter_image_quarter_turns_clockwise_from_calibration_display": 2,
                        "selected_camera_adapter_image_degrees_clockwise_from_calibration_display": 180,
                        "selected_confidence": 0.3,
                        "confidence_margin": 0.01,
                        "preferred_confidence": 0.5,
                        "preferred_margin": 0.08,
                    },
                    {},
                ),
            ):
                result = camera.correct_game_orientation(
                    np.zeros((2, 3, 3), np.uint8),
                    np.zeros((2, 3, 3), np.uint8),
                )
            self.assertFalse(result["applied"])
            self.assertEqual(0, camera._game_upright_turns)
            self.assertEqual(1, len(FakeFacadeReader.instances))
            camera.close()

    def test_runtime_orientation_request_is_non_blocking_and_retryable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._config_path(directory)
            camera = hikcam.HikCamera(
                config={
                    "diagnostic_calibration_override": path,
                    "reader_factory": FakeFacadeReader,
                    "color_order": "BGR",
                }
            )
            ambiguous = {
                "status": "selected_low_confidence",
                "applied": False,
                "application_status": "not_applied_ambiguous_image_evidence",
            }
            with mock.patch.object(
                camera, "correct_game_orientation", return_value=ambiguous
            ) as correction:
                scheduled = camera.request_game_orientation_correction(
                    np.zeros((2, 3, 3), np.uint8),
                    np.zeros((2, 3, 3), np.uint8),
                )
                self.assertEqual("running", scheduled["status"])
                camera._orientation_job_thread.join(timeout=1.0)
                completed = camera.get_game_orientation_correction()
                self.assertEqual("completed_not_applied", completed["status"])
                self.assertTrue(completed["retryable"])
                retry = camera.request_game_orientation_correction(
                    np.zeros((2, 3, 3), np.uint8),
                    np.zeros((2, 3, 3), np.uint8),
                )
                self.assertTrue(retry["non_blocking"])
                camera._orientation_job_thread.join(timeout=1.0)
            self.assertEqual(2, correction.call_count)

    def test_android_surface_orientation_composes_with_saved_rig_relation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._config_path(directory)
            document = json.loads(path.read_text(encoding="utf-8"))
            document["phone"] = {"orientation_quarter_turns": 3}
            path.write_text(json.dumps(document), encoding="utf-8")
            FakeFacadeReader.instances = []
            camera = hikcam.HikCamera(
                config={
                    "diagnostic_calibration_override": path,
                    "reader_factory": FakeFacadeReader,
                    "color_order": "BGR",
                }
            )
            camera.open()
            result = camera.correct_game_orientation_from_android_surface(
                1, foreground_package="org.example.game"
            )
            self.assertEqual(2, camera._game_upright_turns)
            self.assertEqual(
                "foreground_game_android_surface_and_saved_rig_relation",
                result["selection_basis"],
            )
            self.assertFalse(result["image_evidence_used"])
            self.assertTrue(result["adapter_reopened"])
            self.assertEqual(2, len(FakeFacadeReader.instances))
            camera.close()


if __name__ == "__main__":
    unittest.main()
