import json
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
import acquisition.rig_calibration.hik.camera as hikcam
from acquisition.rig_calibration.hik.driver import (
    HikMvsCameraAdapter,
    MvsPythonBackend,
    RectifiedHikCamera,
    create_camera_adapter,
)
from acquisition.rig_calibration.hik.display import AdbDisplayTarget
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
from acquisition.rig_calibration.hik.workflow import (
    HikCalibrationOptions,
    HikRigCalibrationSession,
    screen_filling_charuco_layout,
)
from acquisition.rig_calibration.geometry import estimate_screen_geometry


class HikAlgorithmTests(unittest.TestCase):
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
        self.assertFalse(cannot_qualify(0, 1, 1000, 0.999))
        self.assertTrue(cannot_qualify(0, 2, 1000, 0.999))
        self.assertFalse(cannot_qualify(999, 1000, 1000, 0.999))

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

    def test_phone_charuco_layout_fills_portrait_and_landscape(self):
        portrait = screen_filling_charuco_layout((1080, 2400))
        landscape = screen_filling_charuco_layout((2400, 1080))
        self.assertEqual((portrait.squares_x, portrait.squares_y), (9, 20))
        self.assertEqual(portrait.board_size_px, (1080, 2400))
        self.assertEqual((landscape.squares_x, landscape.squares_y), (20, 9))
        self.assertEqual(landscape.board_size_px, (2400, 1080))

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

    def test_default_hik_options_allow_two_complete_refresh_periods(self):
        with tempfile.TemporaryDirectory() as directory:
            options = HikCalibrationOptions(
                "fake", "phone", Path(directory) / "output"
            )
        self.assertEqual(options.maximum_exposure_periods, 2)
        self.assertAlmostEqual(
            refresh_quantized_exposure_us(
                120.0, 1.0 / options.maximum_exposure_periods
            ),
            16666.666667,
            places=3,
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
                return FrameSample(image, 1, receive_time_ns=1, source_id="fake")

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

    def test_visible_region_is_wholly_inside_skewed_footprint(self):
        polygon = np.asarray([[25, 10], [190, 35], [160, 190], [5, 150]], np.float32)
        result = camera_visible_screen_region(polygon, (200, 200), margin_px=3)
        x, y, width, height = result["xywh"]
        corners = [(x, y), (x + width - 1, y), (x + width - 1, y + height - 1), (x, y + height - 1)]
        contour = polygon.reshape((-1, 1, 2))
        self.assertTrue(all(cv2.pointPolygonTest(contour, point, False) >= 0 for point in corners))
        self.assertGreater(width, 32)
        self.assertGreater(height, 32)

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
        self.assertAlmostEqual(
            pose["lens_to_panel_distance_mm"], expected_distance, delta=1.0
        )
        self.assertAlmostEqual(pose["pitch_deg"], expected_pitch, delta=0.2)
        self.assertAlmostEqual(pose["yaw_deg"], expected_yaw, delta=0.2)
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

    def test_hik_one_shot_auto_limits_allow_refresh_safe_exposure_and_full_gain(self):
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
        result = adapter.configure_once_auto_limits(8333.333)
        self.assertEqual(result["exposure_upper_us"], 8333)
        self.assertAlmostEqual(result["gain_upper"], 16.9806995)

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
        if args[:4] == ["shell", "settings", "get", "system"] or args[:4] == ["shell", "settings", "get", "global"]:
            return self.settings.get((args[3], args[4])) or "null"
        if args[:4] == ["shell", "settings", "put", "system"] or args[:4] == ["shell", "settings", "put", "global"]:
            self.settings[(args[3], args[4])] = args[5]
        if args[:4] == ["shell", "settings", "delete", "system"] or args[:4] == ["shell", "settings", "delete", "global"]:
            self.settings[(args[3], args[4])] = None
        return ""


class HikPhoneTests(unittest.TestCase):
    def test_subprocess_runner_decodes_adb_output_as_utf8_without_locale_dependency(self):
        completed = mock.Mock(returncode=0, stdout="display � text", stderr="")
        with mock.patch("subprocess.run", return_value=completed) as run:
            self.assertEqual(_subprocess_runner(["adb", "devices"], 1.0), "display � text")
        self.assertEqual(run.call_args[1]["encoding"], "utf-8")
        self.assertEqual(run.call_args[1]["errors"], "replace")

    def test_adb_discovery_and_connected_device_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            adb = Path(directory) / "adb.exe"
            adb.write_bytes(b"")
            self.assertEqual(resolve_adb_executable(str(adb)), str(adb.resolve()))
            output = "List of devices attached\nPHONE-1\tdevice\nPHONE-2\toffline\n"
            with mock.patch(
                "acquisition.rig_calibration.hik.phone._subprocess_runner",
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
        self.assertFalse(runner.display_on)

    def test_physical_display_is_verified_after_wakeup(self):
        runner = FakeAdbRunner()
        phone = AdbPhoneSession(
            "SERIAL-1", runner=runner, sleeper=lambda _seconds: None
        )
        phone.wake_and_hold_display()
        evidence = phone.ensure_display_on()
        self.assertEqual(evidence["state"], "ON")
        self.assertTrue(runner.display_on)
        phone.cleanup(turn_display_off=True)

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
        return FrameSample(self.image.copy(), 1, receive_time_ns=1, source_id="fake")

    def close(self):
        self.closed = True

    def align_roi(self, roi):
        return list(map(int, roi))


class HikRectifiedStreamTests(unittest.TestCase):
    def test_data_matrix_high_failure_rate_skips_each_size_after_two_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            options = HikCalibrationOptions(
                "fake",
                "phone",
                Path(directory) / "output",
                data_matrix_trials_per_size=1000,
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
            session.visible_region = {"xywh": [10, 10, 80, 80]}
            session._wait_painted = lambda _shown: None
            session._capture_settled = lambda: np.zeros((80, 80, 3), np.uint8)
            rendered = mock.Mock(
                image=np.zeros((100, 100, 3), np.uint8),
                trial_id="dm-test",
            )
            failed_grade = {
                "grade": 0.0,
                "grade_letter": "F",
                "exact_payload_decoded": False,
            }
            with mock.patch(
                "acquisition.rig_calibration.hik.workflow.render_data_matrix_target",
                return_value=rendered,
            ), mock.patch(
                "acquisition.rig_calibration.hik.workflow.grade_data_matrix_decode",
                return_value=failed_grade,
            ):
                result = session.grade_data_matrix()
        self.assertGreater(len(result["per_size"]), 1)
        self.assertTrue(all(row["early_rejected"] for row in result["per_size"]))
        self.assertTrue(all(row["trial_count"] == 2 for row in result["per_size"]))

    def test_optional_data_matrix_decoder_failure_returns_unavailable_result(self):
        with tempfile.TemporaryDirectory() as directory:
            options = HikCalibrationOptions(
                "fake",
                "phone",
                Path(directory) / "output",
                data_matrix_trials_per_size=1,
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
            session.visible_region = {"xywh": [10, 10, 80, 80]}
            session._wait_painted = lambda _shown: None
            session._capture_settled = lambda: np.zeros((80, 80, 3), np.uint8)
            rendered = mock.Mock(
                image=np.zeros((100, 100, 3), np.uint8),
                trial_id="dm-test",
            )
            with mock.patch(
                "acquisition.rig_calibration.hik.workflow.render_data_matrix_target",
                return_value=rendered,
            ), mock.patch(
                "acquisition.rig_calibration.hik.workflow.grade_data_matrix_decode",
                side_effect=RuntimeError("decoder unavailable"),
            ):
                result = session.grade_data_matrix()
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["failed_trial_index"], 0)
        self.assertIn("decoder unavailable", result["error"])
        self.assertFalse(session._preview_disabled)
        self.assertIn("Data Matrix grading", session._preview_stage)

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
            saved = session.save()
            self.assertEqual(saved, output.resolve())
            self.assertTrue((saved / "calibration.yaml").is_file())
            self.assertTrue((saved / "rectification_maps.npz").is_file())
            config = json.loads((saved / "hik_camera_calibration.json").read_text(encoding="utf-8"))
            self.assertEqual(config["normalization"]["output_size_px"], session.visible_region["xywh"][2:])
            self.assertEqual(
                config["normalization"]["orientation"]["source"],
                "charuco_correspondences",
            )
            self.assertEqual(
                config["normalization"]["orientation"]["adapter_output_up"], "app_up"
            )
            self.assertEqual(config["camera"]["controls"]["gain"]["maximum"], 24.0)
            self.assertEqual(config["imaging"]["black_level"], 240)


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
                config={"calibration": path, "reader_factory": FakeFacadeReader}
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

    def test_calibration_can_be_first_argument_or_environment_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._config_path(directory)
            first_argument = hikcam.HikCamera(
                str(path), config={"reader_factory": FakeFacadeReader}
            )
            self.assertEqual(first_argument.ip, "CAMERA-1")
            with mock.patch.dict("os.environ", {"ARIA_HIK_CALIBRATION": str(path)}):
                self.assertEqual(hikcam.HikCamera.get_all_ips(), ["CAMERA-1"])
                default = hikcam.HikCamera(config={"reader_factory": FakeFacadeReader})
                self.assertEqual(default.ip, "CAMERA-1")

    def test_explicit_camera_identifier_must_match_saved_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._config_path(directory)
            with self.assertRaisesRegex(ValueError, "does not match"):
                hikcam.HikCamera(
                    "OTHER-CAMERA",
                    config={"calibration": path, "reader_factory": FakeFacadeReader},
                )


if __name__ == "__main__":
    unittest.main()
