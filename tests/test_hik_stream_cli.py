import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from aria_trace.adapters.rig.devices import CameraDevice

from aria_trace.apps import hik_stream as stream


class HikStreamCliTests(unittest.TestCase):
    def test_wait_for_game_surface_requires_stable_foreground_rotation(self):
        phone = Mock()
        phone.capture_surface.side_effect = [
            {"quarter_turns_clockwise_from_natural": 0, "logical_size_px": [1080, 2400]},
            {"quarter_turns_clockwise_from_natural": 1, "logical_size_px": [2400, 1080]},
            {"quarter_turns_clockwise_from_natural": 1, "logical_size_px": [2400, 1080]},
            {"quarter_turns_clockwise_from_natural": 1, "logical_size_px": [2400, 1080]},
        ]
        with patch.object(
            stream,
            "foreground_component",
            return_value="org.example.game/.MainActivity",
        ), patch.object(stream.time, "sleep"):
            result = stream.wait_for_foreground_game_surface(
                phone, "org.example.game", stable_probes=3
            )
        self.assertEqual(1, result["surface"]["quarter_turns_clockwise_from_natural"])
        self.assertEqual("org.example.game", result["foreground_package"])

    def test_live_telemetry_reports_measured_fps_read_time_and_same_clock_age(self):
        telemetry = stream.LiveStreamTelemetry(history=4)
        telemetry.observe(
            0,
            10_000_000,
            {
                "host_capture_time_ns": 9_000_000,
                "host_timestamp_clock_id": "host_perf_counter_ns",
            },
        )
        telemetry.observe(
            30_000_000,
            40_000_000,
            {
                "host_capture_time_ns": 38_000_000,
                "host_timestamp_clock_id": "host_perf_counter_ns",
            },
        )
        self.assertAlmostEqual(33.333, telemetry.fps, places=2)
        self.assertEqual(10.0, telemetry.read_latency_ms)
        self.assertEqual(2.0, telemetry.frame_age_ms)
        self.assertIn("FPS 33.3", telemetry.label())
        self.assertIn("read 10.0 ms", telemetry.label())
        self.assertIn("age 2.0 ms", telemetry.label())

    def test_live_telemetry_does_not_mix_unknown_timestamp_clocks(self):
        telemetry = stream.LiveStreamTelemetry()
        telemetry.observe(
            0,
            10_000_000,
            {"host_capture_time_ns": 9_000_000, "host_timestamp_clock_id": "device"},
        )
        self.assertIsNone(telemetry.frame_age_ms)
        self.assertIn("age n/a", telemetry.label())

    def test_telemetry_overlay_preserves_input_frame(self):
        telemetry = stream.LiveStreamTelemetry()
        frame = np.zeros((80, 320, 3), np.uint8)
        rendered = stream.overlay_stream_telemetry(frame, telemetry)
        self.assertEqual(frame.shape, rendered.shape)
        self.assertFalse(np.shares_memory(frame, rendered))
        self.assertEqual(0, int(frame.max()))
        self.assertGreater(int(rendered.max()), 0)

    def test_telemetry_label_is_averaged_and_visually_latched(self):
        telemetry = stream.LiveStreamTelemetry(history=8, display_interval_ms=500)
        telemetry.observe(0, 10_000_000, {})
        telemetry.observe(30_000_000, 40_000_000, {})
        first = telemetry.label()
        telemetry.observe(45_000_000, 50_000_000, {})
        self.assertEqual(first, telemetry.label())
        telemetry.observe(590_000_000, 600_000_000, {})
        self.assertNotEqual(first, telemetry.label())
        self.assertIn("avg", telemetry.label())

    def test_geometry_overlay_draws_only_matching_runtime_space(self):
        frame = np.zeros((80, 100, 3), np.uint8)
        camera = Mock()
        camera.get_minimap_geometry.return_value = {
            "available_in_stream_space": True,
            "center_xy_px": [50.0, 40.0],
            "boundary_size_xy_px": [60.0, 60.0],
            "image_space": {"stored_size_px": [100, 80]},
        }
        camera.get_cursor_geometry.return_value = {
            "available_in_stream_space": True,
            "center_xy_px": [50.0, 40.0],
            "rotating_cursor_envelope_size_xy_px": [12.0, 10.0],
            "image_space": {"stored_size_px": [100, 80]},
        }
        state = stream.GeometryOverlayState()
        rendered = stream.overlay_stream_geometry(frame, camera, "minimap", state)
        self.assertGreater(int(rendered.max()), 0)
        camera.get_minimap_geometry.return_value["image_space"] = {
            "stored_size_px": [99, 80]
        }
        camera.get_cursor_geometry.return_value["image_space"] = {
            "stored_size_px": [99, 80]
        }
        rejected = stream.overlay_stream_geometry(frame, camera, "minimap", state)
        self.assertEqual(0, int(rejected.max()))

    def test_geometry_overlay_runtime_keys_toggle_components(self):
        state = stream.GeometryOverlayState()
        self.assertIn("off", state.handle_key(ord("g")))
        self.assertFalse(state.enabled)
        self.assertIn("boundary off", state.handle_key(ord("b")))
        self.assertTrue(state.enabled)
        self.assertFalse(state.minimap_boundary)
        self.assertIsNone(state.handle_key(ord("x")))

    def test_open_camera_delegates_profiled_modes_to_existing_game_adapter(self):
        adapter = object()
        opened = object()
        profiled = Mock()
        profiled.open.return_value = opened
        with patch.object(stream, "HikMvsCameraAdapter", return_value=adapter), patch.object(
            stream, "ProfiledHikGameCamera", return_value=profiled
        ) as constructor:
            result = stream.open_camera(
                diagnostic_calibration_override=Path("rig.json"),
                diagnostic_rig_game_profile_override=Path("rig-game-profile.json"),
                mode="dual",
            )
        self.assertIs(opened, result)
        constructor.assert_called_once_with(
            Path("rig.json"),
            Path("rig-game-profile.json"),
            mode="dual",
            rectify_minimap=True,
            adapter=adapter,
            mask_policy="none",
        )
        profiled.open.assert_called_once_with()

    def test_non_full_mode_requires_a_rig_game_profile(self):
        with self.assertRaisesRegex(ValueError, "diagnostic-rig-game-profile"):
            stream.open_camera(
                diagnostic_calibration_override=Path("rig.json"), mode="dual"
            )

    def test_parser_accepts_profiled_dual_demo(self):
        arguments = stream.parser().parse_args(
            [
                "--diagnostic-calibration-override",
                "rig.json",
                "--diagnostic-rig-game-profile-override",
                "profile.json",
                "--mode",
                "dual",
                "--gui",
            ]
        )
        self.assertEqual("dual", arguments.mode)
        self.assertEqual("none", arguments.mask_policy)
        self.assertEqual(
            Path("profile.json"),
            arguments.diagnostic_rig_game_profile_override,
        )

    def test_parser_accepts_automatic_game_launch_orientation(self):
        arguments = stream.parser().parse_args(
            ["--gui", "--launch-game", "--game-id", "game-1"]
        )
        self.assertTrue(arguments.launch_game)
        self.assertEqual("game-1", arguments.game_id)

    def test_parser_accepts_precomposed_minimap_mask(self):
        arguments = stream.parser().parse_args(
            [
                "--game-id",
                "game-1",
                "--mode",
                "dual",
                "--mask-policy",
                "minimap_circle",
            ]
        )
        self.assertEqual("minimap_circle", arguments.mask_policy)

    def test_parser_allows_registry_selected_game_without_calibration_path(self):
        arguments = stream.parser().parse_args(
            ["--game-id", "game-1", "--mode", "dual", "--gui"]
        )
        self.assertIsNone(arguments.diagnostic_calibration_override)
        self.assertEqual("game-1", arguments.game_id)
        self.assertEqual("dual", arguments.mode)

    def test_parser_accepts_native_hik_library(self):
        arguments = stream.parser().parse_args(
            ["--camera-library", "native", "--camera-id", "DA9066154", "--gui"]
        )
        self.assertEqual("native", arguments.camera_library)
        self.assertEqual("DA9066154", arguments.camera_id)
        self.assertEqual("full", arguments.mode)

    def test_adapter_hik_class_is_loaded_from_drop_in_module(self):
        adapter_class = object()
        module = Mock(HikCamera=adapter_class)
        with patch.object(stream.importlib, "import_module", return_value=module) as importer:
            self.assertIs(adapter_class, stream.adapter_hik_camera_class())
        importer.assert_called_once_with("hikcam")

    def test_native_hik_source_uses_existing_mvs_adapter_and_full_sensor_source(self):
        adapter = Mock()
        source = Mock()
        with patch.object(
            stream, "HikMvsCameraAdapter", return_value=adapter
        ) as adapter_class, patch.object(
            stream, "NativeHikFrameSource", return_value=source
        ) as source_class:
            self.assertIs(
                source,
                stream.open_native_mvs_source("DA9066154", "MVS/MvImport"),
            )
        adapter_class.assert_called_once_with(sdk_python_path="MVS/MvImport")
        source_class.assert_called_once_with("DA9066154", adapter=adapter)
        source.start.assert_called_once_with()

    def test_native_hik_source_requires_an_unambiguous_mvs_camera(self):
        adapter = Mock()
        adapter.devices.return_value = (
            CameraDevice("camera-1", "HIK 1"),
            CameraDevice("camera-2", "HIK 2"),
        )
        with patch.object(stream, "HikMvsCameraAdapter", return_value=adapter):
            with self.assertRaisesRegex(RuntimeError, "pass --camera-id"):
                stream.open_native_mvs_source(None, None)

    def test_native_hik_source_auto_selects_one_mvs_camera(self):
        adapter = Mock()
        adapter.devices.return_value = (
            CameraDevice("DA9066154", "USB3 MV-CS016-10UC"),
        )
        source = Mock()
        with patch.object(
            stream, "HikMvsCameraAdapter", return_value=adapter
        ), patch.object(
            stream, "NativeHikFrameSource", return_value=source
        ) as source_class:
            self.assertIs(source, stream.open_native_mvs_source(None, None))
        adapter.devices.assert_called_once_with(probe=True)
        source_class.assert_called_once_with("DA9066154", adapter=adapter)
        source.start.assert_called_once_with()

    def test_native_hik_demo_uses_mvs_source_without_adapter_profiles(self):
        source = Mock()
        with patch.object(
            stream, "open_native_mvs_source", return_value=source
        ) as opener:
            self.assertEqual(
                0,
                stream.main(
                    [
                        "--camera-library",
                        "native",
                        "--camera-id",
                        "DA9066154",
                        "--mvs-python-path",
                        "MVS/MvImport",
                    ]
                ),
            )
        opener.assert_called_once_with("DA9066154", "MVS/MvImport")
        source.stop.assert_called_once_with()

    def test_legacy_positional_paths_are_obsolete(self):
        with self.assertRaises(SystemExit):
            stream.parser().parse_args(["rig.json"])


if __name__ == "__main__":
    unittest.main()
