import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from aria_trace.adapters.rig.devices import CameraDevice

from aria_trace.apps import hik_stream as stream


class HikStreamCliTests(unittest.TestCase):
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
        self.assertEqual(
            Path("profile.json"),
            arguments.diagnostic_rig_game_profile_override,
        )

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
