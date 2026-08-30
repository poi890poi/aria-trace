import unittest
from pathlib import Path
from unittest.mock import Mock, patch

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

    def test_legacy_positional_paths_are_obsolete(self):
        with self.assertRaises(SystemExit):
            stream.parser().parse_args(["rig.json"])


if __name__ == "__main__":
    unittest.main()
