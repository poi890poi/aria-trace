import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from acquisition.rig_calibration.hik import stream


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
                Path("rig.json"),
                minimap_calibration=Path("rig-game-current.json"),
                mode="dual",
            )
        self.assertIs(opened, result)
        constructor.assert_called_once_with(
            Path("rig.json"),
            Path("rig-game-current.json"),
            mode="dual",
            rectify_minimap=True,
            adapter=adapter,
        )
        profiled.open.assert_called_once_with()

    def test_non_full_mode_requires_a_rig_game_profile(self):
        with self.assertRaisesRegex(ValueError, "requires --minimap-calibration"):
            stream.open_camera(Path("rig.json"), mode="dual")

    def test_parser_accepts_profiled_dual_demo(self):
        arguments = stream.parser().parse_args(
            [
                "rig.json",
                "--minimap-calibration",
                "current.json",
                "--mode",
                "dual",
                "--gui",
            ]
        )
        self.assertEqual("dual", arguments.mode)
        self.assertEqual(Path("current.json"), arguments.minimap_calibration)


if __name__ == "__main__":
    unittest.main()
