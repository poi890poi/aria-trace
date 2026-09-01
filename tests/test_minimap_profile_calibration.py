import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aria_trace.workflows.minimap_profile_calibration import main


class MinimapProfileCalibrationTests(unittest.TestCase):
    def test_cli_derives_output_and_publishes_through_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = root / "profiles"
            published = {
                "phone_game": {"revision_id": "phone-game-1"},
                "rig_game": None,
            }
            with patch(
                "aria_trace.workflows.minimap_profile_calibration.calibrate_session"
            ) as calibrate, patch(
                "aria_trace.workflows.minimap_profile_calibration.publish_minimap_profiles",
                return_value=published,
            ) as publish:
                self.assertEqual(
                    0,
                    main(
                        [
                            str(root / "session"),
                            "--profile-root",
                            str(profiles),
                            "--game-id",
                            "game-1",
                            "--rotation",
                            "0",
                            "2",
                            "--movement",
                            "3",
                            "5",
                        ]
                    ),
                )
            calibrate_positional, _ = calibrate.call_args
            publish_positional, publish_keyword = publish.call_args
            output = Path(calibrate_positional[1])
            self.assertEqual(profiles / "calibrations" / "minimap", output.parent)
            self.assertTrue(output.name.startswith("game-1-"))
            self.assertEqual(output / "calibration.json", publish_positional[0])
            self.assertTrue(publish_keyword["activate"])
            self.assertEqual("game-1", publish_keyword["game_id"])
            self.assertIsNone(publish_keyword["camera_id"])


if __name__ == "__main__":
    unittest.main()
