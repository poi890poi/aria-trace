import json
import tempfile
import unittest
from pathlib import Path

from acquisition.detect_game_display_dim import _active_rig_calibration, parser
from acquisition.profile_registry import ProfileContext, ProfileRegistry


class AutoConfigPolicyTests(unittest.TestCase):
    def test_wake_helper_resolves_exact_active_registry_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "arbitrary-artifact" / "hik_camera_calibration.json"
            source.parent.mkdir()
            source.write_text(json.dumps({"camera": {"device_id": "CAM-1"}}), encoding="utf-8")
            registry = ProfileRegistry(root / "profiles")
            profile = registry.publish(
                "rig",
                ProfileContext(
                    camera_id="CAM-1",
                    phone_id="PHONE-1",
                    panel_display={"natural_panel_px": [1080, 2400]},
                ),
                {"profile_kind": "rig"},
                runtime_files={"hik_camera_calibration": source},
                review_state="accepted",
                activate=True,
            )
            selected = _active_rig_calibration(
                root / "profiles", camera_id="CAM-1", phone_serial="PHONE-1"
            )
            self.assertEqual(
                ProfileRegistry.runtime_file(profile, "hik_camera_calibration").resolve(),
                selected,
            )
            self.assertNotEqual(source.resolve(), selected)

    def test_wake_helper_no_longer_accepts_positional_calibration_path(self):
        with self.assertRaises(SystemExit):
            parser().parse_args(["artifact/hik_camera_calibration.json"])


if __name__ == "__main__":
    unittest.main()
