import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from aria_trace.adapters.filesystem.profile_registry import ProfileContext, ProfileRegistry
from aria_trace.adapters.filesystem.system_configuration import (
    default_system_configuration,
    load_system_configuration,
    resolve_rig_repeatability_policy,
    save_system_configuration,
)
from aria_trace.workflows.system_setup import main as setup_main


class SystemConfigurationTests(unittest.TestCase):
    def test_default_repeatability_policy_relaxes_geometry_reuse_not_save_protection(self):
        policy = resolve_rig_repeatability_policy(default_system_configuration())
        self.assertEqual("relaxed", policy["name"])
        self.assertEqual(16.0, policy["reuse_max_displacement_px"])
        self.assertEqual(12.0, policy["save_max_displacement_px"])
        self.assertEqual(3, policy["save_movement_consecutive_frames"])

    def test_environment_profile_root_owns_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "shared-profiles"
            with mock.patch.dict(os.environ, {"ARIA_PROFILE_ROOT": str(root)}):
                configured = load_system_configuration()
                self.assertEqual(str(root.resolve()), configured["effective_profile_root"])
                self.assertEqual("ARIA_PROFILE_ROOT", configured["profile_root_source"])
                configured["devices"]["camera_id"] = "CAM-7"
                configured["rig_calibration"]["repeatability_policy"] = "balanced"
                save_system_configuration(configured)
                loaded = load_system_configuration()
            self.assertEqual("CAM-7", loaded["devices"]["camera_id"])
            self.assertEqual(
                "balanced", loaded["rig_calibration"]["repeatability_policy"]
            )
            self.assertEqual(
                8.0,
                resolve_rig_repeatability_policy(loaded)[
                    "reuse_max_displacement_px"
                ],
            )
            self.assertTrue((root / ".registry" / "settings.json").is_file())
            self.assertTrue((root / ".registry" / "settings.yaml").is_file())

    def test_setup_cli_configures_defaults_and_lists_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "profiles"
            self.assertEqual(
                0,
                setup_main(
                    [
                        "--profile-root",
                        str(root),
                        "configure",
                        "--camera-id",
                        "CAM-1",
                        "--phone-id",
                        "PHONE-1",
                        "--rig-repeatability",
                        "strict",
                    ]
                ),
            )
            configured = load_system_configuration(root)
            self.assertEqual("CAM-1", configured["devices"]["camera_id"])
            self.assertEqual(
                "strict", configured["rig_calibration"]["repeatability_policy"]
            )
            self.assertEqual(0, setup_main(["--profile-root", str(root), "profiles"]))

    def test_registry_lists_active_immutable_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "profiles"
            calibration = Path(directory) / "calibration.json"
            calibration.write_text(json.dumps({"ok": True}), encoding="utf-8")
            registry = ProfileRegistry(root)
            profile = registry.publish(
                "rig",
                ProfileContext(
                    camera_id="CAM-1",
                    phone_id="PHONE-1",
                    panel_display={
                        "natural_panel_px": [64, 48],
                        "logical_frame_px": [64, 48],
                    },
                ),
                {"profile_kind": "rig"},
                runtime_files={"hik_camera_calibration": calibration},
                review_state="accepted",
                activate=True,
            )
            rows = registry.list_revisions(kind="rig", active_only=True)
            self.assertEqual(1, len(rows))
            self.assertEqual(profile["revision_id"], rows[0]["revision_id"])
            self.assertTrue(rows[0]["active"])


if __name__ == "__main__":
    unittest.main()
