import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from acquisition.profile_registry import (
    AdapterRequest,
    ProfileContext,
    ProfileRegistry,
    ProfileResolutionError,
)
from aria_trace.adapters.hik import compat as hikcam
from aria_trace.adapters.hik import game_camera


class HikAutoProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.registry = ProfileRegistry(self.root / "profiles")
        self.calibration = self.root / "hik_camera_calibration.json"
        self.calibration.write_text(
            json.dumps(
                {
                    "camera": {
                        "device_id": "CAM-1",
                        "full_sensor_mode": {"width_px": 8, "height_px": 8, "fps": 30},
                        "hardware_roi_xywh": [0, 0, 8, 8],
                    },
                    "phone": {"serial": "PHONE-1"},
                    "imaging": {
                        "exposure_us": 1000, "gain": 1,
                        "white_balance": {"ratio_red": 1, "ratio_green": 1, "ratio_blue": 1},
                    },
                    "normalization": {
                        "output_size_px": [8, 8],
                        "full_sensor_camera_to_output_3x3": np.eye(3).tolist(),
                    },
                }
            ),
            encoding="utf-8",
        )
        self.context = ProfileContext(
            game_id="game-1", camera_id="CAM-1", phone_id="PHONE-1",
            panel_display={"natural_panel_px": [100, 200], "refresh_hz": 120},
            game_display={
                "logical_frame_px": [200, 100],
                "game_viewport_xywh": [0, 0, 200, 100],
                "rotation_quarter_turns": 1,
            },
        )
        self.rig = self.registry.publish(
            "rig", self.context, {"profile_kind": "rig"},
            runtime_files={"hik_camera_calibration": self.calibration},
            review_state="accepted", activate=True,
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_full_mode_resolves_rig_once_without_phone_operations(self):
        reader = Mock()
        reader.open.return_value = reader
        reader.is_calibrated.return_value = True
        with patch.object(hikcam, "RectifiedHikCamera", return_value=reader):
            camera = hikcam.HikCamera(
                config={
                    "profile_root": self.root / "profiles",
                    "camera_id": "CAM-1",
                    "phone_id": "PHONE-1",
                    "panel_display": self.context.panel_display,
                    "mode": "full",
                }
            )
            camera.open()
        self.assertEqual(self.rig["revision_id"], camera.resolved_config["profiles"]["rig"])
        self.assertEqual("none", camera.resolved_config["adapter_plan"]["phone_operations"])
        self.assertEqual(0, camera.resolved_config["adapter_plan"]["registry_reads_per_frame"])

    def test_dual_mode_resolves_exact_rig_game_and_passes_runtime_options(self):
        phone_game = self.registry.publish(
            "phone_game", self.context,
            {"canonical_phone_crop_xywh": [10, 20, 30, 30]},
            review_state="accepted", activate=True,
        )
        rig_game = self.registry.publish(
            "rig_game", self.context,
            {
                "canonical_phone_crop_xywh": [10, 20, 30, 30],
                "hik_bayer_conversion": {"status": "selected", "gamma": 0.8},
            },
            dependencies={
                "rig": self.rig["revision_id"],
                "phone_game": phone_game["revision_id"],
            },
            review_state="accepted", activate=True,
        )
        profiled = Mock()
        profiled.open.return_value = profiled
        with patch.object(game_camera, "ProfiledHikGameCamera", return_value=profiled) as constructor:
            camera = hikcam.HikCamera(
                config={
                    "profile_root": self.root / "profiles",
                    "game_id": "game-1", "camera_id": "CAM-1", "phone_id": "PHONE-1",
                    "panel_display": self.context.panel_display,
                    "game_display": self.context.game_display,
                    "mode": "dual", "color_order": "BGR",
                    "color_policy": "rig_locked", "minimap_margin_px": 12,
                }
            )
            camera.open()
        self.assertEqual(rig_game["revision_id"], camera.resolved_config["profiles"]["rig_game"])
        constructor.assert_called_once_with(
            camera.calibration_path,
            camera.config["minimap_calibration"],
            mode="dual",
            rectify_minimap=True,
                minimap_margin_px=12,
                apply_game_color=False,
                output_quarter_turns_clockwise=0,
                mask_policy="none",
            )

    def test_adapter_rejects_rig_game_pinned_to_superseded_active_rig(self):
        phone_game = self.registry.publish(
            "phone_game",
            self.context,
            {"canonical_phone_crop_xywh": [10, 20, 30, 30]},
            review_state="accepted",
            activate=True,
        )
        self.registry.publish(
            "rig_game",
            self.context,
            {"canonical_phone_crop_xywh": [10, 20, 30, 30]},
            dependencies={
                "rig": self.rig["revision_id"],
                "phone_game": phone_game["revision_id"],
            },
            review_state="accepted",
            activate=True,
        )
        replacement = self.registry.publish(
            "rig",
            self.context,
            {"profile_kind": "rig", "generation": 2},
            runtime_files={"hik_camera_calibration": self.calibration},
            review_state="accepted",
            activate=True,
        )

        full = self.registry.resolve_adapter(
            self.context, AdapterRequest(mode="full")
        )
        self.assertEqual(replacement["revision_id"], full["profiles"]["rig"])
        with self.assertRaisesRegex(
            ProfileResolutionError, "stale.*superseded active rig"
        ):
            self.registry.resolve_adapter(
                self.context,
                AdapterRequest(mode="dual", color_policy="rig_locked"),
            )

    def test_facade_rejects_transform_policy_it_cannot_enforce(self):
        with self.assertRaisesRegex(ValueError, "not implemented"):
            hikcam.HikCamera(
                config={
                    "profile_root": self.root / "profiles",
                    "camera_id": "CAM-1",
                    "phone_id": "PHONE-1",
                    "panel_display": self.context.panel_display,
                    "mode": "full",
                    "normalization": "homography",
                }
            )


if __name__ == "__main__":
    unittest.main()
