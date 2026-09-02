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

    def test_caller_can_list_and_select_an_explicit_rig_revision(self):
        replacement = self.registry.publish(
            "rig",
            self.context,
            {"profile_kind": "rig", "generation": 2},
            runtime_files={"hik_camera_calibration": self.calibration},
            review_state="accepted",
            activate=True,
        )
        listed = hikcam.HikCamera.list_profiles(
            {
                "profile_root": self.root / "profiles",
                "camera_id": "CAM-1",
                "game_id": "game-1",
                "panel_display": self.context.panel_display,
                "game_display": self.context.game_display,
            },
            kinds=["rig"],
            active_only=False,
        )
        self.assertEqual(
            {self.rig["revision_id"], replacement["revision_id"]},
            {item["revision_id"] for item in listed["rig"]},
        )
        camera = hikcam.HikCamera(
            config={
                "profile_root": self.root / "profiles",
                "camera_id": "CAM-1",
                "game_id": "game-1",
                "panel_display": self.context.panel_display,
                "game_display": self.context.game_display,
                "profile_revisions": {"rig": self.rig["revision_id"]},
                "mode": "full",
                "color_policy": "rig_locked",
            }
        )
        self.assertEqual(
            self.rig["revision_id"], camera.resolved_config["profiles"]["rig"]
        )
        self.assertEqual(
            {"rig": self.rig["revision_id"]},
            camera.resolved_config["manual_profile_revisions"],
        )

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
                runtime_surface_quarter_turns_clockwise_from_natural=None,
                best_effort_initialization=True,
                mask_policy="none",
            )

    def test_initialization_recovery_updates_only_rig_orientation_profile(self):
        phone_game_payload = {"canonical_phone_crop_xywh": [10, 20, 30, 30]}
        phone_game = self.registry.publish(
            "phone_game",
            self.context,
            phone_game_payload,
            review_state="accepted",
            activate=True,
        )
        rig_game_payload = {
            "canonical_phone_crop_xywh": [10, 20, 30, 30],
            "portable_marker": "must-not-change",
        }
        rig_game = self.registry.publish(
            "rig_game",
            self.context,
            rig_game_payload,
            dependencies={
                "rig": self.rig["revision_id"],
                "phone_game": phone_game["revision_id"],
            },
            review_state="accepted",
            activate=True,
        )
        orientation_payload = {
            "profile_kind": "rig_game_orientation",
            "game_surface_quarter_turns_clockwise_from_phone_natural": 1,
            "camera_adapter_image_quarter_turns_clockwise_from_calibration_display": 0,
            "portable_marker": {"game": "unchanged"},
        }
        orientation = self.registry.publish(
            "rig_game_orientation",
            self.context,
            orientation_payload,
            dependencies={"rig": self.rig["revision_id"]},
            review_state="accepted",
            activate=True,
        )
        reader = Mock()
        reader.open.return_value = reader
        reader.initialization_orientation_recovery.return_value = {
            "status": "four_orientation_intersection_fallback",
            "selected_surface_quarter_turns": 3,
            "previous_output_quarter_turns": 0,
            "selected_output_quarter_turns": 2,
            "orientation_recovered": True,
            "runtime_cost": "initialization_only",
        }
        camera = hikcam.HikCamera(
            config={
                "profile_root": self.root / "profiles",
                "game_id": "game-1",
                "camera_id": "CAM-1",
                "phone_id": "PHONE-1",
                "panel_display": self.context.panel_display,
                "game_display": self.context.game_display,
                "mode": "dual",
                "color_policy": "rig_locked",
                "reader_factory": lambda _path: reader,
            }
        ).open()

        recovery = camera.get_iris_initialization_recovery()
        self.assertEqual("recovered_and_profile_updated", recovery["status"])
        active_orientation = self.registry.resolve(
            "rig_game_orientation", self.context
        )
        self.assertNotEqual(orientation["revision_id"], active_orientation["revision_id"])
        self.assertEqual(
            self.rig["revision_id"], active_orientation["dependencies"]["rig"]
        )
        self.assertEqual(
            2,
            active_orientation["payload"][
                "camera_adapter_image_quarter_turns_clockwise_from_calibration_display"
            ],
        )
        self.assertEqual(
            1,
            active_orientation["payload"][
                "game_surface_quarter_turns_clockwise_from_phone_natural"
            ],
        )
        self.assertEqual(
            {"game": "unchanged"},
            active_orientation["payload"]["portable_marker"],
        )
        self.assertEqual(
            phone_game["revision_id"],
            self.registry.resolve("phone_game", self.context)["revision_id"],
        )
        self.assertEqual(
            rig_game["revision_id"],
            self.registry.resolve("rig_game", self.context)["revision_id"],
        )
        self.assertEqual(2, camera.resolved_config["adapter_plan"][
            "game_upright_quarter_turns_clockwise"
        ])
        self.assertEqual(3, camera.resolved_config["adapter_plan"][
            "initialization_surface_quarter_turns_clockwise_from_natural"
        ])
        next_run = self.registry.resolve_adapter(
            self.context,
            AdapterRequest(mode="dual", color_policy="rig_locked"),
        )
        self.assertEqual(
            2,
            next_run["adapter_plan"][
                "game_upright_quarter_turns_clockwise"
            ],
        )
        self.assertEqual(
            3,
            next_run["adapter_plan"][
                "initialization_surface_quarter_turns_clockwise_from_natural"
            ],
        )

    def test_initialization_recovery_and_writeback_can_be_disabled(self):
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
        profiled = Mock()
        profiled.open.return_value = profiled
        with patch.object(
            game_camera, "ProfiledHikGameCamera", return_value=profiled
        ) as constructor:
            camera = hikcam.HikCamera(
                config={
                    "profile_root": self.root / "profiles",
                    "game_id": "game-1",
                    "camera_id": "CAM-1",
                    "phone_id": "PHONE-1",
                    "panel_display": self.context.panel_display,
                    "game_display": self.context.game_display,
                    "mode": "minimap",
                    "color_policy": "rig_locked",
                    "best_effort_initialization": False,
                    "persist_initialization_recovery": False,
                }
            ).open()
        self.assertFalse(constructor.call_args[1]["best_effort_initialization"])
        self.assertEqual({}, camera.get_iris_initialization_recovery())

    def test_initialization_profile_write_failure_is_non_gating(self):
        reader = Mock()
        reader.open.return_value = reader
        reader.initialization_orientation_recovery.return_value = {
            "selected_surface_quarter_turns": 2,
            "previous_output_quarter_turns": 0,
            "selected_output_quarter_turns": 2,
            "orientation_recovered": True,
        }
        camera = hikcam.HikCamera(
            config={
                "diagnostic_calibration_override": self.calibration,
                "reader_factory": lambda _path: reader,
            }
        )
        with patch.object(
            camera,
            "_persist_recovered_orientation",
            side_effect=PermissionError("read-only profile store"),
        ), self.assertWarnsRegex(RuntimeWarning, "could not update"):
            opened = camera.open()
        self.assertIs(camera, opened)
        self.assertTrue(camera.is_open)
        self.assertEqual(
            "recovered_profile_update_failed_non_gating",
            camera.get_iris_initialization_recovery()["status"],
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
