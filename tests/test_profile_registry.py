import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import aria_trace.adapters.filesystem.profile_registry as profile_registry_module

from acquisition.profile_registry import (
    AdapterRequest,
    ProfileContext,
    ProfileRegistry,
    default_profile_root,
)


def context(width=2400, refresh=120.0, game="genshin-impact"):
    return ProfileContext(
        game_id=game,
        camera_id="CAM-1",
        phone_id="PHONE-1",
        panel_display={
            "natural_panel_px": [1080, 2400],
            "logical_frame_px": [1080, 2400],
            "refresh_hz": refresh,
            "density_dpi": 480,
        },
        game_display={
            "natural_panel_px": [1080, 2400],
            "logical_frame_px": [width, 1080],
            "game_viewport_xywh": [0, 0, width, 1080],
            "rotation_quarter_turns": 1,
            "ui_layout_id": "default",
        },
    )


class ProfileRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.registry = ProfileRegistry(self.root / "profiles")
        self.calibration = self.root / "hik_camera_calibration.json"
        self.calibration.write_text(json.dumps({"camera": {}}), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def publish_rig(self, value=None, activate=True):
        return self.registry.publish(
            "rig",
            value or context(),
            {"kind": "rig"},
            runtime_files={"hik_camera_calibration": self.calibration},
            review_state="accepted",
            activate=activate,
        )

    def test_local_working_directory_is_profile_root_fallback(self):
        local = self.root / "application"
        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch.object(profile_registry_module.Path, "cwd", return_value=local):
                self.assertEqual((local / "profiles").resolve(), default_profile_root())

    def test_profile_root_environment_has_priority_over_local_directory(self):
        configured = self.root / "configured-profiles"
        local = self.root / "application"
        with mock.patch.dict("os.environ", {"IRIS_PROFILE_ROOT": str(configured)}):
            with mock.patch.object(profile_registry_module.Path, "cwd", return_value=local):
                self.assertEqual(configured.resolve(), default_profile_root())

    def test_explicit_profile_root_has_priority_over_environment(self):
        explicit = self.root / "explicit-profiles"
        configured = self.root / "configured-profiles"
        with mock.patch.dict("os.environ", {"IRIS_PROFILE_ROOT": str(configured)}):
            self.assertEqual(explicit.resolve(), default_profile_root(explicit))

    def test_display_dimensions_and_refresh_are_compatibility_dimensions(self):
        base = context()
        self.assertNotEqual(base.game_display_signature, context(width=1920).game_display_signature)
        self.assertNotEqual(base.panel_signature, context(refresh=60.0).panel_signature)

    def test_candidate_publication_does_not_replace_active_profile(self):
        active = self.publish_rig()
        candidate = self.registry.publish(
            "rig",
            context(),
            {"kind": "rig", "value": 2},
            runtime_files={"hik_camera_calibration": self.calibration},
        )
        resolved = self.registry.resolve("rig", context())
        self.assertEqual(active["revision_id"], resolved["revision_id"])
        self.assertNotEqual(candidate["revision_id"], resolved["revision_id"])

    def test_activation_is_atomic_and_supports_compare_and_swap(self):
        first = self.publish_rig()
        second = self.registry.publish(
            "rig",
            context(),
            {"kind": "rig", "value": 2},
            runtime_files={"hik_camera_calibration": self.calibration},
        )
        with self.assertRaisesRegex(RuntimeError, "activation conflict"):
            self.registry.activate(second["revision_id"], expected_current="wrong")
        self.registry.activate(second["revision_id"], expected_current=first["revision_id"])
        self.assertEqual(second["revision_id"], self.registry.resolve("rig", context())["revision_id"])

    def test_unchanged_publication_reuses_existing_revision(self):
        first = self.publish_rig()
        repeated = self.publish_rig()
        self.assertEqual(first["revision_id"], repeated["revision_id"])
        self.assertEqual("unchanged_existing_revision", repeated["publication"])

    def test_incomplete_context_uses_newest_active_variant(self):
        first = self.publish_rig(context())
        newest = self.publish_rig(context(refresh=60.0))
        incomplete = ProfileContext(camera_id="CAM-1", phone_id="PHONE-1")
        resolved = self.registry.resolve("rig", incomplete)
        self.assertNotEqual(first["revision_id"], resolved["revision_id"])
        self.assertEqual(newest["revision_id"], resolved["revision_id"])
        self.assertEqual(
            "newest_active_compatible_revision",
            resolved["resolution"]["selection"],
        )

    def test_phone_game_is_owned_by_platform_panel_and_game_not_phone_serial(self):
        source = context()
        profile = self.registry.publish(
            "phone_game",
            source,
            {"canonical_phone_crop_xywh": [10, 20, 30, 30]},
            review_state="accepted",
            activate=True,
        )
        other_phone = ProfileContext(
            game_id=source.game_id,
            camera_id="OTHER-CAMERA",
            phone_id="PHONE-2",
            phone_model="another handset",
            panel_display=source.panel_display,
            game_display=source.game_display,
        )
        resolved = self.registry.resolve("phone_game", other_phone)
        self.assertEqual(profile["revision_id"], resolved["revision_id"])
        compatibility = resolved["resolution"]["compatibility"]
        self.assertEqual([], compatibility["warnings"])
        self.assertTrue(compatibility["provenance_notes"])

    def test_game_model_is_game_scoped_and_resolved_into_adapter_plan(self):
        rig = self.publish_rig()
        model = self.registry.publish(
            "game_model",
            ProfileContext(game_id="genshin-impact"),
            {
                "cursor_follows": "camera",
                "cursor_behavior_by_acquisition": {
                    "zigzag": "rotating",
                    "micro_movement": "static",
                },
                "minimap_orientation": "rotating",
            },
            review_state="accepted",
            activate=True,
        )
        resolved = self.registry.resolve_adapter(context(), AdapterRequest(mode="full"))
        self.assertEqual(rig["revision_id"], resolved["profiles"]["rig"])
        self.assertEqual(model["revision_id"], resolved["profiles"]["game_model"])
        self.assertEqual(
            "rotating",
            resolved["adapter_plan"]["game_model"][
                "cursor_behavior_by_acquisition"
            ]["zigzag"],
        )

    def test_mask_policy_requires_rectified_minimap_output(self):
        with self.assertRaisesRegex(ValueError, "masking requires"):
            AdapterRequest(
                mode="minimap", normalization="none", mask_policy="minimap_circle"
            )
        with self.assertRaisesRegex(ValueError, "minimap or dual"):
            AdapterRequest(mode="full", mask_policy="minimap_circle")

    def test_explicit_incompatible_revision_warns_but_is_returned(self):
        profile = self.registry.publish(
            "phone_game",
            context(),
            {"canonical_phone_crop_xywh": [10, 20, 30, 30]},
            review_state="accepted",
            activate=True,
        )
        requested = context(width=1920, game="another-game")
        selected = self.registry.resolve_revision(
            profile["revision_id"], requested, expected_kind="phone_game"
        )
        compatibility = selected["resolution"]["compatibility"]
        self.assertEqual("incompatible_override", compatibility["status"])
        self.assertGreaterEqual(len(compatibility["warnings"]), 2)

    def test_adapter_mode_controls_dependencies_not_profile_identity(self):
        rig = self.publish_rig()
        phone_game = self.registry.publish(
            "phone_game",
            context(),
            {"canonical_phone_crop_xywh": [10, 20, 30, 30]},
            review_state="accepted",
            activate=True,
        )
        rig_game = self.registry.publish(
            "rig_game",
            context(),
            {"canonical_phone_crop_xywh": [10, 20, 30, 30]},
            dependencies={
                "rig": rig["revision_id"],
                "phone_game": phone_game["revision_id"],
            },
            review_state="accepted",
            activate=True,
        )
        full = self.registry.resolve_adapter(context(), AdapterRequest(mode="full"))
        dual = self.registry.resolve_adapter(context(), AdapterRequest(mode="dual"))
        self.assertIsNone(full["profiles"]["rig_game"])
        self.assertEqual(rig_game["revision_id"], dual["profiles"]["rig_game"])
        self.assertEqual(0, dual["adapter_plan"]["registry_reads_per_frame"])
        self.assertEqual("none", dual["adapter_plan"]["phone_operations"])

    def test_game_color_profile_is_independent_of_minimap_geometry(self):
        rig = self.publish_rig()
        color = self.registry.publish(
            "rig_game_color",
            context(),
            {
                "hik_bayer_conversion": {
                    "status": "selected",
                    "gamma": 0.8,
                    "ccm_rgb_3x3": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                }
            },
            dependencies={"rig": rig["revision_id"]},
            review_state="accepted",
            activate=True,
        )
        resolved = self.registry.resolve_adapter(
            context(),
            AdapterRequest(mode="full", color_policy="game_matched"),
        )
        self.assertEqual(
            color["revision_id"], resolved["profiles"]["rig_game_color"]
        )
        self.assertIsNone(resolved["profiles"]["rig_game"])
        self.assertEqual("game_matched", resolved["adapter_plan"]["color_policy"])
        self.assertTrue(Path(resolved["paths"]["game_color_profile"]).is_file())

    def test_auto_color_falls_back_to_locked_rig_without_game_profile(self):
        self.publish_rig()
        resolved = self.registry.resolve_adapter(
            context(), AdapterRequest(mode="full", color_policy="auto")
        )
        self.assertEqual("rig_locked", resolved["adapter_plan"]["color_policy"])
        self.assertIsNone(resolved["paths"]["game_color_profile"])

    def test_options_do_not_change_selected_revision(self):
        rig = self.publish_rig()
        first = self.registry.resolve_adapter(
            context(), AdapterRequest(mode="full", color_order="RGB")
        )
        second = self.registry.resolve_adapter(
            context(), AdapterRequest(
                mode="full", color_order="BGR", normalization="none",
                minimap_margin_px=20,
            )
        )
        self.assertEqual(rig["revision_id"], first["profiles"]["rig"])
        self.assertEqual(rig["revision_id"], second["profiles"]["rig"])
        self.assertTrue(first["adapter_plan"]["rectify"])
        self.assertEqual("auto", first["adapter_plan"]["normalization"])
        self.assertFalse(second["adapter_plan"]["rectify"])
        self.assertEqual("none", second["adapter_plan"]["normalization"])


if __name__ == "__main__":
    unittest.main()
