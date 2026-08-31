import json
import tempfile
import unittest
from pathlib import Path

from acquisition.profile_registry import AdapterRequest, ProfileContext, ProfileRegistry
from aria_trace.workflows.portable_profiles import (
    export_portable_profile,
    import_portable_profile,
)


def phone_context(game="game-1", phone="PHONE-1", panel=(1080, 2400)):
    width, height = panel
    return ProfileContext(
        game_id=game,
        package="game.package",
        camera_id="CAM-1",
        phone_id=phone,
        phone_model="source handset",
        panel_display={"natural_panel_px": [width, height]},
        game_display={
            "natural_panel_px": [width, height],
            "logical_frame_px": [height, width],
            "game_viewport_xywh": [0, 0, height, width],
            "rotation_quarter_turns": 1,
            "ui_layout_id": "default",
        },
    )


class PortableProfileTests(unittest.TestCase):
    def test_export_import_composes_with_local_rig_and_does_not_activate_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_registry = ProfileRegistry(root / "source-profiles")
            mask = root / "mask.png"
            mask.write_bytes(b"portable-mask")
            source = source_registry.publish(
                "phone_game",
                phone_context(),
                {
                    "profile_kind": "phone_game",
                    "canonical_phone_crop_xywh": [100, 80, 220, 220],
                },
                runtime_files={"shift_estimation_mask": mask},
                review_state="accepted",
                activate=True,
            )
            package = root / "portable.zip"
            export_portable_profile(
                source["revision_id"], package, registry=source_registry
            )

            target_registry = ProfileRegistry(root / "target-profiles")
            calibration = root / "rig.json"
            calibration.write_text(json.dumps({"camera": {}}), encoding="utf-8")
            target_context = phone_context(phone="PHONE-2")
            rig = target_registry.publish(
                "rig",
                target_context,
                {"profile_kind": "rig"},
                runtime_files={"hik_camera_calibration": calibration},
                review_state="accepted",
                activate=True,
            )
            imported = import_portable_profile(
                package,
                registry=target_registry,
                requested_context=target_context,
            )
            self.assertEqual([], imported["compatibility"]["warnings"])
            self.assertIsNotNone(imported["rig_game"])
            self.assertEqual(
                rig["revision_id"],
                imported["rig_game"]["dependencies"]["rig"],
            )
            self.assertEqual(
                imported["portable_profile"]["revision_id"],
                imported["rig_game"]["dependencies"]["phone_game"],
            )
            self.assertEqual(
                "review_required", imported["portable_profile"]["review_state"]
            )
            self.assertEqual(
                "review_required", imported["rig_game"]["review_state"]
            )
            self.assertEqual(
                "PHONE-2",
                imported["portable_profile"]["context"]["devices"]["phone"]["id"],
            )
            self.assertEqual(
                "PHONE-1",
                imported["portable_profile"]["provenance"]["portable_source"][
                    "phone_provenance"
                ]["id"],
            )
            copied = target_registry.runtime_file(
                imported["portable_profile"], "shift_estimation_mask"
            )
            self.assertEqual(b"portable-mask", copied.read_bytes())
            activated = import_portable_profile(
                package,
                registry=target_registry,
                requested_context=target_context,
                activate=True,
            )
            resolved = target_registry.resolve_adapter(
                target_context, AdapterRequest(mode="dual")
            )
            self.assertEqual(
                activated["rig_game"]["revision_id"],
                resolved["profiles"]["rig_game"],
            )
            self.assertEqual(rig["revision_id"], resolved["profiles"]["rig"])

    def test_incompatible_import_is_allowed_and_records_warnings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_registry = ProfileRegistry(root / "source-profiles")
            source = source_registry.publish(
                "phone_game",
                phone_context(),
                {"canonical_phone_crop_xywh": [1, 2, 30, 30]},
                review_state="accepted",
                activate=True,
            )
            package = root / "portable"
            export_portable_profile(
                source["revision_id"], package, registry=source_registry
            )
            target_registry = ProfileRegistry(root / "target-profiles")
            target = phone_context(game="different-game", panel=(720, 1600))
            imported = import_portable_profile(
                package,
                registry=target_registry,
                requested_context=target,
                compose_local_rig=False,
            )
            self.assertEqual(
                "incompatible_override", imported["compatibility"]["status"]
            )
            self.assertGreaterEqual(len(imported["warnings"]), 2)
            stored = imported["portable_profile"]["payload"]["portable_import"]
            self.assertEqual(
                "incompatible_override", stored["compatibility"]["status"]
            )


if __name__ == "__main__":
    unittest.main()
