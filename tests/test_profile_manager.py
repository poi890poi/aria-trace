import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from acquisition.profile_manager import (
    publish_minimap_profiles,
    publish_rig_calibration,
)
from acquisition.profile_registry import AdapterRequest, ProfileContext, ProfileRegistry
from rig_runtime.services.calibration.minimap.spatial import minimap_crop_space


def write_rig(
    root: Path,
    *,
    camera_x_offset: float = 0.0,
    calibration_display_turns: int = 0,
) -> Path:
    root.mkdir()
    calibration = root / "hik_camera_calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "camera": {
                    "adapter_id": "hik_mvs",
                    "device_id": "CAM-1",
                    "full_sensor_mode": {"width_px": 100, "height_px": 80, "fps": 30},
                    "hardware_roi_xywh": [0, 0, 100, 80],
                },
                "phone": {
                    "serial": "PHONE-1",
                    "model": "phone",
                    "natural_screen_size_px": [100, 200],
                    "screen_size_px": (
                        [200, 100]
                        if int(calibration_display_turns) % 2 else [100, 200]
                    ),
                    "orientation_quarter_turns": int(
                        calibration_display_turns
                    ) % 4,
                    "refresh_hz": 120,
                    "density_dpi": 400,
                },
                "imaging": {
                    "exposure_us": 1000,
                    "gain": 1,
                    "white_balance": {"ratio_red": 1, "ratio_green": 1, "ratio_blue": 1},
                },
                "geometry": {
                    "screen_to_full_sensor_camera_3x3": np.asarray(
                        [[1, 0, camera_x_offset], [0, 1, 0], [0, 0, 1]],
                        dtype=float,
                    ).tolist(),
                    "full_sensor_camera_to_screen_3x3": np.asarray(
                        [[1, 0, -camera_x_offset], [0, 1, 0], [0, 0, 1]],
                        dtype=float,
                    ).tolist(),
                },
                "normalization": {
                    "output_size_px": [100, 80],
                    "full_sensor_camera_to_output_3x3": np.eye(3).tolist(),
                    "valid_mask_file": "valid_screen_mask.png",
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "last_camera_frame.png").write_bytes(b"snapshot")
    (root / "valid_screen_mask.png").write_bytes(b"mask")
    return calibration


class ProfileManagerTests(unittest.TestCase):
    def test_verified_calibration_document_publishes_through_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ProfileRegistry(root / "profiles")
            rig_profile = publish_rig_calibration(
                write_rig(root / "rig"), registry=registry
            )
            session = root / "session"
            session.mkdir()
            manifest = {
                "session_id": "image-series-session",
                "frame_sources": [
                    {
                        "stream_id": "android_phone",
                        "serial": "PHONE-1",
                        "preferred_frame_storage": "image_series",
                    }
                ],
                "context": {
                    "game_id": "game-1",
                    "phone_surface_orientation": {
                        "quarter_turns_clockwise_from_natural": 1,
                        "logical_size_px": [200, 100],
                        "natural_size_px": [100, 200],
                    },
                },
            }
            (session / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            output = root / "calibration"
            output.mkdir()
            (output / "model.npz").write_bytes(b"model")
            (output / "boundary.png").write_bytes(b"evidence")
            space = minimap_crop_space(
                [40, 40],
                parent_space_id="android_logical_display_pixels",
                crop_xywh=[10, 5, 40, 40],
            )
            calibration = {
                "schema_version": "2.0",
                "status": "review_required",
                "provenance": {"session_path": str(session)},
                "config": {"crop_xywh": [10, 5, 40, 40]},
                "outer_boundary": {
                    "spatial_schema_version": 1,
                    "geometry_type": "circle",
                    "space": space,
                    "center_x": 20.0,
                    "center_y": 20.0,
                    "radius": 18.0,
                },
                "rotation_center": {
                    "spatial_schema_version": 1,
                    "geometry_type": "point",
                    "space": space,
                    "x": 20.0,
                    "y": 20.0,
                },
                "model_file": "model.npz",
                "evidence": [{"name": "boundary.png"}],
            }
            calibration_path = output / "calibration.json"
            calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
            published = publish_minimap_profiles(
                calibration_path,
                registry=registry,
                camera_id="CAM-1",
                activate=True,
            )
            phone = published["phone_game"]
            self.assertEqual("accepted", phone["review_state"])
            self.assertEqual(
                "minimap_model.npz",
                phone["runtime_files"]["minimap_model"]["path"].split("/")[-1],
            )
            resolved = registry.resolve(
                "phone_game",
                ProfileContext.from_dict(phone["context"]),
            )
            self.assertEqual(phone["revision_id"], resolved["revision_id"])
            self.assertEqual(
                rig_profile["revision_id"],
                published["rig_game"]["dependencies"]["rig"],
            )
            self.assertEqual(
                phone["revision_id"],
                published["rig_game"]["dependencies"]["phone_game"],
            )
            self.assertTrue(phone["payload"]["capabilities"]["cursor_rotation_center"])
            self.assertFalse(phone["payload"]["capabilities"]["cursor_shape"])
            self.assertEqual(
                "rotation_center_only",
                phone["payload"]["capabilities"]["cursor_result_level"],
            )
            self.assertEqual(
                phone["payload"]["rotation_center"],
                phone["payload"]["cursor_geometry"]["rotation_center"],
            )

    def test_rig_publication_copies_runtime_files_and_activates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ProfileRegistry(root / "profiles")
            profile = publish_rig_calibration(write_rig(root / "rig"), registry=registry)
            resolved = registry.resolve(
                "rig",
                ProfileContext(
                    camera_id="CAM-1", phone_id="PHONE-1",
                    panel_display={
                        "natural_panel_px": [100, 200],
                        "logical_frame_px": [100, 200],
                        "refresh_hz": 120, "density_dpi": 400,
                    },
                ),
            )
            self.assertEqual(profile["revision_id"], resolved["revision_id"])
            self.assertTrue(registry.runtime_file(resolved, "hik_camera_calibration").is_file())
            self.assertTrue(registry.runtime_file(resolved, "last_camera_frame").is_file())
            mask = registry.runtime_file(resolved, "valid_screen_mask")
            self.assertTrue(mask.is_file())
            calibration = json.loads(
                registry.runtime_file(resolved, "hik_camera_calibration").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                mask,
                registry.runtime_file(resolved, "hik_camera_calibration").parent
                / calibration["normalization"]["valid_mask_file"],
            )

    def test_new_rig_recomposes_active_phone_game_without_game_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ProfileRegistry(root / "profiles")
            first_rig = publish_rig_calibration(
                write_rig(root / "rig-1"), registry=registry
            )
            phone_context = ProfileContext(
                game_id="game-1",
                camera_id="CAM-1",
                phone_id="PHONE-1",
                panel_display={
                    "natural_panel_px": [100, 200],
                },
                game_display={
                    "logical_frame_px": [100, 200],
                    "rotation_quarter_turns": 0,
                },
            )
            registry.publish(
                "phone_game",
                ProfileContext(
                    game_id="game-1",
                    camera_id="CAM-1",
                    phone_id="PHONE-1",
                    panel_display={
                        "natural_panel_px": [100, 200],
                        "logical_frame_px": [100, 200],
                        "refresh_hz": 120,
                        "density_dpi": 400,
                    },
                    game_display=phone_context.game_display,
                ),
                {
                    "profile_kind": "phone_game",
                    "canonical_phone_crop_xywh": [1, 2, 30, 32],
                },
                review_state="accepted",
                activate=True,
            )
            phone_game = registry.publish(
                "phone_game",
                phone_context,
                {
                    "profile_kind": "phone_game",
                    "canonical_phone_crop_xywh": [5, 6, 40, 42],
                    "outer_boundary": {"center_x": 25, "center_y": 27, "radius": 18},
                    "rotation_center": {"x": 24, "y": 26},
                },
                review_state="accepted",
                activate=True,
            )
            old_composition = registry.publish(
                "rig_game",
                phone_context,
                {"profile_kind": "rig_game", "canonical_phone_crop_xywh": [5, 6, 40, 42]},
                dependencies={
                    "rig": first_rig["revision_id"],
                    "phone_game": phone_game["revision_id"],
                },
                review_state="accepted",
                activate=True,
            )
            old_orientation = registry.publish(
                "rig_game_orientation",
                phone_context,
                {
                    "profile_kind": "rig_game_orientation",
                    "camera_adapter_image_quarter_turns_clockwise_from_calibration_display": 3,
                },
                dependencies={"rig": first_rig["revision_id"]},
                review_state="accepted",
                activate=True,
            )
            old_color = registry.publish(
                "rig_game_color",
                phone_context,
                {
                    "profile_kind": "rig_game_color",
                    "hik_bayer_conversion": {
                        "status": "selected",
                        "gamma": 0.9,
                        "ccm_rgb_3x3": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                    },
                },
                dependencies={"rig": first_rig["revision_id"]},
                review_state="accepted",
                activate=True,
            )

            second_rig = publish_rig_calibration(
                write_rig(root / "rig-2", camera_x_offset=3.0),
                registry=registry,
            )

            self.assertNotEqual(first_rig["revision_id"], second_rig["revision_id"])
            self.assertEqual(1, len(second_rig["recomposed_rig_game_profiles"]))
            recomposed = second_rig["recomposed_rig_game_profiles"][0]
            self.assertNotEqual(old_composition["revision_id"], recomposed["revision_id"])
            self.assertEqual(
                second_rig["revision_id"], recomposed["dependencies"]["rig"]
            )
            self.assertEqual(
                phone_game["revision_id"], recomposed["dependencies"]["phone_game"]
            )
            self.assertEqual(
                phone_game["payload"]["rotation_center"],
                recomposed["payload"]["rotation_center"],
            )
            active = registry.resolve(
                "rig_game",
                ProfileContext(game_id="game-1", camera_id="CAM-1"),
            )
            self.assertEqual(recomposed["revision_id"], active["revision_id"])
            self.assertEqual(
                1, len(second_rig["recomposed_rig_game_orientation_profiles"])
            )
            recomposed_orientation = second_rig[
                "recomposed_rig_game_orientation_profiles"
            ][0]
            self.assertNotEqual(
                old_orientation["revision_id"], recomposed_orientation["revision_id"]
            )
            self.assertEqual(
                second_rig["revision_id"],
                recomposed_orientation["dependencies"]["rig"],
            )
            stale_color = second_rig["rig_dependent_reconciliation"][
                "requires_fresh_evidence"
            ]["rig_game_color"]
            self.assertEqual(1, len(stale_color))
            self.assertEqual(
                old_color["revision_id"], stale_color[0]["profile_revision"]
            )
            with self.assertWarnsRegex(RuntimeWarning, "rig-locked color"):
                resolved = registry.resolve_adapter(
                    ProfileContext(
                        game_id="game-1",
                        camera_id="CAM-1",
                        game_display=phone_context.game_display,
                    ),
                    AdapterRequest(mode="minimap", color_policy="game_matched"),
                )
            self.assertEqual(
                3,
                resolved["adapter_plan"][
                    "game_upright_quarter_turns_clockwise"
                ],
            )
            self.assertEqual("rig_locked", resolved["adapter_plan"]["color_policy"])
            self.assertIsNone(resolved["profiles"]["rig_game_color"])
            self.assertTrue(resolved["compatibility"]["warnings"])

    def test_orientation_recomposition_changes_relative_turn_for_new_rig_display(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ProfileRegistry(root / "profiles")
            first_rig = publish_rig_calibration(
                write_rig(root / "rig-1", calibration_display_turns=1),
                registry=registry,
            )
            context = ProfileContext(
                game_id="game-1",
                camera_id="CAM-1",
                phone_id="PHONE-1",
                panel_display={"natural_panel_px": [100, 200]},
                game_display={
                    "logical_frame_px": [100, 200],
                    "rotation_quarter_turns": 0,
                },
            )
            registry.publish(
                "rig_game_orientation",
                context,
                {
                    "profile_kind": "rig_game_orientation",
                    # Legacy payload: game surface 0 minus old rig display 1.
                    "camera_adapter_image_quarter_turns_clockwise_from_calibration_display": 3,
                },
                dependencies={"rig": first_rig["revision_id"]},
                review_state="accepted",
                activate=True,
            )

            second_rig = publish_rig_calibration(
                write_rig(root / "rig-2", calibration_display_turns=0),
                registry=registry,
            )
            recomposed = second_rig[
                "recomposed_rig_game_orientation_profiles"
            ][0]
            self.assertEqual(
                0,
                recomposed["payload"][
                    "game_surface_quarter_turns_clockwise_from_phone_natural"
                ],
            )
            self.assertEqual(
                0,
                recomposed["payload"][
                    "camera_adapter_image_quarter_turns_clockwise_from_calibration_display"
                ],
            )
            resolved = registry.resolve_adapter(
                ProfileContext(
                    game_id="game-1",
                    camera_id="CAM-1",
                    game_display=context.game_display,
                ),
                AdapterRequest(mode="full"),
            )
            self.assertEqual(
                0,
                resolved["adapter_plan"]["game_upright_quarter_turns_clockwise"],
            )
            self.assertEqual(
                "derived_from_legacy_source_rig_and_relative_turn",
                recomposed["provenance"]["portable_orientation_basis"],
            )

    def test_orientation_recomposition_preserves_relative_turn_when_rigs_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ProfileRegistry(root / "profiles")
            first_rig = publish_rig_calibration(
                write_rig(root / "rig-1", calibration_display_turns=1),
                registry=registry,
            )
            context = ProfileContext(
                game_id="game-1",
                camera_id="CAM-1",
                phone_id="PHONE-1",
                panel_display={"natural_panel_px": [100, 200]},
                game_display={"logical_frame_px": [200, 100]},
            )
            registry.publish(
                "rig_game_orientation",
                context,
                {
                    "profile_kind": "rig_game_orientation",
                    "game_surface_quarter_turns_clockwise_from_phone_natural": 0,
                    "camera_adapter_image_quarter_turns_clockwise_from_calibration_display": 3,
                },
                dependencies={"rig": first_rig["revision_id"]},
                review_state="accepted",
                activate=True,
            )

            second_rig = publish_rig_calibration(
                write_rig(root / "rig-2", calibration_display_turns=1),
                registry=registry,
            )
            recomposed = second_rig[
                "recomposed_rig_game_orientation_profiles"
            ][0]
            self.assertEqual(
                3,
                recomposed["payload"][
                    "camera_adapter_image_quarter_turns_clockwise_from_calibration_display"
                ],
            )
            self.assertEqual(
                "stored_portable_game_surface_orientation",
                recomposed["provenance"]["portable_orientation_basis"],
            )

    def test_localization_publishes_display_variant_candidates_then_resolves_when_activated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ProfileRegistry(root / "profiles")
            rig = write_rig(root / "rig")
            session = root / "session"
            session.mkdir()
            manifest = {
                "session_id": "s1",
                "frame_sources": [{
                    "stream_id": "android_phone",
                    "shared_capture": {"serial": "PHONE-1"},
                }],
                "context": {
                    "game_id": "game-1",
                    "game_launch": {"game_id": "game-1", "package": "game.package"},
                    "phone_surface_orientation": {
                        "quarter_turns_clockwise_from_natural": 1,
                        "logical_size_px": [200, 100],
                        "natural_size_px": [100, 200],
                    },
                    "hik_capture": {"rig_calibration": str(rig)},
                },
            }
            (session / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "localization"
            output.mkdir()
            summary = {
                "status": "review_required",
                "provenance": {"session_path": str(session)},
                "android": {
                    "frame_size_px": [200, 100],
                    "outer_boundary": {"center_x": 30, "center_y": 25, "radius": 20},
                    "shift_estimation_mask": "android/mask.png",
                    "verified_backend_evidence": ["android/overlay.png"],
                },
                "hik_bayer_conversion": {"status": "selected", "gamma": 0.8},
                "hik_session_observation": {"status": "available"},
            }
            summary_path = output / "localization_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            published = publish_minimap_profiles(summary_path, registry=registry)
            phone_payload = published["phone_game"]["payload"]
            self.assertEqual(
                "android_phone_natural_display_pixels",
                phone_payload["outer_boundary"]["space"]["space_id"],
            )
            self.assertEqual(
                "circle", phone_payload["outer_boundary"]["geometry_type"]
            )
            self.assertEqual("review_required", published["phone_game"]["review_state"])
            with self.assertRaises(Exception):
                registry.resolve_adapter(
                    ProfileContext(game_id="game-1", camera_id="CAM-1", phone_id="PHONE-1"),
                    AdapterRequest(mode="dual"),
                )
            registry.activate(published["phone_game"]["revision_id"])
            registry.activate(published["rig_game"]["revision_id"])
            resolved = registry.resolve_adapter(
                ProfileContext(game_id="game-1", camera_id="CAM-1", phone_id="PHONE-1"),
                AdapterRequest(mode="dual"),
            )
            self.assertEqual(published["rig_game"]["revision_id"], resolved["profiles"]["rig_game"])


if __name__ == "__main__":
    unittest.main()
