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


def write_rig(root: Path) -> Path:
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
                    "screen_size_px": [100, 200],
                    "refresh_hz": 120,
                    "density_dpi": 400,
                },
                "imaging": {
                    "exposure_us": 1000,
                    "gain": 1,
                    "white_balance": {"ratio_red": 1, "ratio_green": 1, "ratio_blue": 1},
                },
                "geometry": {
                    "screen_to_full_sensor_camera_3x3": np.eye(3).tolist(),
                    "full_sensor_camera_to_screen_3x3": np.eye(3).tolist(),
                },
                "normalization": {
                    "output_size_px": [100, 80],
                    "full_sensor_camera_to_output_3x3": np.eye(3).tolist(),
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "last_camera_frame.png").write_bytes(b"snapshot")
    return calibration


class ProfileManagerTests(unittest.TestCase):
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
