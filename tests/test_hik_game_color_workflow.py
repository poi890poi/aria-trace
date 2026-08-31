import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from aria_trace.adapters.filesystem.profile_registry import (
    ProfileContext,
    ProfileRegistry,
    context_from_rig_calibration,
)
from aria_trace.workflows.hik_game_color_calibration import (
    calibrate_game_color_session,
)
from aria_trace.workflows.profile_management import publish_rig_calibration


class HikGameColorWorkflowTests(unittest.TestCase):
    def test_session_color_calibration_publishes_exact_rig_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rig = root / "rig" / "hik_camera_calibration.json"
            rig.parent.mkdir()
            rig_document = {
                "camera": {
                    "adapter_id": "hik_mvs",
                    "device_id": "CAM-1",
                    "full_sensor_mode": {"width_px": 8, "height_px": 8, "fps": 30},
                    "hardware_roi_xywh": [0, 0, 8, 8],
                },
                "phone": {
                    "serial": "PHONE-1",
                    "model": "phone",
                    "natural_screen_size_px": [8, 16],
                    "screen_size_px": [8, 16],
                    "refresh_hz": 120,
                },
                "imaging": {
                    "exposure_us": 1000,
                    "gain": 1,
                    "white_balance": {
                        "ratio_red": 1024,
                        "ratio_green": 1024,
                        "ratio_blue": 1024,
                    },
                },
                "normalization": {
                    "output_size_px": [8, 8],
                    "full_sensor_camera_to_output_3x3": [
                        [1, 0, 0], [0, 1, 0], [0, 0, 1]
                    ],
                },
            }
            rig.write_text(json.dumps(rig_document), encoding="utf-8")
            registry = ProfileRegistry(root / "profiles")
            rig_profile = publish_rig_calibration(rig, registry=registry)

            session = root / "session"
            session.mkdir()
            manifest = {
                "schema_version": "1.0",
                "status": "complete",
                "context": {
                    "game_id": "game-1",
                    "game_launch": {"game_id": "game-1", "package": "game.package"},
                    "hik_capture": {"rig_calibration": str(rig)},
                    "phone_surface_orientation": {
                        "quarter_turns_clockwise_from_natural": 1,
                        "logical_size_px": [16, 8],
                        "natural_size_px": [8, 16],
                    },
                },
                "videos": {
                    "android_phone": "android.mkv",
                    "hik_phone": "hik.mkv",
                },
            }
            (session / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            rows = []
            for index in range(4):
                rows.append({
                    "stream_id": "android_phone", "frame_index": index,
                    "host_capture_time_ns": 1000 + index * 100,
                    "session_time_ns": index * 100, "width": 16, "height": 8,
                })
                rows.append({
                    "stream_id": "hik_phone", "frame_index": index,
                    "host_capture_time_ns": 1010 + index * 100,
                    "session_time_ns": 10 + index * 100, "width": 8, "height": 8,
                })
            (session / "frames.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            (session / "inputs.jsonl").write_text("", encoding="utf-8")
            (session / "coordinate_spaces.yaml").write_text(
                yaml.safe_dump({
                    "conversions": {"adb_to_hik_phone_video_3x3": np.eye(3).tolist()},
                    "streams": {"hik_phone": {"content_size_px": [8, 8]}},
                }),
                encoding="utf-8",
            )
            frames = np.full((4, 8, 8, 3), 100, np.uint8)
            conversion = {
                "status": "selected",
                "gamma": 0.8,
                "ccm_rgb_3x3": np.eye(3).tolist(),
                "fit": {
                    "baseline_validation": {"rgb_mae_dn": 10.0},
                    "selected_validation": {"rgb_mae_dn": 4.0},
                },
            }
            with patch(
                "aria_trace.workflows.hik_game_color_calibration._decode_indices",
                return_value=frames,
            ), patch(
                "aria_trace.workflows.hik_game_color_calibration.optimize_mvs_bayer_conversion",
                return_value=(conversion, {"review.png": frames[0]}),
            ):
                result = calibrate_game_color_session(
                    session,
                    root / "output",
                    profile_root=root / "profiles",
                )
            profile = registry.revision(result["profile_revision"])
            portable_color = registry.revision(
                result["portable_phone_game_color_revision"]
            )
            self.assertEqual(
                rig_profile["revision_id"], profile["dependencies"]["rig"]
            )
            self.assertEqual(
                portable_color["revision_id"],
                profile["dependencies"]["phone_game_color"],
            )
            self.assertEqual(
                "phone_game_color", portable_color["identity"]["kind"]
            )
            self.assertTrue(
                registry.runtime_file(
                    portable_color, "adb_game_color_reference"
                ).is_file()
            )
            self.assertTrue((root / "output" / "hikcam_adapter.py").is_file())


if __name__ == "__main__":
    unittest.main()
