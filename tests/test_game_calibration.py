import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from aria_trace.adapters.filesystem.profile_registry import (
    AdapterRequest,
    ProfileContext,
    ProfileRegistry,
)
from aria_trace.services.calibration.minimap.discovery import (
    discover_android_minimap_crop,
)


class GameCalibrationTests(unittest.TestCase):
    def test_bounded_discovery_finds_dynamic_upper_left_circle(self):
        height, width = 360, 640
        frames = []
        rng = np.random.default_rng(4)
        for index in range(16):
            image = np.full((height, width, 3), 30, np.uint8)
            texture = rng.integers(20, 220, (112, 112, 3), dtype=np.uint8)
            mask = np.zeros((112, 112), np.uint8)
            cv2.circle(mask, (56, 56), 52, 255, -1)
            area = image[24:136, 28:140]
            area[mask > 0] = texture[mask > 0]
            cv2.circle(image, (84, 80), 52, (220, 220, 220), 2)
            cv2.putText(image, str(index), (300, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 90), 2)
            frames.append(image)
        result = discover_android_minimap_crop(np.stack(frames))
        center = result["selected_candidate"]["center_xy"]
        self.assertLess(abs(center[0] - 84), 12)
        self.assertLess(abs(center[1] - 80), 12)
        self.assertLess(abs(result["selected_candidate"]["radius_px"] - 52), 14)

    def test_orientation_profile_is_optional_and_resolved_for_full_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = root / "rig.json"
            calibration.write_text(json.dumps({"camera": {}}), encoding="utf-8")
            context = ProfileContext(
                game_id="game",
                camera_id="CAM",
                panel_display={"natural_panel_px": [100, 200]},
                game_display={
                    "logical_frame_px": [200, 100],
                    "rotation_quarter_turns": 1,
                },
            )
            registry = ProfileRegistry(root / "profiles")
            rig = registry.publish(
                "rig", context, {},
                runtime_files={"hik_camera_calibration": calibration},
                review_state="accepted", activate=True,
            )
            orientation = registry.publish(
                "rig_game_orientation",
                context,
                {"camera_adapter_image_quarter_turns_clockwise_from_calibration_display": 3},
                dependencies={"rig": rig["revision_id"]},
                review_state="accepted", activate=True,
            )
            resolved = registry.resolve_adapter(context, AdapterRequest(mode="full"))
            self.assertEqual(
                orientation["revision_id"],
                resolved["profiles"]["rig_game_orientation"],
            )
            self.assertEqual(
                3,
                resolved["adapter_plan"][
                    "game_upright_quarter_turns_clockwise"
                ],
            )


if __name__ == "__main__":
    unittest.main()
