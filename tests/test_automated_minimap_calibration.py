import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from acquisition.automated_minimap_calibration import (
    calibrate_source_frames,
    compose_rig_game_profile,
    logical_crop_to_natural,
)


def frames(count=48):
    height, width = 180, 220
    center = (112, 83)
    radius = 69
    yy, xx = np.ogrid[:height, :width]
    inside = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius ** 2
    values = []
    for index in range(count):
        image = np.full((height, width, 3), 20, np.uint8)
        phase = (np.sin(xx * 0.10 + index * 0.22) + np.cos(yy * 0.12 - index * 0.16) + 2.0) * 38.0
        for channel, offset in enumerate((25, 40, 55)):
            layer = np.clip(offset + phase, 0, 255).astype(np.uint8)
            image[:, :, channel][inside] = layer[inside]
        cv2.circle(image, center, radius, (225, 225, 225), 2, cv2.LINE_AA)
        values.append(image)
    return np.stack(values)


class AutomatedMinimapCalibrationTests(unittest.TestCase):
    def test_android_result_has_commented_yaml_and_exact_shift_mask(self):
        orientation = {
            "quarter_turns_clockwise_from_natural": 0,
            "natural_size_px": [220, 180],
        }
        with tempfile.TemporaryDirectory() as temporary:
            result = calibrate_source_frames(
                frames(),
                Path(temporary),
                image_source="android_scrcpy",
                coordinate_space_id="android_logical_display_pixels",
                expected_circle_xy_radius=[112, 83, 69],
                phone_surface_orientation=orientation,
            )
            self.assertEqual(result["crop_xywh"], result["canonical_phone_crop_xywh"])
            self.assertTrue((Path(temporary) / "actual_shift_estimation_mask.png").is_file())
            yaml_text = (Path(temporary) / "minimap_calibration.yaml").read_text(encoding="utf-8")
            self.assertIn("# Authority for every source-space crop", yaml_text)
            self.assertNotIn("cursor", result["scope"]["includes"])

    def test_logical_crop_rotation_matches_adapter_coordinate_convention(self):
        self.assertEqual(
            [10, 75, 15, 20],
            logical_crop_to_natural(
                [5, 10, 20, 15],
                {
                    "quarter_turns_clockwise_from_natural": 1,
                    "natural_size_px": [80, 100],
                },
            ),
        )

    def test_rig_game_composition_uses_saved_matrix_without_fitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rig_path = root / "hik_camera_calibration.json"
            rig_path.write_text(
                json.dumps(
                    {
                        "camera": {"device_id": "camera-1"},
                        "phone": {"serial": "phone-1"},
                        "geometry": {
                            "full_sensor_camera_to_screen_3x3": [
                                [2, 0, 10], [0, 2, 20], [0, 0, 1]
                            ]
                        },
                    }
                ),
                encoding="utf-8",
            )
            phone = {
                "canonical_phone_crop_xywh": [90, 120, 40, 40],
                "outer_boundary": {
                    "canonical_phone_center_xy": [110, 140],
                    "radius": 18,
                },
            }
            hik = {
                "crop_xywh": [40, 50, 20, 20],
                "outer_boundary": {"source_center_xy": [50, 60], "radius": 10},
            }
            result = compose_rig_game_profile(rig_path, phone, hik)
            self.assertEqual([110.0, 140.0], result["cross_source_coordinate_check"]["mapped_hik_center_in_phone_xy"])
            self.assertEqual(
                "apply_saved_rig_homography_only_no_fitting",
                result["cross_source_coordinate_check"]["method"],
            )
            self.assertIn("No optical transform is fitted", result["composition_rule"])


if __name__ == "__main__":
    unittest.main()
