import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from acquisition.automated_minimap_calibration import (
    _circle_candidates,
    android_discovery_config,
    calibrate_source_frames,
    compose_rig_game_profile,
    create_current_hik_observation,
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
    def test_android_relative_prior_is_configurable_and_filters_distractors(self):
        source = np.zeros((16, 200, 400, 3), np.uint8)
        candidates = np.asarray(
            [[[90.0, 60.0, 36.0], [30.0, 30.0, 10.0], [300.0, 80.0, 36.0], [5.0, 40.0, 36.0]]],
            np.float32,
        )
        discovery = android_discovery_config(
            center_region_xyxy_fraction="0,0,0.35,0.4",
            radius_fraction_range="0.07,0.22",
            minimum_circle_visible_fraction=0.85,
        )
        with mock.patch(
            "acquisition.automated_minimap_calibration.cv2.HoughCircles",
            return_value=candidates,
        ):
            ranked = _circle_candidates(source, discovery=discovery)
        self.assertEqual(1, len(ranked))
        self.assertAlmostEqual(90.0, ranked[0]["center_x"])
        self.assertAlmostEqual(36.0, ranked[0]["radius"])
        self.assertEqual("stable_disc_boundary", ranked[0]["score_kind"])

    def test_android_relative_prior_validation_rejects_precise_invalid_bounds(self):
        with self.assertRaisesRegex(ValueError, "center region"):
            android_discovery_config(center_region_xyxy_fraction="0,0,1.1,0.5")
        with self.assertRaisesRegex(ValueError, "radius fractions"):
            android_discovery_config(radius_fraction_range="0.22,0.07")

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

    def test_rig_game_composition_does_not_reapply_sensor_matrix_to_normalized_hik(self):
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
                "coordinate_space": {"id": "rig_normalized_hik_phone_pixels"},
                "crop_xywh": [40, 50, 20, 20],
                "outer_boundary": {"source_center_xy": [50, 60], "radius": 10},
            }
            result = compose_rig_game_profile(rig_path, phone, hik)
            self.assertEqual(
                "not_applicable_to_rig_normalized_hik_observation",
                result["cross_source_coordinate_check"]["method"],
            )
            self.assertNotIn(
                "mapped_hik_center_in_phone_xy",
                result["cross_source_coordinate_check"],
            )

    def test_current_hik_observation_is_clipped_and_never_reusable(self):
        hik_frames = np.zeros((16, 100, 100, 3), np.uint8)
        for index in range(len(hik_frames)):
            hik_frames[index, :, :, :] = index
        phone = {
            "outer_boundary": {
                "source_center_xy": [50.0, 40.0],
                "radius": 30.0,
                "confidence": 0.9,
            }
        }
        alignment = {
            "method": "synthetic_translation",
            "android_to_hik_3x3": np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 50.0], [0.0, 0.0, 1.0]]
            ),
            "android_keypoints": 100,
            "hik_keypoints": 120,
            "ratio_match_count": 80,
            "inlier_count": 70,
            "inlier_rate": 0.875,
            "median_inlier_reprojection_px": 0.5,
            "p95_inlier_reprojection_px": 1.2,
            "confidence": 0.95,
            "android_average": np.zeros((100, 100, 3), np.uint8),
            "hik_average": hik_frames.mean(axis=0).astype(np.uint8),
            "match_visualization": np.zeros((100, 200, 3), np.uint8),
        }
        with tempfile.TemporaryDirectory() as temporary:
            result = create_current_hik_observation(
                phone,
                hik_frames,
                alignment,
                Path(temporary),
                image_source="hik_test",
                coordinate_space_id="native_hik_sensor_bgr_pixels",
            )
            self.assertFalse(result["reuse"]["persistent"])
            self.assertTrue(result["visibility"]["clipped_by_source_frame"])
            self.assertEqual([50.0, 90.0], result["outer_boundary"]["source_center_xy"])
            left, top, width, height = result["crop_xywh"]
            self.assertGreater(width, 0)
            self.assertGreater(height, 0)
            self.assertGreaterEqual(left, 0)
            self.assertGreaterEqual(top, 0)
            self.assertLessEqual(left + width, 100)
            self.assertLessEqual(top + height, 100)
            self.assertTrue((Path(temporary) / "mapped_boundary_overlay.png").is_file())
            self.assertTrue((Path(temporary) / "actual_shift_estimation_mask.png").is_file())


if __name__ == "__main__":
    unittest.main()
