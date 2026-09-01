import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from aria_trace.domain.packets import FramePacket
from aria_trace.domain.spatial import bind_geometry, raster_space
from aria_trace.services.calibration.game_repeatability import (
    compare_thresholded_app_geometry,
    evaluate_minimap_static_geometry,
)
from aria_trace.workflows import game_repeatability as workflow


class GameRepeatabilityGeometryTests(unittest.TestCase):
    @staticmethod
    def boundary(center_x=110, center_y=105, radius=80):
        return bind_geometry(
            {"center_x": center_x, "center_y": center_y, "radius": radius},
            "circle",
            raster_space("current_minimap_crop_pixels", [220, 220]),
        )

    def test_minimap_check_uses_static_rim_and_tolerates_contrast_change(self):
        dark = np.full((220, 220, 3), 25, np.uint8)
        cv2.circle(dark, (110, 105), 80, (110, 110, 110), 3)
        bright = np.clip(dark.astype(np.int16) * 2 + 30, 0, 255).astype(np.uint8)
        result, images = evaluate_minimap_static_geometry(
            bright,
            [0, 0, 220, 220],
            self.boundary(),
        )
        self.assertTrue(result["matches"])
        self.assertGreater(result["angular_coverage"], 0.9)
        self.assertFalse(result["dynamic_game_content_used"])
        self.assertIn("static_geometry_overlay.png", images)

    def test_minimap_check_rejects_missing_static_boundary(self):
        image = np.full((220, 220, 3), 80, np.uint8)
        result, _images = evaluate_minimap_static_geometry(
            image,
            [0, 0, 220, 220],
            self.boundary(),
        )
        self.assertFalse(result["matches"])
        self.assertEqual(0.0, result["angular_coverage"])

    def test_minimap_check_rejects_geometry_without_space(self):
        image = np.full((220, 220, 3), 80, np.uint8)
        with self.assertRaisesRegex(ValueError, "spatial schema"):
            evaluate_minimap_static_geometry(
                image,
                [0, 0, 220, 220],
                {"center_x": 110, "center_y": 105, "radius": 80},
            )

    def test_diagnostic_app_check_compares_fixed_threshold_features(self):
        reference = np.full((160, 240, 3), 20, np.uint8)
        cv2.rectangle(reference, (30, 25), (205, 130), (190, 190, 190), 3)
        current = np.clip(reference.astype(np.int16) + 45, 0, 255).astype(np.uint8)
        result, images = compare_thresholded_app_geometry(
            reference, current, minimum_score=0.4
        )
        self.assertTrue(result["matches"])
        self.assertIn("diagnostic_geometry_overlay.png", images)


class GameRepeatabilityWorkflowTests(unittest.TestCase):
    def test_profile_circle_is_transformed_to_current_crop_space(self):
        profile = {
            "payload": {
                "canonical_phone_crop_xywh": [10, 20, 40, 40],
                "android_logical_crop_xywh": [10, 20, 40, 40],
                "phone_surface_orientation": {
                    "quarter_turns_clockwise_from_natural": 0
                },
                "outer_boundary": {
                    "center_x": 30.0,
                    "center_y": 40.0,
                    "radius": 18.0,
                },
            }
        }
        crop, boundary = workflow._logical_profile_crop(
            profile,
            {
                "natural_size_px": [100, 200],
                "quarter_turns_clockwise_from_natural": 1,
            },
        )
        self.assertEqual([140, 10, 40, 40], crop)
        self.assertAlmostEqual(19.0, boundary["center_x"])
        self.assertAlmostEqual(20.0, boundary["center_y"])
        self.assertEqual(
            "current_minimap_crop_pixels", boundary["space"]["space_id"]
        )
        self.assertEqual("circle", boundary["geometry_type"])

    def test_camera_candidate_uses_inverse_android_surface_convention(self):
        self.assertTrue(
            workflow._camera_candidate_agrees_with_android_surface(0, 0)
        )
        self.assertTrue(
            workflow._camera_candidate_agrees_with_android_surface(3, 1)
        )
        self.assertTrue(
            workflow._camera_candidate_agrees_with_android_surface(2, 2)
        )
        self.assertTrue(
            workflow._camera_candidate_agrees_with_android_surface(1, 3)
        )
        self.assertFalse(
            workflow._camera_candidate_agrees_with_android_surface(1, 1)
        )

    def test_orientation_result_is_applied_without_rig_recalibration(self):
        class Source:
            applied = None

            def set_output_orientation(self, turns, evidence):
                self.applied = (turns, evidence)

        source = Source()
        turns = workflow.apply_orientation_result(
            source,
            {
                "status": "selected",
                "selection_basis": "image_evidence",
                "selected_confidence": 0.9,
                "confidence_margin": 0.2,
                "selected_camera_adapter_image_quarter_turns_clockwise_from_calibration_display": 3,
            },
        )
        self.assertEqual(3, turns)
        self.assertEqual(3, source.applied[0])
        self.assertEqual("image_evidence", source.applied[1]["selection_basis"])

    def test_diagnostic_adb_only_result_is_machine_readable_and_space_aware(self):
        image = np.full((80, 120, 3), 20, np.uint8)
        cv2.rectangle(image, (15, 10), (100, 65), (220, 220, 220), 3)
        image_space = {
            "space_id": "android_phone_natural_display_pixels",
            "canonical_space_id": "android_phone_natural_display_pixels",
            "canonical_size_px": [120, 80],
            "local_to_canonical_3x3": np.eye(3).tolist(),
        }
        packet = FramePacket(
            "android_phone", image.copy(), 1, 2, metadata={"image_space": image_space}
        )

        class FakePhone:
            def __init__(self, *_args, **_kwargs):
                pass

            def orientation_settings(self):
                return {
                    "accelerometer_rotation": 0,
                    "user_rotation_quarter_turns": 0,
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            reference.mkdir()
            cv2.imwrite(str(reference / "adb_current.png"), image)
            (reference / "result.json").write_text(
                json.dumps(
                    {
                        "capture": {"adb_image_space": image_space},
                        "evidence": {"adb_current": "adb_current.png"},
                    }
                ),
                encoding="utf-8",
            )
            output = root / "output"
            surface = {
                "natural_size_px": [120, 80],
                "logical_size_px": [120, 80],
                "quarter_turns_clockwise_from_natural": 0,
                "source": "unit_test",
            }
            with patch.object(
                workflow,
                "load_system_configuration",
                return_value={
                    "effective_profile_root": str(root / "profiles"),
                    "game": {"game_id": None},
                    "devices": {"camera_id": None, "phone_id": "phone-1"},
                    "tools": {"adb": "adb.exe", "mvs_python_path": None},
                },
            ), patch.object(
                workflow, "resolve_adb_executable", return_value="adb.exe"
            ), patch.object(
                workflow,
                "probe_android_capture_surface",
                return_value=("phone-1", surface),
            ), patch.object(
                workflow, "AdbPhoneSession", FakePhone
            ), patch.object(
                workflow, "foreground_component", return_value="com.example/.Main"
            ), patch.object(
                workflow, "_capture_adb_frame", return_value=packet
            ):
                result = workflow.run_game_repeatability_check(
                    output,
                    adb_only=True,
                    expected_package="com.example",
                    diagnostic_reference_result=reference,
                )
            stored = json.loads((output / "result.json").read_text(encoding="utf-8"))
        self.assertEqual("match", result["status"])
        self.assertEqual("com.example", stored["application"]["foreground_package"])
        self.assertEqual(
            "android_phone_natural_display_pixels",
            stored["capture"]["adb_image_space"]["canonical_space_id"],
        )
        self.assertEqual(
            "none", stored["operation"]["app_launch_or_input"]
        )


if __name__ == "__main__":
    unittest.main()
