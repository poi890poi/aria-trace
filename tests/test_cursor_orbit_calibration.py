import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from rig_runtime.adapters.android.cursor_orbit import CursorOrbitTouchPlan
from rig_runtime.domain.spatial import bind_geometry, raster_space
from rig_runtime.services.calibration.minimap.calibration import (
    calibrate_cursor_orbit_frames,
    calibrate_cursor_static_frames,
)
from rig_runtime.workflows.profile_management import cursor_behavior_by_acquisition
from rig_runtime.workflows.minimap_capture import _build_control_plan, parser


class CursorOrbitCalibrationTests(unittest.TestCase):
    def test_game_model_separates_acquisition_pattern_from_cursor_behavior(self):
        self.assertEqual(
            {"zigzag": "static", "micro_movement": "rotating"},
            cursor_behavior_by_acquisition("character"),
        )
        self.assertEqual(
            {"zigzag": "rotating", "micro_movement": "static"},
            cursor_behavior_by_acquisition("camera"),
        )

    def test_micro_movement_is_the_nonsemantic_public_capture_name(self):
        arguments = parser().parse_args(["--capture-mode", "micro-movement"])
        plan = _build_control_plan(arguments, 2400, 1080)
        self.assertEqual("balanced_micro_movement", plan.as_dict()["plan_kind"])

    def test_capture_defaults_use_short_pulses_and_twelve_directions(self):
        arguments = parser().parse_args(["--capture-mode", "cursor-orbit"])
        plan = _build_control_plan(arguments, 2400, 1080)
        self.assertEqual([432, 842], plan.center_xy)
        self.assertEqual(65, plan.radius_px)
        self.assertEqual(12, plan.direction_count)
        self.assertEqual(2, plan.repeats)
        self.assertAlmostEqual(0.12, plan.step_seconds)

    def test_touch_plan_pairs_opposite_short_pulses(self):
        plan = CursorOrbitTouchPlan(
            center_xy=[432, 842],
            radius_px=65,
            direction_count=12,
            repeats=2,
            step_seconds=0.12,
            settle_seconds=0.0,
            reset_seconds=0.0,
        )
        strokes = plan.strokes()
        self.assertEqual(24, len(strokes))
        for first, second in zip(strokes[0::2], strokes[1::2]):
            endpoint_sum = np.asarray(first["end_xy"]) + np.asarray(second["end_xy"])
            np.testing.assert_allclose(endpoint_sum, np.asarray([864, 1684]), atol=1)
            self.assertEqual([432, 842], first["start_xy"])
            self.assertEqual([432, 842], second["start_xy"])

    def test_calibrates_cursor_center_and_shape_without_refitting_boundary(self):
        height, width = 180, 220
        pivot = np.asarray([111.5, 82.5])
        boundary_center = np.asarray([112.0, 83.0])
        cursor_color = tuple(
            int(value)
            for value in cv2.cvtColor(
                np.uint8([[[93, 230, 240]]]), cv2.COLOR_HSV2BGR
            )[0, 0]
        )
        base = np.asarray([[-7.0, -6.0], [-7.0, 6.0], [9.0, 0.0]])
        frames = []
        for index in range(24):
            angle = 2.0 * np.pi * index / 12.0
            rotation = np.asarray(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            polygon = np.round(base.dot(rotation.T) + pivot).astype(np.int32)
            image = np.full((height, width, 3), 30, np.uint8)
            cv2.circle(image, tuple(boundary_center.astype(int)), 69, (100, 100, 100), 1)
            cv2.fillConvexPoly(image, polygon, cursor_color, cv2.LINE_AA)
            frames.append(image)
        frames = np.stack(frames)
        space = raster_space("current_minimap_crop_pixels", [width, height])
        boundary = bind_geometry(
            {
                "center_x": float(boundary_center[0]),
                "center_y": float(boundary_center[1]),
                "radius": 69.0,
                "confidence": 0.95,
            },
            "circle",
            space,
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = calibrate_cursor_orbit_frames(
                frames,
                Path(temporary),
                outer_boundary=boundary,
                frame_space=space,
            )
            self.assertAlmostEqual(112.0, result["outer_boundary"]["center_x"])
            self.assertAlmostEqual(69.0, result["outer_boundary"]["radius"])
            self.assertLess(
                np.linalg.norm(
                    np.asarray(
                        [result["rotation_center"]["x"], result["rotation_center"]["y"]]
                    )
                    - pivot
                ),
                1.0,
            )
            self.assertEqual(
                "cursor_orbit_masks_aligned_to_canonical_direction",
                result["cursor_shape"]["source"],
            )
            declared = {item["name"] for item in result["evidence"]}
            self.assertIn("cursor_center_orbit.png", declared)
            self.assertIn("cursor_shape_polar_correlation.png", declared)
            self.assertTrue((Path(temporary) / "model.npz").is_file())

    def test_preserves_rotation_center_when_later_shape_fit_is_ineligible(self):
        height, width = 180, 220
        pivot = np.asarray([111.5, 82.5])
        cursor_color = tuple(
            int(value)
            for value in cv2.cvtColor(
                np.uint8([[[93, 230, 240]]]), cv2.COLOR_HSV2BGR
            )[0, 0]
        )
        frames = []
        for index in range(12):
            angle = 2.0 * np.pi * index / 12.0
            centroid = np.round(
                pivot + 7.0 * np.asarray([np.cos(angle), np.sin(angle)])
            ).astype(np.int32)
            image = np.full((height, width, 3), 30, np.uint8)
            cv2.circle(image, tuple(centroid), 4, cursor_color, -1)
            frames.append(image)
        space = raster_space("current_minimap_crop_pixels", [width, height])
        boundary = bind_geometry(
            {"center_x": 112.0, "center_y": 83.0, "radius": 69.0},
            "circle",
            space,
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "rig_runtime.services.calibration.minimap.calibration._cursor_shape",
            side_effect=RuntimeError("persistent contour is empty"),
        ):
            result = calibrate_cursor_orbit_frames(
                np.stack(frames),
                Path(temporary),
                outer_boundary=boundary,
                frame_space=space,
            )
            self.assertEqual("partial", result["status"])
            self.assertEqual("rotation_center_only", result["result_level"])
            self.assertIsNotNone(result["rotation_center"])
            self.assertIsNone(result["cursor_shape"])
            self.assertEqual(
                "available", result["capabilities"]["rotation_center"]["status"]
            )
            self.assertEqual(
                "unavailable", result["capabilities"]["cursor_shape"]["status"]
            )
            self.assertIn(
                "persistent contour is empty",
                result["capabilities"]["cursor_shape"]["reason"],
            )
            self.assertTrue((Path(temporary) / "cursor_center_orbit.png").is_file())
            with np.load(str(Path(temporary) / "model.npz")) as model:
                self.assertIn("rotation_center", model.files)
                self.assertNotIn("cursor_polygon_relative_xy", model.files)

    def test_non_hsv_cursor_preserves_color_agnostic_rotation_center(self):
        height, width = 180, 220
        space = raster_space("current_minimap_crop_pixels", [width, height])
        boundary = bind_geometry(
            {"center_x": 110.0, "center_y": 90.0, "radius": 70.0},
            "circle",
            space,
        )
        # Red is intentionally outside the legacy cyan HSV shape interval.
        # Center geometry must still be recovered from temporal symmetry.
        pivot = np.asarray([112.0, 86.0])
        base = np.asarray([[-7.0, -5.0], [-7.0, 5.0], [10.0, 0.0]])
        frames = []
        for index in range(24):
            angle = 2.0 * np.pi * index / 12.0
            rotation = np.asarray(
                [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            )
            polygon = np.round(base.dot(rotation.T) + pivot).astype(np.int32)
            image = np.full((height, width, 3), 25, np.uint8)
            cv2.fillConvexPoly(image, polygon, (20, 20, 240), cv2.LINE_AA)
            frames.append(image)
        with tempfile.TemporaryDirectory() as temporary:
            result = calibrate_cursor_orbit_frames(
                np.stack(frames),
                Path(temporary),
                outer_boundary=boundary,
                frame_space=space,
            )
            fitted = result["rotation_center"]
            self.assertLess(
                np.linalg.norm(np.asarray([fitted["x"], fitted["y"]]) - pivot),
                1.5,
            )
            self.assertEqual(
                "color_agnostic_temporal_centrosymmetry", fitted["method"]
            )
            self.assertEqual("rotation_center_only", result["result_level"])
            self.assertEqual("partial", result["status"])
            self.assertEqual(
                "available", result["capabilities"]["rotation_center"]["status"]
            )
            self.assertEqual(
                "unavailable", result["capabilities"]["cursor_shape"]["status"]
            )
            self.assertIn(
                "shape component gate",
                result["capabilities"]["cursor_shape"]["reason"],
            )
            declared = {item["name"] for item in result["evidence"]}
            self.assertIn("cursor_center_heatmap.png", declared)
            self.assertIn("cursor_center_symmetry.png", declared)

    def test_static_series_reports_shape_without_inventing_rotation_center(self):
        height, width = 180, 220
        space = raster_space("current_minimap_crop_pixels", [width, height])
        boundary = bind_geometry(
            {"center_x": 110.0, "center_y": 90.0, "radius": 70.0},
            "circle",
            space,
        )
        cursor_color = tuple(
            int(value)
            for value in cv2.cvtColor(
                np.uint8([[[93, 230, 240]]]), cv2.COLOR_HSV2BGR
            )[0, 0]
        )
        frames = []
        for value in range(8):
            image = np.full((height, width, 3), 25 + value, np.uint8)
            cv2.fillConvexPoly(
                image,
                np.asarray([[104, 83], [104, 97], [122, 90]], np.int32),
                cursor_color,
                cv2.LINE_AA,
            )
            frames.append(image)
        with tempfile.TemporaryDirectory() as temporary:
            result = calibrate_cursor_static_frames(
                np.stack(frames),
                Path(temporary),
                outer_boundary=boundary,
                frame_space=space,
            )
            self.assertIsNone(result["rotation_center"])
            self.assertEqual(
                "not_observable_from_static_cursor",
                result["rotation_center_status"],
            )
            self.assertEqual("shape_only", result["result_level"])
            self.assertEqual(
                "unavailable", result["capabilities"]["rotation_center"]["status"]
            )
            self.assertEqual(
                "available", result["capabilities"]["cursor_shape"]["status"]
            )
            self.assertIsNone(
                result["cursor_shape"]["rotating_cursor_envelope_diameter_px"]
            )
            self.assertGreater(
                result["cursor_shape"]["observed_static_cursor_max_span_px"], 10.0
            )
            self.assertTrue(
                (Path(temporary) / "cursor_static_shape_overlay.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
