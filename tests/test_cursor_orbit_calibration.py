import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from aria_trace.adapters.android.cursor_orbit import CursorOrbitTouchPlan
from aria_trace.domain.spatial import bind_geometry, raster_space
from aria_trace.services.calibration.minimap.calibration import (
    calibrate_cursor_orbit_frames,
)
from aria_trace.workflows.minimap_capture import _build_control_plan, parser


class CursorOrbitCalibrationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
