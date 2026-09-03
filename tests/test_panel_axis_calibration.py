import unittest

import cv2
import numpy as np

from rig_runtime.services.calibration.rig.hik.panel_axis import (
    aggregate_panel_axis_measurements,
    measure_panel_axis_edges,
    panel_axis_correction_matrix,
    refine_camera_to_screen_matrix,
)
from rig_runtime.services.calibration.rig.hik.patterns import (
    focus_panel_axis_edges,
    focus_pattern,
    panel_axis_edges,
    panel_axis_pattern,
)


class PanelAxisCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.size = [480, 320]
        self.region = [30, 24, 420, 272]
        self.pattern = panel_axis_pattern(self.size, self.region)
        self.edges = panel_axis_edges(self.region)

    def rotated(self, clockwise_degrees):
        width, height = self.size
        matrix = cv2.getRotationMatrix2D(
            ((width - 1) / 2.0, (height - 1) / 2.0),
            -float(clockwise_degrees),
            1.0,
        )
        return cv2.warpAffine(
            self.pattern,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(24, 24, 24),
        )

    def test_broad_edges_measure_small_clockwise_residual(self):
        measured = measure_panel_axis_edges(
            self.rotated(2.5), self.edges, [0, 0]
        )

        self.assertEqual("accepted", measured["status"])
        self.assertAlmostEqual(2.5, measured["residual_clockwise_degrees"], delta=0.15)
        self.assertEqual(4, len(measured["lines"]))

    def test_correction_is_precomposable_and_removes_residual(self):
        rotated = self.rotated(-1.75)
        measured = measure_panel_axis_edges(rotated, self.edges, [0, 0])
        correction = panel_axis_correction_matrix(
            self.size, measured["correction_counterclockwise_degrees"]
        )
        corrected = cv2.warpPerspective(rotated, correction, tuple(self.size))
        verified = measure_panel_axis_edges(corrected, self.edges, [0, 0])

        self.assertLess(abs(verified["residual_clockwise_degrees"]), 0.15)

    def test_output_correction_refines_the_authoritative_screen_mapping(self):
        camera_to_screen = np.asarray(
            [[1.2, 0.03, 14.0], [-0.02, 0.9, 8.0], [0.0001, 0.0002, 1.0]],
            dtype=np.float64,
        )
        origin = [30.0, 24.0]
        scale = [1.25, 0.8]
        correction = panel_axis_correction_matrix(self.size, 1.5)
        refined = refine_camera_to_screen_matrix(
            camera_to_screen, origin, scale, correction
        )
        screen_to_output = np.asarray(
            [
                [1.0 / scale[0], 0.0, -origin[0] / scale[0]],
                [0.0, 1.0 / scale[1], -origin[1] / scale[1]],
                [0.0, 0.0, 1.0],
            ]
        )

        expected_output = correction.dot(screen_to_output).dot(camera_to_screen)
        expected_output /= expected_output[2, 2]
        actual_output = screen_to_output.dot(refined)
        actual_output /= actual_output[2, 2]

        np.testing.assert_allclose(expected_output, actual_output, atol=1.0e-10)
        np.testing.assert_allclose(
            np.linalg.inv(refined).dot(refined), np.eye(3), atol=1.0e-10
        )

    def test_thirty_degree_disagreement_is_not_applied(self):
        measurement = {
            "status": "accepted",
            "residual_clockwise_degrees": 30.0,
            "confidence": 0.9,
            "lines": [],
            "failures": [],
        }

        aggregate = aggregate_panel_axis_measurements([measurement])

        self.assertEqual("rejected_charuco_disagreement", aggregate["status"])
        self.assertFalse(aggregate["applied"])

    def test_large_axis_disagreement_is_not_aliased_to_small_rotation(self):
        measurement = {
            "status": "accepted",
            "residual_clockwise_degrees": 80.0,
            "confidence": 0.9,
            "lines": [],
            "failures": [],
        }

        aggregate = aggregate_panel_axis_measurements([measurement])

        self.assertEqual("rejected_charuco_disagreement", aggregate["status"])
        self.assertFalse(aggregate["applied"])

    def test_raw_axis_reference_is_reported_from_output_to_raw_map(self):
        rows, columns = np.indices((self.size[1], self.size[0]))
        measured = measure_panel_axis_edges(
            self.pattern,
            self.edges,
            [0, 0],
            output_to_raw_maps=(
                columns.astype(np.float32), rows.astype(np.float32)
            ),
        )

        self.assertEqual("accepted", measured["status"])
        self.assertAlmostEqual(
            0.0, measured["camera_up_to_panel_up_clockwise_degrees"], delta=0.1
        )
        np.testing.assert_allclose(
            [0.0, -1.0],
            measured["panel_up_unit_vector_full_sensor_camera_xy"],
            atol=0.01,
        )

    def test_raw_axis_reference_preserves_panel_direction_at_quarter_turn(self):
        rows, columns = np.indices((self.size[1], self.size[0]))
        measured = measure_panel_axis_edges(
            self.pattern,
            self.edges,
            [0, 0],
            output_to_raw_maps=(
                (self.size[1] - 1 - rows).astype(np.float32),
                columns.astype(np.float32),
            ),
        )

        self.assertEqual("accepted", measured["status"])
        np.testing.assert_allclose(
            [1.0, 0.0],
            measured["panel_up_unit_vector_full_sensor_camera_xy"],
            atol=0.01,
        )
        self.assertAlmostEqual(
            90.0,
            measured["camera_up_to_panel_up_clockwise_degrees"],
            delta=0.1,
        )

    def test_gui_focus_frame_edges_support_same_measurement(self):
        chart = focus_pattern(self.size, self.region)
        width, height = self.size
        transform = cv2.getRotationMatrix2D(
            ((width - 1) / 2.0, (height - 1) / 2.0), -1.25, 1.0
        )
        rotated = cv2.warpAffine(
            chart,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(24, 24, 24),
        )

        measured = measure_panel_axis_edges(
            rotated, focus_panel_axis_edges(self.region), [0, 0]
        )

        self.assertEqual("accepted", measured["status"])
        self.assertAlmostEqual(
            1.25, measured["residual_clockwise_degrees"], delta=0.2
        )


if __name__ == "__main__":
    unittest.main()
