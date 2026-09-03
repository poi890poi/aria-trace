import unittest

import cv2
import numpy as np

from rig_runtime.services.calibration.rig.distortion import (
    combined_output_to_raw_maps,
    distorted_screen_region_roi,
    fit_evidence_gated_distortion,
    undistort_pixel_points,
)
from rig_runtime.services.calibration.rig.image_quality import (
    measure_slanted_edge_esfr,
)
from rig_runtime.services.calibration.rig.image_quality import (
    generate_slanted_edge_target,
)


def synthetic_views(distortion):
    camera_matrix = np.asarray(
        [[900.0, 0.0, 640.0], [0.0, 910.0, 480.0], [0.0, 0.0, 1.0]]
    )
    xx, yy = np.meshgrid(np.linspace(0, 240, 9), np.linspace(0, 400, 12))
    screen = np.column_stack([xx.reshape(-1), yy.reshape(-1)]).astype(np.float32)
    objects = np.column_stack([screen, np.zeros(len(screen))]).astype(np.float32)
    camera_views, screen_views = [], []
    poses = [
        ([0.02, -0.10, 0.01], [-120, -180, 850]),
        ([0.08, 0.05, -0.03], [-80, -160, 900]),
        ([-0.07, 0.09, 0.04], [-150, -120, 820]),
        ([0.11, -0.04, -0.05], [-60, -210, 930]),
        ([-0.09, -0.08, 0.02], [-170, -190, 880]),
        ([0.05, 0.12, -0.02], [-100, -130, 860]),
    ]
    for rotation, translation in poses:
        projected, _ = cv2.projectPoints(
            objects,
            np.asarray(rotation, dtype=np.float64),
            np.asarray(translation, dtype=np.float64),
            camera_matrix,
            np.asarray(distortion, dtype=np.float64),
        )
        camera_views.append(projected.reshape((-1, 2)))
        screen_views.append(screen.copy())
    return camera_views, screen_views


class RigDistortionTests(unittest.TestCase):
    def test_distortion_candidate_must_improve_independent_holdout(self):
        camera, screen = synthetic_views([-0.24, 0.08, 0.001, -0.001, 0.0])
        result = fit_evidence_gated_distortion(camera, screen, [1280, 960], 0.05)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["source"], "measured")
        self.assertGreater(result["holdout"]["relative_p95_improvement"], 0.05)

    def test_unnecessary_distortion_is_rejected_by_holdout(self):
        camera, screen = synthetic_views([0.0, 0.0, 0.0, 0.0, 0.0])
        result = fit_evidence_gated_distortion(camera, screen, [1280, 960], 0.05)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["source"], "rejected_holdout")

    def test_combined_map_is_one_output_to_raw_lookup(self):
        lens = {
            "source": "measured",
            "camera_matrix_3x3": [[100, 0, 4], [0, 100, 3], [0, 0, 1]],
            "distortion_coefficients": [-0.2, 0.05, 0.0, 0.0, 0.0],
        }
        map_x, map_y = combined_output_to_raw_maps(
            np.eye(3), [9, 7], lens, chunk_size=10
        )
        self.assertEqual(map_x.shape, (7, 9))
        self.assertEqual(map_y.shape, (7, 9))
        self.assertAlmostEqual(float(map_x[3, 4]), 4.0, places=4)
        self.assertAlmostEqual(float(map_y[3, 4]), 3.0, places=4)

    def test_distorted_roi_and_esfr_keep_native_raw_samples(self):
        screen_size = (320, 240)
        camera_size = (640, 480)
        rect = (40, 40, 240, 160)
        camera_to_screen = np.asarray(
            [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        lens = {
            "source": "measured",
            "camera_matrix_3x3": [
                [520.0, 0.0, 319.5],
                [0.0, 515.0, 239.5],
                [0.0, 0.0, 1.0],
            ],
            "distortion_coefficients": [-0.18, 0.05, 0.0, 0.0, 0.0],
        }
        target = generate_slanted_edge_target(
            screen_size, rect, edge_angle_deg=5.0, channel="luminance"
        )
        ideal = cv2.resize(target, camera_size, interpolation=cv2.INTER_NEAREST)
        ideal = cv2.GaussianBlur(ideal, (0, 0), 1.2)
        yy, xx = np.mgrid[0 : camera_size[1], 0 : camera_size[0]]
        raw_points = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
        ideal_points = undistort_pixel_points(raw_points, lens)
        raw = cv2.remap(
            ideal,
            ideal_points[:, 0].reshape(camera_size[1], camera_size[0]).astype(np.float32),
            ideal_points[:, 1].reshape(camera_size[1], camera_size[0]).astype(np.float32),
            cv2.INTER_LINEAR,
        )
        roi = distorted_screen_region_roi(
            camera_size, rect, np.linalg.inv(camera_to_screen), lens
        )
        self.assertGreater(roi[2], 400)
        result, _ = measure_slanted_edge_esfr(
            raw,
            camera_to_screen,
            rect,
            edge_angle_deg=5.0,
            camera_lens_model=lens,
        )
        self.assertEqual(result["measurement_input_space"], "camera_raw_distorted_px")
        self.assertTrue(result["lens_geometry_applied_without_image_resampling"])
        self.assertIsNotNone(result["display_referred"]["mtf50"])


if __name__ == "__main__":
    unittest.main()
