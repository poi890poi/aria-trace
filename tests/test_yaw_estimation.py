import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "poc"))

from yaw_estimation import KltAngularYawEstimator, camera_matrix, yaw_rotation


class YawEstimatorTest(unittest.TestCase):
    def test_known_pure_yaw(self):
        height, width = 480, 640
        rng = np.random.RandomState(7)
        image = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        k = camera_matrix(width, height)
        estimator = KltAngularYawEstimator(k, use_essential_gate=False)
        estimator.update(image)

        expected = 0.7
        homography = k @ yaw_rotation(expected) @ np.linalg.inv(k)
        rotated = cv2.warpPerspective(image, homography, (width, height))
        estimate = estimator.update(rotated)

        self.assertEqual(estimate.status, "ok")
        self.assertAlmostEqual(estimate.delta_deg, expected, delta=0.08)


if __name__ == "__main__":
    unittest.main()
