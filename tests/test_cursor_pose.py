import unittest

import numpy as np

from acquisition.cursor_pose import CursorPoseEstimator, circular_difference_degrees


class CircularGaussianFitTests(unittest.TestCase):
    def test_vectorized_grid_matches_legacy_fit(self):
        random = np.random.RandomState(13)
        angles = np.arange(360, dtype=np.float64)
        for center, sigma in ((-179.2, 8.0), (-72.4, 19.0), (0.3, 31.0), (133.7, 45.0)):
            distance = circular_difference_degrees(angles, center)
            response = (
                0.12
                + 0.75 * np.exp(-0.5 * (distance / sigma) ** 2)
                + random.normal(0.0, 0.008, len(angles))
            ).astype(np.float32)
            legacy = CursorPoseEstimator._fit_circular_gaussian_legacy(response)
            vectorized = CursorPoseEstimator._fit_circular_gaussian_vectorized(
                response
            )
            self.assertLessEqual(
                abs(
                    float(
                        circular_difference_degrees(
                            vectorized["center_deg"], legacy["center_deg"]
                        )
                    )
                ),
                0.051,
            )
            self.assertAlmostEqual(
                vectorized["r_squared"], legacy["r_squared"], places=6
            )


if __name__ == "__main__":
    unittest.main()
