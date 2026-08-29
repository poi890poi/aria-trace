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

    def test_fast_grid_still_fits_the_gaussian_center(self):
        angles = np.arange(360, dtype=np.float64)
        distance = circular_difference_degrees(angles, -83.35)
        response = (
            0.18 + 0.71 * np.exp(-0.5 * (distance / 17.0) ** 2)
        ).astype(np.float32)

        fitted = CursorPoseEstimator._fit_circular_gaussian_fast(response)

        self.assertLess(
            abs(float(circular_difference_degrees(fitted["center_deg"], -83.35))),
            0.1,
        )
        self.assertGreater(fitted["r_squared"], 0.999)


if __name__ == "__main__":
    unittest.main()
