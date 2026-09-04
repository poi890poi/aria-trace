import unittest

import numpy as np

from acquisition.cursor_pose import CursorPoseEstimator, circular_difference_degrees


class CircularGaussianFitTests(unittest.TestCase):
    def test_parabolic_circular_peak_refines_between_bins(self):
        response = np.zeros(360, dtype=np.float32)
        response[40:43] = [0.8, 1.0, 0.9]

        peak = CursorPoseEstimator._parabolic_circular_peak(response)

        self.assertGreater(peak, 41.0)
        self.assertLess(peak, 41.5)

    def test_angular_projection_correlation_recovers_circular_shift(self):
        estimator = object.__new__(CursorPoseEstimator)
        template = np.zeros(360, dtype=np.float32)
        template[4:11] = np.linspace(0.1, 1.0, 7)
        template[11:19] = np.linspace(0.9, 0.05, 8)
        estimator.angular_template_fft = np.fft.fft(template)
        estimator.angular_template_energy = float(np.linalg.norm(template))
        estimator.x_map = np.arange(360, dtype=np.float32)[:, None]
        estimator.y_map = np.zeros((360, 1), dtype=np.float32)
        observed = np.roll(template, 73).reshape(1, 360)

        shift, response, _ = estimator._angular_projection_correlate(observed)

        self.assertAlmostEqual(shift, 73.0, places=4)
        self.assertGreater(float(np.max(response)), 0.99)

    def test_pixel_validation_policy_only_expands_ambiguous_frames(self):
        estimator = object.__new__(CursorPoseEstimator)
        strong = {
            "r_squared": 0.98,
            "center_std_deg": 0.2,
            "fallback_used": False,
        }
        estimator.validation_policy = "ambiguous"
        self.assertFalse(
            estimator._needs_pixel_validation(strong, 0.10, 2.0, 0.85)
        )
        self.assertTrue(
            estimator._needs_pixel_validation(strong, 0.01, 2.0, 0.85)
        )
        estimator.validation_policy = "full"
        self.assertTrue(
            estimator._needs_pixel_validation(strong, 0.10, 2.0, 0.85)
        )
        estimator.validation_policy = "minimal"
        self.assertFalse(
            estimator._needs_pixel_validation(strong, 0.01, 40.0, 0.2)
        )

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

    def test_analytic_lm_fits_subdegree_center_without_grid_refinement(self):
        random = np.random.RandomState(41)
        angles = np.arange(360, dtype=np.float64)
        distance = circular_difference_degrees(angles, 43.25)
        response = (
            0.11
            + 0.77 * np.exp(-0.5 * (distance / 11.5) ** 2)
            + random.normal(0.0, 0.004, len(angles))
        ).astype(np.float32)

        fitted = CursorPoseEstimator._fit_circular_gaussian_lm(response)

        self.assertLess(
            abs(float(circular_difference_degrees(fitted["center_deg"], 43.25))),
            0.25,
        )
        self.assertEqual(fitted["fit_method"], "analytic_lm")
        self.assertGreater(fitted["r_squared"], 0.99)

    def test_temporal_window_selects_the_plausible_lobe(self):
        angles = np.arange(360, dtype=np.float64)
        near = circular_difference_degrees(angles, 31.0)
        far = circular_difference_degrees(angles, 172.0)
        response = (
            0.1
            + 0.60 * np.exp(-0.5 * (near / 9.0) ** 2)
            + 0.80 * np.exp(-0.5 * (far / 9.0) ** 2)
        ).astype(np.float32)

        fitted = CursorPoseEstimator._fit_circular_gaussian_cascade(
            response, center_prior_deg=30.0, search_half_width_deg=20.0
        )

        self.assertLess(
            abs(float(circular_difference_degrees(fitted["center_deg"], 31.0))),
            0.5,
        )

    def test_batched_symmetric_chamfer_matches_angle_loop(self):
        random = np.random.RandomState(23)
        polygon_edges = random.rand(9, 11, 11) > 0.88
        polygon_edges[:, 5, 5] = True
        polygon_distance_transforms = random.rand(9, 11, 11).astype(np.float32)
        observed_edge = random.rand(11, 11) > 0.9
        observed_edge[5, 5] = True
        observed_distance = random.rand(11, 11).astype(np.float32)
        expected = []
        for angle in range(len(polygon_edges)):
            expected.append(
                0.5
                * (
                    float(np.mean(observed_distance[polygon_edges[angle]]))
                    + float(
                        np.mean(
                            polygon_distance_transforms[angle][observed_edge]
                        )
                    )
                )
            )

        actual = CursorPoseEstimator._symmetric_chamfer_curve(
            observed_edge,
            observed_distance,
            polygon_edges,
            polygon_distance_transforms,
        )

        np.testing.assert_allclose(actual, expected, rtol=1.0e-7, atol=1.0e-7)


if __name__ == "__main__":
    unittest.main()
