import unittest

import cv2
import numpy as np

from acquisition.minimap_transition_analysis import MapScaleMatcher, _likelihoods


class MinimapTransitionAnalysisTests(unittest.TestCase):
    def test_cached_map_matcher_measures_displayed_scale(self):
        random = np.random.RandomState(71)
        mosaic = random.randint(0, 256, (360, 480, 3), dtype=np.uint8)
        cv2.line(mosaic, (25, 310), (450, 40), (20, 240, 90), 6)
        coverage = np.full(mosaic.shape[:2], 255, np.uint8)
        reference = cv2.resize(
            mosaic[80:280, 140:340],
            (100, 100),
            interpolation=cv2.INTER_AREA,
        )
        mask = np.full((100, 100), 255, np.uint8)

        estimate = MapScaleMatcher(mosaic, coverage).estimate(reference, mask)

        self.assertAlmostEqual(
            estimate["map_pixels_per_minimap_pixel"], 2.0, delta=0.06
        )
        self.assertAlmostEqual(estimate["canonical_xy"][0], 240.0, delta=3.0)
        self.assertAlmostEqual(estimate["canonical_xy"][1], 180.0, delta=3.0)
        self.assertGreaterEqual(estimate["inlier_count"], 8)

    def test_scale_likelihoods_have_universal_mode_ordering(self):
        scales = {"world": 2.64, "town": 0.88}
        world = _likelihoods(2.60, scales, 0.18)
        town = _likelihoods(0.90, scales, 0.18)

        self.assertGreater(world["world"], 0.99)
        self.assertGreater(town["town"], 0.99)
        self.assertAlmostEqual(sum(world.values()), 1.0)


if __name__ == "__main__":
    unittest.main()
