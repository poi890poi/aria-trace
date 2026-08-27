import unittest

import cv2
import numpy as np

from acquisition.minimap_verification import (
    benchmark_masked_shift,
    estimate_masked_shift,
)


class MinimapVerificationTests(unittest.TestCase):
    def test_estimates_masked_start_to_end_translation(self):
        rng = np.random.RandomState(7)
        first = rng.randint(0, 255, (120, 140, 3), dtype=np.uint8)
        transform = np.float32([[1, 0, 7], [0, 1, -4]])
        last = cv2.warpAffine(
            first,
            transform,
            (first.shape[1], first.shape[0]),
            borderMode=cv2.BORDER_WRAP,
        )
        mask = np.zeros(first.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (70, 60), 48, 255, -1)
        shift, response = estimate_masked_shift(first, last, mask)
        self.assertAlmostEqual(shift[0], 7.0, delta=0.6)
        self.assertAlmostEqual(shift[1], -4.0, delta=0.6)
        self.assertGreater(response, 0.20)

        measured_shift, measured_response, benchmark = benchmark_masked_shift(
            first, last, mask, repeat_count=3
        )
        self.assertAlmostEqual(measured_shift[0], shift[0], places=6)
        self.assertAlmostEqual(measured_shift[1], shift[1], places=6)
        self.assertAlmostEqual(measured_response, response, places=6)
        self.assertEqual(benchmark["sample_count"], 3)
        self.assertEqual(benchmark["warmup_count"], 1)
        self.assertEqual(benchmark["image_size_wh"], [140, 120])
        self.assertGreater(benchmark["median_ms"], 0.0)
        self.assertGreaterEqual(benchmark["p95_ms"], benchmark["median_ms"])


if __name__ == "__main__":
    unittest.main()
