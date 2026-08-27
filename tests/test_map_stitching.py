import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from acquisition.map_stitching import stitch_map_frames


class MapStitchingTests(unittest.TestCase):
    def test_stitches_overlapping_translated_map_views_with_evidence(self):
        rng = np.random.RandomState(11)
        panorama = rng.randint(0, 255, (260, 520, 3), dtype=np.uint8)
        for x in range(30, 500, 55):
            cv2.circle(panorama, (x, 130), 12, (20, 240, 80), 2)
        frames = [panorama[20:220, offset : offset + 280].copy() for offset in range(0, 181, 20)]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = stitch_map_frames(frames, output, provenance={"fixture": True})
            self.assertEqual(result["status"], "review_required")
            self.assertGreater(result["accepted_registrations"], 5)
            self.assertGreater(result["mosaic_size_wh"][0], 350)
            self.assertEqual(result["coverage_scope"], "observed_viewports_only")
            for item in result["evidence"]:
                self.assertGreater((output / item["name"]).stat().st_size, 0)
            self.assertTrue((output / "map_stitch.json").is_file())


if __name__ == "__main__":
    unittest.main()
