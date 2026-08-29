import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from acquisition.map_stitching import (
    _build_localization_derivative,
    stitch_map_frames,
)


class MapStitchingTests(unittest.TestCase):
    def test_builds_verified_localization_raster_at_minimap_scale(self):
        rng = np.random.RandomState(29)
        mosaic = rng.randint(0, 255, (640, 920, 3), dtype=np.uint8)
        for index in range(18):
            center = (45 + index * 47, 55 + (index * 83) % 520)
            cv2.circle(mosaic, center, 8 + index % 9, (30, 230, 80), 2)
        original_patch = mosaic[220:500, 330:610]
        minimap_patch = cv2.resize(original_patch, (112, 112), interpolation=cv2.INTER_AREA)
        reference_frame = np.zeros((180, 220, 3), np.uint8)
        reference_frame[34:146, 54:166] = minimap_patch
        reference = {
            "image": reference_frame,
            "calibration": {
                "outer_boundary": {"center_x": 110, "center_y": 90, "radius": 56}
            },
            "calibration_id": "fixture-calibration",
            "source_image_name": "forward_start.png",
        }
        coverage = np.full(mosaic.shape[:2], 255, np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = _build_localization_derivative(
                mosaic, coverage, output, reference
            )
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["map_orientation_model"], "fixed_north_up")
            self.assertFalse(result["north_normalization_applied"])
            self.assertEqual(result["applied_map_rotation_deg"], 0.0)
            self.assertEqual(result["minimap_to_map_rotation_deg"], 0.0)
            self.assertIn("diagnostic_residual_rotation_deg", result)
            self.assertAlmostEqual(
                result["map_pixels_per_minimap_pixel"], 2.5, delta=0.08
            )
            self.assertEqual(result["source_minimap_calibration_id"], "fixture-calibration")
            self.assertGreaterEqual(result["quality"]["inlier_count"], 6)
            self.assertGreater(result["quality"]["gradient_correlation_margin"], 0.06)
            for name in (
                "localization_mosaic.png",
                "localization_coverage.png",
                "localization_scale_evidence.png",
            ):
                self.assertGreater((output / name).stat().st_size, 0)

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
