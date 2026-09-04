import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from acquisition.map_stitching import (
    _build_localization_derivative,
    _estimate_minimap_similarity,
    _fit_fixed_north_up_similarity,
    _minimap_reference,
    _masked_oriented_gradient_zncc,
    _prepare_localization_mosaic,
    _root_sift,
    _scale_consensus,
    load_localization_reference_candidates,
    stitch_map_frames,
)
from aria_trace.services.mapping.stitching import _select_session_keyframes


class MapStitchingTests(unittest.TestCase):
    class _Capture:
        def __init__(self, frames):
            self.frames = iter(frames)

        def read(self):
            try:
                return True, next(self.frames)
            except StopIteration:
                return False, None

    def test_oriented_gradient_zncc_prefers_matching_edge_polarity(self):
        template = np.zeros((30, 30, 3), np.uint8)
        template[:, 15:] = 220
        source = np.zeros((70, 100, 3), np.uint8)
        source[20:50, 20:35] = 220
        source[20:50, 65:80] = 220
        source[20:50, 80:95] = 0
        mask = np.full(template.shape[:2], 255, np.uint8)
        response = _masked_oriented_gradient_zncc(source, template, mask)
        _, score, _, location = cv2.minMaxLoc(response)
        self.assertGreater(score, 0.65)
        self.assertEqual(location, (5, 20))
        self.assertLess(float(response[20, 65]), 0.0)

    def test_scale_consensus_selects_largest_consistent_reference_cluster(self):
        rows = []
        for index, scale in enumerate((3.80, 3.82, 3.81, 7.2)):
            rows.append(
                {
                    "candidate": {"source_image_name": "frame-{}".format(index)},
                    "estimate": {
                        "map_pixels_per_minimap_pixel": scale,
                        "inlier_count": 5 + index,
                        "inlier_ratio": 0.8,
                        "reprojection_p95_px": 2.0,
                    },
                }
            )
        consensus = _scale_consensus(rows)
        self.assertEqual(len(consensus["members"]), 3)
        self.assertAlmostEqual(consensus["scale"], 3.81, places=6)
        self.assertEqual(
            consensus["selected"]["candidate"]["source_image_name"], "frame-2"
        )
        preferred = _scale_consensus(rows, preferred_name="frame-0")
        self.assertEqual(
            preferred["selected"]["candidate"]["source_image_name"], "frame-0"
        )
        rows[1]["estimate"]["inlier_count"] = 2
        strict = _scale_consensus(rows, minimum_inliers=3)
        self.assertEqual(len(strict["members"]), 2)

    def test_loads_both_persisted_forward_reference_endpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cv2.imwrite(str(root / "forward_start.png"), np.full((20, 30, 3), 40, np.uint8))
            cv2.imwrite(str(root / "forward_end.png"), np.full((20, 30, 3), 80, np.uint8))
            calibration = {
                "forward_verification": {
                    "source_frames": {
                        "start": {"frame_index": 5},
                        "end": {"frame_index": 95},
                    }
                }
            }
            references = load_localization_reference_candidates(root, calibration)
            self.assertEqual(
                [item["source_image_name"] for item in references],
                ["forward_start.png", "forward_end.png"],
            )
            self.assertEqual(
                [item["source_frame_index"] for item in references], [5, 95]
            )

    def test_fixed_north_up_fit_rejects_outlier_without_rotation(self):
        source = np.asarray(
            [[0, 0], [10, 0], [0, 10], [10, 10], [20, 15]], dtype=np.float32
        )
        target = source * 2.5 + np.asarray([40, 70], dtype=np.float32)
        target[-1] = [300, 12]
        matrix, inliers = _fit_fixed_north_up_similarity(source, target)
        self.assertIsNotNone(matrix)
        self.assertAlmostEqual(float(matrix[0, 0]), 2.5, places=6)
        self.assertEqual(float(matrix[0, 1]), 0.0)
        self.assertEqual(float(matrix[1, 0]), 0.0)
        self.assertEqual(int(np.count_nonzero(inliers)), 4)

    def test_root_sift_descriptors_have_unit_l2_norm(self):
        descriptors = np.asarray([[1.0, 3.0], [4.0, 0.0]], dtype=np.float32)
        transformed = _root_sift(descriptors)
        np.testing.assert_allclose(
            np.linalg.norm(transformed, axis=1), np.ones(2), atol=1.0e-6
        )

    def test_minimap_reference_masks_calibrated_cursor_and_boundary_ui(self):
        image = np.full((180, 220, 3), (80, 120, 90), np.uint8)
        cv2.circle(image, (110, 90), 56, (100, 140, 110), -1)
        calibration = {
            "outer_boundary": {"center_x": 110, "center_y": 90, "radius": 56}
        }
        _, mask = _minimap_reference(image, calibration)
        self.assertEqual(int(mask[56, 56]), 0)
        self.assertEqual(int(mask[56, 71]), 0)
        self.assertEqual(int(mask[56, 76]), 255)
        self.assertEqual(int(mask[0, 0]), 0)

    def test_reuses_precomputed_mosaic_features_without_changing_estimate(self):
        rng = np.random.RandomState(19)
        mosaic = rng.randint(0, 255, (420, 610, 3), dtype=np.uint8)
        reference_image = np.zeros((180, 220, 3), np.uint8)
        reference_image[34:146, 54:166] = cv2.resize(
            mosaic[120:400, 210:490], (112, 112), interpolation=cv2.INTER_AREA
        )
        calibration = {
            "outer_boundary": {"center_x": 110, "center_y": 90, "radius": 56}
        }
        coverage = np.full(mosaic.shape[:2], 255, np.uint8)
        patch, mask = _minimap_reference(reference_image, calibration)
        cache = _prepare_localization_mosaic(mosaic, coverage)
        first = _estimate_minimap_similarity(
            patch, mask, mosaic, coverage, mosaic_cache=cache
        )
        second = _estimate_minimap_similarity(
            patch, mask, mosaic, coverage, mosaic_cache=cache
        )
        self.assertAlmostEqual(
            first["map_pixels_per_minimap_pixel"],
            second["map_pixels_per_minimap_pixel"],
            places=9,
        )
        self.assertEqual(first["inlier_count"], second["inlier_count"])

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
            self.assertTrue(result["method"].startswith("rootsift_"))
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
                "localization_reference_consensus.png",
                "localization_correlation_heatmap.png",
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

    def test_session_keyframe_selection_does_not_retain_every_video_frame(self):
        rng = np.random.RandomState(12)
        panorama = rng.randint(0, 255, (220, 600, 3), dtype=np.uint8)
        frames = [
            panorama[:, offset : offset + 300].copy()
            for offset in range(0, 181, 3)
        ]
        selected, indices, decoded = _select_session_keyframes(
            self._Capture(frames), len(frames)
        )

        self.assertEqual(decoded, len(frames))
        self.assertEqual(len(selected), len(indices))
        self.assertLess(len(selected), len(frames) // 3)
        self.assertEqual(indices[0], 0)
        self.assertGreater(indices[-1], 50)


if __name__ == "__main__":
    unittest.main()
