import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from acquisition.map_layers import (
    LayeredGlobalLocalizer,
    build_map_atlas,
    estimate_map_layer_alignment,
)


class MapLayerTests(unittest.TestCase):
    @staticmethod
    def _mosaics():
        random = np.random.RandomState(44)
        canonical = random.randint(0, 256, (360, 480, 3), dtype=np.uint8)
        cv2.circle(canonical, (170, 140), 45, (10, 240, 80), 5)
        cv2.line(canonical, (30, 300), (440, 40), (240, 40, 20), 7)
        layer = cv2.resize(canonical, (336, 252), interpolation=cv2.INTER_AREA)
        return canonical, layer

    @staticmethod
    def _write_stitch(root: Path, mosaic: np.ndarray, calibration_id: str):
        root.mkdir(parents=True)
        coverage = np.full(mosaic.shape[:2], 255, np.uint8)
        cv2.imwrite(str(root / "mosaic.png"), mosaic)
        cv2.imwrite(str(root / "coverage.png"), coverage)
        cv2.imwrite(str(root / "localization_mosaic.png"), mosaic)
        cv2.imwrite(str(root / "localization_coverage.png"), coverage)
        manifest = {
            "stitch_id": root.name,
            "source_minimap_calibration_id": calibration_id,
            "localization": {
                "status": "ready",
                "mosaic_file": "localization_mosaic.png",
                "coverage_file": "localization_coverage.png",
                "localization_to_original_map_3x3": np.eye(3).tolist(),
            },
        }
        (root / "map_stitch.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    def test_aligns_different_render_scales(self):
        canonical, layer = self._mosaics()
        estimate = estimate_map_layer_alignment(canonical, layer)

        matrix = estimate["layer_original_to_canonical_3x3"]
        scale = estimate["quality"]["canonical_pixels_per_layer_pixel"]
        self.assertAlmostEqual(scale, 1.0 / 0.7, delta=0.03)
        point = matrix.dot(np.asarray([70.0, 70.0, 1.0]))
        self.assertAlmostEqual(point[0], 100.0, delta=2.0)
        self.assertAlmostEqual(point[1], 100.0, delta=2.0)
        self.assertGreaterEqual(estimate["quality"]["inlier_count"], 8)

    def test_builds_portable_two_layer_atlas(self):
        canonical, layer = self._mosaics()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            world = root / "world-stitch"
            town = root / "town-stitch"
            output = root / "atlas"
            self._write_stitch(world, canonical, "cal-world")
            self._write_stitch(town, layer, "cal-town")

            manifest = build_map_atlas(
                [
                    {"mode_id": "world", "stitch_root": world},
                    {"mode_id": "town", "stitch_root": town},
                ],
                output,
                canonical_mode_id="world",
                atlas_id="two-scale-a",
            )

            self.assertEqual(manifest["canonical_mode_id"], "world")
            self.assertEqual(len(manifest["layers"]), 2)
            self.assertTrue((output / "map_atlas.json").is_file())
            self.assertTrue((output / "layers" / "town" / "localization_mosaic.png").is_file())
            town_layer = next(
                item for item in manifest["layers"] if item["mode_id"] == "town"
            )
            self.assertIsNotNone(town_layer["alignment_evidence_file"])

    def test_layered_localizer_returns_canonical_coordinates_and_mode(self):
        canonical, layer = self._mosaics()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            world = root / "world-stitch"
            town = root / "town-stitch"
            output = root / "atlas"
            self._write_stitch(world, canonical, "cal-world")
            self._write_stitch(town, layer, "cal-town")
            build_map_atlas(
                [
                    {"mode_id": "world", "stitch_root": world},
                    {"mode_id": "town", "stitch_root": town},
                ],
                output,
                canonical_mode_id="world",
            )
            localizer = LayeredGlobalLocalizer(output)
            observation = layer[76:176, 118:218].copy()
            mask = np.zeros((100, 100), np.uint8)
            cv2.circle(mask, (50, 50), 47, 255, -1)
            try:
                localizer.set_active_mode("town")
                fix = localizer.localize(observation, mask, yaw_prior_deg=0.0)
            finally:
                localizer.close()

            self.assertTrue(fix.valid)
            self.assertAlmostEqual(fix.x, (118 + 50) / 0.7, delta=5.0)
            self.assertAlmostEqual(fix.y, (76 + 50) / 0.7, delta=5.0)
            self.assertEqual(
                fix.diagnostics["map_layer"]["selected_mode_id"], "town"
            )

    def test_transition_endpoints_normalize_each_layer_independently(self):
        canonical, _ = self._mosaics()
        mask = np.full((100, 100), 255, np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            world = root / "world-stitch"
            output = root / "atlas"
            self._write_stitch(world, canonical, "cal-world")
            town_reference = cv2.resize(
                canonical[60:260, 100:300],
                (100, 100),
                interpolation=cv2.INTER_AREA,
            )

            manifest = build_map_atlas(
                [
                    {
                        "mode_id": "world",
                        "stitch_root": world,
                        "minimap_reference": canonical[80:180, 120:220].copy(),
                        "minimap_reference_mask": mask,
                    },
                    {
                        "mode_id": "town",
                        "stitch_root": world,
                        "minimap_reference": town_reference,
                        "minimap_reference_mask": mask,
                    },
                ],
                output,
                canonical_mode_id="world",
            )

            for atlas_layer in manifest["layers"]:
                self.assertEqual(
                    atlas_layer["localization_source"],
                    "transition_endpoint_minimap_reference",
                )
                self.assertTrue(
                    (output / atlas_layer["minimap_reference_file"]).is_file()
                )
                self.assertGreater(
                    atlas_layer["map_pixels_per_minimap_pixel"], 0.0
                )
            town_layer = next(
                item for item in manifest["layers"] if item["mode_id"] == "town"
            )
            self.assertEqual(
                town_layer["alignment_quality"]["method"],
                "shared_source_identity",
            )

            localizer = LayeredGlobalLocalizer(output)
            try:
                world_observation = canonical[80:180, 120:220].copy()
                world_result = localizer.observe_modes(
                    world_observation, mask, (170.0, 130.0), 20.0
                )
                town_result = localizer.observe_modes(
                    town_reference, mask, (200.0, 160.0), 20.0
                )
            finally:
                localizer.close()

            self.assertTrue(world_result["valid"])
            self.assertEqual(world_result["selected_mode_id"], "world")
            self.assertTrue(town_result["valid"])
            self.assertEqual(town_result["selected_mode_id"], "town")
            self.assertEqual(town_result["pose_authority"], "none")
            self.assertEqual(
                town_result["canonical_xy_read_only"], [200.0, 160.0]
            )


if __name__ == "__main__":
    unittest.main()
