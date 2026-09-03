import unittest

import numpy as np

from rig_runtime.domain.spatial import (
    bind_geometry,
    normalize_legacy_geometry,
    raster_space,
    require_same_space,
    require_spatial_geometry,
    transform_circle_similarity,
)
from rig_runtime.services.calibration.minimap.spatial import (
    minimap_crop_space,
    normalize_minimap_geometry,
)


class SpatialGeometryTests(unittest.TestCase):
    def test_geometry_requires_its_own_valid_space(self):
        with self.assertRaisesRegex(ValueError, "spatial schema"):
            require_spatial_geometry(
                {"center_x": 10, "center_y": 12, "radius": 8}, "circle"
            )
        space = raster_space("crop-a", [100, 80])
        circle = bind_geometry(
            {"center_x": 10, "center_y": 12, "radius": 8}, "circle", space
        )
        self.assertEqual("crop-a", circle["space"]["space_id"])

    def test_mixed_spaces_are_rejected(self):
        first = bind_geometry(
            {"x": 3, "y": 4}, "point", raster_space("first", [20, 10])
        )
        second = bind_geometry(
            {"center_x": 3, "center_y": 4, "radius": 2},
            "circle",
            raster_space("second", [20, 10]),
        )
        with self.assertRaisesRegex(ValueError, "across coordinate spaces"):
            require_same_space(first, second)

    def test_circle_transform_requires_similarity_and_rebinds_space(self):
        source = raster_space("source", [100, 80])
        target = raster_space("target", [200, 160])
        circle = bind_geometry(
            {"center_x": 10, "center_y": 12, "radius": 8}, "circle", source
        )
        converted = transform_circle_similarity(
            circle,
            [[0, -2, 100], [2, 0, 20], [0, 0, 1]],
            target,
        )
        np.testing.assert_allclose(
            [converted["center_x"], converted["center_y"], converted["radius"]],
            [76, 40, 16],
        )
        self.assertEqual("target", converted["space"]["space_id"])
        with self.assertRaisesRegex(ValueError, "ellipse"):
            transform_circle_similarity(
                circle, [[2, 0, 0], [0, 1, 0], [0, 0, 1]], target
            )

    def test_legacy_geometry_requires_explicit_fallback_space(self):
        normalized = normalize_legacy_geometry(
            {"x": 2, "y": 5}, "point", raster_space("known-owner", [10, 10])
        )
        self.assertEqual("known-owner", normalized["space"]["space_id"])

    def test_minimap_consumer_rejects_same_id_with_different_raster(self):
        wrong = bind_geometry(
            {"center_x": 10, "center_y": 12, "radius": 8},
            "circle",
            minimap_crop_space([90, 80]),
        )
        with self.assertRaisesRegex(ValueError, "containing raster"):
            normalize_minimap_geometry(
                {"outer_boundary": wrong}, minimap_crop_space([100, 80])
            )


if __name__ == "__main__":
    unittest.main()
