import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from acquisition.session_minimap_localization import (
    load_session_registration,
    nearest_synchronized_pair,
    parser,
    project_android_boundary,
)


class SessionMinimapLocalizationTests(unittest.TestCase):
    def test_registration_uses_only_current_session_yaml(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            matrix = [[0.5, 0, -10], [0, 0.5, -5], [0, 0, 1]]
            (root / "coordinate_spaces.yaml").write_text(
                json.dumps(
                    {
                        "streams": {
                            "android_phone": {"stored_size_px": [2400, 1080]},
                            "hik_phone": {"stored_size_px": [700, 880]},
                        },
                        "conversions": {
                            "adb_to_hik_phone_video_3x3": matrix
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = load_session_registration(
                root, [2400, 1080], [700, 880]
            )
            np.testing.assert_allclose(result["matrix"], matrix)

    def test_missing_registration_is_not_replaced_by_image_fitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "will not be guessed"):
                load_session_registration(
                    Path(temporary), [2400, 1080], [700, 880]
                )

    def test_nearest_pair_reports_timing_distribution(self):
        result = nearest_synchronized_pair(
            np.asarray([0, 10_000_000, 20_000_000], np.int64),
            np.asarray([2_000_000, 12_000_000, 22_000_000], np.int64),
        )
        self.assertEqual(-2.0, result["best_signed_delta_ms"])
        self.assertEqual(2.0, result["absolute_delta_p50_ms"])

    def test_projection_maps_android_circle_without_camera_prior(self):
        result = project_android_boundary(
            {"center_x": 100.0, "center_y": 80.0, "radius": 30.0},
            [[0.5, 0, 10], [0, 0.5, 20], [0, 0, 1]],
            [200, 150],
        )
        np.testing.assert_allclose(result["center_xy"], [60.0, 60.0])
        self.assertEqual(1.0, result["visible_circumference_fraction"])
        self.assertGreater(np.count_nonzero(result["mask"]), 0)

    def test_cli_has_only_android_discovery_bounds(self):
        destinations = {action.dest for action in parser()._actions}
        self.assertIn("android_center_region", destinations)
        self.assertIn("android_radius_fraction", destinations)


if __name__ == "__main__":
    unittest.main()
