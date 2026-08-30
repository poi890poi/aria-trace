import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from acquisition.session_minimap_localization import (
    localize_session_minimap,
    load_session_registration,
    nearest_synchronized_pair,
    parser,
    project_android_boundary,
)


class SessionMinimapLocalizationTests(unittest.TestCase):
    def test_android_only_session_runs_verified_boundary_without_hik_projection(self):
        frames = np.zeros((12, 120, 200, 3), np.uint8)
        times = np.arange(12, dtype=np.int64) * 1_000_000
        fitted = {
            "outer_boundary": {
                "center_x": 30.0,
                "center_y": 28.0,
                "radius": 18.0,
                "confidence": 0.8,
            },
            "model": {"discovery": {"method": "automatic", "selected": {}}},
            "evidence": [],
        }
        fake_session = mock.Mock()
        fake_session.manifest = {
            "status": "complete",
            "session_id": "adb-only",
            "context": {"capture_kind": "zigzag_minimap_source_data"},
        }
        fake_session.frames_by_stream = {"android_phone": [{}] * 12}
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "acquisition.session_minimap_localization.SessionReader",
            return_value=fake_session,
        ), mock.patch(
            "acquisition.session_minimap_localization.read_representative_frames",
            return_value=(frames, times),
        ), mock.patch(
            "acquisition.session_minimap_localization.calibrate_minimap_boundary_frames",
            return_value=fitted,
        ):
            output = Path(temporary) / "result"
            result = localize_session_minimap(Path(temporary), output)
            with np.load(str(output / "minimap_geometry.npz")) as saved:
                saved_files = set(saved.files)
        self.assertEqual("android_only", result["provenance"]["capture_mode"])
        self.assertEqual(
            "not_applicable", result["cross_source_registration"]["status"]
        )
        self.assertEqual("not_available", result["hik_session_observation"]["status"])
        self.assertEqual({"android_boundary", "android_mask"}, saved_files)

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
