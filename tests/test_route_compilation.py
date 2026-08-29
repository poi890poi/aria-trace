import unittest

import numpy as np

from acquisition.live_tracker import GlobalFix
from acquisition.route_compilation import localize_route_frames


class FakeExtractor:
    def extract(self, image):
        return image.copy(), np.full(image.shape[:2], 255, np.uint8)


class FakeLayeredLocalizer:
    def __init__(self, fixes):
        self.fixes = iter(fixes)
        self.searches = []

    def localize(
        self,
        observation,
        mask,
        yaw_prior_deg=None,
        search_center_xy=None,
        search_radius_px=None,
    ):
        self.searches.append((search_center_xy, search_radius_px))
        return next(self.fixes)


class RouteCompilationTests(unittest.TestCase):
    @staticmethod
    def _fix(x, mode="world", valid=True):
        return GlobalFix(
            x,
            20.0,
            5.0,
            1.0,
            0.8,
            0.2,
            1.0,
            valid=valid,
            rejection_reasons=() if valid else ("weak",),
            diagnostics={
                "map_layer": {
                    "selected_mode_id": mode,
                    "mode_likelihoods": {mode: 0.8},
                }
            },
        )

    def test_keeps_continuous_samples_and_records_rejections(self):
        records = [
            {"frame_index": index, "session_time_ns": index * 100_000_000}
            for index in range(6)
        ]
        images = [
            np.random.RandomState(index).randint(0, 255, (64, 64, 3), np.uint8)
            for index in range(6)
        ]
        localizer = FakeLayeredLocalizer(
            [
                self._fix(10.0),
                self._fix(20.0),
                self._fix(200.0),
                self._fix(30.0, valid=False),
                self._fix(30.0, mode="town"),
                self._fix(40.0, mode="town"),
            ]
        )

        accepted, rejected = localize_route_frames(
            records,
            images,
            FakeExtractor(),
            localizer,
            max_step_px=50.0,
        )

        self.assertEqual([item["x"] for item in accepted], [10.0, 20.0, 30.0, 40.0])
        self.assertEqual([item["mode_id"] for item in accepted], ["world", "world", "town", "town"])
        self.assertEqual(len(rejected), 2)
        self.assertTrue(rejected[0]["reason"].startswith("discontinuous:"))
        self.assertEqual(rejected[1]["reason"], "localization:weak")
        self.assertIsNone(localizer.searches[0][0])
        self.assertEqual(localizer.searches[-1][0], (30.0, 20.0))


if __name__ == "__main__":
    unittest.main()
