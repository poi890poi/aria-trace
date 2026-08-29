import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from acquisition.route_tracker import (
    RouteCandidateAdvisor,
    RouteGlobalLocalizer,
    RouteLockedStateEstimator,
)
from replay.route_tracking import (
    RouteTrackingPackage,
    compile_route_tracking_package,
    describe_minimap,
)


class RouteTrackerTests(unittest.TestCase):
    @staticmethod
    def _package(root: Path):
        observations = []
        images = []
        for index in range(12):
            image = np.zeros((64, 64, 3), np.uint8)
            cv2.line(
                image,
                (5 + index * 2, 4),
                (30 + index * 2, 58),
                (100 + index * 10, 230, 40),
                3,
            )
            images.append(image)
            observations.append(
                {
                    "source_frame_index": index,
                    "session_time_ns": 1_000_000_000 + index * 200_000_000,
                    "x": index * 10.0,
                    "y": 0.0,
                    "mode_id": "world" if index < 6 else "town",
                    "map_alignment_deg": 15.0,
                    "descriptor": describe_minimap(image),
                }
            )
        compile_route_tracking_package(
            observations,
            root,
            route_id="route-a",
            atlas_id="atlas-a",
            coordinate_space_id="map-atlas:atlas-a:canonical-map-px",
            corridor_radius_px=20.0,
        )
        return RouteTrackingPackage(root), images

    def test_global_localization_searches_only_route_templates(self):
        with tempfile.TemporaryDirectory() as temporary:
            package, images = self._package(Path(temporary) / "route")
            localizer = RouteGlobalLocalizer(package, margin_min=0.0)
            mask = np.full((64, 64), 255, np.uint8)

            fix = localizer.localize(images[8], mask)

            self.assertTrue(fix.valid)
            self.assertAlmostEqual(fix.x, 80.0)
            self.assertEqual(
                fix.diagnostics["route"]["selected_state_index"], 8
            )
            self.assertLessEqual(fix.search_area_fraction, 3.0 / 12.0)

    def test_candidate_advisor_clusters_adjacent_route_states(self):
        with tempfile.TemporaryDirectory() as temporary:
            package, images = self._package(Path(temporary) / "route")
            advisor = RouteCandidateAdvisor(
                package, top_k=6, adjacent_state_gap=2, search_padding_px=50.0
            )

            proposal = advisor.propose(
                images[8], np.full((64, 64), 255, np.uint8)
            )

            self.assertEqual(proposal["policy"], "candidate-window-only")
            self.assertIn(8, proposal["cluster_state_indexes"])
            self.assertGreater(proposal["cluster_candidate_count"], 1)
            self.assertGreaterEqual(proposal["radius_px"], 50.0)

    def test_state_estimator_advances_locally_and_projects_to_route(self):
        with tempfile.TemporaryDirectory() as temporary:
            package, images = self._package(Path(temporary) / "route")
            estimator = RouteLockedStateEstimator(package)
            estimator.initialize_near(40.0, 0.0)
            mask = np.full((64, 64), 255, np.uint8)

            result = estimator.update(images[5], mask, (51.0, 8.0))

            self.assertTrue(result["accepted"])
            self.assertGreaterEqual(result["state_index"], 4)
            self.assertLessEqual(result["state_index"], 6)
            self.assertAlmostEqual(result["canonical_xy"][1], 0.0, delta=0.01)
            self.assertAlmostEqual(result["cross_track_error_px"], 8.0, delta=0.1)
            self.assertLess(result["route_progress"], 0.60)

    def test_mode_switch_is_tied_to_route_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            package, images = self._package(Path(temporary) / "route")
            estimator = RouteLockedStateEstimator(package)
            estimator.initialize_near(50.0, 0.0)
            mask = np.full((64, 64), 255, np.uint8)

            result = estimator.update(images[7], mask, (70.0, 0.0))

            self.assertTrue(result["mode_switched"])
            self.assertEqual(result["active_mode_id"], "town")
            self.assertTrue(result["reset_local_reference"])


if __name__ == "__main__":
    unittest.main()
