import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from acquisition.route_tracker import (
    RouteCandidateAdvisor,
    RouteGlobalLocalizer,
    RouteLockedStateEstimator,
    RouteVisualTracker,
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

    def test_visual_tracker_uses_route_as_proposal_and_map_as_pose(self):
        class RefiningMap:
            def __init__(self):
                self.centers = []

            def refine_near(
                self, observation, mask, center, search_radius_px, score_min
            ):
                self.centers.append(tuple(center))
                return {
                    "valid": True,
                    "x": float(center[0]) + 3.0,
                    "y": float(center[1]) - 4.0,
                    "score": 0.8,
                    "margin": 0.2,
                    "selected_mode_id": "world",
                    "pose_authority": "current-frame-map-correlation",
                }

        with tempfile.TemporaryDirectory() as temporary:
            package, images = self._package(Path(temporary) / "route")
            map_localizer = RefiningMap()
            tracker = RouteVisualTracker(
                package, map_localizer, recovery_top_k=1
            )

            first = tracker.track(
                images[8], np.full((64, 64), 255, np.uint8)
            )
            second = tracker.track(
                images[8], np.full((64, 64), 255, np.uint8)
            )

            self.assertTrue(first["measurement_accepted"])
            self.assertEqual((first["x"], first["y"]), (83.0, -4.0))
            self.assertEqual(first["route_role"], "bounded-search-proposal-only")
            self.assertEqual(second["source"], "continuous-local")
            self.assertEqual(map_localizer.centers[-1], (83.0, -4.0))

    def test_visual_tracker_prefers_active_layer_refinement(self):
        class ActiveLayerMap:
            def __init__(self):
                self.active_calls = 0

            def refine_near(self, *args, **kwargs):
                raise AssertionError("all-layer refinement must not be used")

            def refine_active_near(
                self, observation, mask, center, search_radius_px, score_min
            ):
                self.active_calls += 1
                return {
                    "valid": True,
                    "x": float(center[0]),
                    "y": float(center[1]),
                    "score": 0.2,
                    "selected_mode_id": "world",
                }

        with tempfile.TemporaryDirectory() as temporary:
            package, images = self._package(Path(temporary) / "route")
            map_localizer = ActiveLayerMap()
            tracker = RouteVisualTracker(
                package, map_localizer, score_min=0.0, local_radius_px=12.0
            )
            tracker.seed(40.0, 0.0)

            result = tracker.track(
                images[4], np.full((64, 64), 255, np.uint8)
            )

            self.assertTrue(result["measurement_accepted"])
            self.assertEqual(map_localizer.active_calls, 1)

    def test_visual_tracker_holds_then_confirms_at_trained_transition_anchor(self):
        class TransitionMap:
            def __init__(self):
                self.centers = []
                self.valid = True

            def refine_near(
                self, observation, mask, center, search_radius_px, score_min
            ):
                self.centers.append((tuple(center), float(search_radius_px)))
                return {
                    "valid": self.valid,
                    "x": float(center[0]) + 2.0,
                    "y": float(center[1]) + 1.0,
                    "score": 0.9,
                    "margin": 0.2,
                    "selected_mode_id": "town",
                    "pose_authority": "current-frame-map-correlation",
                }

        with tempfile.TemporaryDirectory() as temporary:
            package, images = self._package(Path(temporary) / "route")
            map_localizer = TransitionMap()
            tracker = RouteVisualTracker(package, map_localizer)
            tracker.seed(50.0, 0.0, timestamp_ns=1_000_000_000)

            pending = tracker.arm_trained_transition("world", "town")
            held = tracker.track(
                images[6],
                np.full((64, 64), 255, np.uint8),
                timestamp_ns=1_100_000_000,
            )

            self.assertEqual(pending["target_state_index"], 6)
            self.assertEqual(pending["target_canonical_xy"], [60.0, 0.0])
            self.assertTrue(held["held"])
            self.assertEqual(held["decision"], "held:trained-map-transition")
            self.assertEqual((held["x"], held["y"]), (50.0, 0.0))
            self.assertEqual(map_localizer.centers, [])

            self.assertTrue(tracker.confirm_trained_transition_layer("town"))
            map_localizer.valid = False
            awaiting = tracker.track(
                images[6],
                np.full((64, 64), 255, np.uint8),
                timestamp_ns=1_200_000_000,
            )

            self.assertTrue(awaiting["transition_waiting"])
            self.assertEqual(
                awaiting["decision"],
                "held:trained-transition-awaiting-visual-confirmation",
            )
            self.assertEqual(len(map_localizer.centers), 1)

            map_localizer.valid = True
            confirmed = tracker.track(
                images[6],
                np.full((64, 64), 255, np.uint8),
                timestamp_ns=1_300_000_000,
            )

            self.assertTrue(confirmed["measurement_accepted"])
            self.assertEqual(confirmed["source"], "route-transition-anchor")
            self.assertEqual(map_localizer.centers, [((60.0, 0.0), 20.0)] * 2)
            self.assertEqual((confirmed["x"], confirmed["y"]), (62.0, 1.0))
            self.assertIsNone(confirmed["trained_transition"])

    def test_visual_tracker_holds_prior_pose_on_implausible_jump(self):
        class JumpingMap:
            def __init__(self):
                self.calls = 0

            def refine_near(
                self, observation, mask, center, search_radius_px, score_min
            ):
                self.calls += 1
                x = 21.0 if self.calls == 1 else 70.0
                return {
                    "valid": True,
                    "x": x,
                    "y": 20.0,
                    "score": 0.85,
                    "margin": 0.2,
                    "selected_mode_id": "world",
                    "pose_authority": "current-frame-map-correlation",
                }

        with tempfile.TemporaryDirectory() as temporary:
            package, images = self._package(Path(temporary) / "route")
            tracker = RouteVisualTracker(package, JumpingMap())
            tracker.seed(20.0, 20.0)
            first = tracker.track(
                images[0],
                np.full((64, 64), 255, np.uint8),
                timestamp_ns=1_000_000_000,
            )
            second = tracker.track(
                images[0],
                np.full((64, 64), 255, np.uint8),
                timestamp_ns=1_010_000_000,
            )

            self.assertTrue(first["measurement_accepted"])
            self.assertFalse(second["measurement_accepted"])
            self.assertTrue(second["pose_available"])
            self.assertTrue(second["held"])
            self.assertTrue(second["continuity_rejected"])
            self.assertEqual(second["decision"], "held:continuity-jump")
            self.assertEqual((second["x"], second["y"]), (21.0, 20.0))
            self.assertEqual((second["measured_x"], second["measured_y"]), (70.0, 20.0))

    def test_visual_tracker_continuity_uses_time_since_last_accepted_pose(self):
        class DelayedMap:
            def refine_near(
                self, observation, mask, center, search_radius_px, score_min
            ):
                return {
                    "valid": True,
                    "x": 70.0,
                    "y": 20.0,
                    "score": 0.85,
                    "margin": 0.2,
                    "selected_mode_id": "world",
                    "pose_authority": "current-frame-map-correlation",
                }

        with tempfile.TemporaryDirectory() as temporary:
            package, images = self._package(Path(temporary) / "route")
            tracker = RouteVisualTracker(package, DelayedMap())
            tracker.continuity_speed_limit_px_s = 120.0
            tracker.seed(20.0, 20.0, timestamp_ns=1_000_000_000)
            mask = np.full((64, 64), 255, np.uint8)

            for timestamp_ns in (
                1_010_000_000,
                1_100_000_000,
                1_200_000_000,
                1_300_000_000,
                1_400_000_000,
            ):
                held = tracker.track(
                    images[0], mask, timestamp_ns=timestamp_ns
                )
                self.assertTrue(held["continuity_rejected"])
                self.assertEqual((held["x"], held["y"]), (20.0, 20.0))

            accepted = tracker.track(
                images[0], mask, timestamp_ns=1_500_000_000
            )

            self.assertTrue(accepted["measurement_accepted"])
            self.assertAlmostEqual(accepted["continuity_step_limit_px"], 60.0)
            self.assertEqual((accepted["x"], accepted["y"]), (70.0, 20.0))

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
