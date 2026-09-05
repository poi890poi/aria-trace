"""Production regressions distinct from the frozen candidate comparisons."""

import unittest
from types import SimpleNamespace
from concurrent.futures import Future

import numpy as np

from aria_trace.services.tracking import Pose2D
from aria_trace.services.tracking.runtime import TwoRateRealtimeTracker
from aria_trace.services.localization.route.tracker import RouteVisualTracker
from rig_runtime.services.calibration.minimap.transition import TransitionController
from tests import test_live_tracker as fixtures


class AtlasTrackingIntegrationTests(unittest.TestCase):
    def test_atlas_tracking_needs_no_route_and_does_not_accumulate_scene_motion(self):
        class Localizer(fixtures.ImmediateLocalizer):
            def refine_near(self, observation, mask, center, **kwargs):
                return {"valid": True, "x": center[0]+1, "y": center[1], "score": .9}

        tracker = fixtures.TwoRateTrackerTests._tracker(Localizer())
        try:
            tracker.fusion.initialize(Pose2D(10, 20, 0))
            tracker.scene_estimator = SimpleNamespace(update=lambda *args: self.fail("scene integration ran"))
            result = tracker.update(np.zeros((100, 100, 3), np.uint8), 1)
            self.assertIsNone(tracker.route_visual_tracker.package)
            self.assertIsNone(tracker.global_candidate_advisor)
            self.assertTrue(result["xy_measurement_fresh_accepted"])
            self.assertFalse(result["local_motion"]["applied"])
            self.assertEqual(result["route_tracking"]["route_role"], "none")
            self.assertEqual(result["pose"]["x"], 11)
        finally:
            tracker.close()

    def test_non_refining_map_retains_relative_tracker(self):
        tracker = fixtures.TwoRateTrackerTests._tracker(fixtures.ImmediateLocalizer())
        try:
            self.assertIsNone(tracker.route_visual_tracker)
        finally:
            tracker.close()

    def test_delayed_seed_refines_current_image_then_uses_local_radius(self):
        radii = []
        class Localizer:
            def refine_near(self, observation, mask, center, **kwargs):
                radii.append(kwargs["search_radius_px"])
                return {"valid": abs(45-center[0]) <= kwargs["search_radius_px"], "x": 45, "y": 20, "score": .9}
        tracker = RouteVisualTracker(None, Localizer(), local_radius_px=12)
        tracker.seed(10, 20)
        self.assertIsNone(tracker.arm_trained_transition("world", "town"))
        self.assertTrue(tracker.track(None, None, 1_000_000_000)["measurement_accepted"])
        self.assertTrue(tracker.track(None, None, 2_000_000_000)["measurement_accepted"])
        self.assertEqual(radii, [55, 12])


if __name__ == "__main__":
    unittest.main()
