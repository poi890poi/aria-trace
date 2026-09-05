"""Production regressions distinct from the frozen candidate comparisons."""

import unittest
import time
from types import SimpleNamespace
from concurrent.futures import Future

import numpy as np

from aria_trace.services.tracking import Pose2D
from aria_trace.services.tracking.runtime import TwoRateRealtimeTracker
from aria_trace.services.localization.route.tracker import RouteVisualTracker
from rig_runtime.services.calibration.minimap.transition import TransitionController
from tests import test_live_tracker as fixtures


class AtlasTrackingIntegrationTests(unittest.TestCase):
    def test_armed_transition_observes_at_control_cadence_only_while_armed(self):
        def exercise(armed):
            class Localizer(fixtures.ImmediateLocalizer):
                transition_model = {"source_mode_id":"world", "target_mode_id":"town"}
                calls = 0
                def refine_near(self, observation, mask, center, **kwargs):
                    return {"valid": True, "x": center[0], "y": center[1], "score": .9}
                def observe_modes(self, *args):
                    self.calls += 1
                    return {"valid": False}
            localizer = Localizer()
            tracker = fixtures.TwoRateTrackerTests._tracker(localizer)
            tracker._representation_executor.shutdown(wait=True)
            def submit(function, *args):
                future = Future()
                future.set_result(function(*args))
                return future
            tracker._representation_executor = SimpleNamespace(submit=submit, shutdown=lambda **kwargs: None)
            try:
                tracker.fusion.initialize(Pose2D(10, 20, 0))
                tracker._activate_map_mode("world", update_scale=False)
                tracker.last_representation_ns = 1_000_000_000
                tracker._last_representation_observation = {"controller":{"transition_armed":armed}}
                tracker.update(np.zeros((100,100,3),np.uint8), 1_050_000_000)
                return localizer.calls
            finally:
                tracker.close()
        self.assertEqual(exercise(False),0)
        self.assertEqual(exercise(True),1)

    def test_completed_current_frame_heading_is_published_in_same_update(self):
        tracker = fixtures.TwoRateTrackerTests._tracker(
            fixtures.ImmediateLocalizer(), cursor_pose_estimator=fixtures.FakeCursorPoseEstimator())
        tracker._cursor_executor.shutdown(wait=True)
        def submit(function, *args):
            future = Future()
            future.set_result(function(*args))
            return future
        tracker._cursor_executor = SimpleNamespace(submit=submit, shutdown=lambda **kwargs: None)
        try:
            tracker.fusion.initialize(Pose2D(10, 20, 15))
            timestamp = time.perf_counter_ns()
            result = tracker.update(np.zeros((100, 100, 3), np.uint8), timestamp)
            self.assertTrue(result["cursor_pose_fresh"])
            self.assertEqual(result["cursor_pose"]["session_time_ns"], timestamp)
        finally:
            tracker.close()

    def test_expired_frame_does_not_wait_for_pending_heading(self):
        tracker = fixtures.TwoRateTrackerTests._tracker(fixtures.ImmediateLocalizer())
        future = Future()
        future.result = lambda *args, **kwargs: self.fail("waited beyond source deadline")
        tracker._cursor_future = future
        try:
            result = tracker.update(np.zeros((100, 100, 3), np.uint8), 1)
            self.assertFalse(result["cursor_pose_fresh"])
            self.assertIs(tracker._cursor_future, future)
        finally:
            tracker._cursor_future = None
            tracker.close()

    def test_transition_uses_uncertainty_from_observed_frame(self):
        tracker = object.__new__(TwoRateRealtimeTracker)
        tracker.transition_controller = TransitionController({
            "source_mode_id": "world", "target_mode_id": "town",
            "transition_zones": [{"zone_id": "tiny", "center_xy": [0, 0], "radius_px": 1.65}],
            "runtime": {"confirmation_count": 2, "minimum_mode_margin": .2},
        }, confirmation_count=2)
        tracker.transition_controller.set_active_mode("town")
        tracker.route_visual_tracker = None
        tracker._representation_position_sigma = 3.0
        tracker.fusion = SimpleNamespace(state=SimpleNamespace(position_sigma_m=100))
        future = Future()
        future.set_result({"valid": True, "likelihoods": {"world": 1, "town": 0}, "canonical_xy_read_only": [2.48, 0]})
        tracker._representation_future = future
        tracker._consume_representation_observation(1)
        self.assertTrue(tracker._last_representation_observation["controller"]["transition_armed"])

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
