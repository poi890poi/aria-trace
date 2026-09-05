from concurrent.futures import Future
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch

import numpy as np

from aria_trace.services.tracking.runtime import GlobalFix, TwoRateRealtimeTracker
from aria_trace.services.localization.route.tracker import RouteVisualTracker
from rig_runtime.services.calibration.minimap.transition import TransitionController
from benchmarks.localization.tracking_candidates import candidate_sources, installed_candidates
from aria_trace.services.tracking import Pose2D
from tests import test_live_tracker as fixtures


class TrackingCandidateTests(unittest.TestCase):
    def test_all_declared_edits_compile_and_restore_production_methods(self):
        original = TwoRateRealtimeTracker.update
        for variant in ("A", "B", "C", "D", "E", "F", "G", "H", "AB", "CD", "ABCDE", "ABF", "CDH", "AG", "AEG"):
            for key, source in candidate_sources(variant).items():
                compile(source, str(key), "exec")
            with installed_candidates(variant):
                pass
            self.assertIs(TwoRateRealtimeTracker.update, original)

    def test_transition_authority_uses_observation_uncertainty_not_newer_pose(self):
        def exercise(variant):
            controller = TransitionController({"source_mode_id":"world", "target_mode_id":"town",
                "transition_zones":[{"zone_id":"tiny", "center_xy":[0,0], "radius_px":1.65}],
                "runtime":{"confirmation_count":2, "minimum_mode_margin":.2}}, confirmation_count=2)
            controller.set_active_mode("town")
            tracker = object.__new__(TwoRateRealtimeTracker)
            tracker.transition_controller = controller
            tracker.route_visual_tracker = None
            tracker._representation_position_sigma = 3.0
            tracker.fusion = SimpleNamespace(state=SimpleNamespace(position_sigma_m=100))
            future = Future()
            future.set_result({"valid":True,"likelihoods":{"world":1,"town":0},
                               "canonical_xy_read_only":[2.48,0]})
            tracker._representation_future = future
            with installed_candidates(variant):
                tracker._consume_representation_observation(1)
            return tracker._last_representation_observation["controller"]
        self.assertFalse(exercise("")["transition_armed"])
        self.assertTrue(exercise("A")["transition_armed"])

    def test_reverse_anchor_is_only_a_proposal_and_still_requires_visual_confirmation(self):
        tracker = object.__new__(RouteVisualTracker)
        tracker.previous_xy = (20,20)
        tracker._trained_transition = None
        tracker.package = SimpleNamespace(states=[{"canonical_xy":[10,10]}, {"canonical_xy":[20,20]}],
            transitions=[{"source_mode_id":"world", "target_mode_id":"town", "last_source_state_index":0,
                          "first_target_state_index":1, "last_source_canonical_xy":[10,10], "first_target_canonical_xy":[20,20]}])
        self.assertIsNone(tracker.arm_trained_transition("town","world"))
        with installed_candidates("B"):
            proposal = tracker.arm_trained_transition("town","world")
        self.assertEqual(proposal["target_canonical_xy"], [10,10])
        self.assertFalse(proposal["target_layer_confirmed"])
        self.assertEqual(tracker.previous_xy, (20,20))

    def test_latest_consensus_does_not_average_positions_from_different_times(self):
        fixes = [GlobalFix(10,20,0,1,.9,.2,1), GlobalFix(20,20,10,1,.9,.2,1)]
        self.assertEqual(TwoRateRealtimeTracker._mean_fix(fixes).x, 15)
        with installed_candidates("C"):
            self.assertEqual(TwoRateRealtimeTracker._mean_fix(fixes).x, 20)

    def test_accepted_local_motion_must_not_erase_pending_global_recovery(self):
        def exercise(variant):
            tracker = fixtures.TwoRateTrackerTests._tracker(fixtures.ImmediateLocalizer())
            frame = np.zeros((100,100,3), np.uint8)
            tracker.fusion.initialize(Pose2D(10,20,15))
            tracker.fusion.state.position_sigma_m = 40
            tracker._recovery_request_active = True
            tracker.recovery_consensus_count = 2
            tracker.last_global_ns = 1
            tracker.sequence = 1
            tracker.previous_minimap = tracker.extractor.extract(frame)[0]
            try:
                with installed_candidates(variant), patch("aria_trace.services.tracking.runtime.estimate_masked_shift", return_value=((0,0),.9)):
                    for timestamp, x in ((2,12),(3,13)):
                        future = Future()
                        future.set_result(GlobalFix(x,21,15,1,.9,.2,2))
                        tracker._global_future = future
                        result = tracker.update(frame, timestamp)
                    return result["global_fix"]["decision"]
            finally:
                tracker.close()
        self.assertEqual(exercise(""), "awaiting-recovery-consensus:1/2")
        self.assertEqual(exercise("D"), "recovered-consensus")

    def test_arming_alone_need_not_freeze_a_valid_current_layer_measurement(self):
        def exercise(variant):
            tracker = object.__new__(RouteVisualTracker)
            tracker.previous_xy = (10,20)
            tracker.previous_time_ns = 1
            tracker.local_radius_px = 12
            tracker.continuity_speed_limit_px_s = 120
            tracker._trained_transition = {"target_layer_confirmed":False, "target_mode_id":"town"}
            tracker._refine = lambda *args: {"valid":True,"x":11,"y":20,"score":.9}
            with installed_candidates(variant):
                result = tracker.track(None, None, timestamp_ns=100_000_000)
            return result, tracker._trained_transition
        baseline, _ = exercise("")
        self.assertTrue(baseline["held"])
        changed, pending = exercise("F")
        self.assertTrue(changed["measurement_accepted"])
        self.assertEqual(changed["x"], 11)
        self.assertFalse(pending["target_layer_confirmed"])

    def test_scene_motion_removal_preserves_minimap_motion_measurement(self):
        tracker = fixtures.TwoRateTrackerTests._tracker(fixtures.ImmediateLocalizer())
        frame = np.zeros((100,100,3), np.uint8)
        tracker.fusion.initialize(Pose2D(10,20,0))
        tracker.sequence = 1
        tracker.previous_minimap = tracker.extractor.extract(frame)[0]
        tracker.scene_estimator = SimpleNamespace(update=lambda frame: self.fail("scene motion branch ran"))
        try:
            with installed_candidates("H"), patch("aria_trace.services.tracking.runtime.estimate_masked_shift", return_value=((2,0),.9)):
                result = tracker.update(frame, 2)
            self.assertTrue(result["local_motion"]["applied"])
            self.assertAlmostEqual(result["pose"]["x"], 8)
            self.assertEqual(result["scene_yaw"]["status"], "bypassed:scene-motion-ablation")
        finally:
            tracker.close()

    def test_cursor_alternatives_publish_completed_current_frame_result(self):
        def exercise(variant):
            tracker = fixtures.TwoRateTrackerTests._tracker(fixtures.ImmediateLocalizer(), cursor_pose_estimator=fixtures.FakeCursorPoseEstimator())
            tracker._cursor_executor.shutdown(wait=True)
            def submit(function, *args):
                future = Future()
                future.set_result(function(*args))
                return future
            tracker._cursor_executor = SimpleNamespace(submit=submit, shutdown=lambda **kwargs:None)
            tracker.fusion.initialize(Pose2D(10,20,15))
            timestamp = time.perf_counter_ns()
            try:
                with installed_candidates(variant):
                    result = tracker.update(np.zeros((100,100,3), np.uint8), timestamp)
                return result, timestamp
            finally:
                tracker.close()
        self.assertFalse(exercise("")[0]["cursor_pose_fresh"])
        for variant in ("E", "G", "EG"):
            result, timestamp = exercise(variant)
            self.assertTrue(result["cursor_pose_fresh"])
            self.assertEqual(result["cursor_pose"]["session_time_ns"], timestamp)


if __name__ == "__main__":
    unittest.main()
