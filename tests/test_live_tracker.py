import threading
import time
import json
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from acquisition.live_tracker import (
    GlobalFix,
    GlobalMapLocalizer,
    TwoRateRealtimeTracker,
    render_map_overlay,
    render_minimap_route_overlay,
)
from acquisition.map_layers import LayeredGlobalLocalizer
from acquisition.models import FramePacket
from acquisition.profiles import ProfileCatalog
from acquisition.workbench import AcquisitionWorkbench
from aria_trace.services.tracking import Pose2D
from replay.route_tracking import compile_route_tracking_package


class GlobalMapLocalizerTests(unittest.TestCase):
    def test_rejected_fix_reports_unavailable_metrics_as_json_null(self):
        random = np.random.RandomState(7)
        mosaic = random.randint(0, 256, (220, 280, 3), dtype=np.uint8)
        localizer = GlobalMapLocalizer(mosaic)
        observation = np.zeros((100, 100, 3), np.uint8)
        mask = np.full((100, 100), 255, np.uint8)

        fix = localizer.localize(observation, mask)

        self.assertFalse(fix.valid)
        self.assertEqual(fix.rejection_reasons, ("too-few-observation-features",))
        self.assertIsNone(fix.reprojection_p95_px)
        self.assertIsNone(fix.center_agreement_px)
        json.dumps(fix.__dict__, allow_nan=False)

    def test_localizes_exact_observed_patch_on_mosaic(self):
        random = np.random.RandomState(42)
        mosaic = random.randint(0, 256, (300, 360, 3), dtype=np.uint8)
        center_x, center_y = 220, 150
        observation = mosaic[100:200, 170:270].copy()
        mask = np.zeros((100, 100), np.uint8)
        cv2.circle(mask, (50, 50), 47, 255, -1)
        localizer = GlobalMapLocalizer(
            mosaic,
        )

        fix = localizer.localize(observation, mask, yaw_prior_deg=0.0)

        self.assertAlmostEqual(fix.x, center_x, delta=2.0)
        self.assertAlmostEqual(fix.y, center_y, delta=2.0)
        self.assertAlmostEqual(fix.yaw_deg, 0.0, delta=2.1)
        self.assertGreater(fix.score, 0.90)
        self.assertTrue(fix.valid)
        self.assertGreaterEqual(fix.inlier_count, 6)
        self.assertLess(fix.elapsed_ms, 1000.0)
        self.assertIsNotNone(fix.diagnostics)
        self.assertEqual(
            fix.diagnostics["observation"].shape, observation.shape
        )
        self.assertEqual(
            fix.diagnostics["candidate_overlay"].shape, mosaic.shape
        )
        self.assertEqual(
            fix.diagnostics["correlation_heatmap"].ndim, 3
        )

    def test_position_prior_bounds_correlation_without_changing_coordinates(self):
        random = np.random.RandomState(43)
        mosaic = random.randint(0, 256, (500, 700, 3), dtype=np.uint8)
        center_x, center_y = 410, 260
        observation = mosaic[210:310, 360:460].copy()
        mask = np.zeros((100, 100), np.uint8)
        cv2.circle(mask, (50, 50), 47, 255, -1)
        localizer = GlobalMapLocalizer(
            mosaic,
            localization_to_original_3x3=[
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        )

        fix = localizer.localize(
            observation,
            mask,
            search_center_xy=(2 * (center_x + 8), 2 * (center_y - 5)),
            search_radius_px=160,
        )

        self.assertTrue(fix.valid)
        self.assertAlmostEqual(fix.x, 2 * center_x, delta=4.0)
        self.assertAlmostEqual(fix.y, 2 * center_y, delta=4.0)
        self.assertLess(fix.search_area_fraction, 0.25)
        self.assertIsNotNone(fix.search_bounds_xyxy)


class BlockingLocalizer:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def localize(self, observation, mask, yaw_prior_deg=None):
        self.started.set()
        if not self.release.wait(2.0):
            raise RuntimeError("test did not release global localization")
        return GlobalFix(80.0, 70.0, 15.0, 1.0, 0.91, 0.12, 123.0)


class ImmediateLocalizer:
    def __init__(self):
        self.calls = 0

    def localize(self, observation, mask, yaw_prior_deg=None):
        self.calls += 1
        offset = 0.5 * (self.calls - 1)
        return GlobalFix(80.0 + offset, 70.0, 15.0, 1.0, 0.91, 0.12, 8.0)


class FakeCursorPoseEstimator:
    def __init__(self):
        self.last_frame_shape = None

    def estimate(self, frame, frame_index=None, session_time_ns=None):
        self.last_frame_shape = frame.shape
        return {
            "detected": True,
            "angle_screen_deg": 35.0,
            "confidence": 0.80,
            "session_time_ns": session_time_ns,
        }

    @staticmethod
    def public_result(value):
        return dict(value)


class TemporalCursorPoseEstimator(FakeCursorPoseEstimator):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.calibration = {
            "cursor_temporal_dynamics": {
                "recommended_runtime_envelope": {
                    "calibrated_turn_rate_p99_deg_s": 180.0,
                    "normal_turn_rate_p95_deg_s": 60.0,
                    "normal_turn_rate_p99_deg_s": 90.0,
                    "calibrated_angular_acceleration_p99_deg_s2": 360.0,
                    "ordinary_heading_jump_p99_deg": 2.0,
                }
            }
        }

    def estimate(
        self,
        frame,
        frame_index=None,
        session_time_ns=None,
        angle_prior_deg=None,
        search_half_width_deg=None,
    ):
        self.calls.append((angle_prior_deg, search_half_width_deg))
        return super().estimate(frame, frame_index, session_time_ns)


class TwoRateTrackerTests(unittest.TestCase):
    @staticmethod
    def _tracker(
        localizer,
        initial_consensus_count=1,
        global_interval_s=10.0,
        cursor_pose_estimator=None,
    ):
        mosaic = np.full((160, 180, 3), 40, np.uint8)
        minimap_config = {"crop_xywh": [0, 0, 80, 80]}
        minimap_calibration = {
            "outer_boundary": {"center_x": 40, "center_y": 40, "radius": 34}
        }
        scene_yaw = {
            "focal_ratio": 0.9,
            "camera_matrix": [[72, 0, 40], [0, 72, 40], [0, 0, 1]],
            "config": {"excluded_rects": []},
        }
        return TwoRateRealtimeTracker(
            mosaic,
            minimap_config,
            minimap_calibration,
            scene_yaw,
            global_interval_s=global_interval_s,
            localizer=localizer,
            initial_consensus_count=initial_consensus_count,
            cursor_pose_estimator=cursor_pose_estimator,
        )

    def test_initial_pose_requires_two_agreeing_global_fixes(self):
        localizer = ImmediateLocalizer()
        tracker = self._tracker(
            localizer, initial_consensus_count=2, global_interval_s=0.001
        )
        frame = np.zeros((100, 100, 3), np.uint8)
        decisions = []
        result = None
        try:
            for index in range(100):
                result = tracker.update(frame, index * 2_000_000 + 1)
                decision = (result.get("global_fix") or {}).get("decision")
                if decision and (not decisions or decisions[-1] != decision):
                    decisions.append(decision)
                if result["pose"] is not None:
                    break
                time.sleep(0.002)
            self.assertIn("awaiting-consensus:1/2", decisions)
            self.assertEqual(decisions[-1], "initialized-consensus")
            self.assertIsNotNone(result["pose"])
            self.assertGreaterEqual(localizer.calls, 2)
        finally:
            tracker.close()

    def test_calibrated_cursor_motion_supplies_a_bounded_next_search(self):
        estimator = TemporalCursorPoseEstimator()
        tracker = TwoRateRealtimeTracker(
            np.full((160, 180, 3), 40, np.uint8),
            {"crop_xywh": [0, 0, 80, 80]},
            {"outer_boundary": {"center_x": 40, "center_y": 40, "radius": 34}},
            {
                "focal_ratio": 0.9,
                "config": {"excluded_rects": []},
            },
            global_interval_s=10.0,
            localizer=ImmediateLocalizer(),
            initial_consensus_count=1,
            cursor_pose_estimator=estimator,
            cursor_interval_s=0.001,
            temporal_pose_search=True,
        )
        frame = np.zeros((100, 100, 3), np.uint8)
        try:
            for index in range(100):
                tracker.update(frame, 1_000_000_000 + index * 10_000_000)
                if len(estimator.calls) >= 2:
                    break
                time.sleep(0.001)
            self.assertIsNone(estimator.calls[0][0])
            self.assertAlmostEqual(estimator.calls[1][0], 35.0)
            self.assertGreater(estimator.calls[1][1], 2.0)
            self.assertLess(estimator.calls[1][1], 20.0)
            self.assertEqual(
                tracker._last_cursor_search["state"], "stable"
            )
            tracker._cursor_last_time_ns = 2_000_000_000
            tracker._cursor_tracking_state = "stable"
            _, stable_width = tracker._cursor_search(2_050_000_000)
            tracker._cursor_tracking_state = "turning"
            _, turning_width = tracker._cursor_search(2_050_000_000)
            self.assertGreater(turning_width, stable_width)
        finally:
            tracker.close()

    def test_global_search_does_not_block_high_rate_update(self):
        localizer = BlockingLocalizer()
        tracker = self._tracker(localizer)
        frame = np.zeros((100, 100, 3), np.uint8)
        try:
            first = tracker.update(frame, 1)
            self.assertTrue(localizer.started.wait(0.5))
            self.assertTrue(first["global_localization_running"])
            self.assertIsNone(first["pose"])

            second = tracker.update(frame, 2)
            self.assertTrue(second["global_localization_running"])
            self.assertIsNone(second["pose"])

            localizer.release.set()
            result = second
            for timestamp in range(3, 50):
                result = tracker.update(frame, timestamp)
                if result["global_fix_fresh"]:
                    break
                time.sleep(0.005)
            self.assertTrue(result["global_fix_fresh"])
            self.assertEqual(
                result["global_fix"]["decision"], "initialized-consensus"
            )
            self.assertAlmostEqual(result["pose"]["x"], 80.0)
            self.assertAlmostEqual(result["pose"]["y"], 70.0)
        finally:
            localizer.release.set()
            tracker.close()

    def test_scene_yaw_does_not_force_rotation_of_static_minimap(self):
        localizer = BlockingLocalizer()
        tracker = self._tracker(localizer)
        tracker.scene_estimator = SimpleNamespace(
            update=lambda _frame: SimpleNamespace(
                delta_deg=20.0,
                confidence=1.0,
                tracks=100,
                inliers=90,
                status="ok",
            )
        )
        frame = np.random.RandomState(5).randint(
            0, 255, (100, 100, 3), dtype=np.uint8
        )
        try:
            tracker.update(frame, 1)
            result = tracker.update(frame, 2)
            self.assertEqual(
                result["local_motion"]["rotation_compensation_sign"], 0.0
            )
            self.assertEqual(
                result["local_motion"]["map_alignment_delta_deg"], 0.0
            )
        finally:
            localizer.release.set()
            tracker.close()

    def test_implausible_local_shift_holds_the_last_pose(self):
        tracker = self._tracker(ImmediateLocalizer())
        tracker.fusion.initialize(Pose2D(80.0, 70.0, 15.0))
        tracker.scene_estimator = SimpleNamespace(
            update=lambda _frame: SimpleNamespace(
                delta_deg=0.0,
                confidence=1.0,
                tracks=100,
                inliers=90,
                status="ok",
            )
        )
        frame = np.random.RandomState(8).randint(
            0, 255, (100, 100, 3), dtype=np.uint8
        )
        try:
            tracker.update(frame, 1)
            with patch(
                "aria_trace.services.tracking.runtime.estimate_masked_shift",
                return_value=((50.0, -40.0), 0.95),
            ):
                result = tracker.update(frame, 2)

            self.assertEqual(result["mode"], "HOLD")
            self.assertEqual(
                result["local_motion"]["decision"],
                "rejected:implausible-displacement",
            )
            self.assertFalse(result["local_motion"]["applied"])
            self.assertAlmostEqual(result["pose"]["x"], 80.0)
            self.assertAlmostEqual(result["pose"]["y"], 70.0)
        finally:
            tracker.close()

    def test_weak_local_correlation_holds_the_last_pose(self):
        tracker = self._tracker(ImmediateLocalizer())
        tracker.fusion.initialize(Pose2D(80.0, 70.0, 15.0))
        frame = np.random.RandomState(9).randint(
            0, 255, (100, 100, 3), dtype=np.uint8
        )
        try:
            tracker.update(frame, 1)
            with patch(
                "aria_trace.services.tracking.runtime.estimate_masked_shift",
                return_value=((2.0, 1.0), 0.05),
            ):
                result = tracker.update(frame, 2)

            self.assertEqual(result["mode"], "HOLD")
            self.assertEqual(
                result["local_motion"]["decision"],
                "rejected:weak-correlation",
            )
            self.assertFalse(result["local_motion"]["applied"])
            self.assertAlmostEqual(result["pose"]["x"], 80.0)
            self.assertAlmostEqual(result["pose"]["y"], 70.0)
        finally:
            tracker.close()

    def test_global_search_stops_after_initial_lock(self):
        localizer = ImmediateLocalizer()
        tracker = self._tracker(
            localizer, initial_consensus_count=1, global_interval_s=0.001
        )
        frame = np.random.RandomState(10).randint(
            0, 255, (100, 100, 3), dtype=np.uint8
        )
        try:
            result = None
            for index in range(100):
                result = tracker.update(frame, index * 2_000_000 + 1)
                if result["pose"] is not None:
                    break
                time.sleep(0.002)
            self.assertIsNotNone(result["pose"])
            locked_call_count = localizer.calls

            for index in range(20):
                tracker.update(frame, 1_000_000_000 + index * 10_000_000)
                time.sleep(0.001)

            self.assertEqual(localizer.calls, locked_call_count)
        finally:
            tracker.close()

    def test_scale_transition_switches_layer_without_pose_jump(self):
        class ScaleTransitionLocalizer:
            transition_model = {
                "source_mode_id": "world",
                "target_mode_id": "town",
                "runtime": {
                    "confirmation_count": 2,
                    "minimum_mode_margin": 0.20,
                },
            }

            def __init__(self):
                self.calls = 0
                self.observations = 0
                self.active_mode_id = None

            def localize(self, observation, mask, yaw_prior_deg=None):
                self.calls += 1
                return GlobalFix(
                    80.0,
                    70.0,
                    15.0,
                    2.64,
                    0.91,
                    0.12,
                    8.0,
                    diagnostics={
                        "map_layer": {
                            "selected_mode_id": "world",
                            "mode_likelihoods": {"world": 0.92, "town": 0.08},
                        }
                    },
                )

            def observe_modes(
                self, observation, mask, canonical_xy, search_radius_px
            ):
                self.observations += 1
                likelihoods = (
                    {"world": 1.0, "town": 0.1}
                    if self.observations == 1
                    else {"world": 0.1, "town": 1.0}
                )
                return {
                    "valid": True,
                    "likelihoods": likelihoods,
                    "raw_correlation_scores": likelihoods,
                    "elapsed_ms": 1.0,
                    "pose_authority": "none",
                }

            def set_active_mode(self, mode_id):
                self.active_mode_id = mode_id

            @staticmethod
            def map_scale_for_mode(mode_id):
                return 2.64 if mode_id == "world" else 0.88

        localizer = ScaleTransitionLocalizer()
        tracker = TwoRateRealtimeTracker(
            np.full((160, 180, 3), 40, np.uint8),
            {"crop_xywh": [0, 0, 80, 80]},
            {"outer_boundary": {"center_x": 40, "center_y": 40, "radius": 34}},
            {"focal_ratio": 0.9, "config": {"excluded_rects": []}},
            global_interval_s=0.001,
            localizer=localizer,
            initial_consensus_count=1,
            relocalize_after_rejections=1,
            recovery_consensus_count=2,
            representation_interval_s=0.001,
        )
        frame = np.random.RandomState(15).randint(
            0, 255, (100, 100, 3), dtype=np.uint8
        )
        result = None
        try:
            with patch(
                "aria_trace.services.tracking.runtime.estimate_masked_shift",
                return_value=((0.0, 0.0), 0.90),
            ):
                for index in range(200):
                    result = tracker.update(frame, index * 10_000_000 + 1)
                    if result.get("map_transition"):
                        break
                    time.sleep(0.002)

            self.assertIsNotNone(result["map_transition"])
            self.assertEqual(result["active_map_mode_id"], "town")
            self.assertEqual(
                result["map_transition"]["position_policy"],
                "held-continuous-pose",
            )
            self.assertEqual(
                result["map_transition"]["evidence_source"],
                "continuous-local-representation-observer",
            )
            self.assertAlmostEqual(result["pose"]["x"], 80.0)
            self.assertAlmostEqual(result["pose"]["y"], 70.0)
            self.assertAlmostEqual(tracker.map_scale, 0.88)
            self.assertAlmostEqual(result["map_scale"], 0.88)
            self.assertEqual(localizer.active_mode_id, "town")
            self.assertEqual(
                result["map_representation_observation"]["pose_authority"],
                "none",
            )
        finally:
            tracker.close()

    def test_fusion_relocalize_state_requests_and_latches_recovery(self):
        tracker = self._tracker(ImmediateLocalizer())
        try:
            tracker.fusion.initialize(Pose2D(10.0, 20.0, 0.0))
            tracker.fusion.state.position_sigma_m = 41.0
            tracker.fusion._refresh_mode()

            self.assertTrue(tracker._recovery_requested())
            tracker.fusion.state.position_sigma_m = 3.0
            tracker.fusion._refresh_mode()
            self.assertTrue(tracker._recovery_requested())

            tracker._clear_recovery_request()
            self.assertFalse(tracker._recovery_requested())
        finally:
            tracker.close()

    def test_accepted_global_recovery_reseeds_route_visual_tracker(self):
        class VisualTracker:
            def __init__(self):
                self.previous_xy = (1.0, 1.0)
                self.seed_calls = []

            def seed(self, x, y):
                self.previous_xy = (float(x), float(y))
                self.seed_calls.append(self.previous_xy)

            def track(self, observation, mask, timestamp_ns=None):
                return {
                    "measurement_accepted": False,
                    "pose_available": True,
                    "held": True,
                    "x": self.previous_xy[0],
                    "y": self.previous_xy[1],
                    "score": 0.0,
                    "decision": "held-no-map-measurement",
                }

        visual = VisualTracker()
        tracker = TwoRateRealtimeTracker(
            np.full((160, 180, 3), 40, np.uint8),
            {"crop_xywh": [0, 0, 80, 80]},
            {"outer_boundary": {"center_x": 40, "center_y": 40, "radius": 34}},
            {"focal_ratio": 0.9, "config": {"excluded_rects": []}},
            global_interval_s=10.0,
            localizer=ImmediateLocalizer(),
            recovery_consensus_count=2,
            route_visual_tracker=visual,
        )
        frame = np.zeros((100, 100, 3), np.uint8)
        try:
            tracker.fusion.initialize(Pose2D(10.0, 20.0, 15.0))
            tracker.fusion.state.position_sigma_m = 40.0
            tracker.fusion._refresh_mode()
            tracker._recovery_request_active = True
            tracker.last_global_ns = 1

            first = Future()
            first.set_result(GlobalFix(80.0, 70.0, 15.0, 1.0, 0.9, 0.2, 2.0))
            tracker._global_future = first
            tracker.update(frame, 2)
            self.assertEqual(visual.seed_calls, [])

            second = Future()
            second.set_result(GlobalFix(81.0, 70.0, 15.0, 1.0, 0.9, 0.2, 2.0))
            tracker._global_future = second
            result = tracker.update(frame, 3)

            self.assertEqual(
                result["global_fix"]["decision"], "recovered-consensus"
            )
            self.assertEqual(len(visual.seed_calls), 1)
            self.assertEqual(
                visual.seed_calls[0],
                (tracker.fusion.state.pose.x, tracker.fusion.state.pose.y),
            )
        finally:
            tracker.close()

    def test_route_pose_projection_hook_is_not_supported(self):
        class RouteEstimator:
            def update(self, observation, mask, predicted_xy, timestamp_ns=None):
                return {
                    "accepted": True,
                    "canonical_xy": [42.0, 24.0],
                    "active_mode_id": "town",
                    "mode_switched": True,
                    "reset_local_reference": True,
                    "state": "route_track",
                }

        class ModeLocalizer(ImmediateLocalizer):
            def __init__(self):
                super().__init__()
                self.active_mode_id = None

            def set_active_mode(self, mode_id):
                self.active_mode_id = mode_id

        with self.assertRaisesRegex(TypeError, "route_state_estimator"):
            TwoRateRealtimeTracker(
                np.full((160, 180, 3), 40, np.uint8),
                {"crop_xywh": [0, 0, 80, 80]},
                {"outer_boundary": {"center_x": 40, "center_y": 40, "radius": 34}},
                {"focal_ratio": 0.9, "config": {"excluded_rects": []}},
                localizer=ModeLocalizer(),
                route_state_estimator=RouteEstimator(),
            )

    def test_route_visual_tracking_replaces_phase_xy_after_initial_lock(self):
        class VisualTracker:
            def __init__(self):
                self.previous_xy = None
                self.seeded = None
                self.calls = 0

            def seed(self, x, y):
                self.previous_xy = (float(x), float(y))
                self.seeded = self.previous_xy

            def track(self, observation, mask, timestamp_ns=None):
                self.calls += 1
                self.previous_xy = (87.0, 73.0)
                return {
                    "measurement_accepted": True,
                    "pose_available": True,
                    "held": False,
                    "x": 87.0,
                    "y": 73.0,
                    "score": 0.82,
                    "decision": "accepted-current-frame-map-pose",
                    "route_role": "bounded-search-proposal-only",
                }

        visual = VisualTracker()
        tracker = TwoRateRealtimeTracker(
            np.full((160, 180, 3), 40, np.uint8),
            {"crop_xywh": [0, 0, 80, 80]},
            {"outer_boundary": {"center_x": 40, "center_y": 40, "radius": 34}},
            {"focal_ratio": 0.9, "config": {"excluded_rects": []}},
            global_interval_s=0.001,
            localizer=ImmediateLocalizer(),
            initial_consensus_count=1,
            route_visual_tracker=visual,
        )
        frame = np.random.RandomState(18).randint(
            0, 255, (100, 100, 3), dtype=np.uint8
        )
        result = None
        try:
            with patch(
                "aria_trace.services.tracking.runtime.estimate_masked_shift"
            ) as phase_shift:
                for index in range(100):
                    result = tracker.update(frame, index * 2_000_000 + 1)
                    if result.get("route_tracking_fresh"):
                        break
                    time.sleep(0.002)

            self.assertEqual(visual.seeded, (80.0, 70.0))
            self.assertGreater(visual.calls, 0)
            self.assertEqual(
                (result["pose"]["x"], result["pose"]["y"]), (87.0, 73.0)
            )
            self.assertEqual(
                result["local_motion"]["decision"],
                "bypassed:route-map-correlation",
            )
            self.assertFalse(result["local_motion"]["applied"])
            self.assertTrue(result["route_tracking"]["measurement_accepted"])
            phase_shift.assert_not_called()
        finally:
            tracker.close()

    def test_exposes_player_heading_from_cursor_and_map_alignment(self):
        cursor_pose_estimator = FakeCursorPoseEstimator()
        tracker = self._tracker(
            ImmediateLocalizer(), cursor_pose_estimator=cursor_pose_estimator
        )
        tracker.cursor_interval_ns = 1
        frame = np.zeros((100, 100, 3), np.uint8)
        result = None
        try:
            for index in range(100):
                result = tracker.update(frame, index * 2_000_000 + 1)
                if (
                    result.get("pose")
                    and result["pose"].get("player_heading_map_deg") is not None
                ):
                    break
                time.sleep(0.002)
            self.assertAlmostEqual(result["pose"]["map_alignment_deg"], 15.0)
            self.assertAlmostEqual(result["pose"]["cursor_screen_deg"], 35.0)
            self.assertAlmostEqual(
                result["pose"]["player_heading_map_deg"], 50.0
            )
            self.assertEqual(
                result["pose"]["heading_source"],
                "calibrated_cursor_plus_map_alignment",
            )
            self.assertEqual(cursor_pose_estimator.last_frame_shape, (80, 80, 3))
        finally:
            tracker.close()

    def test_route_candidate_window_is_verified_and_falls_back_on_miss(self):
        class Advisor:
            def propose(self, observation, mask):
                return {"center_xy": [20.0, 30.0], "radius_px": 40.0,
                        "policy": "candidate-window-only"}

        class Verifier:
            supports_bounded_search = True

            def __init__(self):
                self.calls = []

            def localize(self, observation, mask, yaw_prior_deg=None,
                         search_center_xy=None, search_radius_px=None):
                self.calls.append((search_center_xy, search_radius_px))
                if len(self.calls) == 1:
                    return GlobalFix(0, 0, 0, 1, 0.1, 0, 1,
                                     valid=False,
                                     rejection_reasons=("weak",))
                return GlobalFix(55, 66, 7, 1, 0.8, 0.2, 2)

        verifier = Verifier()
        tracker = TwoRateRealtimeTracker(
            np.full((100, 100, 3), 40, np.uint8),
            {"crop_xywh": [0, 0, 80, 80]},
            {"outer_boundary": {"center_x": 40, "center_y": 40, "radius": 34}},
            {"focal_ratio": 0.9, "config": {"excluded_rects": []}},
            localizer=verifier,
            global_candidate_advisor=Advisor(),
        )
        try:
            fix = tracker._localize_global(
                np.zeros((68, 68, 3), np.uint8),
                np.full((68, 68), 255, np.uint8),
                None,
                None,
                None,
            )
            self.assertTrue(fix.valid)
            self.assertEqual(verifier.calls[0], ([20.0, 30.0], 40.0))
            self.assertEqual(verifier.calls[1], (None, None))
            self.assertEqual(
                tracker._last_route_assistance["status"],
                "bounded-miss-fell-back",
            )
        finally:
            tracker.close()

    def test_route_assistance_cannot_change_verified_pose(self):
        class Advisor:
            def propose(self, observation, mask):
                return {"center_xy": [40.0, 50.0], "radius_px": 60.0,
                        "policy": "candidate-window-only"}

        class DeterministicVerifier:
            supports_bounded_search = True

            def localize(self, observation, mask, yaw_prior_deg=None,
                         search_center_xy=None, search_radius_px=None):
                return GlobalFix(42.0, 24.0, 13.0, 0.88, 0.91, 0.12, 3.0)

        arguments = (
            np.full((100, 100, 3), 40, np.uint8),
            {"crop_xywh": [0, 0, 80, 80]},
            {"outer_boundary": {"center_x": 40, "center_y": 40, "radius": 34}},
            {"focal_ratio": 0.9, "config": {"excluded_rects": []}},
        )
        free = TwoRateRealtimeTracker(
            *arguments, localizer=DeterministicVerifier()
        )
        assisted = TwoRateRealtimeTracker(
            *arguments,
            localizer=DeterministicVerifier(),
            global_candidate_advisor=Advisor(),
        )
        observation = np.zeros((68, 68, 3), np.uint8)
        mask = np.full((68, 68), 255, np.uint8)
        try:
            free_fix = free._localize_global(
                observation, mask, None, None, None
            )
            assisted_fix = assisted._localize_global(
                observation, mask, None, None, None
            )
            self.assertEqual(
                (assisted_fix.x, assisted_fix.y, assisted_fix.yaw_deg,
                 assisted_fix.scale, assisted_fix.valid),
                (free_fix.x, free_fix.y, free_fix.yaw_deg,
                 free_fix.scale, free_fix.valid),
            )
        finally:
            free.close()
            assisted.close()

    def test_renders_pose_and_quality_overlay(self):
        mosaic = np.full((420, 620, 3), (35, 45, 55), np.uint8)
        state = {
            "mode": "TRACK",
            "pose": {"x": 310.0, "y": 210.0, "yaw_deg": 30.0},
            "trail": [[300.0, 205.0], [305.0, 208.0]],
            "global_fix": {"score": 0.9, "margin": 0.1, "elapsed_ms": 120},
            "local_motion": {"response": 0.8},
            "scene_yaw": {"confidence": 0.7},
            "update_elapsed_ms": 6.0,
            "position_sigma_map_px": 3.0,
            "yaw_sigma_deg": 2.0,
        }
        image = render_map_overlay(mosaic, state)
        self.assertEqual(image.shape, (360, 520, 3))
        self.assertGreater(int(np.max(image[:, :, 1])), 200)

    def test_renders_demonstrated_route_without_changing_overlay_contract(self):
        mosaic = np.full((420, 620, 3), (35, 45, 55), np.uint8)
        state = {
            "mode": "TRACK",
            "pose": {"x": 310.0, "y": 210.0, "yaw_deg": 0.0},
            "global_fix": {},
        }
        route = [[250.0, 210.0], [310.0, 210.0], [370.0, 230.0]]

        plain = render_map_overlay(mosaic, state)
        guided = render_map_overlay(mosaic, state, route_points=route)

        self.assertEqual(guided.shape, (360, 520, 3))
        self.assertGreater(int(np.count_nonzero(guided != plain)), 100)

    def test_projects_demo_route_into_cursor_centered_minimap_overlay(self):
        image = render_minimap_route_overlay(
            [[50.0, 100.0], [100.0, 100.0], [150.0, 100.0]],
            {
                "pose": {
                    "x": 100.0,
                    "y": 100.0,
                    "map_alignment_deg": 0.0,
                },
                "map_scale": 1.0,
            },
            {
                "rotation_center": {"x": 110.0, "y": 83.0},
                "outer_boundary": {
                    "center_x": 111.0,
                    "center_y": 83.0,
                    "radius": 69.0,
                },
            },
            [0, 0, 220, 180],
        )

        self.assertEqual(image.shape, (180, 220, 4))
        self.assertGreater(int(np.max(image[:, :, 3])), 200)
        self.assertGreater(int(image[83, 110, 3]), 0)
        self.assertEqual(int(image[0, 0, 3]), 0)

    def test_renders_rejected_global_candidate_without_claiming_pose(self):
        mosaic = np.full((420, 620, 3), (35, 45, 55), np.uint8)
        image = render_map_overlay(
            mosaic,
            {
                "mode": "INITIALIZING",
                "pose": None,
                "global_fix": {
                    "x": 300.0,
                    "y": 200.0,
                    "score": 0.42,
                    "margin": 0.01,
                    "inlier_count": 3,
                    "ratio_match_count": 8,
                    "decision": "rejected-quality:ambiguous-correlation",
                },
            },
        )
        self.assertEqual(image.shape, (360, 520, 3))
        self.assertGreater(int(np.max(image[:, :, 2])), 240)


class FakeLiveFrameSource:
    def __init__(self):
        self.stopped = threading.Event()
        self.sequence = 0

    def start(self):
        return None

    def read(self):
        if self.stopped.wait(0.005):
            return None
        self.sequence += 1
        now = time.perf_counter_ns()
        return FramePacket(
            "main", np.full((240, 260, 3), 70, np.uint8), now, now
        )

    def stop(self):
        self.stopped.set()


class FakeLiveEngine:
    def __init__(self, *args, **kwargs):
        self.sequence = 0
        self.closed = False
        self._diagnostics = None
        self.args = args
        self.kwargs = kwargs

    def update(self, frame, host_time_ns):
        self.sequence += 1
        fresh = self.sequence == 2
        if fresh:
            self._diagnostics = {
                "candidate_overlay": np.full((40, 50, 3), 140, np.uint8),
                "correlation_heatmap": np.full((30, 40, 3), 180, np.uint8),
            }
        return {
            "sequence": self.sequence,
            "host_time_ns": host_time_ns,
            "mode": "TRACK",
            "pose": {"x": 45.0, "y": 55.0, "yaw_deg": 10.0},
            "trail": [[45.0, 55.0]],
            "global_fix": {
                "score": 0.9,
                "margin": 0.1,
                "elapsed_ms": 80.0,
                "decision": "consistent",
                "alternatives": [],
                "fusion": {
                    "accepted": True,
                    "reason": "consistent",
                    "applied_position_change_map_px": 10.0,
                },
            },
            "global_fix_fresh": fresh,
            "local_motion": {"response": 0.8},
            "scene_yaw": {"confidence": 0.7},
            "update_elapsed_ms": 4.0,
            "position_sigma_map_px": 3.0,
            "yaw_sigma_deg": 2.0,
        }

    def take_global_diagnostics(self):
        diagnostics = self._diagnostics
        self._diagnostics = None
        return diagnostics

    def close(self):
        self.closed = True


class WorkbenchLiveTrackerTests(unittest.TestCase):
    def test_tracker_lifecycle_is_serializable_and_recording_is_exclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_root = root / "artifacts"
            game_id = "genshin-impact-pc"
            calibration_root = artifact_root / "minimap_calibrations" / game_id / "mini-a"
            scene_root = artifact_root / "scene_yaw_calibrations" / game_id / "yaw-a"
            stitch_root = artifact_root / "map_stitches" / game_id / "map-a"
            for path in (calibration_root, scene_root, stitch_root):
                path.mkdir(parents=True)
            (calibration_root / "calibration.json").write_text(
                json.dumps({"outer_boundary": {"center_x": 111, "center_y": 83, "radius": 68}}),
                encoding="utf-8",
            )
            (scene_root / "scene_yaw_calibration.json").write_text(
                json.dumps({"focal_ratio": 0.9, "camera_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}),
                encoding="utf-8",
            )
            (stitch_root / "map_stitch.json").write_text(
                json.dumps(
                    {
                        "evidence": [
                            {"name": "mosaic.png"},
                            {"name": "localization_mosaic.png"},
                            {"name": "localization_coverage.png"},
                        ],
                        "localization": {
                            "status": "ready",
                            "source_minimap_calibration_id": "mini-a",
                            "mosaic_file": "localization_mosaic.png",
                            "coverage_file": "localization_coverage.png",
                            "localization_to_original_map_3x3": [
                                [1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1],
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            tracker_map = np.random.RandomState(17).randint(
                0, 255, (180, 220, 3), dtype=np.uint8
            )
            cv2.imwrite(str(stitch_root / "mosaic.png"), tracker_map)
            cv2.imwrite(
                str(stitch_root / "localization_mosaic.png"),
                tracker_map,
            )
            cv2.imwrite(
                str(stitch_root / "localization_coverage.png"),
                np.full((180, 220), 255, np.uint8),
            )
            state = AcquisitionWorkbench(
                root / "sessions", artifact_root, profiles=ProfileCatalog()
            )
            source = FakeLiveFrameSource()
            state.sources.capture_sources = lambda frame, inputs: (source, None)
            try:
                with patch(
                    "aria_trace.apps.workbench.live_tracking.TwoRateRealtimeTracker", FakeLiveEngine
                ), patch(
                    "aria_trace.apps.workbench.live_tracking.CursorPoseEstimator",
                    return_value=object(),
                ) as cursor_constructor:
                    cursor_constructor.GAUSSIAN_FIT_METHODS = (
                        "vectorized_grid",
                        "fast_grid",
                        "legacy_grid",
                    )
                    cursor_constructor.POSE_METHODS = (
                        "polygon_gaussian",
                        "angular_projection_ncc_parabolic",
                    )
                    state.start_live_tracker(
                        {
                            "game_profile_id": game_id,
                            "minimap_calibration_id": "mini-a",
                            "scene_yaw_calibration_id": "yaw-a",
                            "map_stitch_id": "map-a",
                            "tracking_profile": "real-time",
                            "cursor_pose_method": "fast_grid",
                            "frame_source": {
                                "adapter": "windows_window",
                                "window_title": "Genshin Impact",
                            },
                        }
                    )
                    for _ in range(100):
                        descriptor = state.descriptor()
                        if (descriptor.get("live_tracker") or {}).get("latest"):
                            break
                        time.sleep(0.005)
                    runtime = descriptor["live_tracker"]
                    self.assertEqual(runtime["status"], "running")
                    self.assertEqual(runtime["cursor_pose_method"], "fast_grid")
                    self.assertEqual(runtime["tracking_profile"], "real-time")
                    self.assertEqual(
                        runtime["resolved_tracking_profile"]["global_interval_s"],
                        2.0,
                    )
                    cursor_constructor.assert_not_called()
                    engine = state._live_tracker_engine
                    process_config = engine.kwargs["cursor_pose_process_config"]
                    self.assertEqual(
                        process_config["calibration_path"], calibration_root
                    )
                    self.assertEqual(
                        process_config["gaussian_fit_method"], "fast_grid"
                    )
                    self.assertEqual(
                        process_config["pose_method"],
                        "angular_projection_ncc_parabolic",
                    )
                    self.assertEqual(process_config["opencv_threads"], 1)
                    self.assertEqual(runtime["latest"]["mode"], "TRACK")
                    self.assertIn("capture_dropped_before_processing", runtime)
                    json.dumps(descriptor)
                    overlay = cv2.imdecode(
                        np.frombuffer(state.live_tracker_overlay_image(), np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    self.assertEqual(overlay.shape, (360, 520, 3))
                    with self.assertRaisesRegex(RuntimeError, "Stop live tracking"):
                        state.start_session({"game_profile_id": game_id})
                    state.stop_live_tracker()
                    for _ in range(100):
                        if state.descriptor()["live_tracker"]["status"] == "stopped":
                            break
                        time.sleep(0.005)
                    self.assertEqual(
                        state.descriptor()["live_tracker"]["status"], "stopped"
                    )
                    descriptor = state.descriptor()
                    evidence = descriptor["live_tracker"]["evidence"]
                    self.assertGreater(evidence["counts"]["telemetry_rows"], 0)
                    run = descriptor["live_tracking_runs"][game_id][0]
                    self.assertEqual(run["status"], "stopped")
                    run_root = artifact_root / run["artifact_relative_path"]
                    self.assertTrue((run_root / "telemetry.jsonl").is_file())
                    self.assertTrue((run_root / "live_tracking.json").is_file())
                    fix = run["recent_global_fixes"][0]
                    content_type, image = state.live_tracking_image(
                        game_id,
                        run["tracking_id"],
                        fix["fix_id"],
                        "candidate_overlay.png",
                    )
                    self.assertEqual(content_type, "image/png")
                    self.assertGreater(len(image), 0)
            finally:
                source.stop()
                state.close()

    def test_route_locked_mode_loads_atlas_and_compiled_route(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_root = root / "artifacts"
            game_id = "genshin-impact-pc"
            calibration_root = (
                artifact_root / "minimap_calibrations" / game_id / "mini-a"
            )
            scene_root = artifact_root / "scene_yaw_calibrations" / game_id / "yaw-a"
            atlas_root = artifact_root / "map_atlases" / game_id / "atlas-a"
            route_root = artifact_root / "route_tracking" / game_id / "route-a"
            for path in (calibration_root, scene_root, atlas_root):
                path.mkdir(parents=True)
            (calibration_root / "calibration.json").write_text(
                json.dumps(
                    {"outer_boundary": {"center_x": 111, "center_y": 83, "radius": 68}}
                ),
                encoding="utf-8",
            )
            (scene_root / "scene_yaw_calibration.json").write_text(
                json.dumps({"focal_ratio": 0.9}), encoding="utf-8"
            )
            atlas_image = np.random.RandomState(29).randint(
                0, 255, (180, 220, 3), dtype=np.uint8
            )
            cv2.imwrite(str(atlas_root / "canonical_mosaic.png"), atlas_image)
            layer_root = atlas_root / "layers" / "world"
            layer_root.mkdir(parents=True)
            cv2.imwrite(
                str(layer_root / "localization_mosaic.png"),
                atlas_image,
            )
            cv2.imwrite(
                str(layer_root / "localization_coverage.png"),
                np.full((180, 220), 255, np.uint8),
            )
            (atlas_root / "map_atlas.json").write_text(
                json.dumps(
                    {
                        "atlas_id": "atlas-a",
                        "coordinate_space_id": "map-atlas:atlas-a:canonical-map-px",
                        "canonical_mosaic_file": "canonical_mosaic.png",
                        "layers": [
                            {
                                "mode_id": "world",
                                "localization_mosaic_file": (
                                    "layers/world/localization_mosaic.png"
                                ),
                                "localization_coverage_file": (
                                    "layers/world/localization_coverage.png"
                                ),
                                "localization_to_canonical_3x3": [
                                    [1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1],
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            observations = []
            for index in range(4):
                descriptor = np.zeros(16, np.float32)
                descriptor[index] = 1.0
                observations.append(
                    {
                        "source_frame_index": index,
                        "session_time_ns": 1_000_000_000 + index * 100_000_000,
                        "x": index * 10.0,
                        "y": 0.0,
                        "mode_id": "world",
                        "descriptor": descriptor,
                    }
                )
            compile_route_tracking_package(
                observations,
                route_root,
                route_id="route-a",
                atlas_id="atlas-a",
                coordinate_space_id="map-atlas:atlas-a:canonical-map-px",
            )
            state = AcquisitionWorkbench(
                root / "sessions", artifact_root, profiles=ProfileCatalog()
            )
            source = FakeLiveFrameSource()
            state.sources.capture_sources = lambda frame, inputs: (source, None)
            try:
                with patch(
                    "aria_trace.apps.workbench.live_tracking.TwoRateRealtimeTracker", FakeLiveEngine
                ), patch(
                    "aria_trace.apps.workbench.live_tracking.CursorPoseEstimator",
                    return_value=object(),
                ) as cursor_constructor:
                    cursor_constructor.GAUSSIAN_FIT_METHODS = (
                        "vectorized_grid",
                        "fast_grid",
                        "legacy_grid",
                    )
                    cursor_constructor.POSE_METHODS = (
                        "polygon_gaussian",
                        "angular_projection_ncc_parabolic",
                    )
                    descriptor = state.start_live_tracker(
                        {
                            "game_profile_id": game_id,
                            "minimap_calibration_id": "mini-a",
                            "scene_yaw_calibration_id": "yaw-a",
                            "map_atlas_id": "atlas-a",
                            "route_package_id": "route-a",
                            "tracking_mode": "route-locked",
                            "tracking_profile": "fast",
                            "cursor_pose_method": "fast_grid",
                            "frame_source": {
                                "adapter": "windows_window",
                                "window_title": "Genshin Impact",
                            },
                        }
                    )
                    runtime = descriptor["live_tracker"]
                    self.assertEqual(runtime["tracking_mode"], "route-assisted")
                    self.assertEqual(
                        runtime["route_policy"], "current-frame-map-correlation"
                    )
                    self.assertEqual(runtime["map_atlas_id"], "atlas-a")
                    self.assertEqual(runtime["route_package_id"], "route-a")
                    self.assertEqual(runtime["tracking_profile"], "fast")
                    engine = state._live_tracker_engine
                    self.assertIsInstance(
                        engine.kwargs["localizer"], LayeredGlobalLocalizer
                    )
                    self.assertIsNotNone(
                        engine.kwargs["global_candidate_advisor"]
                    )
                    self.assertIsNotNone(engine.kwargs["route_visual_tracker"])
                    self.assertEqual(
                        engine.kwargs["route_visual_tracker"].score_min, 0.50
                    )
                    self.assertEqual(
                        engine.kwargs["cursor_pose_process_config"][
                            "opencv_threads"
                        ],
                        1,
                    )
                    self.assertNotIn("route_state_estimator", engine.kwargs)
                    with state._lock:
                        state._live_tracker["status"] = "running"
                        state._live_tracker["latest"] = {
                            "mode": "TRACK",
                            "pose": {
                                "x": 10.0,
                                "y": 0.0,
                                "yaw_deg": 0.0,
                                "map_alignment_deg": 0.0,
                            },
                            "map_scale": 1.0,
                        }
                        state._live_tracker["frame_size_wh"] = [1920, 1080]
                    hud = state.hud_descriptor()
                    self.assertEqual(
                        hud["minimap_route_overlay_url"],
                        "/api/tracker/minimap-route-overlay",
                    )
                    self.assertEqual(
                        hud["minimap_route_role"], "visual-guidance-only"
                    )
                    guide = cv2.imdecode(
                        np.frombuffer(
                            state.live_tracker_minimap_route_overlay_image(),
                            np.uint8,
                        ),
                        cv2.IMREAD_UNCHANGED,
                    )
                    self.assertEqual(guide.shape[2], 4)
            finally:
                source.stop()
                if state._tracker_running():
                    state.stop_live_tracker()
                    for _ in range(100):
                        if state.descriptor()["live_tracker"]["status"] == "stopped":
                            break
                        time.sleep(0.005)
                state.close()


if __name__ == "__main__":
    unittest.main()
