import tempfile
import unittest
from pathlib import Path

import numpy as np

from acquisition.workbench import AcquisitionWorkbench
from tests import test_live_tracker as fixtures


class StartLocalizer(fixtures.ImmediateLocalizer):
    localizers = {"world": object()}

    def __init__(self):
        super().__init__()
        self.score = 0.4
        self.centers = []

    def refine_active_near(self, observation, mask, center, **kwargs):
        self.centers.append(tuple(center))
        valid = self.score >= kwargs["score_min"]
        return {"valid": valid, "x": 15 if valid else None,
                "y": 22 if valid else None, "score": self.score}

    refine_near = refine_active_near


class RouteStartTests(unittest.TestCase):
    def test_first_current_measurement_owns_pose_and_start_is_not_reapplied(self):
        localizer = StartLocalizer()
        tracker = fixtures.TwoRateTrackerTests._tracker(localizer)
        frame = np.zeros((100, 100, 3), np.uint8)
        try:
            tracker.set_route_start({"canonical_xy": [10, 20], "mode_id": "world"})
            self.assertIsNone(tracker.fusion._state)
            waiting = tracker.update(frame, 1_000_000_000)
            self.assertIsNone(waiting["pose"])
            self.assertFalse(waiting["xy_measurement_fresh_accepted"])
            self.assertFalse(waiting["control_output_available"])
            self.assertEqual(localizer.calls, 0)
            localizer.score = 0.8
            matched = tracker.update(frame, 1_033_333_333)
            self.assertTrue(matched["xy_measurement_fresh_accepted"])
            self.assertEqual((matched["pose"]["x"], matched["pose"]["y"]), (15, 22))
            self.assertIsNone(matched["pose"]["player_heading_map_deg"])
            self.assertEqual(matched["route_start"]["status"], "image-acquired")
            tracker.update(frame, 1_066_666_666)
            self.assertEqual(localizer.centers, [(10, 20), (10, 20), (15, 22)])
            with self.assertRaisesRegex(ValueError, "before tracking"):
                tracker.set_route_start({"canonical_xy": [50, 60], "mode_id": "world"})
        finally:
            tracker.close()

    def test_invalid_start_does_not_create_a_pose(self):
        tracker = fixtures.TwoRateTrackerTests._tracker(StartLocalizer())
        try:
            for xy, mode in (([float("nan"), 20], "world"), ([10, 20], "missing"), ([10], "world")):
                with self.subTest(xy=xy, mode=mode), self.assertRaises(ValueError):
                    tracker.set_route_start({"canonical_xy": xy, "mode_id": mode})
                self.assertIsNone(tracker.fusion._state)
                self.assertIsNone(tracker._route_start)
        finally:
            tracker.close()

    def test_global_recovery_remains_available_after_start_acquisition(self):
        localizer = StartLocalizer()
        localizer.score = 0.8
        tracker = fixtures.TwoRateTrackerTests._tracker(localizer)
        frame = np.zeros((100, 100, 3), np.uint8)
        try:
            tracker.set_route_start({"canonical_xy": [10, 20], "mode_id": "world"})
            tracker.update(frame, 1_000_000_000)
            localizer.score = -1.0
            for index in range(tracker.relocalize_after_rejections):
                tracker.update(frame, 1_033_333_333 + index * 33_333_333)
            self.assertIsNotNone(tracker._global_future)
            tracker._global_future.result(timeout=2)
            self.assertGreater(localizer.calls, 0)
        finally:
            tracker.close()

    def test_workbench_rejects_start_without_route_before_opening_capture(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            state = AcquisitionWorkbench(root/"sessions", root/"artifacts")
            try:
                for mode, policy in (("free-roam", "demonstrated-start"), ("route-assisted", "invalid")):
                    with self.subTest(mode=mode, policy=policy), self.assertRaises(ValueError):
                        state.start_live_tracker({"game_profile_id": "genshin-impact-pc",
                            "tracking_mode": mode, "route_start_policy": policy})
                self.assertFalse(state._tracker_running())
            finally:
                state.close()
