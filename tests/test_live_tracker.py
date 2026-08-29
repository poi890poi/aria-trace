import threading
import time
import json
import tempfile
import unittest
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
)
from acquisition.models import FramePacket
from acquisition.profiles import ProfileCatalog
from acquisition.workbench import AcquisitionWorkbench


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

    def update(self, frame, host_time_ns):
        self.sequence += 1
        return {
            "sequence": self.sequence,
            "mode": "TRACK",
            "pose": {"x": 45.0, "y": 55.0, "yaw_deg": 10.0},
            "trail": [[45.0, 55.0]],
            "global_fix": {
                "score": 0.9,
                "margin": 0.1,
                "elapsed_ms": 80.0,
            },
            "local_motion": {"response": 0.8},
            "scene_yaw": {"confidence": 0.7},
            "update_elapsed_ms": 4.0,
            "position_sigma_map_px": 3.0,
            "yaw_sigma_deg": 2.0,
        }

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
                    "acquisition.workbench.TwoRateRealtimeTracker", FakeLiveEngine
                ), patch(
                    "acquisition.workbench.CursorPoseEstimator",
                    return_value=object(),
                ) as cursor_constructor:
                    cursor_constructor.GAUSSIAN_FIT_METHODS = (
                        "vectorized_grid",
                        "fast_grid",
                        "legacy_grid",
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
                    cursor_constructor.assert_called_once_with(
                        calibration_root,
                        gaussian_fit_method="fast_grid",
                        validation_policy="ambiguous",
                    )
                    self.assertEqual(runtime["latest"]["mode"], "TRACK")
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
            finally:
                source.stop()
                state.close()


if __name__ == "__main__":
    unittest.main()
