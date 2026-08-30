import unittest
from unittest.mock import Mock

import numpy as np

from poc.benchmark_route_tracer import CausalRouteTracer, _loss_metrics


class _Package:
    def __init__(self):
        self.calls = []
        self.manifest = {"motion_envelope": {}}

    def candidates(self, descriptor, **kwargs):
        self.calls.append(kwargs)
        return [
            {
                "state_index": 7,
                "score": 0.9,
                "state": {
                    "canonical_xy": [10.0, 20.0],
                    "mode_id": "world",
                },
            },
            {
                "state_index": 8,
                "score": 0.8,
                "state": {
                    "canonical_xy": [11.0, 20.0],
                    "mode_id": "world",
                },
            },
        ]


class RouteTracerBenchmarkTests(unittest.TestCase):
    def test_route_query_never_uses_demo_progress_or_mode(self):
        package = _Package()
        tracer = CausalRouteTracer(package, Mock(localizers={}), "route_descriptor")
        observation = np.zeros((32, 32, 3), np.uint8)

        result = tracer.track(observation, np.full((32, 32), 255, np.uint8))

        self.assertTrue(result.valid)
        self.assertEqual(package.calls[0]["top_k"], 2)
        self.assertNotIn("previous_state_index", package.calls[0])
        self.assertNotIn("mode_id", package.calls[0])

    def test_refined_pose_comes_from_current_map_match(self):
        package = _Package()
        atlas = Mock()
        atlas.localizers = {"world": object()}
        atlas._observe_one_mode.return_value = {
            "valid": True,
            "score": 0.8,
            "best_offset_canonical_xy": [3.0, -4.0],
        }
        tracer = CausalRouteTracer(package, atlas, "route_refine_top1")
        observation = np.zeros((32, 32, 3), np.uint8)

        result = tracer.track(observation, np.full((32, 32), 255, np.uint8))

        self.assertTrue(result.valid)
        self.assertEqual((result.x, result.y), (13.0, 16.0))
        self.assertEqual(result.source, "route_recovery")

    def test_continuous_variant_uses_local_match_before_route_recovery(self):
        package = _Package()
        atlas = Mock()
        atlas.localizers = {"world": object()}
        atlas._observe_one_mode.return_value = {
            "valid": True,
            "score": 0.8,
            "best_offset_canonical_xy": [1.0, 0.0],
        }
        tracer = CausalRouteTracer(package, atlas, "continuous_local")
        tracer.previous_xy = (50.0, 60.0)
        observation = np.zeros((32, 32, 3), np.uint8)

        result = tracer.track(observation, np.full((32, 32), 255, np.uint8))

        self.assertTrue(result.valid)
        self.assertEqual((result.x, result.y), (51.0, 60.0))
        self.assertEqual(result.source, "local")
        self.assertEqual(package.calls, [])

    def test_continuity_gate_holds_previous_pose_on_large_jump(self):
        package = _Package()
        package.manifest = {"motion_envelope": {}}
        atlas = Mock()
        atlas.localizers = {"world": object()}
        atlas._observe_one_mode.return_value = {
            "valid": True,
            "score": 0.8,
            "best_offset_canonical_xy": [20.0, 0.0],
        }
        tracer = CausalRouteTracer(package, atlas, "continuous_gated")
        tracer.previous_xy = (50.0, 60.0)
        tracer.previous_time_ns = 1_000_000_000
        observation = np.zeros((32, 32, 3), np.uint8)

        result = tracer.track(
            observation,
            np.full((32, 32), 255, np.uint8),
            session_time_ns=1_033_000_000,
        )

        self.assertTrue(result.valid)
        self.assertFalse(result.measurement_accepted)
        self.assertEqual((result.x, result.y), (50.0, 60.0))
        self.assertEqual(result.source, "continuity_hold")

    def test_loss_metrics_count_episodes_and_longest_streak(self):
        metrics = _loss_metrics([True, False, False, True, False])

        self.assertEqual(metrics["loss_episode_count"], 2)
        self.assertEqual(metrics["longest_loss_frames"], 2)
        self.assertEqual(metrics["lost_frame_count"], 3)
        self.assertAlmostEqual(metrics["tracked_fraction"], 0.4)


if __name__ == "__main__":
    unittest.main()
