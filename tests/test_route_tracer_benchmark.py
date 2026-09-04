import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from poc.benchmark_route_tracer import (
    CausalRouteTracer,
    _correlation_feature,
    _loss_metrics,
)


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

    def test_two_layer_variant_rejects_low_score_in_single_final_gate(self):
        package = _Package()
        atlas = Mock()
        atlas.localizers = {"world": object()}
        atlas._observe_one_mode.return_value = {
            "valid": True,
            "score": 0.20,
            "best_offset_canonical_xy": [0.0, 0.0],
        }
        tracer = CausalRouteTracer(
            package,
            atlas,
            "local_primary_gated",
            score_min=0.50,
        )
        tracer.previous_xy = (50.0, 60.0)
        tracer.previous_time_ns = 1_000_000_000

        result = tracer.track(
            np.zeros((32, 32, 3), np.uint8),
            np.full((32, 32), 255, np.uint8),
            session_time_ns=1_033_000_000,
        )

        self.assertTrue(result.valid)
        self.assertFalse(result.measurement_accepted)
        self.assertTrue(result.primary_candidate_produced)
        self.assertTrue(result.final_gate_rejected)
        self.assertEqual(result.source, "confidence_hold")
        self.assertEqual((result.x, result.y), (50.0, 60.0))
        self.assertEqual(package.calls, [])

    def test_accepted_continuity_clock_accumulates_time_while_held(self):
        package = _Package()
        atlas = Mock()
        atlas.localizers = {"world": object()}
        atlas._observe_one_mode.return_value = {
            "valid": True,
            "score": 0.8,
            "best_offset_canonical_xy": [20.0, 0.0],
        }
        tracer = CausalRouteTracer(
            package,
            atlas,
            "continuous_gated",
            continuity_clock="accepted",
        )
        tracer.previous_xy = (50.0, 60.0)
        tracer.previous_time_ns = 1_000_000_000

        result = tracer.track(
            np.zeros((32, 32, 3), np.uint8),
            np.full((32, 32), 255, np.uint8),
            session_time_ns=1_033_000_000,
        )

        self.assertFalse(result.measurement_accepted)
        self.assertEqual(tracer.previous_time_ns, 1_000_000_000)

    def test_global_recovery_consensus_reseeds_after_rejection_streak(self):
        package = _Package()
        atlas = Mock()
        atlas.localizers = {"world": object()}
        atlas._observe_one_mode.return_value = {
            "valid": True,
            "score": 0.8,
            "best_offset_canonical_xy": [20.0, 0.0],
        }
        atlas.localize.return_value = SimpleNamespace(
            valid=True,
            x=70.0,
            y=60.0,
            score=0.9,
            margin=0.2,
            diagnostics={"map_layer": {"selected_mode_id": "world"}},
        )
        tracer = CausalRouteTracer(
            package,
            atlas,
            "continuous_gated",
            recovery_policy="global_consensus",
        )
        tracer.previous_xy = (50.0, 60.0)
        tracer.previous_time_ns = 1_000_000_000
        result = None
        for index in range(7):
            result = tracer.track(
                np.zeros((32, 32, 3), np.uint8),
                np.full((32, 32), 255, np.uint8),
                session_time_ns=1_033_000_000 + index * 33_000_000,
            )

        self.assertEqual(result.source, "global_recovery")
        self.assertTrue(result.measurement_accepted)
        self.assertEqual((result.x, result.y), (70.0, 60.0))
        self.assertEqual(atlas.localize.call_count, 2)

    def test_loss_metrics_count_episodes_and_longest_streak(self):
        metrics = _loss_metrics([True, False, False, True, False])

        self.assertEqual(metrics["loss_episode_count"], 2)
        self.assertEqual(metrics["longest_loss_frames"], 2)
        self.assertEqual(metrics["lost_frame_count"], 3)
        self.assertAlmostEqual(metrics["tracked_fraction"], 0.4)

    def test_experimental_correlation_features_preserve_image_size(self):
        image = np.random.RandomState(3).randint(
            0, 255, (32, 40, 3), dtype=np.uint8
        )
        for name in ("gradient", "intensity", "canny", "laplacian"):
            feature = _correlation_feature(image, name)
            self.assertEqual(feature.shape, (32, 40))
            self.assertEqual(feature.dtype, np.float32)

    def test_experimental_feature_does_not_corrupt_global_initialization_map(self):
        package = _Package()
        mosaic = np.random.RandomState(4).randint(
            0, 255, (48, 52, 3), dtype=np.uint8
        )
        canonical_gradient = np.full((48, 52), 17.0, np.float32)
        localizer = SimpleNamespace(
            mosaic=mosaic,
            map_gradient=canonical_gradient.copy(),
        )
        atlas = Mock(localizers={"world": localizer})
        atlas._observe_one_mode.return_value = {
            "valid": True,
            "score": 0.8,
            "best_offset_canonical_xy": [0.0, 0.0],
        }
        tracer = CausalRouteTracer(
            package,
            atlas,
            "route_refine_top1",
            correlation_feature="intensity",
        )

        np.testing.assert_array_equal(localizer.map_gradient, canonical_gradient)
        tracer.track(
            np.zeros((32, 32, 3), np.uint8),
            np.full((32, 32), 255, np.uint8),
        )

        np.testing.assert_array_equal(
            localizer.map_gradient,
            _correlation_feature(mosaic, "intensity"),
        )

    def test_phase_local_match_reports_new_atlas_center(self):
        random = np.random.RandomState(8)
        atlas_feature = random.rand(100, 100).astype(np.float32)
        localizer = SimpleNamespace(
            map_gradient=atlas_feature,
            coverage=np.full((100, 100), 255, np.uint8),
            original_to_localization=np.eye(3),
            _localization_xy=lambda value: tuple(value),
            _original_xy=lambda value: tuple(value),
        )
        package = _Package()
        atlas = Mock(localizers={"world": localizer})
        tracer = CausalRouteTracer(
            package,
            atlas,
            "local_primary_gated",
            local_matcher="phase_correlation",
        )

        result = tracer._observe_one_mode(
            localizer,
            atlas_feature[30:70, 33:73],
            np.full((40, 40), 255, np.uint8),
            (50.0, 50.0),
            12.0,
        )

        self.assertTrue(result["valid"])
        self.assertAlmostEqual(result["best_offset_canonical_xy"][0], 3.0, delta=0.2)
        self.assertAlmostEqual(result["best_offset_canonical_xy"][1], 0.0, delta=0.2)
        self.assertGreater(result["score"], 0.5)


if __name__ == "__main__":
    unittest.main()
