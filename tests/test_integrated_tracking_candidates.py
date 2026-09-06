"""Candidate ownership and compatibility without patching tracker algorithms."""

import copy
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

from aria_trace.services.mapping.candidates import VisualTransitionLocalizer
from aria_trace.services.mapping.layers import LayeredGlobalLocalizer
from aria_trace.services.tracking.runtime import TwoRateRealtimeTracker
from rig_runtime.services.calibration.minimap.transition import TransitionController


class IntegratedTransitionTests(unittest.TestCase):
    def test_visual_policy_keeps_map_observations_and_confirmation_in_both_directions(self):
        model = {"source_mode_id": "world", "target_mode_id": "town",
                 "runtime": {"minimum_mode_margin": .2},
                 "transition_zones": [{"center_xy": [0, 0], "radius_px": 2}]}
        original = copy.deepcopy(model)
        controller = TransitionController(model, 2, spatial_gating=False)
        self.assertEqual(1, len(controller.transition_zones))
        for source, target in [('world', 'town'), ('town', 'world')]:
            self.assertTrue(controller.observation_required((100,100)))
            self.assertTrue(controller.observation_required(None))
            self.assertFalse(controller.update({source:.9,target:1.})['switched'])
            scores = {source:.1,target:.9}
            self.assertFalse(controller.update(scores,canonical_xy=(100,100))['switched'])
            result = controller.update(scores,canonical_xy=(100,100))
            self.assertTrue(result['switched'])
            self.assertEqual(target,result['active_mode_id'])
            self.assertEqual([0.,0.],result['position_delta_xy'])
            self.assertFalse(result['within_observed_zone'])
        self.assertEqual(original,model)
        default = TransitionController(model,2)
        self.assertFalse(default.observation_required((100,100)))
        self.assertFalse(default.update({'world':.1,'town':.9},canonical_xy=(100,100))['transition_armed'])

    def test_visual_candidate_reuses_existing_xy_without_mutating_default(self):
        self.assertIs(VisualTransitionLocalizer.refine_active_near,LayeredGlobalLocalizer.refine_active_near)
        self.assertFalse(VisualTransitionLocalizer.spatial_transition_gating)
        self.assertTrue(getattr(LayeredGlobalLocalizer,'spatial_transition_gating',True))

    def test_retained_zone_diagnostics_do_not_reintroduce_route_anchor_hold(self):
        model = {"source_mode_id": "world", "target_mode_id": "town",
                 "transition_zones": [{"zone_id": "crossing", "center_xy": [0, 0], "radius_px": 2}]}
        for gated in (True, False):
            tracker = TwoRateRealtimeTracker.__new__(TwoRateRealtimeTracker)
            tracker.transition_controller = TransitionController(model, 2, spatial_gating=gated)
            tracker._representation_position_sigma = 0.
            tracker._active_map_mode_id = 'world'
            tracker.route_visual_tracker = SimpleNamespace(arm_trained_transition=Mock(), cancel_trained_transition=Mock())
            tracker._representation_future = Future()
            tracker._representation_future.set_result({"valid": True, "canonical_xy_read_only": [0, 0],
                                                       "likelihoods": {"world": .9, "town": .1}})
            self.assertTrue(tracker._consume_representation_observation(1))
            self.assertTrue(tracker._last_representation_observation['controller']['within_observed_zone'])
            self.assertEqual(int(gated), tracker.route_visual_tracker.arm_trained_transition.call_count)


if __name__ == '__main__':
    unittest.main()
