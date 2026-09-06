import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from aria_trace.services.mapping.layers import LayeredGlobalLocalizer
from benchmarks.localization.precision_candidates import fractional_match, quadratic_peak, installed


class PrecisionCandidateTests(unittest.TestCase):
    def test_known_fractional_translations_reduce_error_without_changing_score(self):
        rng = np.random.default_rng(547)
        field = cv2.GaussianBlur(rng.uniform(0, 255, (380, 420)).astype(np.float32), (7, 7), 1.4)
        localizer = SimpleNamespace(map_gradient=field, coverage=np.full(field.shape, 255, np.uint8),
                                    original_to_localization=np.eye(3), _localization_xy=lambda p:p, _original_xy=lambda p:p)
        mask = np.zeros((138, 138), np.uint8)
        cv2.circle(mask, (69, 69), 66, 255, -1); cv2.circle(mask, (69, 69), 14, 0, -1)
        integer_errors, fractional_errors = [], []
        for dx, dy in [(a, b) for a in (.15, .35, .65, .85) for b in (.2, .4, .7)]:
            truth = np.array([201+dx, 184+dy])
            # getRectSubPix centers odd/even rasters on (width-1)/2; matcher
            # coordinates use width/2. The half-pixel convention is explicit.
            observation = cv2.getRectSubPix(field, (138, 138), tuple(truth-.5))
            before = LayeredGlobalLocalizer._observe_one_mode(localizer, observation, mask, (200., 184.), 12.)
            after = fractional_match(localizer, observation, mask, (200., 184.), 12.)
            self.assertEqual(before['valid'], after['valid'])
            self.assertEqual(before['score'], after['score'])
            integer_errors.append(np.linalg.norm(np.array([200.,184.])+before['best_offset_canonical_xy']-truth))
            fractional_errors.append(np.linalg.norm(np.array([200.,184.])+after['best_offset_canonical_xy']-truth))
        self.assertLess(np.mean(fractional_errors), .1)
        self.assertLess(np.mean(fractional_errors), np.mean(integer_errors)/3)

    def test_peak_edges_and_flat_responses_do_not_extrapolate(self):
        self.assertEqual((0., 0.), quadratic_peak(np.ones((5,5)), (0,0)))
        self.assertEqual((0., 0.), quadratic_peak(np.ones((5,5)), (2,2)))

    def test_coverage_gate_is_preserved(self):
        field = np.ones((200,200), np.float32)
        localizer = SimpleNamespace(map_gradient=field, coverage=np.zeros(field.shape,np.uint8),
                                    original_to_localization=np.eye(3), _localization_xy=lambda p:p, _original_xy=lambda p:p)
        result=fractional_match(localizer,np.ones((138,138),np.float32),np.ones((138,138),np.uint8),(100,100),12)
        self.assertFalse(result['valid'])
        self.assertEqual('insufficient-observed-coverage',result['reason'])

    def test_selective_expansion_preserves_mode_owner_and_requires_image_match(self):
        layered = LayeredGlobalLocalizer.__new__(LayeredGlobalLocalizer)
        layered.active_mode_id = 'town'
        layered.map_scales = {'town': 1., 'world': 3.}
        layered._precision_pyramid = [(1., .4), (2., .8), (3., .3)]
        endpoint = {'valid': True, 'score': .6, 'x': 10., 'y': 20.}
        calls = []
        def match(layer, *args):
            calls.append(layer)
            return {'valid': True, 'score': layer, 'coverage_fraction': 1., 'best_offset_canonical_xy': [2., 3.]}
        def refine(*args):
            return dict(endpoint)
        with patch.object(LayeredGlobalLocalizer, '_observe_one_mode', staticmethod(match)), patch.object(LayeredGlobalLocalizer, 'refine_active_near', refine):
            with installed(scale_sweep=True, selective_scale=True):
                image = np.zeros((12, 12, 3), np.uint8)
                mask = np.ones((12, 12), np.uint8)
                result = layered.refine_active_near(image, mask, (10., 20.))
                self.assertEqual((10., 20.), (result['x'], result['y']))
                self.assertEqual([], calls)
                endpoint['score'] = .4
                result = layered.refine_active_near(image, mask, (10., 20.))
                self.assertEqual((12., 23.), (result['x'], result['y']))
                self.assertEqual(2., result['precision_scale']['estimated_scale'])
                self.assertEqual('town', layered.active_mode_id)
                self.assertEqual('town', result['selected_mode_id'])
                result = layered.refine_active_near(image, mask, (10., 20.), score_min=.9)
                self.assertFalse(result['valid'])
                self.assertIsNone(result['x'])


if __name__=='__main__':
    unittest.main()
