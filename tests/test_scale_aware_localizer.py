"""Known-image geometry and authority checks for the reusable scale candidate."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from aria_trace.services.mapping.fractional_template import fractional_match, quadratic_peak
from aria_trace.services.mapping.layers import LayeredGlobalLocalizer
from aria_trace.services.mapping.scale_aware import ScaleAwareLocalizer, _build_pyramid


class ScaleAwareTests(unittest.TestCase):
    def test_known_fractional_displacements_preserve_acceptance_and_reduce_error(self):
        field = cv2.GaussianBlur(np.random.default_rng(547).uniform(0, 255, (380, 420)).astype(np.float32), (7, 7), 1.4)
        layer = SimpleNamespace(map_gradient=field, coverage=np.full(field.shape, 255, np.uint8),
                                original_to_localization=np.eye(3), _localization_xy=lambda p:p, _original_xy=lambda p:p)
        mask = np.zeros((138, 138), np.uint8)
        cv2.circle(mask, (69, 69), 66, 255, -1)
        cv2.circle(mask, (69, 69), 14, 0, -1)
        errors, baseline = [], []
        for dx in (.15, .35, .65, .85):
            for dy in (.2, .4, .7):
                truth = np.array([201+dx, 184+dy])
                # OpenCV patch centers use (width-1)/2; the atlas uses width/2.
                observation = cv2.getRectSubPix(field, (138, 138), tuple(truth-.5))
                before = LayeredGlobalLocalizer._observe_one_mode(layer, observation, mask, (200.,184.), 12.)
                after = fractional_match(layer, observation, mask, (200.,184.), 12.)
                for key in ('valid', 'score', 'coverage_fraction'):
                    self.assertEqual(before[key], after[key])
                baseline.append(np.linalg.norm(np.array([200.,184.])+before['best_offset_canonical_xy']-truth))
                errors.append(np.linalg.norm(np.array([200.,184.])+after['best_offset_canonical_xy']-truth))
        self.assertLess(np.mean(errors), .1)
        self.assertLess(np.mean(errors), np.mean(baseline)/3)
        layer.coverage[:] = 0
        self.assertFalse(fractional_match(layer, observation, mask, (200.,184.), 12.)['valid'])

    def test_peak_edges_and_flat_responses_do_not_extrapolate(self):
        for location in ((0, 0), (2, 2)):
            self.assertEqual((0., 0.), quadratic_peak(np.ones((5, 5)), location))

    def test_single_scale_reuses_layer_without_reading_or_resampling(self):
        endpoint = object()
        atlas = SimpleNamespace(map_scales={'world': 3.}, localizers={'world': endpoint})
        layers, metadata = _build_pyramid(atlas)
        self.assertEqual([(3., endpoint)], layers)
        self.assertEqual(0, metadata['extra_array_bytes'])
        atlas.map_scales['world'] = float('nan')
        with self.assertRaises(ValueError):
            _build_pyramid(atlas)

    def test_image_scores_own_xy_without_changing_discrete_mode(self):
        localizer = ScaleAwareLocalizer.__new__(ScaleAwareLocalizer)
        localizer.active_mode_id = 'town'
        localizer._scale_pyramid = [(1., .4), (2., .8), (3., .3)]
        def match(layer, *args):
            return {'valid': True, 'score': layer, 'coverage_fraction': 1., 'best_offset_canonical_xy': [2., 3.]}
        image, mask = np.zeros((12,12,3), np.uint8), np.ones((12,12), np.uint8)
        with patch.object(ScaleAwareLocalizer, '_observe_one_mode', staticmethod(match)):
            result = localizer.refine_active_near(image, mask, (10.,20.))
            self.assertEqual((12.,23.), (result['x'],result['y']))
            self.assertEqual(2., result['precision_scale']['estimated_scale'])
            self.assertEqual('town', localizer.active_mode_id)
            self.assertEqual('town', result['selected_mode_id'])
            rejected = localizer.refine_active_near(image, mask, (10.,20.), score_min=.9)
            self.assertFalse(rejected['valid'])
            self.assertIsNone(rejected['x'])
        with patch.object(ScaleAwareLocalizer, '_observe_one_mode', return_value={'valid':False}):
            self.assertIsNone(localizer.refine_active_near(image, mask, (10.,20.))['x'])


if __name__ == '__main__':
    unittest.main()
