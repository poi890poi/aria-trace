import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np

from aria_trace.services.mapping.scale_aware import ScaleAwareLocalizer
from benchmarks.localization.scale_cost import ParallelScaleLocalizer


class ScaleCostTests(unittest.TestCase):
    def test_completion_order_does_not_change_tied_scale_winner(self):
        localizer = ParallelScaleLocalizer.__new__(ParallelScaleLocalizer)
        localizer._matching_pool = ThreadPoolExecutor(max_workers=2)
        localizer._scale_pyramid = [(1.,1),(2.,2)]
        localizer.active_mode_id = 'world'
        second_finished = threading.Event()
        def match(layer,*args):
            if layer == 1:
                self.assertTrue(second_finished.wait(2))
            else:
                second_finished.set()
            return {'valid':True,'score':.8,'coverage_fraction':1.,'best_offset_canonical_xy':[layer,0.]}
        try:
            with patch.object(localizer,'_observe_one_mode',side_effect=match):
                result = localizer.refine_active_near(np.zeros((10,10,3),np.uint8),np.ones((10,10),np.uint8),(20.,30.))
            self.assertEqual(21.,result['x'])
            self.assertEqual(1.,result['precision_scale']['estimated_scale'])
            self.assertEqual('world',result['selected_mode_id'])
        finally:
            with patch.object(ScaleAwareLocalizer,'close'):
                localizer.close()
        self.assertIsNone(localizer._matching_pool)

    def test_worker_exception_is_not_converted_to_fresh_pose(self):
        localizer = ParallelScaleLocalizer.__new__(ParallelScaleLocalizer)
        localizer._matching_pool = ThreadPoolExecutor(max_workers=2)
        localizer._scale_pyramid = [(1.,1),(2.,2)]
        try:
            with patch.object(localizer,'_observe_one_mode',side_effect=RuntimeError('matching failed')):
                with self.assertRaisesRegex(RuntimeError,'matching failed'):
                    localizer.refine_active_near(np.zeros((10,10,3),np.uint8),np.ones((10,10),np.uint8),(20.,30.))
        finally:
            with patch.object(ScaleAwareLocalizer,'close'):
                localizer.close()


if __name__ == '__main__':
    unittest.main()
