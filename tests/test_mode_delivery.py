from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace
import unittest

import numpy as np

from aria_trace.services.tracking import Pose2D
from aria_trace.services.tracking.runtime import TwoRateRealtimeTracker
from benchmarks.localization.mode_delivery import consume_after_xy


class ModeDeliveryTests(unittest.TestCase):
    def test_result_finishing_during_xy_is_published_without_relaxing_confirmation(self):
        for late in (False,True):
            localizer = SimpleNamespace(
                transition_model={'source_mode_id':'world','target_mode_id':'town',
                                  'runtime':{'confirmation_count':2,'minimum_mode_margin':.2}},
                map_scale_for_mode=lambda mode: 3. if mode == 'world' else 1.)
            pending = Future()
            def track(*args,**kwargs):
                pending.set_result({'valid':True,'likelihoods':{'world':.1,'town':.9},
                                    'canonical_xy_read_only':[80.,70.]})
                return {'measurement_accepted':True,'x':80.,'y':70.,'score':.9}
            route = SimpleNamespace(previous_xy=(80.,70.),track=track)
            tracker = TwoRateRealtimeTracker(np.full((160,180,3),40,np.uint8),
                {'crop_xywh':[0,0,80,80]}, {'outer_boundary':{'center_x':40,'center_y':40,'radius':34}},
                {'focal_ratio':.9},localizer=localizer,route_visual_tracker=route)
            try:
                tracker.fusion.initialize(Pose2D(80.,70.,0.))
                tracker._activate_map_mode('world',update_scale=True)
                first = tracker.transition_controller.update({'world':.1,'town':.9},canonical_xy=(80.,70.))
                self.assertFalse(first['switched'])
                tracker._representation_future = pending
                with consume_after_xy() if late else nullcontext():
                    result = tracker.update(np.full((100,100,3),40,np.uint8),1)
                self.assertEqual('town' if late else 'world',result['active_map_mode_id'])
                self.assertEqual(1. if late else 3.,result['map_scale'])
                self.assertAlmostEqual(80.,result['pose']['x'])
                self.assertAlmostEqual(70.,result['pose']['y'])
            finally:
                tracker.close()


if __name__ == '__main__':
    unittest.main()
