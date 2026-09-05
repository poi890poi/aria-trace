import unittest
from unittest.mock import patch
from types import SimpleNamespace
import threading
import cv2
import numpy as np
from aria_trace.services.tracking.runtime import GlobalFix, GlobalMapLocalizer
from aria_trace.services.localization.route.tracker import RouteVisualTracker
from benchmarks.localization.tracking_followups import installed, sources
from tests import test_live_tracker as fixtures


class FollowupTests(unittest.TestCase):
    def test_all_candidates_compile(self):
        for variant in ("", "K", "L", "KL", "M", "KLM", "N", "MN", "LN"):
            for key, source in sources(variant).items():
                compile(source,str(key),"exec")

    def test_invalid_geometry_never_runs_map_correlation(self):
        localizer=object.__new__(GlobalMapLocalizer)
        localizer._cancel=threading.Event()
        points=[cv2.KeyPoint(float(i*3),float(i*2),1) for i in range(6)]
        descriptors=np.zeros((6,128),np.float32)
        localizer.sift=SimpleNamespace(detectAndCompute=lambda *args:(points,descriptors))
        localizer.map_points=points
        localizer.map_descriptors=descriptors
        localizer._transform=lambda *args:self.fail("transformed known-invalid geometry")
        pairs=[(cv2.DMatch(i,i,1),cv2.DMatch(i,(i+1)%6,10)) for i in range(6)]
        matcher=SimpleNamespace(knnMatch=lambda *args,**kwargs:pairs)
        affine=np.array([[2.,0,0],[0,2.,0]])
        with installed("N"), patch('aria_trace.services.tracking.runtime.cv2.BFMatcher',return_value=matcher), patch('aria_trace.services.tracking.runtime.cv2.estimateAffinePartial2D',return_value=(affine,np.ones((6,1),np.uint8))):
            fix=localizer.localize(np.zeros((30,30,3),np.uint8),np.full((30,30),255,np.uint8))
        self.assertFalse(fix.valid)
        self.assertIn("scale-out-of-range",fix.rejection_reasons)

    def test_layer_change_widens_search_without_disabling_continuity(self):
        class Map:
            def refine_near(self, observation, mask, center, search_radius_px, **kwargs):
                return {"valid":True,"x":40 if search_radius_px==55 else 11,"y":20,"score":.9}
        tracker=RouteVisualTracker(None,Map(),local_radius_px=12)
        tracker.seed(10,20,timestamp_ns=1_000_000_000)
        with installed("K"):
            tracker.confirm_trained_transition_layer("town")
            rejected=tracker.track(None,None,timestamp_ns=1_100_000_000)
            self.assertTrue(rejected["continuity_rejected"])
            self.assertEqual(rejected["x"],10)
            accepted=tracker.track(None,None,timestamp_ns=1_300_000_000)
            self.assertTrue(accepted["measurement_accepted"])
            self.assertEqual(accepted["x"],40)
            self.assertFalse(tracker._wide_after_mode_change)

    def test_initial_window_requires_a_valid_prior_hypothesis(self):
        tracker=fixtures.TwoRateTrackerTests._tracker(fixtures.ImmediateLocalizer())
        try:
            with installed("M"):
                self.assertEqual(tracker._global_search(),(None,None))
                tracker._initial_hypotheses=[GlobalFix(10,20,0,1,.9,.2,1)]
                self.assertEqual(tracker._global_search(),((10,20),150))
                self.assertIsNone(tracker.fusion._state)
                tracker._initial_hypotheses=[]
                self.assertEqual(tracker._global_search(),(None,None))
        finally:
            tracker.close()


if __name__=="__main__":
    unittest.main()
