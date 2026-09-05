import unittest
from aria_trace.services.tracking.runtime import GlobalFix
from aria_trace.services.localization.route.tracker import RouteVisualTracker
from benchmarks.localization.tracking_followups import installed, sources
from tests import test_live_tracker as fixtures


class FollowupTests(unittest.TestCase):
    def test_all_candidates_compile(self):
        for variant in ("", "K", "L", "KL", "M", "KLM"):
            for key, source in sources(variant).items():
                compile(source,str(key),"exec")

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
