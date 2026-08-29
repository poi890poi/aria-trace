import unittest

from replay.route_similarity import route_similarity_report


class RouteSimilarityTests(unittest.TestCase):
    def test_reports_cross_track_rmse_without_alignment(self):
        demonstrated = [[0, 0], [10, 0], [20, 0]]
        live = [[0, 3], [10, 3], [20, 3]]

        report = route_similarity_report(live, demonstrated, 5.0)

        self.assertEqual(report["status"], "complete")
        self.assertAlmostEqual(report["cross_track_rmse_px"], 3.0)
        self.assertFalse(report["feeds_tracker"])
        self.assertEqual(report["alignment"], "none")
        self.assertEqual(report["time_synchronization"], "none")

    def test_stationary_frames_do_not_dominate_spatial_rmse(self):
        demonstrated = [[0, 0], [10, 0], [20, 0]]
        live = [[0, 2]] * 100 + [[10, 2], [20, 2]]

        report = route_similarity_report(live, demonstrated, 5.0)

        self.assertAlmostEqual(report["cross_track_rmse_px"], 2.0)
        self.assertEqual(report["live_sample_count"], 3)


if __name__ == "__main__":
    unittest.main()
