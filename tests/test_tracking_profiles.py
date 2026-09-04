import unittest

from acquisition.tracking_profiles import resolve_tracking_profile


class TrackingProfileTests(unittest.TestCase):
    def test_profiles_resolve_independently(self):
        realtime = resolve_tracking_profile("real-time")
        offline = resolve_tracking_profile("offline")
        self.assertFalse(realtime["temporal_pose_search"])
        self.assertFalse(offline["temporal_pose_search"])
        self.assertEqual(realtime["cursor_pose_method"], "cascade")
        self.assertEqual(
            realtime["cursor_pose_core"],
            "angular_projection_ncc_parabolic",
        )
        self.assertEqual(realtime["cursor_validation_policy"], "minimal")
        self.assertEqual(realtime["cursor_interval_s"], 0.0)
        self.assertEqual(realtime["pose_confidence_min"], 0.0)
        legacy = resolve_tracking_profile("real-time-legacy")
        self.assertEqual(legacy["cursor_pose_core"], "polygon_gaussian")
        self.assertEqual(legacy["cursor_interval_s"], 0.05)
        self.assertEqual(legacy["pose_confidence_min"], 0.45)
        self.assertEqual(
            resolve_tracking_profile("fast")["cursor_validation_policy"],
            "minimal",
        )
        self.assertEqual(offline["cursor_interval_s"], 0.0)
        self.assertTrue(realtime["cursor_worker_process"])
        self.assertEqual(realtime["cursor_opencv_threads"], 1)
        self.assertEqual(offline["cursor_opencv_threads"], 2)
        self.assertEqual(realtime["route_map_score_min"], 0.0)
        self.assertEqual(realtime["route_local_radius_px"], 12.0)
        self.assertEqual(offline["route_map_score_min"], 0.50)

    def test_explicit_developer_override_is_visible(self):
        value = resolve_tracking_profile(
            "real-time",
            {
                "cursor_pose_core": "polygon_gaussian",
                "cursor_pose_method": "vectorized_grid",
            },
        )
        self.assertEqual(value["profile"], "real-time")
        self.assertEqual(value["cursor_pose_core"], "polygon_gaussian")
        self.assertEqual(value["cursor_pose_method"], "vectorized_grid")

    def test_unknown_profile_and_override_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown tracking profile"):
            resolve_tracking_profile("turbo")
        with self.assertRaisesRegex(ValueError, "Unknown tracking-profile override"):
            resolve_tracking_profile("fast", {"magic": True})

    def test_route_map_threshold_is_validated(self):
        with self.assertRaisesRegex(ValueError, "route_map_score_min"):
            resolve_tracking_profile(
                "real-time", {"route_map_score_min": 1.1}
            )
        with self.assertRaisesRegex(ValueError, "route_local_radius_px"):
            resolve_tracking_profile(
                "real-time", {"route_local_radius_px": 2.0}
            )


if __name__ == "__main__":
    unittest.main()
