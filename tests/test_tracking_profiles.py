import unittest

from acquisition.tracking_profiles import resolve_tracking_profile


class TrackingProfileTests(unittest.TestCase):
    def test_profiles_resolve_independently(self):
        realtime = resolve_tracking_profile("real-time")
        offline = resolve_tracking_profile("offline")
        self.assertTrue(realtime["temporal_pose_search"])
        self.assertFalse(offline["temporal_pose_search"])
        self.assertEqual(realtime["cursor_pose_method"], "cascade")
        self.assertEqual(offline["diagnostics_stride"], 1)

    def test_explicit_developer_override_is_visible(self):
        value = resolve_tracking_profile(
            "real-time", {"cursor_pose_method": "vectorized_grid"}
        )
        self.assertEqual(value["profile"], "real-time")
        self.assertEqual(value["cursor_pose_method"], "vectorized_grid")

    def test_unknown_profile_and_override_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown tracking profile"):
            resolve_tracking_profile("turbo")
        with self.assertRaisesRegex(ValueError, "Unknown tracking-profile override"):
            resolve_tracking_profile("fast", {"magic": True})


if __name__ == "__main__":
    unittest.main()
