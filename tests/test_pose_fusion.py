import unittest

from poc.pose_fusion import Pose2D, PoseFusionGate, angle_difference_deg


class PoseFusionTests(unittest.TestCase):
    def test_angle_difference_wraps(self):
        self.assertAlmostEqual(angle_difference_deg(-179.0, 179.0), 2.0)

    def test_prediction_uses_heading(self):
        fusion = PoseFusionGate()
        fusion.initialize(Pose2D(0.0, 0.0, 90.0))
        fusion.predict((1.0, 0.0), 0.0)
        self.assertAlmostEqual(fusion.state.pose.x, 0.0, places=8)
        self.assertAlmostEqual(fusion.state.pose.y, 1.0, places=8)

    def test_strong_measurements_accumulate_less_uncertainty(self):
        weak = PoseFusionGate()
        strong = PoseFusionGate()
        weak.initialize(Pose2D(0.0, 0.0, 0.0))
        strong.initialize(Pose2D(0.0, 0.0, 0.0))

        for _ in range(20):
            weak.predict((0.1, 0.0), 0.2, measurement_quality=0.0)
            strong.predict((0.1, 0.0), 0.2, measurement_quality=1.0)

        self.assertLess(
            strong.state.position_sigma_m,
            weak.state.position_sigma_m * 0.25,
        )
        self.assertLess(strong.state.yaw_sigma_deg, weak.state.yaw_sigma_deg)

    def test_accepts_consistent_absolute_pose(self):
        fusion = PoseFusionGate()
        fusion.initialize(Pose2D(0.0, 0.0, 0.0))
        fusion.predict((0.2, 0.0), 1.0)
        decision = fusion.consider_absolute(
            Pose2D(0.21, 0.01, 1.2), Pose2D(0.3, -0.1, 4.0)
        )
        self.assertTrue(decision.accepted)

    def test_position_measurement_updates_xy_without_changing_yaw(self):
        fusion = PoseFusionGate()
        fusion.initialize(Pose2D(4.0, 5.0, 37.0))
        fusion.accept_position_measurement(8.0, 9.0, measurement_quality=0.8)

        self.assertEqual((fusion.state.pose.x, fusion.state.pose.y), (8.0, 9.0))
        self.assertEqual(fusion.state.pose.yaw_deg, 37.0)

    def test_rejects_position_jump(self):
        fusion = PoseFusionGate()
        fusion.initialize(Pose2D(0.0, 0.0, 0.0))
        decision = fusion.consider_absolute(
            Pose2D(4.0, 0.0, 0.0), Pose2D(0.1, 0.0, 0.0)
        )
        self.assertFalse(decision.accepted)
        self.assertIn("position", decision.reason)

    def test_rejects_wrong_heading(self):
        fusion = PoseFusionGate()
        fusion.initialize(Pose2D(0.0, 0.0, 0.0))
        decision = fusion.consider_absolute(
            Pose2D(0.0, 0.0, 160.0), Pose2D(0.0, 0.0, 5.0)
        )
        self.assertFalse(decision.accepted)
        self.assertIn("heading", decision.reason)


if __name__ == "__main__":
    unittest.main()
