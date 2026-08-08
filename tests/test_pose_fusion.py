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

    def test_accepts_consistent_absolute_pose(self):
        fusion = PoseFusionGate()
        fusion.initialize(Pose2D(0.0, 0.0, 0.0))
        fusion.predict((0.2, 0.0), 1.0)
        decision = fusion.consider_absolute(
            Pose2D(0.21, 0.01, 1.2), Pose2D(0.3, -0.1, 4.0)
        )
        self.assertTrue(decision.accepted)

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
