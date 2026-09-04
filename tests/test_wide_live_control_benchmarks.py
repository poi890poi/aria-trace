import unittest

from benchmarks.cursor_pose.run_wide_temporal import _apply
from benchmarks.localization.run_wide_temporal import _temporal


class WideLiveControlBenchmarkTests(unittest.TestCase):
    def test_pose_ema_uses_shortest_circular_path(self):
        rows = [
            {"session_time_ns": 0, "angle_deg": 359.0, "confidence": 1.0},
            {"session_time_ns": 33_333_333, "angle_deg": 1.0, "confidence": 1.0},
        ]
        output = _apply(rows, "ema_085", 0.0)
        self.assertLess(abs(output[-1]["output_angle_deg"] - 0.7), 0.01)

    def test_pose_physical_gate_rejects_jump_and_holds(self):
        rows = [
            {"session_time_ns": 0, "angle_deg": 10.0, "confidence": 1.0},
            {
                "session_time_ns": 10_000_000,
                "angle_deg": 100.0,
                "confidence": 1.0,
            },
        ]

        output = _apply(
            rows, "physical_gate", 0.0, turn_rate_limit_deg_s=180.0
        )

        self.assertEqual(output[-1]["output_provenance"], "held")
        self.assertEqual(output[-1]["output_angle_deg"], 10.0)
        self.assertTrue(output[-1]["final_physical_gate_rejected"])

    def test_localization_confidence_hold_is_not_fresh(self):
        rows = [
            {
                "session_time_ns": 0,
                "valid": True,
                "x": 10.0,
                "y": 20.0,
                "score": 0.8,
            },
            {
                "session_time_ns": 33_333_333,
                "valid": True,
                "x": 11.0,
                "y": 20.0,
                "score": 0.2,
            },
        ]
        output = _temporal(rows, "hold_below_050")
        self.assertEqual(output[-1]["output_provenance"], "held")
        self.assertEqual(output[-1]["output_x"], 10.0)
        self.assertFalse(output[-1]["measurement_accepted_by_temporal_policy"])


if __name__ == "__main__":
    unittest.main()
