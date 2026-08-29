import unittest

from acquisition.cursor_dynamics import summarize_cursor_dynamics


def poses(step_deg, count=40, dt_ns=50_000_000):
    return [
        {
            "detected": True,
            "confidence": 0.9,
            "session_time_ns": index * dt_ns,
            "angle_screen_deg": (index * step_deg) % 360.0,
        }
        for index in range(count)
    ]


class CursorDynamicsTests(unittest.TestCase):
    def test_separates_ordinary_and_stress_turning_envelopes(self):
        result = summarize_cursor_dynamics(
            {
                "ordinary_cruise": poses(1.0),
                "movement_only": poses(4.0),
            },
            {"ordinary_cruise": {"session_id": "ordinary-1"}},
        )
        envelope = result["recommended_runtime_envelope"]
        self.assertAlmostEqual(envelope["normal_turn_rate_p95_deg_s"], 20.0)
        self.assertAlmostEqual(envelope["calibrated_turn_rate_p99_deg_s"], 80.0)
        self.assertEqual(
            result["sources"]["ordinary_cruise"]["provenance"]["session_id"],
            "ordinary-1",
        )
        self.assertEqual(result["runtime_scope"], "temporal search bounds only; no teleport detection")

    def test_rejects_missing_standard_sources(self):
        with self.assertRaisesRegex(ValueError, "ordinary_cruise"):
            summarize_cursor_dynamics({"movement_only": poses(2.0)})


if __name__ == "__main__":
    unittest.main()
