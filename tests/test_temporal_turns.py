import math
import unittest

from benchmarks.temporal_turns import (
    align_turn_signals,
    evaluate_reversal_response,
    find_sharp_reversals,
)


def _rows(values, step_s=0.02, lag_s=0.0):
    return [
        {"session_time_ns": int((index * step_s + lag_s) * 1.0e9), "value": value}
        for index, value in enumerate(values)
    ]


class TemporalTurnTests(unittest.TestCase):
    def test_alignment_finds_opposite_sign_and_response_lag(self):
        evidence = [math.sin(index * 0.07) + 0.4 * math.sin(index * 0.19) for index in range(500)]
        observed = [-value for value in evidence]
        result = align_turn_signals(_rows(evidence), _rows(observed, lag_s=0.18))
        self.assertEqual(result["sign_relation"], "opposite")
        self.assertAlmostEqual(result["lag_ms"], 166.67, delta=35.0)
        self.assertGreater(result["correlation"], 0.98)

    def test_reversal_detection_needs_no_pause_or_boundary(self):
        values = [1.0] * 50 + [-1.0] * 50 + [1.0] * 50 + [-1.0] * 50
        events = find_sharp_reversals(_rows(values), window_s=0.3)
        self.assertGreaterEqual(len(events), 3)

    def test_response_reports_hysteresis_delay(self):
        evidence = [1.0] * 50 + [-1.0] * 50 + [1.0] * 50 + [-1.0] * 50
        delay = 8
        observed = [1.0] * delay + evidence[:-delay]
        result = evaluate_reversal_response(
            _rows(evidence), _rows(observed), maximum_lag_ms=400.0
        )
        self.assertGreaterEqual(result["responded_reversal_count"], 2)
        self.assertGreater(result["alignment"]["lag_ms"], 100.0)
        self.assertIn("settling_time_ms", result)
        self.assertEqual(result["parameters"]["stable_window_ms"], 200.0)


if __name__ == "__main__":
    unittest.main()
