import unittest

from benchmarks.cursor_pose.run_candidate_e2e import (
    _apply_threshold,
    _response_prominence,
    _select_threshold,
)


def _row(index, angle, confidence):
    return {
        "profile": "candidate",
        "session": "development",
        "ordinal": index,
        "frame_index": index,
        "session_time_ns": index * 100_000_000,
        "angle_deg": angle,
        "confidence": confidence,
        "primary_candidate_produced": True,
        "confidence_gate_passed": False,
        "natural_primary_accepted": False,
        "natural_rejection_reason": None,
        "confidence_threshold": None,
        "primary_latency_ns": 1_000_000,
        "pixel_validation_performed": False,
        "gaussian_fitter_fallback_used": False,
        "reference_valid": True,
        "reference_angle_deg": 0.0,
        "reference_response": 1.0,
        "reference_displacement_px": 10.0,
    }


class CursorPoseCandidateE2ETests(unittest.TestCase):
    def test_response_prominence_distinguishes_a_unique_peak(self):
        flat = [1.0] * 360
        peaked = list(flat)
        peaked[30] = 10.0

        self.assertEqual(_response_prominence(flat, 30.0), 0.0)
        self.assertGreater(_response_prominence(peaked, 30.0), 0.9)

    def test_threshold_selection_cannot_hold_everything(self):
        rows = [
            _row(index, 0.0, 0.9) if index < 95 else _row(index, 90.0, 0.1)
            for index in range(100)
        ]
        threshold, trials = _select_threshold(
            rows,
            {
                "innovation_limit_deg": 20.0,
                "large_innovation_confidence_min": 0.8,
            },
        )
        selected = next(row for row in trials if row["threshold"] == threshold)

        self.assertGreater(threshold, 0.1)
        self.assertGreaterEqual(
            selected["primary_measurement_accepted_rate"], 0.95
        )
        self.assertEqual(selected["final_output_available_rate"], 1.0)
        self.assertEqual(selected["e2e_absolute_error_deg"]["worst"], 0.0)

    def test_threshold_reports_fresh_rejection_separately_from_output(self):
        rows = [_row(0, 0.0, 0.9), _row(1, 30.0, 0.2)]
        thresholded = _apply_threshold(rows, 0.5)

        self.assertTrue(thresholded[0]["natural_primary_accepted"])
        self.assertFalse(thresholded[1]["natural_primary_accepted"])
        self.assertEqual(
            thresholded[1]["natural_rejection_reason"],
            "confidence_below_threshold",
        )


if __name__ == "__main__":
    unittest.main()
