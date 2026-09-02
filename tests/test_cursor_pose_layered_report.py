import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.cursor_pose.build_layered_report import build


def _e2e(solution, outage="natural"):
    return {
        "solution": solution,
        "outage_scenario": outage,
        "split": "holdout",
        "primary_candidate_produced_rate": 1.0,
        "primary_measurement_accepted_rate": 0.98,
        "output_provenance_rate": {
            "fresh_measurement": 0.98,
            "held": 0.02,
            "predicted": 0.0,
            "unavailable": 0.0,
        },
        "final_output_available_rate": 1.0,
        "e2e_absolute_error_deg": {
            "mean": 2.0,
            "median": 1.5,
            "p95": 4.0,
            "worst": 8.0,
        },
        "e2e_latency_ms": {"median": 3.0, "p95": 4.0},
    }


class CursorPoseLayeredReportTests(unittest.TestCase):
    def test_report_is_mobile_cards_and_preserves_source_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            method_path = root / "methods.json"
            e2e_path = root / "e2e.json"
            output = root / "report"
            method_path.write_text(
                json.dumps(
                    {
                        "aggregate": [
                            {
                                "method": "current_realtime_cascade_ambiguous",
                                "group": "production_baseline",
                                "split": "holdout",
                                "pose_production_rate": 1.0,
                                "mean_abs_error_deg": 2.0,
                                "median_abs_error_deg": 1.5,
                                "p95_abs_error_deg": 4.0,
                                "latency": {"median_ms": 3.0, "p95_ms": 4.0},
                            }
                        ],
                        "decisions": {
                            "current_realtime_cascade_ambiguous": "HOLD"
                        },
                    }
                ),
                encoding="utf-8",
            )
            e2e_path.write_text(
                json.dumps(
                    {
                        "aggregate": [
                            _e2e("accurate_confidence_hold"),
                            _e2e("realtime_confidence_hold"),
                            _e2e("fast_confidence_hold"),
                            _e2e("realtime_strict_hold"),
                            _e2e(
                                "realtime_confidence_hold",
                                "three_frame_burst_every_90",
                            ),
                            _e2e(
                                "realtime_confidence_predict",
                                "three_frame_burst_every_90",
                            ),
                            _e2e(
                                "realtime_confidence_reject",
                                "three_frame_burst_every_90",
                            ),
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = build(method_path, e2e_path, output)
            report = (output / "REPORT.md").read_text(encoding="utf-8")

            self.assertNotIn("|---", report)
            self.assertIn("## Layer 1A", report)
            self.assertIn("## Layer 2", report)
            self.assertIn("## Layer 3", report)
            self.assertEqual(len(result["method_results"]["sha256"]), 64)
            self.assertEqual(len(result["e2e_results"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
