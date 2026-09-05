import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.localization.build_realtime_control_report import build


class LocalizationRealtimeReportTests(unittest.TestCase):
    def test_held_positions_do_not_count_as_fresh(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "inputs"
            case = root / "gradient12" / "replay-a"
            case.mkdir(parents=True)
            report = {
                "rows_file": "telemetry.jsonl",
                "session": {"session_id": "a"},
                "parameters": {"feature": "gradient"},
                "initialization": {"elapsed_ms": 40.0},
                "method_traceability": {"variant": "local_primary_gated"},
            }
            (case / "report.json").write_text(json.dumps(report), encoding="utf-8")
            rows = [
                {
                    "valid": True,
                    "measurement_accepted": True,
                    "primary_candidate_produced": True,
                    "final_gate_rejected": False,
                    "localization_core_elapsed_ms": 4.0,
                    "end_to_end_serial_elapsed_ms": 7.0,
                    "reference_error_px": 1.0,
                },
                {
                    "valid": True,
                    "measurement_accepted": False,
                    "primary_candidate_produced": True,
                    "final_gate_rejected": True,
                    "localization_core_elapsed_ms": 4.0,
                    "end_to_end_serial_elapsed_ms": 8.0,
                    "reference_error_px": 7.0,
                },
            ]
            (case / "telemetry.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            result = build(root, Path(temporary) / "output")

            candidate = result["candidates"][0]
            self.assertEqual(candidate["fresh_measurement_accepted_rate"], 0.5)
            self.assertEqual(candidate["fresh_within_33_3ms_rate"], 0.5)
            self.assertEqual(candidate["fresh_reference_error_px"]["worst"], 1.0)
            self.assertEqual(
                candidate["served_e2e_reference_error_px"]["worst"], 7.0
            )
            self.assertEqual(
                candidate["serial_decode_to_xy_latency_ms"]["worst"], 8.0
            )
            self.assertEqual(candidate["worst_replay_fresh_rate"], 0.5)
            self.assertIsNone(result["recommendation"]["candidate"])


if __name__ == "__main__":
    unittest.main()
