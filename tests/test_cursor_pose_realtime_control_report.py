import csv
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.cursor_pose.build_realtime_control_report import build


class CursorPoseRealtimeControlReportTests(unittest.TestCase):
    def test_held_output_is_not_used_and_profile_caps_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark = root / "benchmark"
            sessions = root / "sessions"
            output = root / "output"
            benchmark.mkdir()
            run = sessions / "run_13"
            run.mkdir(parents=True)
            aggregate = []
            for confidence_mode in ("selected", "accept_all"):
                for fallback in (
                    "reuse_previous_state",
                    "constant_velocity_last_2_accepted",
                    "no_fallback",
                ):
                    for outage in ("natural", "three_frame_burst_every_90"):
                        aggregate.append(
                            {
                                "split": "holdout",
                                "profile": "fast_cascade_minimal",
                                "confidence_mode": confidence_mode,
                                "fallback_strategy": fallback,
                                "outage_scenario": outage,
                                "primary_measurement_accepted_rate": 0.5,
                                "output_provenance_rate": {
                                    "fresh_measurement": 0.5,
                                    "held": 0.5 if fallback == "reuse_previous_state" else 0.0,
                                    "predicted": 0.5 if fallback == "constant_velocity_last_2_accepted" else 0.0,
                                    "unavailable": 0.5 if fallback == "no_fallback" else 0.0,
                                },
                                "e2e_absolute_error_deg": {
                                    "sample_count": 2,
                                    "mean": 1.0,
                                    "median": 1.0,
                                    "p95": 2.0,
                                    "worst": 3.0,
                                },
                            }
                        )
            (benchmark / "results.json").write_text(
                json.dumps(
                    {
                        "sessions": ["run_13"],
                        "holdout_sessions": ["run_13"],
                        "aggregate": aggregate,
                    }
                ),
                encoding="utf-8",
            )
            with (benchmark / "primary_measurements.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "profile",
                        "session",
                        "confidence_mode",
                        "natural_primary_accepted",
                        "primary_latency_ns",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "profile": "fast_cascade_minimal",
                        "session": "run_13",
                        "confidence_mode": "selected",
                        "natural_primary_accepted": "True",
                        "primary_latency_ns": "7000000",
                    }
                )
                writer.writerow(
                    {
                        "profile": "fast_cascade_minimal",
                        "session": "run_13",
                        "confidence_mode": "selected",
                        "natural_primary_accepted": "False",
                        "primary_latency_ns": "7000000",
                    }
                )
            with (run / "frames.jsonl").open("w", encoding="utf-8") as stream:
                for index in range(4):
                    stream.write(
                        json.dumps(
                            {
                                "stream_id": "main",
                                "frame_index": index,
                                "session_time_ns": index * 33_000_000,
                            }
                        )
                        + "\n"
                    )

            # The standard report compares these promising profiles.  Reuse the
            # same controlled fixture rows under their IDs.
            source = json.loads((benchmark / "results.json").read_text(encoding="utf-8"))
            expanded = list(source["aggregate"])
            for profile in (
                "realtime_cascade_ambiguous",
                "angular_projection_ncc_parabolic",
                "symmetric_pixel_fft_ncc_parabolic",
            ):
                for row in aggregate:
                    clone = dict(row)
                    clone["profile"] = profile
                    expanded.append(clone)
            source["aggregate"] = expanded
            (benchmark / "results.json").write_text(
                json.dumps(source), encoding="utf-8"
            )

            result = build(benchmark, sessions, output)

            candidate = result["candidates"][0]
            self.assertEqual(candidate["fresh_measurement_accepted_rate"], 0.5)
            self.assertEqual(candidate["estimator_fresh_within_33_3ms_rate"], 0.5)
            self.assertEqual(candidate["production_configured_rate_cap_hz"], 10.0)
            self.assertIn("REJECT FOR LIVE CONTROL", candidate["decision"])
            self.assertFalse(result["premises"]["held_or_predicted_counts_as_fresh"])
            self.assertTrue(result["source_timing"]["run_13"]["can_evaluate_30_fps"])
            report = (output / "REPORT.txt").read_text(encoding="utf-8")
            self.assertIn("Held and predicted states are continuity outputs", report)
            self.assertNotIn("final_output_available_rate", report)


if __name__ == "__main__":
    unittest.main()
