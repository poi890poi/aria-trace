import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.localization.build_cross_session_report import build


class LocalizationCrossSessionReportTests(unittest.TestCase):
    def test_duplicate_family_does_not_vote_twice_and_hold_is_not_a_vote(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "input"
            reference = Path(temporary) / "reference"
            reference.mkdir()
            (reference / "route_states.jsonl").write_text(
                json.dumps({"session_time_ns": 1, "mode_id": "town"}) + "\n",
                encoding="utf-8",
            )
            cases = {
                "gradient12": ("gradient", "ccorr_normed", True, 10.0),
                "gradient18": ("gradient", "ccorr_normed", True, 12.0),
                "intensity18": ("intensity", "ccorr_normed", True, 14.0),
                "phase12": ("gradient", "phase_correlation", False, 99.0),
            }
            for name, (feature, matcher, fresh, x) in cases.items():
                path = root / name / "run" 
                path.mkdir(parents=True)
                report = {
                    "rows_file": "telemetry.jsonl",
                    "session": {"session_id": "session"},
                    "parameters": {
                        "correlation_feature": feature,
                        "local_matcher": matcher,
                        "local_radius_px": 12 if "12" in name else 18,
                    },
                    "reference_package": str(reference),
                }
                (path / "report.json").write_text(json.dumps(report), encoding="utf-8")
                row = {
                    "session_time_ns": 1,
                    "measurement_accepted": fresh,
                    "initialization_frame": False,
                    "x": x,
                    "y": 0.0,
                    "reference_error_px": abs(x - 11.0),
                    "localization_core_elapsed_ms": 2.0,
                }
                (path / "telemetry.jsonl").write_text(
                    json.dumps(row) + "\n", encoding="utf-8"
                )

            result = build(root, Path(temporary) / "output")

            gradient12 = next(
                row for row in result["results"] if row["candidate"] == "gradient12"
            )
            self.assertEqual(gradient12["mode_id"], "town")
            self.assertEqual(
                gradient12["cross_family_consensus_error_px"]["sample_count"], 0
            )
            intensity = next(
                row for row in result["results"] if row["candidate"] == "intensity18"
            )
            self.assertEqual(
                intensity["cross_family_consensus_error_px"]["sample_count"], 0
            )

    def test_consensus_qualification_uses_all_sessions_and_anchor_basin(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "input"
            reference = Path(temporary) / "reference"
            reference.mkdir()
            (reference / "route_states.jsonl").write_text(
                json.dumps({"session_time_ns": 1, "mode_id": "world"}) + "\n",
                encoding="utf-8",
            )
            methods = {
                "gradient12": ("gradient", "ccorr_normed", 12),
                "gradient18": ("gradient", "ccorr_normed", 18),
                "intensity18": ("intensity", "ccorr_normed", 18),
                "phase12": ("gradient", "phase_correlation", 12),
                "laplacian18": ("laplacian", "ccorr_normed", 18),
            }
            for session_index in range(2):
                for name, (feature, matcher, radius) in methods.items():
                    path = root / name / "run{}".format(session_index)
                    path.mkdir(parents=True)
                    (path / "report.json").write_text(
                        json.dumps(
                            {
                                "rows_file": "telemetry.jsonl",
                                "session": {"session_id": "session{}".format(session_index)},
                                "parameters": {
                                    "correlation_feature": feature,
                                    "local_matcher": matcher,
                                    "local_radius_px": radius,
                                },
                                "reference_package": str(reference),
                            }
                        ),
                        encoding="utf-8",
                    )
                    x = 0.0
                    if name == "gradient12" and session_index == 1:
                        x = 30.0
                    if name == "phase12" and session_index == 1:
                        x = 100.0
                    row = {
                        "session_time_ns": 1,
                        "measurement_accepted": True,
                        "initialization_frame": False,
                        "x": x,
                        "y": 0.0,
                        "reference_error_px": abs(x),
                        "localization_core_elapsed_ms": 2.0,
                    }
                    (path / "telemetry.jsonl").write_text(
                        json.dumps(row) + "\n", encoding="utf-8"
                    )

            result = build(root, Path(temporary) / "output")

            selected = result["qualified_consensus_representatives_by_mode"]["world"]
            self.assertEqual(selected["ccorr_normed:gradient"], "gradient18")
            self.assertNotIn("phase_correlation:gradient", selected)


if __name__ == "__main__":
    unittest.main()
