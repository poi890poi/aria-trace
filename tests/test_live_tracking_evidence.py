import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from acquisition.live_tracking_evidence import LiveTrackingEvidenceRecorder


class LiveTrackingEvidenceTests(unittest.TestCase):
    def test_persists_global_fix_and_jump_incident_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "tracking-run"
            recorder = LiveTrackingEvidenceRecorder(
                root,
                {"tracking_id": "fixture-run", "map_stitch_id": "map-a"},
                frame_sample_interval_s=0.0,
                incident_pre_s=2.0,
                incident_post_s=2.0,
                jump_threshold_map_px=8.0,
            )
            frame = np.full((48, 64, 3), 80, np.uint8)
            minimap = np.full((24, 24, 3), 120, np.uint8)
            recorder.record(
                frame,
                minimap,
                {
                    "sequence": 1,
                    "host_time_ns": 1_000_000_000,
                    "mode": "TRACK",
                    "pose": {"x": 10.0, "y": 10.0, "yaw_deg": 0.0},
                    "global_fix_fresh": False,
                    "trail": [[10.0, 10.0]],
                },
            )
            recorder.record(
                frame,
                minimap,
                {
                    "sequence": 2,
                    "host_time_ns": 2_000_000_000,
                    "mode": "TRACK",
                    "pose": {"x": 20.0, "y": 10.0, "yaw_deg": 0.0},
                    "global_fix_fresh": True,
                    "global_fix": {
                        "x": 22.0,
                        "y": 10.0,
                        "score": 0.81,
                        "margin": 0.07,
                        "decision": "consistent",
                        "alternatives": [
                            {"x": 22.0, "y": 10.0, "score": 0.81},
                            {"x": 90.0, "y": 40.0, "score": 0.72},
                        ],
                        "fusion": {
                            "accepted": True,
                            "reason": "consistent",
                            "applied_position_change_map_px": 10.0,
                        },
                    },
                    "trail": [[10.0, 10.0], [20.0, 10.0]],
                },
                diagnostics={
                    "observation": minimap,
                    "mask": np.full((24, 24), 255, np.uint8),
                    "transformed_gradient": np.full((24, 24), 100, np.uint8),
                    "search_region": frame,
                    "correlation_heatmap": frame,
                    "candidate_overlay": frame,
                    "map_overlay": frame,
                },
            )
            recorder.record(
                frame,
                minimap,
                {
                    "sequence": 3,
                    "host_time_ns": 3_000_000_000,
                    "mode": "TRACK",
                    "pose": {"x": 21.0, "y": 10.0, "yaw_deg": 0.0},
                    "global_fix_fresh": False,
                    "trail": [[20.0, 10.0], [21.0, 10.0]],
                },
            )
            summary = recorder.close(status="stopped", processed_frames=3)

            self.assertEqual(summary["status"], "complete")
            manifest = json.loads(
                (root / "live_tracking.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "stopped")
            self.assertEqual(manifest["counts"]["telemetry_rows"], 3)
            self.assertEqual(manifest["counts"]["global_fixes"], 1)
            self.assertEqual(manifest["counts"]["jump_incidents"], 1)
            telemetry = [
                json.loads(line)
                for line in (root / "telemetry.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertNotIn("trail", telemetry[0])
            fix_root = root / "global_fixes" / "fix_000001"
            self.assertTrue((fix_root / "global_fix.json").is_file())
            self.assertTrue((fix_root / "correlation_heatmap.png").is_file())
            self.assertTrue((fix_root / "candidate_overlay.png").is_file())
            incident_root = root / "incidents" / "jump_000001"
            self.assertTrue((incident_root / "incident.json").is_file())
            self.assertGreater(len(list(incident_root.glob("frame_*.jpg"))), 0)

    def test_captures_cursor_recovery_and_scale_transition_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "tracking-run"
            recorder = LiveTrackingEvidenceRecorder(
                root,
                {"tracking_id": "event-run"},
                frame_sample_interval_s=10.0,
                incident_pre_s=1.0,
                incident_post_s=1.0,
            )
            frame = np.full((32, 32, 3), 90, np.uint8)
            base = {
                "pose": {"x": 1.0, "y": 2.0, "yaw_deg": 0.0},
                "global_fix_fresh": False,
                "trail": [],
            }
            recorder.record(
                frame,
                frame,
                dict(
                    base,
                    sequence=1,
                    host_time_ns=1_000_000_000,
                    mode="TRACK",
                    cursor_tracking_state="stable",
                    local_motion={"recovery_requested": False},
                ),
            )
            recorder.record(
                frame,
                frame,
                dict(
                    base,
                    sequence=2,
                    host_time_ns=2_000_000_000,
                    mode="RELOCALIZING",
                    cursor_tracking_state="recovering",
                    local_motion={"recovery_requested": True},
                    map_transition={
                        "host_time_ns": 2_000_000_000,
                        "from_mode_id": "world",
                        "to_mode_id": "town",
                        "evidence_source": "live-minimap-layer-likelihoods",
                    },
                ),
            )
            recorder.close(status="stopped", processed_frames=2)

            manifest = json.loads(
                (root / "live_tracking.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["counts"]["event_incidents"], 4)
            events = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted((root / "events").glob("*/event.json"))
            ]
            self.assertEqual(
                {event["kind"] for event in events},
                {"tracker-mode", "cursor-state", "recovery", "map-scale-transition"},
            )
            self.assertTrue(
                all(list(path.parent.glob("frame_*.jpg")) for path in
                    sorted((root / "events").glob("*/event.json")))
            )


if __name__ == "__main__":
    unittest.main()
