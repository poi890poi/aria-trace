import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.localization.reference_cache import ensure_reference
from benchmarks.localization.run_workbench_replay import score


class ReferenceCacheTests(unittest.TestCase):
    def test_reuses_identical_inputs_invalidates_atlas_and_detects_damage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session, atlas, calibration = [root / name for name in ("session", "atlas", "calibration")]
            for folder in (session, atlas, calibration):
                folder.mkdir()
            for name in ("manifest.json", "frames.jsonl", "video_main.mkv"):
                (session / name).write_text("{}")
            (atlas / "pixels.png").write_bytes(b"first atlas")
            (calibration / "calibration.json").write_text("{}")

            def compile_fake(session, output, **kwargs):
                output.mkdir(parents=True)
                (output / "route_states.jsonl").write_text("original reference")
                return {"state_count": 3}

            args = (root, session, atlas, calibration, {}, root / "cache")
            with patch("benchmarks.localization.reference_cache.compile_route_session", side_effect=compile_fake) as compile_method:
                first, hit = ensure_reference(*args)
                self.assertFalse(hit)
                second, hit = ensure_reference(*args)
                self.assertTrue(hit)
                self.assertEqual(first, second)
                self.assertEqual(compile_method.call_count, 1)
                (atlas / "pixels.png").write_bytes(b"rebuilt atlas")
                third, hit = ensure_reference(*args)
                self.assertFalse(hit)
                self.assertNotEqual(first, third)
                (third / "route_states.jsonl").write_text("damaged reference")
                with self.assertRaisesRegex(RuntimeError, "damaged"):
                    ensure_reference(*args)


class ReplayScoreTests(unittest.TestCase):
    def test_holds_drops_initialization_and_reference_gaps_stay_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory)
            (reference / "manifest.json").write_text(json.dumps({"reference_rate_hz": 2}))
            refs = [{"session_time_ns": int(t*1e9), "canonical_xy": [t, 0], "mode_id": mode}
                    for t, mode in ((0,"world"),(.5,"world"),(1,"town"),(2,"town"))]
            (reference / "route_states.jsonl").write_text("".join(json.dumps(r)+"\n" for r in refs))
            source_rows = [{"host_time_ns": int((t+10)*1e9), "session_time_ns": int(t*1e9), "frame_index": i, "decode_ms": 1, "release_lateness_ms": 0}
                           for i,t in enumerate((0,.25,.5,.75,1,1.5,2))]
            source = SimpleNamespace(rows=source_rows, frames=source_rows, origin=int(10e9))
            rows=[]
            for i in (0,1,2,3,5,6):
                rows.append({"host_time_ns": source_rows[i]["host_time_ns"], "pose": None if i==0 else {"x": 0,"y":0},
                             "xy_measurement_fresh_accepted": i in (1,2,6), "capture_to_control_publish_ms": 10,
                             "update_elapsed_ms": 5, "active_map_mode_id": "world"})
            enriched, result = score(rows, source, reference)
            self.assertEqual(result["unavailable_frames"], 1)
            self.assertEqual(result["held_frames"], 2)
            self.assertEqual(result["steady_fresh_rate"], 3/5)
            self.assertEqual(result["all_source_fresh_rate"], 3/7)
            self.assertAlmostEqual(result["initialization_s"], .25)
            self.assertEqual(len(result["loss_episodes"]), 1)
            self.assertNotIn("reference_error_px", enriched[3])  # transition
            self.assertNotIn("reference_error_px", enriched[4])  # oversized gap
            self.assertEqual(result["reference_error_px"]["count"], 3)


if __name__ == "__main__":
    unittest.main()
