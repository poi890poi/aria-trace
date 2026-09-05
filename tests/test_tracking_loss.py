import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.localization.tracking_loss import calibrate_loss_tolerances, evaluate_tracking_loss


def row(t, error=0, *, mode="world", pose=True, fresh=True):
    result = {"session_time_ns": round(t*1e9), "pose": {"x": 0, "y": 0} if pose else None,
              "xy_measurement_fresh_accepted": fresh, "active_map_mode_id": mode}
    if error is not None:
        result.update(reference_error_px=error, reference_mode="world")
    return result


def evaluate(rows, end=None, **kwargs):
    return evaluate_tracking_loss(rows, start_ns=0,
                                  end_ns=round(end*1e9) if end is not None else rows[-1]["session_time_ns"],
                                  **kwargs)


class TrackingLossTests(unittest.TestCase):
    def test_calibration_uses_other_reference_recordings_and_respects_map_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory)/name for name in ("other", "evaluated")]
            for path in paths:
                path.mkdir()
                (path/"manifest.json").write_text(json.dumps({"atlas_id": "same", "reference_rate_hz": 2}))
                records = [{"session_time_ns": i*500_000_000, "mode_id": mode,
                            "canonical_xy": [i if path.name == "other" else (i%2)*1000, 0],
                            "map_scale": scale}
                           for mode, scale, offset in (("town", 1, 0), ("world", 4, 20))
                           for i in range(offset, offset+15)]
                (path/"route_states.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
            calibration = calibrate_loss_tolerances(paths, exclude_reference=paths[1])
            self.assertEqual(len(calibration["inputs"]), 1)
            self.assertAlmostEqual(calibration["modes"]["world"]["error_limit_px"], 4*2**.5)
            self.assertAlmostEqual(calibration["modes"]["town"]["error_limit_px"], 2**.5)
            rows = [row(0, 3), row(.5, 3)]
            self.assertEqual(evaluate(rows, calibration=calibration, error_limit_px=None)["longest_lost_s"], 0)

    def test_missing_calibration_cannot_certify_correct_tracking(self):
        result = evaluate([row(0), row(.5)], calibration=calibrate_loss_tolerances([]), error_limit_px=None)
        self.assertIsNone(result["longest_lost_s"])

    def test_accepted_wrong_pose_remains_lost_until_end(self):
        result = evaluate([row(t, 60) for t in (0, .5, 1, 1.5, 2)])
        self.assertEqual(result["longest_lost_s"], 2)
        self.assertTrue(result["unrecovered_at_end"])
        self.assertIsNone(result["longest_post_acquisition_lost_s"])

    def test_wrong_layer_counts_even_when_xy_matches(self):
        result = evaluate([row(0), row(.5), row(1, mode="town"), row(1.5, mode="town")])
        self.assertEqual(result["longest_post_acquisition_lost_s"], .5)
        self.assertEqual(result["episodes"][0]["reasons"], ["wrong-map-layer"])

    def test_one_good_frame_cannot_reset_loss_and_confirmation_is_backdated(self):
        rows = [row(0), row(.5), row(1, 30), row(1.25), row(1.5, 30),
                row(2), row(2.25), row(2.5)]
        result = evaluate(rows)
        self.assertEqual(result["episode_count"], 1)
        self.assertEqual(result["longest_lost_s"], 1)
        self.assertEqual(result["episodes"][0]["recovery_confirmed_ns"], 2_500_000_000)
        self.assertFalse(result["unrecovered_at_end"])

    def test_unknown_intervals_do_not_prove_recovery(self):
        result = evaluate([row(0, 30), row(.5, None), row(1, None), row(1.5), row(2)])
        self.assertEqual(result["longest_lost_s"], 1.5)
        self.assertEqual(result["episodes"][0]["unknown_s"], 1)
        self.assertFalse(result["all_intervals_observable"])

    def test_cold_start_and_later_loss_are_separate(self):
        result = evaluate([row(0, pose=False), row(.5, pose=False), row(1), row(1.5),
                           row(2, pose=False), row(2.5)])
        self.assertEqual(result["first_verified_acquisition_s"], 1)
        self.assertEqual(result["longest_lost_s"], 1)
        self.assertEqual(result["longest_post_acquisition_lost_s"], .5)
        self.assertTrue(result["unrecovered_at_end"])  # final good sample unconfirmed

    def test_held_correct_pose_is_not_position_loss_but_cannot_confirm_recovery(self):
        result = evaluate([row(0), row(.5), row(1, fresh=False), row(1.5, fresh=False)])
        self.assertEqual(result["longest_lost_s"], 0)
        result = evaluate([row(0, 30), row(.5, fresh=False), row(1, fresh=False)])
        self.assertTrue(result["unrecovered_at_end"])

    def test_missing_reference_is_unavailable_evaluation_not_zero_loss(self):
        result = evaluate([row(0, None), row(.5, None), row(1, None)])
        self.assertIsNone(result["longest_lost_s"])
        self.assertEqual(result["unknown_s"], 1)

    def test_long_telemetry_gap_breaks_recovery_confirmation(self):
        result = evaluate([row(0, 30), row(.5), row(3)])
        self.assertTrue(result["unrecovered_at_end"])
        self.assertEqual(result["episodes"][0]["unknown_s"], 1.75)

    def test_tolerance_is_explicit_and_boundary_inclusive(self):
        rows = [row(0, 10), row(.5, 10)]
        self.assertEqual(evaluate(rows, error_limit_px=10)["longest_lost_s"], 0)
        self.assertEqual(evaluate(rows, error_limit_px=5)["longest_lost_s"], .5)
        with self.assertRaises(ValueError):
            evaluate(rows, error_limit_px=0)


if __name__ == "__main__":
    unittest.main()
