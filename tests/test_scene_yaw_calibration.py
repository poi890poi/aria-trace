import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from acquisition.scene_yaw_calibration import (
    _estimate,
    _estimate_measurements,
    _measure_tracks,
    _merged_config,
    calibrate_scene_yaw_frames,
)


class SceneYawCalibrationTests(unittest.TestCase):
    def test_fits_full_turn_from_scene_loop_closure_with_evidence(self):
        height, width = 180, 320
        step_px = 16
        frame_count = 73
        panorama_width = step_px * (frame_count - 1)
        rng = np.random.RandomState(19)
        panorama = rng.randint(0, 256, (height, panorama_width, 3), dtype=np.uint8)
        panorama = cv2.GaussianBlur(panorama, (5, 5), 0)
        tiled = np.concatenate((panorama, panorama[:, :width]), axis=1)
        frames = [
            tiled[:, index * step_px : index * step_px + width].copy()
            for index in range(frame_count)
        ]
        progress = []
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = calibrate_scene_yaw_frames(
                frames,
                output,
                config={
                    "focal_ratio_min": 0.40,
                    "focal_ratio_max": 0.90,
                    "focal_ratio_steps": 6,
                    "min_tracks": 12,
                    "max_corners": 500,
                    "use_essential_gate": False,
                },
                provenance={"fixture": "wrapped_scene_panorama"},
                progress=progress.append,
            )
            self.assertEqual(result["status"], "review_required")
            self.assertEqual(result["closure_frame_index"], frame_count - 1)
            self.assertGreater(result["closure_correlation"], 0.95)
            self.assertLess(result["closure_error_deg"], 12.0)
            self.assertGreater(result["valid_frame_rate"], 0.8)
            self.assertEqual(
                result["estimator_benchmark"]["sample_count"], frame_count
            )
            self.assertIn(
                "Fitting camera angular scale: candidate 15 of 15", progress
            )
            self.assertTrue((output / "scene_yaw_estimates.jsonl").is_file())
            for item in result["evidence"]:
                self.assertGreater((output / item["name"]).stat().st_size, 0)

    def test_cached_measurements_match_direct_estimator(self):
        height, width = 120, 200
        rng = np.random.RandomState(23)
        image = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        frames = [np.roll(image, -index * 3, axis=1) for index in range(8)]
        config = _merged_config(
            {
                "min_tracks": 12,
                "max_corners": 400,
                "use_essential_gate": True,
            }
        )

        cv2.setRNGSeed(42)
        direct, _ = _estimate(frames, 0.8, config)
        cv2.setRNGSeed(42)
        measurements = _measure_tracks(frames, config)
        cached, cached_durations = _estimate_measurements(
            measurements, frames[0].shape, 0.8, config
        )

        self.assertEqual(
            [row["status"] for row in cached],
            [row["status"] for row in direct],
        )
        np.testing.assert_allclose(
            [row["relative_yaw_deg"] for row in cached],
            [row["relative_yaw_deg"] for row in direct],
            rtol=0.0,
            atol=1.0e-9,
        )
        self.assertEqual(len(cached_durations), len(frames))


if __name__ == "__main__":
    unittest.main()
