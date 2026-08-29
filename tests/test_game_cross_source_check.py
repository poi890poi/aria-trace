import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from acquisition.game_cross_source_check import (
    GameCrossSourceEvidenceRecorder,
    load_game_alignment_geometry,
    natural_crop_to_logical,
)
from acquisition.models import FramePacket


class GameCrossSourceCheckTests(unittest.TestCase):
    def test_natural_crop_maps_to_each_logical_orientation(self):
        crop = [10, 20, 30, 40]
        natural = [100, 200]
        self.assertEqual([10, 20, 30, 40], natural_crop_to_logical(crop, natural, 0))
        self.assertEqual([140, 10, 40, 30], natural_crop_to_logical(crop, natural, 1))
        self.assertEqual([60, 140, 30, 40], natural_crop_to_logical(crop, natural, 2))
        self.assertEqual([20, 60, 40, 30], natural_crop_to_logical(crop, natural, 3))

    def _calibration(self, root: Path) -> Path:
        config = root / "hik_camera_calibration.json"
        config.write_text(
            json.dumps(
                {
                    "phone": {"natural_screen_size_px": [100, 200]},
                    "normalization": {
                        "origin_screen_xy": [10, 20],
                        "output_size_px": [30, 40],
                        "valid_mask_file": "valid_screen_mask.png",
                    },
                }
            ),
            encoding="utf-8",
        )
        cv2.imwrite(str(root / "valid_screen_mask.png"), np.full((40, 30), 255, np.uint8))
        return config

    def test_loads_and_rotates_saved_rig_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self._calibration(Path(directory))
            geometry = load_game_alignment_geometry(
                config,
                {
                    "natural_size_px": [100, 200],
                    "quarter_turns_clockwise_from_natural": 1,
                },
            )
            self.assertEqual([20, 60, 40, 30], geometry["logical_crop_xywh"])
            self.assertEqual((30, 40), geometry["valid_mask"].shape)

    def test_game_frames_reuse_non_gating_cross_source_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._calibration(root)
            recorder = GameCrossSourceEvidenceRecorder(
                config,
                {
                    "natural_size_px": [100, 200],
                    "quarter_turns_clockwise_from_natural": 1,
                },
                sample_period_seconds=0.0,
            )
            session = root / "session"
            session.mkdir()
            recorder.start(session, "session-id", 0)
            rng = np.random.RandomState(7)
            visible = rng.randint(0, 256, (30, 40, 3), dtype=np.uint8)
            adb = np.zeros((100, 200, 3), np.uint8)
            adb[60:90, 20:60] = visible
            recorder.process(FramePacket("android_phone", adb, 100, 100), 0, 0)
            recorder.process(FramePacket("hik_phone", visible, 110, 110), 0, 10)
            recorder.close()
            summary = json.loads(
                (session / "cross_source_check" / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("measured", summary["status"])
            self.assertTrue(summary["non_gating"])
            self.assertEqual(1, summary["evaluated_pairs"])
            self.assertGreater(summary["metrics"]["confidence"], 0.99)
            self.assertTrue(
                (session / "cross_source_check" / "side_by_side_adb_then_hik.png").is_file()
            )


if __name__ == "__main__":
    unittest.main()
