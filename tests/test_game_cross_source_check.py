import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from acquisition.game_cross_source_check import (
    GameCrossSourceEvidenceRecorder,
    load_game_alignment_geometry,
    match_game_camera_orientation,
    natural_crop_to_logical,
    orient_hik_source_from_first_adb_frame,
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

    def _calibration(self, root: Path, *, write_mask: bool = True) -> Path:
        config = root / "hik_camera_calibration.json"
        config.write_text(
            json.dumps(
                {
                    "phone": {
                        "natural_screen_size_px": [100, 200],
                        "screen_size_px": [200, 100],
                        "orientation_quarter_turns": 1,
                    },
                    "normalization": {
                        "origin_screen_xy": [10, 20],
                        "output_size_px": [30, 40],
                        "valid_mask_file": "valid_screen_mask.png",
                    },
                }
            ),
            encoding="utf-8",
        )
        if write_mask:
            cv2.imwrite(
                str(root / "valid_screen_mask.png"),
                np.full((40, 30), 255, np.uint8),
            )
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
            self.assertEqual([10, 20, 30, 40], geometry["logical_crop_xywh"])
            self.assertEqual((40, 30), geometry["valid_mask"].shape)
            self.assertEqual("available", geometry["valid_mask_status"]["status"])

    def test_missing_mask_does_not_block_geometry_or_orientation_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._calibration(root, write_mask=False)
            geometry = load_game_alignment_geometry(
                config,
                {"quarter_turns_clockwise_from_natural": 1},
            )
            self.assertIsNone(geometry["valid_mask"])
            self.assertEqual("missing", geometry["valid_mask_status"]["status"])

            rng = np.random.RandomState(17)
            hik = rng.randint(0, 256, (40, 30, 3), dtype=np.uint8)
            adb = np.zeros((100, 200, 3), np.uint8)
            adb[20:60, 10:40] = hik
            summary, _images = match_game_camera_orientation(adb, hik, config)
            self.assertEqual(
                "full_output_unmasked_fallback",
                summary["valid_screen_mask"]["orientation_comparison"],
            )
            self.assertIn("unmasked", summary["warning"])

            recorder = GameCrossSourceEvidenceRecorder(
                config,
                {"quarter_turns_clockwise_from_natural": 1},
                sample_period_seconds=0.0,
            )
            session = root / "session"
            session.mkdir()
            recorder.start(session, "session-id", 0)
            recorder.process(FramePacket("android_phone", adb, 100, 100), 0, 0)
            recorder.process(FramePacket("hik_phone", hik, 110, 110), 0, 10)
            recorder.close()
            saved = json.loads(
                (session / "cross_source_check" / "summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("unavailable", saved["status"])
            self.assertEqual(0, saved["evaluated_pairs"])
            self.assertEqual("missing", saved["valid_screen_mask"]["status"])

    def test_first_images_select_orientation_without_trusting_android_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._calibration(root)
            rng = np.random.RandomState(11)
            hik_calibration_display = rng.randint(0, 256, (40, 30, 3), dtype=np.uint8)
            adb = np.zeros((100, 200, 3), np.uint8)
            adb[20:60, 10:40] = hik_calibration_display

            summary, images = match_game_camera_orientation(
                adb,
                hik_calibration_display,
                config,
                android_reported_quarter_turns=3,
            )

            self.assertEqual(
                1,
                summary[
                    "selected_adb_surface_quarter_turns_clockwise_from_phone_natural"
                ],
            )
            self.assertEqual(
                0,
                summary[
                    "selected_camera_adapter_image_quarter_turns_clockwise_from_calibration_display"
                ],
            )
            self.assertEqual(3, summary["android_reported_quarter_turns_clockwise_from_natural"])
            self.assertGreater(summary["selected_confidence"], 0.99)
            self.assertEqual(
                "first_game_adb_and_hik_image_evidence_only",
                summary["selection_basis"],
            )
            self.assertIn(
                "candidate_surface_1_adapter_0deg_side_by_side_adb_then_hik.png",
                images,
            )

    def test_preflight_sets_source_orientation_before_recorder_reuses_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._calibration(root)
            rng = np.random.RandomState(13)
            hik_calibration_display = rng.randint(0, 256, (40, 30, 3), dtype=np.uint8)
            adb = np.zeros((100, 200, 3), np.uint8)
            adb[20:60, 10:40] = hik_calibration_display

            class AdbSource:
                def __init__(self):
                    self.started = False

                def start(self):
                    self.started = True

                def read(self):
                    return FramePacket("android_phone", adb, 100, 100)

            class HikSource:
                def __init__(self):
                    self.started = False
                    self.turns = []

                def start(self):
                    self.started = True

                def set_output_orientation(self, turns, evidence=None):
                    self.turns.append((int(turns), evidence))

                def read(self):
                    return FramePacket("hik_phone", hik_calibration_display, 110, 110)

                def alignment_evidence_image(self, packet):
                    return packet.image

            adb_source = AdbSource()
            hik_source = HikSource()
            summary, _images = orient_hik_source_from_first_adb_frame(
                adb_source,
                hik_source,
                config,
                android_reported_quarter_turns=3,
            )
            self.assertTrue(adb_source.started)
            self.assertTrue(hik_source.started)
            self.assertEqual([0, 0], [value[0] for value in hik_source.turns])
            self.assertEqual(
                1,
                summary[
                    "selected_adb_surface_quarter_turns_clockwise_from_phone_natural"
                ],
            )

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
                orientation_match={
                    "status": "selected",
                    "selection_basis": "first_game_adb_and_hik_image_evidence_only",
                    "selected_adb_surface_quarter_turns_clockwise_from_phone_natural": 1,
                    "selected_camera_adapter_image_quarter_turns_clockwise_from_calibration_display": 0,
                    "selected_confidence": 0.99,
                },
                orientation_evidence_images={
                    "first_adb_game_image.png": np.zeros((2, 2, 3), np.uint8)
                },
            )
            session = root / "session"
            session.mkdir()
            recorder.start(session, "session-id", 0)
            rng = np.random.RandomState(7)
            visible = rng.randint(0, 256, (40, 30, 3), dtype=np.uint8)
            adb = np.zeros((100, 200, 3), np.uint8)
            adb[20:60, 10:40] = visible
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
            self.assertTrue(
                (
                    session
                    / "cross_source_check"
                    / "orientation_match"
                    / "first_adb_game_image.png"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
