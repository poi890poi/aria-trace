import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from aria_trace.services.tracking.runtime import GlobalMapLocalizer
from benchmarks.localization.template_cpu import installed
from benchmarks.localization.known_start import read_start, installed_start, Runtime


class KnownStartTests(unittest.TestCase):
    def test_only_demonstrated_first_position_is_used(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"route_states.jsonl"
            state = {"canonical_xy": [100, 200], "mode_id": "world",
                     "state_index": 0, "source_frame_index": 0, "session_time_ns": 10}
            path.write_text(json.dumps(state)+"\nnot a readable future state\n")
            hint = read_start(path, (40, -40))
            self.assertEqual(hint["center_xy"], [140, 160])
            self.assertEqual(hint["source_state_index"], 0)
            self.assertEqual(hint["role"], "demonstrated-start-prior-not-current-measurement")

    def test_prior_does_not_create_fresh_measurement_or_reset_later_position(self):
        hint = {"center_xy": [100, 200], "mode_id": "world",
                "map_alignment_deg": 0., "role": "demonstrated-start-prior-not-current-measurement"}
        fusion = SimpleNamespace(_state=None)
        fusion.initialize = lambda pose: setattr(fusion, "_state", pose)
        tracker = SimpleNamespace(fusion=fusion, _activate_map_mode=lambda *a, **k: None)
        with patch.object(Runtime, "update", return_value={"xy_measurement_fresh_accepted": False}):
            with installed_start(hint, "prior"):
                row = Runtime.update(tracker)
                self.assertEqual((fusion._state.x, fusion._state.y), (100, 200))
                self.assertFalse(row["xy_measurement_fresh_accepted"])
                fusion._state = "later-measured-position"
                Runtime.update(tracker)
                self.assertEqual(fusion._state, "later-measured-position")


class TranslationTemplateTests(unittest.TestCase):
    def test_known_translation_ignores_masked_pixel_corruption_and_maps_coordinates(self):
        rng=np.random.default_rng(752)
        mosaic=rng.integers(20,230,size=(180,200,3),dtype=np.uint8)
        patch=mosaic[60:108,90:138].copy()
        mask=np.zeros((48,48),np.uint8)
        cv2.circle(mask,(24,24),19,255,-1)
        cv2.circle(mask,(24,24),5,0,-1)
        # Keep one-pixel descriptor/gradient boundary support uncorrupted.
        outer=cv2.dilate(mask,np.ones((3,3),np.uint8))==0
        patch[outer]=255
        transform=np.array([[2.,0,10],[0,2.,20],[0,0,1.]])
        for rep in ("gray","gradient"):
            with self.subTest(rep=rep),tempfile.TemporaryDirectory() as folder:
                with installed(rep,Path(folder)/"candidate"):
                    localizer=GlobalMapLocalizer(mosaic,localization_to_original_3x3=transform)
                    fix=localizer.localize(patch,mask)
                    self.assertTrue(fix.valid,fix.rejection_reasons)
                    self.assertAlmostEqual(fix.x,238.,places=4)
                    self.assertAlmostEqual(fix.y,188.,places=4)
                    self.assertEqual(fix.inlier_count,0)
                    self.assertIsNone(fix.center_agreement_px)

    def test_repeated_pattern_is_rejected_and_bounded_search_respects_prior(self):
        rng=np.random.default_rng(432)
        tile=rng.integers(20,230,size=(100,100,3),dtype=np.uint8)
        mosaic=np.tile(tile,(2,2,1))
        observation=tile[30:70,30:70].copy()
        mask=np.full((40,40),255,np.uint8)
        for rep in ("gray","gradient"):
            with self.subTest(rep=rep),tempfile.TemporaryDirectory() as folder:
                with installed(rep,Path(folder)/"candidate"):
                    localizer=GlobalMapLocalizer(mosaic)
                    fix=localizer.localize(observation,mask)
                    self.assertFalse(fix.valid)
                    self.assertIn("ambiguous-correlation",fix.rejection_reasons)
                    bounded=localizer.localize(observation,mask,search_center_xy=(150,150),search_radius_px=16)
                    self.assertTrue(bounded.valid,bounded.rejection_reasons)
                    self.assertEqual((bounded.x,bounded.y),(150.,150.))
                    self.assertLessEqual(bounded.margin,1.0)
                    self.assertTrue(all(p["score"] > -1 for p in bounded.alternatives))

    def test_blank_observation_and_unobserved_atlas_are_not_positions(self):
        rng=np.random.default_rng(783)
        mosaic=rng.integers(20,230,size=(150,150,3),dtype=np.uint8)
        for rep in ("gray","gradient"):
            with self.subTest(rep=rep),tempfile.TemporaryDirectory() as folder:
                with installed(rep,Path(folder)/"candidate"):
                    localizer=GlobalMapLocalizer(mosaic,np.zeros((150,150),np.uint8))
                    mask=np.full((40,40),255,np.uint8)
                    self.assertFalse(localizer.localize(mosaic[30:70,30:70],mask).valid)
                    fix=localizer.localize(np.zeros((40,40,3),np.uint8),mask)
                    self.assertFalse(fix.valid)
                    self.assertIn("constant-observation",fix.rejection_reasons)

    def test_small_search_window_does_not_invent_suppressed_alternatives(self):
        rng = np.random.default_rng(801)
        mosaic = rng.integers(20, 230, size=(280, 280, 3), dtype=np.uint8)
        observation = mosaic[70:208, 70:208].copy()
        mask = np.full((138, 138), 255, np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            with installed("gradient", Path(folder)/"candidate"):
                fix = GlobalMapLocalizer(mosaic).localize(
                    observation, mask, search_center_xy=(139, 139), search_radius_px=16)
                self.assertTrue(fix.valid, fix.rejection_reasons)
                self.assertEqual(len(fix.alternatives), 1)
                self.assertAlmostEqual(fix.margin, fix.score)


if __name__=="__main__":
    unittest.main()
