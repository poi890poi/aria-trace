import unittest

from benchmarks.localization.transition_zone_ablation import without_spatial_gates
from rig_runtime.services.calibration.minimap.transition import TransitionController


class TransitionZoneAblationTests(unittest.TestCase):
    def test_outside_zone_still_requires_two_visual_confirmations(self):
        model = {"source_mode_id": "world", "target_mode_id": "town",
                 "runtime": {"minimum_mode_margin": .2},
                 "transition_zones": [{"center_xy": [0, 0], "radius_px": 2}]}
        scores = {"world": .6, "town": 1.0}
        original = TransitionController(model, confirmation_count=2)
        self.assertFalse(original.observation_required((100, 100)))
        for _ in range(3):
            self.assertFalse(original.update(scores, canonical_xy=(100, 100))["switched"])
        with without_spatial_gates():
            candidate = TransitionController(model, confirmation_count=2)
            self.assertTrue(candidate.observation_required((100, 100)))
            self.assertFalse(candidate.update(scores, canonical_xy=(100, 100))["switched"])
            self.assertTrue(candidate.update(scores, canonical_xy=(100, 100))["switched"])
            candidate.set_active_mode("world")
            for _ in range(3):
                self.assertFalse(candidate.update({"world": .9, "town": 1.0})["switched"])
        restored = TransitionController(model, confirmation_count=2)
        self.assertFalse(restored.observation_required((100, 100)))


if __name__ == "__main__":
    unittest.main()
