import unittest

from acquisition.minimap_transition import (
    ModeObservation,
    TransitionController,
    learn_transition_model,
)
from acquisition.workbench import AcquisitionWorkbench


class MinimapTransitionTests(unittest.TestCase):
    @staticmethod
    def _observations():
        source = [0.90, 0.86, 0.82, 0.72, 0.55, 0.40, 0.22, 0.12, 0.08]
        target = [0.08, 0.10, 0.14, 0.25, 0.42, 0.58, 0.78, 0.88, 0.92]
        return [
            ModeObservation(
                frame_index=index,
                session_time_ns=index * 100_000_000,
                likelihoods={"world": source[index], "town": target[index]},
                canonical_xy=(100.0 + index, 200.0),
            )
            for index in range(len(source))
        ]

    def test_learns_directed_continuous_transition(self):
        model = learn_transition_model(self._observations(), "world", "town")

        self.assertEqual(model["source_mode_id"], "world")
        self.assertEqual(model["target_mode_id"], "town")
        self.assertEqual(model["position_semantics"], "continuous_no_displacement")
        self.assertLess(
            model["transition"]["first_frame_index"],
            model["transition"]["last_frame_index"],
        )
        self.assertGreater(model["quality"]["confidence"], 0.70)
        self.assertIsNotNone(model["canonical_boundary"])

    def test_controller_switches_only_after_consecutive_target_evidence(self):
        model = learn_transition_model(self._observations(), "world", "town")
        controller = TransitionController(model, confirmation_count=3)

        first = controller.update({"world": 0.4, "town": 0.6})
        interrupted = controller.update({"world": 0.7, "town": 0.3})
        controller.update({"world": 0.3, "town": 0.7})
        controller.update({"world": 0.2, "town": 0.8})
        switched = controller.update({"world": 0.1, "town": 0.9})

        self.assertEqual(first["state"], "transitioning")
        self.assertEqual(interrupted["state"], "source_locked")
        self.assertTrue(switched["switched"])
        self.assertTrue(switched["reset_local_reference"])
        self.assertEqual(switched["position_delta_xy"], [0.0, 0.0])
        self.assertEqual(switched["active_mode_id"], "town")

    def test_workbench_exposes_transition_capture_label(self):
        labels = {item["value"]: item for item in AcquisitionWorkbench.SESSION_LABELS}

        self.assertIn("minimap_transition", labels)
        self.assertEqual(
            labels["minimap_transition"]["capture_kind"], "game_behavior"
        )


if __name__ == "__main__":
    unittest.main()
