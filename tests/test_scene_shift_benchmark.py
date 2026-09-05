import unittest

import cv2
import numpy as np

from benchmarks.scene_shift.methods import KltSceneYaw, PhaseSceneYaw


class SceneShiftBenchmarkTests(unittest.TestCase):
    @staticmethod
    def _frames():
        random = np.random.RandomState(42)
        first = random.randint(0, 255, (240, 400, 3), np.uint8)
        transform = np.float32([[1, 0, 4], [0, 1, 0]])
        second = cv2.warpAffine(first, transform, (400, 240))
        return first, second

    def test_klt_recovers_positive_horizontal_scene_shift(self):
        first, second = self._frames()
        method = KltSceneYaw(
            maximum_width=200,
            max_corners=100,
            forward_backward=True,
            essential_gate=False,
        )
        method.update(first)
        result = method.update(second)

        self.assertEqual(result.status, "ok")
        self.assertGreater(result.delta_deg, 0.0)
        self.assertLess(result.elapsed_ms, 1000.0)

    def test_phase_recovers_positive_horizontal_scene_shift(self):
        first, second = self._frames()
        method = PhaseSceneYaw(maximum_width=200, signal="gradient")
        method.update(first)
        result = method.update(second)

        self.assertEqual(result.status, "ok")
        self.assertGreater(result.delta_deg, 0.0)
        self.assertGreater(result.confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
