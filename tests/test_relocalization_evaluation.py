import unittest

import numpy as np

from poc.evaluate_relocalization import (
    estimate_similarity,
    matrix_to_quaternion,
    quaternion_to_matrix,
    rotation_error_deg,
)


class RelocalizationEvaluationTests(unittest.TestCase):
    def test_recovers_similarity_transform(self):
        source = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        angle = np.deg2rad(31.0)
        rotation = np.array([[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]])
        scale = 2.75
        translation = np.array([4.0, -2.0, 7.0])
        target = (scale * (rotation @ source.T)).T + translation

        actual_scale, actual_rotation, actual_translation = estimate_similarity(source, target)

        self.assertAlmostEqual(actual_scale, scale, places=10)
        np.testing.assert_allclose(actual_rotation, rotation, atol=1e-10)
        np.testing.assert_allclose(actual_translation, translation, atol=1e-10)

    def test_rotation_error(self):
        identity = np.eye(3)
        angle = np.deg2rad(5.0)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]])
        self.assertAlmostEqual(rotation_error_deg(rotation, identity), 5.0, places=8)

    def test_rotation_quaternion_roundtrip(self):
        angle = np.deg2rad(127.0)
        rotation = np.array([[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]])
        quaternion = matrix_to_quaternion(rotation)
        recovered = quaternion_to_matrix(*quaternion)
        np.testing.assert_allclose(recovered, rotation, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
