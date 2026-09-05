"""Experimental adapter boundary checks; no downloaded model needed."""

import unittest
from types import SimpleNamespace

import numpy as np

from benchmarks.localization.xfeat_cpu import XFeatAdapter, mnn_matches, mask_feature_input


class ArrayResult:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class XFeatAdapterTests(unittest.TestCase):
    def test_query_mask_changes_only_excluded_pixels_without_mutating_inputs(self):
        import cv2
        mask = np.zeros((21,21), np.uint8)
        cv2.circle(mask, (10,10), 8, 255, -1)
        cv2.circle(mask, (10,10), 2, 0, -1)
        gray = np.full(mask.shape, 100, np.uint8)
        gray[0,0], gray[10,10] = 200, 250
        saved = gray.copy()
        both = mask_feature_input(gray, mask, "mean")
        np.testing.assert_array_equal(both, np.full(mask.shape,100,np.uint8))
        cursor = mask_feature_input(gray, mask, "mean", "cursor")
        outside = mask_feature_input(gray, mask, "zero", "outside")
        self.assertEqual((cursor[0,0], cursor[10,10]), (200,100))
        self.assertEqual((outside[0,0], outside[10,10]), (0,250))
        np.testing.assert_array_equal(outside[mask>0],gray[mask>0])
        np.testing.assert_array_equal(gray,saved)

    def test_mask_preserves_correspondence_and_rejects_outside_points(self):
        out = {"keypoints": ArrayResult([[2, 3], [5, 5], [-2, 3], [12, 3]]),
               "scores": ArrayResult([.8, .7, .6, .5]),
               "descriptors": ArrayResult(np.eye(4))}
        adapter = XFeatAdapter(SimpleNamespace(detectAndCompute=lambda *a, **kw: [out]))
        mask = np.zeros((10, 10), np.uint8)
        mask[3, 2] = 255
        points, descriptors = adapter.detectAndCompute(mask, mask)
        self.assertEqual([p.pt for p in points], [(2.0, 3.0)])
        np.testing.assert_array_equal(descriptors, [[1, 0, 0, 0]])
        points, descriptors = adapter.detectAndCompute(mask, np.zeros_like(mask))
        self.assertEqual(points, [])
        self.assertIsNone(descriptors)

    def test_mnn_agrees_with_upstream_cosine_rule_for_unit_descriptors(self):
        rng = np.random.default_rng(54)
        query, target = rng.normal(size=(30, 64)), rng.normal(size=(50, 64))
        query = (query/np.linalg.norm(query, axis=1, keepdims=True)).astype(np.float32)
        target = (target/np.linalg.norm(target, axis=1, keepdims=True)).astype(np.float32)
        sim = query @ target.T
        forward, backward = sim.argmax(axis=1), sim.argmax(axis=0)
        expected = {(i, int(j)) for i, j in enumerate(forward) if backward[j] == i}
        self.assertEqual({(m.queryIdx, m.trainIdx) for m in mnn_matches(query, target)}, expected)

    def test_resized_features_return_to_original_mask_coordinates(self):
        seen = []
        def extract(image, **kwargs):
            seen.append(image.shape)
            return [{"keypoints": ArrayResult([[4, 6]]), "scores": ArrayResult([.8]),
                     "descriptors": ArrayResult([[1, 0]])}]
        adapter = XFeatAdapter(SimpleNamespace(detectAndCompute=extract), feature_scale=2)
        mask = np.zeros((10, 10), np.uint8)
        mask[3, 2] = 255
        points, _ = adapter.detectAndCompute(mask, mask)
        self.assertEqual(seen, [(20, 20)])
        self.assertEqual([p.pt for p in points], [(2.0, 3.0)])


if __name__ == "__main__":
    unittest.main()
