import unittest

from acquisition.map_layer_references import _endpoint_candidates


class MapLayerReferenceTests(unittest.TestCase):
    def test_endpoint_sampling_keeps_source_and_target_disjoint(self):
        frames = [{"frame_index": index} for index in range(100)]

        source = _endpoint_candidates(frames, 10, False)
        target = _endpoint_candidates(frames, 10, True)

        self.assertEqual(len(source), 10)
        self.assertEqual(len(target), 10)
        self.assertLess(source[-1]["frame_index"], target[0]["frame_index"])
        self.assertEqual(source[0]["frame_index"], 0)
        self.assertEqual(target[-1]["frame_index"], 99)


if __name__ == "__main__":
    unittest.main()
