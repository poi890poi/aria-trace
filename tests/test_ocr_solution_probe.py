"""The solution experiment must not turn recovery into false authorization."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/android"))
from ocr_solution_probe import CascadeReader, query_image


class FakeCascade(CascadeReader):
    def __init__(self, primary, recovery):
        self.primary = Mock(return_value=primary)
        self.recovery = Mock(return_value=recovery)

    def __call__(self, image, region=None):
        return self.primary(image, region)


class QueryPolicyTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((100, 200, 3), np.uint8)
        self.query = dict(labels=["Continue", "繼續", "继续"], region=[.1,.1,.9,.9])
        self.word = dict(text="Continue", score=.98, box=[30,20,50,20])

    def test_ambiguity_is_terminal_even_if_recovery_would_pick_one(self):
        reader = FakeCascade([self.word, dict(self.word, box=[100,60,50,20])], [self.word])
        result = query_image(reader, self.image, self.query, "roi")
        self.assertEqual(result["outcome"], "ambiguous")
        reader.recovery.assert_not_called()

    def test_recovery_uses_same_image_and_remains_within_declared_scope(self):
        outside = dict(self.word, box=[0,0,50,20])
        reader = FakeCascade([], [outside])
        result = query_image(reader, self.image, self.query, "roi")
        self.assertEqual(result["outcome"], "missing")
        reader.recovery.assert_called_once_with(self.image)
        self.assertEqual(result["calls"], 2)

    def test_missing_label_is_not_replaced_by_expected_answer(self):
        reader = FakeCascade([], [dict(self.word, text="Cancel")])
        self.assertEqual(query_image(reader, self.image, self.query, "roi")["outcome"], "missing")

    def test_recognized_target_is_not_held_on_a_later_frame(self):
        reader = FakeCascade([self.word], [])
        self.assertEqual(query_image(reader, self.image, self.query, "roi")["outcome"], "unique")
        reader.primary.return_value = []
        newer = self.image.copy()
        self.assertEqual(query_image(reader, newer, self.query, "roi")["outcome"], "missing")
        self.assertIs(reader.primary.call_args.args[0], newer)


if __name__ == "__main__":
    unittest.main()
