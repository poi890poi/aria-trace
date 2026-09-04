import unittest

from benchmarks.build_turn_evidence import _sample_indexes


class TurnEvidenceBuildTests(unittest.TestCase):
    def test_sampler_does_not_accumulate_capture_jitter(self):
        records = [
            {"session_time_ns": int(index * 1.0e9 / 30.0 + (index % 3) * 400_000)}
            for index in range(300)
        ]
        selected = _sample_indexes(records, 15.0)
        self.assertGreaterEqual(len(selected), 149)
        self.assertLessEqual(len(selected), 151)


if __name__ == "__main__":
    unittest.main()
