"""Evidence writes overlap transitions but remain part of task completion."""
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from examples.genshin.profile import Runner


class EvidenceTests(unittest.TestCase):
    def test_completion_waits_for_the_original_observed_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner(Mock(), SimpleNamespace(output=directory))
            original = np.zeros((3, 4, 3), dtype=np.uint8)
            runner.frame = original
            entered, release, tapped = threading.Event(), threading.Event(), threading.Event()
            written = []

            def write(path, frame):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("Test did not release evidence writer")
                written.append(frame)
                return True

            def task():
                runner.touch_target((1, 1), "minimap")
                runner.frame = np.ones_like(original)
                return "complete"

            runner.tap = Mock(side_effect=lambda *args: tapped.set())
            runner._run = task
            with patch("cv2.imwrite", side_effect=write), ThreadPoolExecutor(1) as executor:
                future = executor.submit(runner.run)
                try:
                    self.assertTrue(entered.wait(5))
                    self.assertTrue(tapped.wait(5))
                    self.assertFalse(future.done())
                finally:
                    release.set()
                self.assertEqual(future.result(timeout=5), "complete")
            self.assertIs(written[0], original)
            self.assertIsNone(runner._writer)

    def test_failed_evidence_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner(Mock(), SimpleNamespace(output=directory))
            runner.frame = np.zeros((3, 4, 3), dtype=np.uint8)
            runner.tap = Mock(return_value=True)
            runner._run = lambda: runner.touch_target((1, 1), "minimap")
            with patch("cv2.imwrite", return_value=False):
                with self.assertRaisesRegex(OSError, "Evidence image write failed"):
                    runner.run()
            self.assertIsNone(runner._writer)
            self.assertEqual(runner._writes, [])

    def test_failed_workflow_still_finishes_pending_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = Runner(Mock(), SimpleNamespace(output=directory))
            runner.frame = np.zeros((3, 4, 3), dtype=np.uint8)
            runner.tap = Mock(return_value=True)
            written = threading.Event()

            def task():
                runner.touch_target((1, 1), "minimap")
                raise RuntimeError("workflow failed")

            runner._run = task
            with patch("cv2.imwrite", side_effect=lambda *args: written.set() or True):
                with self.assertRaisesRegex(RuntimeError, "workflow failed"):
                    runner.run()
            self.assertTrue(written.is_set())


if __name__ == "__main__":
    unittest.main()
