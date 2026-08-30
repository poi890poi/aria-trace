import unittest
from unittest.mock import MagicMock, patch

from acquisition.cursor_worker import CursorPoseProcessExecutor


class CursorWorkerTests(unittest.TestCase):
    def test_process_executor_uses_spawn_and_one_worker(self):
        pool = MagicMock()
        with patch(
            "aria_trace.services.calibration.cursor.worker.ProcessPoolExecutor",
            return_value=pool,
        ) as constructor:
            worker = CursorPoseProcessExecutor(
                "calibration",
                gaussian_fit_method="cascade",
                validation_policy="minimal",
                opencv_threads=1,
            )
            worker.shutdown()

        arguments = constructor.call_args[1]
        self.assertEqual(arguments["max_workers"], 1)
        self.assertEqual(arguments["mp_context"].get_start_method(), "spawn")
        self.assertEqual(arguments["initargs"][-1], 1)
        pool.shutdown.assert_called_once_with(wait=False)


if __name__ == "__main__":
    unittest.main()
