import unittest
import time
from unittest.mock import MagicMock, patch

from acquisition.cursor_worker import CursorPoseProcessExecutor
from rig_runtime.services.calibration.cursor import worker as worker_module


class CursorWorkerTests(unittest.TestCase):
    def test_worker_result_exposes_estimator_timing(self):
        estimator = MagicMock()
        estimator.estimate.return_value = {"detected": True}
        estimator.public_result.return_value = {"detected": True}
        previous = worker_module._PROCESS_ESTIMATOR
        worker_module._PROCESS_ESTIMATOR = estimator
        try:
            result = worker_module._estimate(None, 1, time.perf_counter_ns(), None, None)
        finally:
            worker_module._PROCESS_ESTIMATOR = previous

        self.assertGreaterEqual(result["estimator_elapsed_ms"], 0.0)
        self.assertGreaterEqual(
            result["estimator_completed_host_time_ns"],
            result["estimator_started_host_time_ns"],
        )

    def test_process_executor_uses_spawn_and_one_worker(self):
        pool = MagicMock()
        with patch(
            "rig_runtime.services.calibration.cursor.worker.ProcessPoolExecutor",
            return_value=pool,
        ) as constructor:
            worker = CursorPoseProcessExecutor(
                "calibration",
                gaussian_fit_method="cascade",
                validation_policy="minimal",
                pose_method="angular_projection_ncc_parabolic",
                opencv_threads=1,
            )
            worker.shutdown()

        arguments = constructor.call_args[1]
        self.assertEqual(arguments["max_workers"], 1)
        self.assertEqual(arguments["mp_context"].get_start_method(), "spawn")
        self.assertEqual(
            arguments["initargs"][-2], "angular_projection_ncc_parabolic"
        )
        self.assertEqual(arguments["initargs"][-1], 1)
        pool.shutdown.assert_called_once_with(wait=False)


if __name__ == "__main__":
    unittest.main()
