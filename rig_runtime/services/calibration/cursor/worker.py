"""Process-isolated cursor pose execution for the live tracker."""

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

import cv2

from .pose import CursorPoseEstimator


_PROCESS_ESTIMATOR = None


def _initialize_worker(
    calibration_path: str,
    gaussian_fit_method: str,
    validation_policy: str,
    pose_method: str,
    opencv_threads: int,
) -> None:
    global _PROCESS_ESTIMATOR
    # OpenCV otherwise creates a thread team sized to the whole host inside the
    # worker process, starving capture, local motion, UI, and global matching.
    cv2.setNumThreads(max(1, int(opencv_threads)))
    _PROCESS_ESTIMATOR = CursorPoseEstimator(
        calibration_path,
        gaussian_fit_method=gaussian_fit_method,
        validation_policy=validation_policy,
        pose_method=pose_method,
    )


def _estimate(
    frame,
    frame_index,
    session_time_ns,
    angle_prior_deg,
    search_half_width_deg,
):
    if _PROCESS_ESTIMATOR is None:
        raise RuntimeError("Cursor pose worker was not initialized")
    started_host_time_ns = time.perf_counter_ns()
    result = _PROCESS_ESTIMATOR.estimate(
        frame,
        frame_index,
        session_time_ns,
        angle_prior_deg,
        search_half_width_deg,
    )
    completed_host_time_ns = time.perf_counter_ns()
    # Diagnostic arrays are useful offline but expensive to copy back for every
    # live frame. Event evidence is reconstructed from the source frame.
    public = _PROCESS_ESTIMATOR.public_result(result)
    public["estimator_started_host_time_ns"] = started_host_time_ns
    public["estimator_completed_host_time_ns"] = completed_host_time_ns
    public["estimator_elapsed_ms"] = (
        completed_host_time_ns - started_host_time_ns
    ) / 1.0e6
    return public


class CursorPoseProcessExecutor:
    """One latest-only cursor worker on an independent CPU process."""

    returns_public_result = True

    def __init__(
        self,
        calibration_path,
        *,
        gaussian_fit_method: str,
        validation_policy: str,
        pose_method: str = "polygon_gaussian",
        opencv_threads: int = 1,
    ) -> None:
        self._executor = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
            initargs=(
                str(calibration_path),
                str(gaussian_fit_method),
                str(validation_policy),
                str(pose_method),
                max(1, int(opencv_threads)),
            ),
        )

    def submit(
        self,
        frame,
        frame_index,
        session_time_ns,
        angle_prior_deg=None,
        search_half_width_deg=None,
    ):
        return self._executor.submit(
            _estimate,
            frame,
            frame_index,
            session_time_ns,
            angle_prior_deg,
            search_half_width_deg,
        )

    def shutdown(self, wait: bool = False) -> None:
        # Python 3.7 (still used by the Windows workbench build) predates the
        # cancel_futures argument.
        self._executor.shutdown(wait=wait)
