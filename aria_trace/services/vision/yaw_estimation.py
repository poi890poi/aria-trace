"""Small, replaceable yaw-estimation backend for camera navigation."""

from dataclasses import dataclass
from time import perf_counter
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class YawEstimate:
    delta_deg: float
    total_deg: float
    tracks: int
    inliers: int
    confidence: float
    elapsed_ms: float
    status: str


@dataclass
class KltTrackMeasurement:
    points0: Optional[np.ndarray]
    points1: Optional[np.ndarray]
    tracks: int
    status: str
    elapsed_ms: float


def _track_klt(
    previous_gray: np.ndarray,
    gray: np.ndarray,
    mask: np.ndarray,
    max_corners: int,
    min_tracks: int,
):
    points0 = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=7,
        mask=mask,
    )
    if points0 is None or len(points0) < min_tracks:
        return None, None, 0, "too_few_features"

    points1, status1, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        gray,
        points0,
        None,
        winSize=(21, 21),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    points0_back, status0, _ = cv2.calcOpticalFlowPyrLK(
        gray,
        previous_gray,
        points1,
        None,
        winSize=(21, 21),
        maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    p0 = points0.reshape(-1, 2)
    p1 = points1.reshape(-1, 2)
    back = points0_back.reshape(-1, 2)
    valid = status1.ravel().astype(bool) & status0.ravel().astype(bool)
    valid &= np.linalg.norm(p0 - back, axis=1) <= 1.5
    valid &= np.isfinite(p1).all(axis=1)
    p0, p1 = p0[valid], p1[valid]
    tracks = len(p0)
    if tracks < min_tracks:
        return None, None, tracks, "too_few_tracks"
    return p0, p1, tracks, "ok"


class KltAngularYawEstimator:
    """Estimate yaw from robust horizontal angular optical flow.

    The class deliberately exposes only reset/update. A future essential-matrix,
    SLAM, or learned backend can implement the same interface.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        max_corners: int = 800,
        min_tracks: int = 20,
        use_essential_gate: bool = True,
        excluded_rects: Sequence[Tuple[float, float, float, float]] = (),
    ) -> None:
        self.k = np.asarray(camera_matrix, dtype=np.float64)
        self.fx = float(self.k[0, 0])
        self.cx = float(self.k[0, 2])
        self.max_corners = max_corners
        self.min_tracks = min_tracks
        self.use_essential_gate = use_essential_gate
        self.excluded_rects = tuple(excluded_rects)
        self.previous_gray: Optional[np.ndarray] = None
        self.total_deg = 0.0

    def reset(self) -> None:
        self.previous_gray = None
        self.total_deg = 0.0

    def _feature_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        height, width = shape
        mask = np.full((height, width), 255, dtype=np.uint8)
        for x0, y0, x1, y1 in self.excluded_rects:
            p0 = (int(x0 * width), int(y0 * height))
            p1 = (int(x1 * width), int(y1 * height))
            cv2.rectangle(mask, p0, p1, 0, thickness=-1)
        return mask

    def update(self, frame: np.ndarray) -> YawEstimate:
        return self.update_measurement(self.measure(frame))

    def measure(self, frame: np.ndarray) -> KltTrackMeasurement:
        """Track one frame without applying camera intrinsics."""
        started = perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.previous_gray is None:
            self.previous_gray = gray
            return KltTrackMeasurement(None, None, 0, "initializing", 0.0)

        p0, p1, tracks, status = _track_klt(
            self.previous_gray,
            gray,
            self._feature_mask(gray.shape),
            self.max_corners,
            self.min_tracks,
        )
        self.previous_gray = gray
        return KltTrackMeasurement(
            p0,
            p1,
            tracks,
            status,
            (perf_counter() - started) * 1000.0,
        )

    def update_measurement(
        self, measurement: KltTrackMeasurement
    ) -> YawEstimate:
        """Apply camera intrinsics to a reusable optical-flow measurement."""
        started = perf_counter()
        tracks = measurement.tracks
        status = measurement.status
        if status == "initializing":
            return YawEstimate(0.0, self.total_deg, 0, 0, 0.0, 0.0, status)
        if status != "ok":
            return self._failure_elapsed(measurement.elapsed_ms, tracks, status)

        p0, p1 = measurement.points0, measurement.points1

        if self.use_essential_gate and tracks >= 8:
            _, essential_mask = cv2.findEssentialMat(
                p0,
                p1,
                self.k,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.5,
            )
            if essential_mask is not None:
                geometric = essential_mask.ravel().astype(bool)
                if geometric.sum() >= self.min_tracks:
                    p0, p1 = p0[geometric], p1[geometric]

        bearing0 = np.arctan2(p0[:, 0] - self.cx, self.fx)
        bearing1 = np.arctan2(p1[:, 0] - self.cx, self.fx)
        deltas = np.unwrap(np.stack((bearing0, bearing1)), axis=0)[1] - bearing0

        median = float(np.median(deltas))
        deviations = np.abs(deltas - median)
        mad = float(np.median(deviations))
        cutoff = max(3.0 * 1.4826 * mad, np.deg2rad(0.08))
        angular_inliers = deviations <= cutoff
        inliers = int(angular_inliers.sum())
        if inliers < self.min_tracks:
            return self._failure_elapsed(
                measurement.elapsed_ms + (perf_counter() - started) * 1000.0,
                tracks,
                "too_few_inliers",
            )

        delta_deg = float(np.degrees(np.median(deltas[angular_inliers])))
        self.total_deg += delta_deg
        scatter_deg = float(np.degrees(mad))
        confidence = min(1.0, inliers / 120.0) * max(0.0, 1.0 - scatter_deg / 1.5)
        return YawEstimate(
            delta_deg,
            self.total_deg,
            tracks,
            inliers,
            confidence,
            measurement.elapsed_ms + (perf_counter() - started) * 1000.0,
            "ok",
        )

    def _failure(self, started: float, tracks: int, status: str) -> YawEstimate:
        return self._failure_elapsed(
            (perf_counter() - started) * 1000.0, tracks, status
        )

    def _failure_elapsed(
        self, elapsed_ms: float, tracks: int, status: str
    ) -> YawEstimate:
        return YawEstimate(
            0.0,
            self.total_deg,
            tracks,
            0,
            0.0,
            elapsed_ms,
            status,
        )


class KltEssentialYawEstimator(KltAngularYawEstimator):
    """Estimate frame rotation from the essential matrix and recoverPose."""

    def __init__(
        self,
        camera_matrix: np.ndarray,
        max_corners: int = 800,
        min_tracks: int = 20,
        excluded_rects: Sequence[Tuple[float, float, float, float]] = (),
    ) -> None:
        super().__init__(
            camera_matrix,
            max_corners=max_corners,
            min_tracks=min_tracks,
            use_essential_gate=False,
            excluded_rects=excluded_rects,
        )

    def update(self, frame: np.ndarray) -> YawEstimate:
        started = perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.previous_gray is None:
            self.previous_gray = gray
            return YawEstimate(0.0, 0.0, 0, 0, 0.0, 0.0, "initializing")

        p0, p1, tracks, status = _track_klt(
            self.previous_gray,
            gray,
            self._feature_mask(gray.shape),
            self.max_corners,
            self.min_tracks,
        )
        self.previous_gray = gray
        if status != "ok":
            return self._failure(started, tracks, status)

        essential, mask = cv2.findEssentialMat(
            p0,
            p1,
            self.k,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.5,
        )
        if essential is None or mask is None:
            return self._failure(started, tracks, "essential_failed")
        try:
            inliers, rotation, _, _ = cv2.recoverPose(
                essential, p0, p1, self.k, mask=mask
            )
        except cv2.error:
            return self._failure(started, tracks, "pose_failed")
        inliers = int(inliers)
        if inliers < self.min_tracks:
            return self._failure(started, tracks, "too_few_pose_inliers")

        delta_deg = float(np.degrees(np.arctan2(rotation[0, 2], rotation[2, 2])))
        if not np.isfinite(delta_deg) or abs(delta_deg) > 5.0:
            return self._failure(started, tracks, "implausible_rotation")
        self.total_deg += delta_deg
        confidence = min(1.0, inliers / 120.0)
        return YawEstimate(
            delta_deg,
            self.total_deg,
            tracks,
            inliers,
            confidence,
            (perf_counter() - started) * 1000.0,
            "ok",
        )


def camera_matrix(width: int, height: int, focal_ratio: float = 0.9) -> np.ndarray:
    focal = focal_ratio * width
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def yaw_rotation(yaw_deg: float) -> np.ndarray:
    angle = np.deg2rad(yaw_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


__all__ = [
    "KltAngularYawEstimator",
    "KltEssentialYawEstimator",
    "KltTrackMeasurement",
    "YawEstimate",
    "camera_matrix",
    "yaw_rotation",
]
