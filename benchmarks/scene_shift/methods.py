"""Replaceable scene-yaw candidates for causal offline comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class SceneYawResult:
    delta_deg: float
    confidence: float
    tracks: int
    inliers: int
    status: str
    elapsed_ms: float

    def as_dict(self) -> dict:
        return asdict(self)


def _resize_gray(frame: np.ndarray, maximum_width: int):
    height, width = frame.shape[:2]
    scale = min(1.0, float(maximum_width) / float(width))
    if scale < 1.0:
        frame = cv2.resize(
            frame,
            (int(round(width * scale)), int(round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scale


def _mask(shape, excluded_rects):
    height, width = shape
    result = np.full((height, width), 255, np.uint8)
    for x0, y0, x1, y1 in excluded_rects:
        cv2.rectangle(
            result,
            (int(x0 * width), int(y0 * height)),
            (int(x1 * width), int(y1 * height)),
            0,
            thickness=-1,
        )
    return result


class KltSceneYaw:
    """Sparse angular-flow estimator with independently switchable costs."""

    family = "sparse_klt"

    def __init__(
        self,
        *,
        maximum_width: int,
        max_corners: int,
        forward_backward: bool = True,
        essential_gate: bool = True,
        min_tracks: int = 20,
        focal_ratio: float = 0.9,
        excluded_rects: Sequence[Tuple[float, float, float, float]] = (),
    ) -> None:
        self.maximum_width = int(maximum_width)
        self.max_corners = int(max_corners)
        self.forward_backward = bool(forward_backward)
        self.essential_gate = bool(essential_gate)
        self.min_tracks = int(min_tracks)
        self.focal_ratio = float(focal_ratio)
        self.excluded_rects = tuple(excluded_rects)
        self.previous_gray = None

    def parameters(self) -> dict:
        return {
            "family": self.family,
            "maximum_width": self.maximum_width,
            "max_corners": self.max_corners,
            "forward_backward": self.forward_backward,
            "essential_gate": self.essential_gate,
            "min_tracks": self.min_tracks,
            "focal_ratio": self.focal_ratio,
        }

    def update(self, frame: np.ndarray) -> SceneYawResult:
        started = perf_counter()
        gray, _ = _resize_gray(frame, self.maximum_width)
        if self.previous_gray is None:
            self.previous_gray = gray
            return SceneYawResult(0.0, 0.0, 0, 0, "initializing", 0.0)
        previous = self.previous_gray
        self.previous_gray = gray
        feature_mask = _mask(gray.shape, self.excluded_rects)
        p0 = cv2.goodFeaturesToTrack(
            previous,
            maxCorners=self.max_corners,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7,
            mask=feature_mask,
        )
        if p0 is None or len(p0) < self.min_tracks:
            return self._failure(started, 0, "too_few_features")
        p1, status1, _ = cv2.calcOpticalFlowPyrLK(
            previous,
            gray,
            p0,
            None,
            winSize=(21, 21),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if p1 is None or status1 is None:
            return self._failure(started, 0, "forward_flow_failed")
        original = p0.reshape(-1, 2)
        current = p1.reshape(-1, 2)
        valid = status1.ravel().astype(bool) & np.isfinite(current).all(axis=1)
        if self.forward_backward:
            back, status0, _ = cv2.calcOpticalFlowPyrLK(
                gray,
                previous,
                p1,
                None,
                winSize=(21, 21),
                maxLevel=4,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    30,
                    0.01,
                ),
            )
            if back is None or status0 is None:
                return self._failure(started, 0, "backward_flow_failed")
            backward = back.reshape(-1, 2)
            valid &= status0.ravel().astype(bool)
            valid &= np.linalg.norm(original - backward, axis=1) <= 1.5
        p0_valid = original[valid]
        p1_valid = current[valid]
        tracks = len(p0_valid)
        if tracks < self.min_tracks:
            return self._failure(started, tracks, "too_few_tracks")
        width = gray.shape[1]
        focal = self.focal_ratio * width
        intrinsic = np.array(
            [[focal, 0.0, width / 2.0], [0.0, focal, gray.shape[0] / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        if self.essential_gate and tracks >= 8:
            _, geometric_mask = cv2.findEssentialMat(
                p0_valid,
                p1_valid,
                intrinsic,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.5,
            )
            if geometric_mask is not None:
                geometric = geometric_mask.ravel().astype(bool)
                if int(geometric.sum()) >= self.min_tracks:
                    p0_valid = p0_valid[geometric]
                    p1_valid = p1_valid[geometric]
        bearing0 = np.arctan2(p0_valid[:, 0] - width / 2.0, focal)
        bearing1 = np.arctan2(p1_valid[:, 0] - width / 2.0, focal)
        deltas = np.unwrap(np.stack((bearing0, bearing1)), axis=0)[1] - bearing0
        median = float(np.median(deltas))
        deviations = np.abs(deltas - median)
        mad = float(np.median(deviations))
        cutoff = max(3.0 * 1.4826 * mad, np.deg2rad(0.08))
        angular = deviations <= cutoff
        inliers = int(angular.sum())
        if inliers < self.min_tracks:
            return self._failure(started, tracks, "too_few_inliers")
        delta_deg = float(np.degrees(np.median(deltas[angular])))
        scatter_deg = float(np.degrees(mad))
        confidence = min(1.0, inliers / 120.0) * max(0.0, 1.0 - scatter_deg / 1.5)
        return SceneYawResult(
            delta_deg,
            confidence,
            tracks,
            inliers,
            "ok",
            (perf_counter() - started) * 1000.0,
        )

    @staticmethod
    def _failure(started, tracks, status):
        return SceneYawResult(
            0.0,
            0.0,
            int(tracks),
            0,
            status,
            (perf_counter() - started) * 1000.0,
        )


class PhaseSceneYaw:
    """Dense translation proxy using OpenCV's FFT phase correlation."""

    family = "phase_correlation"

    def __init__(
        self,
        *,
        maximum_width: int,
        signal: str = "gray",
        focal_ratio: float = 0.9,
        excluded_rects: Sequence[Tuple[float, float, float, float]] = (),
    ) -> None:
        if signal not in ("gray", "gradient"):
            raise ValueError("signal must be gray or gradient")
        self.maximum_width = int(maximum_width)
        self.signal = signal
        self.focal_ratio = float(focal_ratio)
        self.excluded_rects = tuple(excluded_rects)
        self.previous = None
        self.window = None

    def parameters(self) -> dict:
        return {
            "family": self.family,
            "maximum_width": self.maximum_width,
            "signal": self.signal,
            "focal_ratio": self.focal_ratio,
        }

    def _signal(self, gray):
        value = gray.astype(np.float32)
        if self.signal == "gradient":
            gx = cv2.Sobel(value, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(value, cv2.CV_32F, 0, 1, ksize=3)
            value = cv2.magnitude(gx, gy)
        value *= (_mask(gray.shape, self.excluded_rects) > 0).astype(np.float32)
        return value

    def update(self, frame: np.ndarray) -> SceneYawResult:
        started = perf_counter()
        gray, _ = _resize_gray(frame, self.maximum_width)
        signal = self._signal(gray)
        if self.previous is None:
            self.previous = signal
            self.window = cv2.createHanningWindow(
                (gray.shape[1], gray.shape[0]), cv2.CV_32F
            )
            return SceneYawResult(0.0, 0.0, 0, 0, "initializing", 0.0)
        shift, response = cv2.phaseCorrelate(self.previous, signal, self.window)
        self.previous = signal
        if not np.isfinite(shift).all() or not np.isfinite(response):
            return SceneYawResult(
                0.0,
                0.0,
                0,
                0,
                "nonfinite",
                (perf_counter() - started) * 1000.0,
            )
        delta_deg = float(np.degrees(np.arctan2(shift[0], self.focal_ratio * gray.shape[1])))
        return SceneYawResult(
            delta_deg,
            float(np.clip(response, 0.0, 1.0)),
            0,
            0,
            "ok",
            (perf_counter() - started) * 1000.0,
        )

