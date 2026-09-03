"""Small immutable values shared by rig-calibration algorithms."""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


def matrix_3x3(value: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Expected a finite 3x3 matrix")
    if abs(float(np.linalg.det(matrix))) < 1.0e-12:
        raise ValueError("Transform matrix is singular")
    return matrix


def points_xy(value: Sequence[Sequence[float]], minimum: int = 1) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum:
        raise ValueError("Expected at least {} XY points".format(minimum))
    if not np.all(np.isfinite(points)):
        raise ValueError("Points must be finite")
    return points


@dataclass(frozen=True)
class FrameSample:
    """Minimal timestamped image contract supplied by an external adapter."""

    image: np.ndarray
    time_ns: int
    clock_id: str = "host_monotonic_ns"
    receive_time_ns: Optional[int] = None
    source_id: str = "camera"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.image is None or self.image.size == 0:
            raise ValueError("Frame sample image must be non-empty")
        if self.time_ns < 0:
            raise ValueError("Frame sample time must be non-negative")
        if self.receive_time_ns is not None and self.receive_time_ns < 0:
            raise ValueError("Frame receive time must be non-negative")


@dataclass(frozen=True)
class ControlEvent:
    token: str
    state: str
    time_ns: int
    clock_id: str = "host_monotonic_ns"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.token:
            raise ValueError("Control event token is required")
        if not self.state:
            raise ValueError("Control event state is required")
        if self.time_ns < 0:
            raise ValueError("Control event time must be non-negative")


@dataclass(frozen=True)
class SignalObservation:
    time_ns: int
    probabilities: Mapping[str, float]
    clock_id: str = "host_monotonic_ns"
    source_id: str = "camera"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.time_ns < 0:
            raise ValueError("Observation time must be non-negative")
        if not self.probabilities:
            raise ValueError("At least one state probability is required")
        for probability in self.probabilities.values():
            if not np.isfinite(probability) or probability < 0.0 or probability > 1.0:
                raise ValueError("State probabilities must be within [0, 1]")


# Retained only so previously serialized/internal legacy matchability callers
# fail neither at import nor during migration. New calibration code does not
# export or use these project-defined trial/result contracts.
@dataclass(frozen=True)
class MatchResult:
    translation_xy: Tuple[float, float]
    rotation_deg: float = 0.0
    confidence: float = 0.0
    ambiguous: bool = False
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MatchTrial:
    detail_cells_across: int
    reference: np.ndarray
    observed: np.ndarray
    expected_translation_xy: Tuple[float, float] = (0.0, 0.0)
    expected_rotation_deg: float = 0.0
    reference_mode: str = "camera_to_camera"
    pattern_family: str = "luminance"
    moving: bool = False
    trial_id: str = ""

    def __post_init__(self) -> None:
        if self.detail_cells_across <= 0:
            raise ValueError("detail_cells_across must be positive")
        if self.reference.size == 0 or self.observed.size == 0:
            raise ValueError("Match trial images must be non-empty")


@dataclass(frozen=True)
class GeometryEstimate:
    matrix_3x3: np.ndarray
    inverse_matrix_3x3: np.ndarray
    inlier_mask: np.ndarray
    reprojection_errors_px: np.ndarray
    screen_polygon_input_xy: np.ndarray
    viewport_polygon_screen_xy: np.ndarray
    metrics: Dict[str, float]
    confidence: float
    warnings: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        inlier_errors = self.reprojection_errors_px[self.inlier_mask]
        return {
            "matrix_3x3": self.matrix_3x3.tolist(),
            "inverse_matrix_3x3": self.inverse_matrix_3x3.tolist(),
            "inlier_count": int(np.count_nonzero(self.inlier_mask)),
            "correspondence_count": int(len(self.inlier_mask)),
            "reprojection_rmse_px": float(
                np.sqrt(np.mean(inlier_errors ** 2))
            ),
            "reprojection_p95_px": float(
                np.percentile(inlier_errors, 95)
            ),
            "screen_polygon_input_xy": self.screen_polygon_input_xy.tolist(),
            "viewport_polygon_screen_xy": self.viewport_polygon_screen_xy.tolist(),
            "metrics": dict(self.metrics),
            "confidence": float(self.confidence),
            "warnings": list(self.warnings),
        }
