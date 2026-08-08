"""Small, replaceable pose predictor and absolute-correction gate.

The core deliberately models only planar ground navigation.  It does not know
about COLMAP, TartanAir, minimaps, or a particular game.  Those systems provide
measurements through this boundary.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple


def wrap_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def angle_difference_deg(first: float, second: float) -> float:
    """Return the signed shortest difference first - second."""
    return wrap_angle_deg(first - second)


@dataclass
class Pose2D:
    x: float
    y: float
    yaw_deg: float


@dataclass
class FusionState:
    pose: Pose2D
    position_sigma_m: float
    yaw_sigma_deg: float
    mode: str


@dataclass
class FusionConfig:
    initial_position_sigma_m: float = 0.05
    initial_yaw_sigma_deg: float = 1.0
    position_sigma_per_step_m: float = 0.008
    position_sigma_per_meter: float = 0.04
    yaw_sigma_per_step_deg: float = 0.10
    yaw_sigma_per_turn: float = 0.015
    prediction_position_floor_m: float = 0.30
    prediction_position_sigma_factor: float = 3.0
    prediction_position_limit_m: float = 1.25
    coarse_position_limit_m: float = 1.20
    prediction_yaw_floor_deg: float = 10.0
    prediction_yaw_sigma_factor: float = 3.0
    prediction_yaw_limit_deg: float = 40.0
    coarse_yaw_limit_deg: float = 55.0
    position_correction_weight: float = 0.80
    yaw_correction_weight: float = 0.75
    corrected_position_sigma_m: float = 0.08
    corrected_yaw_sigma_deg: float = 2.0
    cautious_position_sigma_m: float = 0.35
    cautious_yaw_sigma_deg: float = 8.0
    relocalize_position_sigma_m: float = 0.75
    relocalize_yaw_sigma_deg: float = 20.0
    stop_position_sigma_m: float = 1.50
    stop_yaw_sigma_deg: float = 45.0


@dataclass
class CorrectionDecision:
    accepted: bool
    reason: str
    predicted_position_innovation_m: float
    predicted_yaw_innovation_deg: float
    coarse_position_innovation_m: Optional[float]
    coarse_yaw_innovation_deg: Optional[float]
    applied_position_change_m: float = 0.0
    applied_yaw_change_deg: float = 0.0


class PoseFusionGate:
    """Predict from local motion and gate occasional absolute pose hypotheses."""

    def __init__(self, config: Optional[FusionConfig] = None) -> None:
        self.config = config or FusionConfig()
        self._state: Optional[FusionState] = None

    @property
    def state(self) -> FusionState:
        if self._state is None:
            raise RuntimeError("PoseFusionGate is not initialized")
        return self._state

    def initialize(self, pose: Pose2D) -> FusionState:
        self._state = FusionState(
            Pose2D(pose.x, pose.y, wrap_angle_deg(pose.yaw_deg)),
            self.config.initial_position_sigma_m,
            self.config.initial_yaw_sigma_deg,
            "TRACK",
        )
        self._refresh_mode()
        return self.state

    def predict(self, local_motion: Tuple[float, float], delta_yaw_deg: float) -> FusionState:
        state = self.state
        local_x, local_y = local_motion
        distance = math.hypot(local_x, local_y)
        # Midpoint integration is adequate for short control intervals and keeps
        # the core independent of a particular game's acceleration model.
        heading = math.radians(state.pose.yaw_deg + 0.5 * delta_yaw_deg)
        world_x = math.cos(heading) * local_x - math.sin(heading) * local_y
        world_y = math.sin(heading) * local_x + math.cos(heading) * local_y
        state.pose = Pose2D(
            state.pose.x + world_x,
            state.pose.y + world_y,
            wrap_angle_deg(state.pose.yaw_deg + delta_yaw_deg),
        )
        state.position_sigma_m += (
            self.config.position_sigma_per_step_m
            + self.config.position_sigma_per_meter * distance
        )
        state.yaw_sigma_deg += (
            self.config.yaw_sigma_per_step_deg
            + self.config.yaw_sigma_per_turn * abs(delta_yaw_deg)
        )
        self._refresh_mode()
        return state

    def consider_absolute(
        self, hypothesis: Pose2D, coarse_prior: Optional[Pose2D] = None
    ) -> CorrectionDecision:
        state = self.state
        predicted_position = math.hypot(
            hypothesis.x - state.pose.x, hypothesis.y - state.pose.y
        )
        predicted_yaw = abs(angle_difference_deg(hypothesis.yaw_deg, state.pose.yaw_deg))
        coarse_position = None
        coarse_yaw = None
        if coarse_prior is not None:
            coarse_position = math.hypot(
                hypothesis.x - coarse_prior.x, hypothesis.y - coarse_prior.y
            )
            coarse_yaw = abs(angle_difference_deg(hypothesis.yaw_deg, coarse_prior.yaw_deg))

        position_limit = min(
            self.config.prediction_position_limit_m,
            max(
                self.config.prediction_position_floor_m,
                self.config.prediction_position_sigma_factor * state.position_sigma_m,
            ),
        )
        yaw_limit = min(
            self.config.prediction_yaw_limit_deg,
            max(
                self.config.prediction_yaw_floor_deg,
                self.config.prediction_yaw_sigma_factor * state.yaw_sigma_deg,
            ),
        )

        reasons = []
        if predicted_position > position_limit:
            reasons.append("prediction-position")
        if predicted_yaw > yaw_limit:
            reasons.append("prediction-heading")
        if coarse_position is not None and coarse_position > self.config.coarse_position_limit_m:
            reasons.append("coarse-position")
        if coarse_yaw is not None and coarse_yaw > self.config.coarse_yaw_limit_deg:
            reasons.append("coarse-heading")
        if reasons:
            return CorrectionDecision(
                False,
                "+".join(reasons),
                predicted_position,
                predicted_yaw,
                coarse_position,
                coarse_yaw,
            )

        old_pose = state.pose
        position_weight = self.config.position_correction_weight
        yaw_change = angle_difference_deg(hypothesis.yaw_deg, old_pose.yaw_deg)
        state.pose = Pose2D(
            old_pose.x + position_weight * (hypothesis.x - old_pose.x),
            old_pose.y + position_weight * (hypothesis.y - old_pose.y),
            wrap_angle_deg(old_pose.yaw_deg + self.config.yaw_correction_weight * yaw_change),
        )
        state.position_sigma_m = min(
            state.position_sigma_m, self.config.corrected_position_sigma_m
        )
        state.yaw_sigma_deg = min(state.yaw_sigma_deg, self.config.corrected_yaw_sigma_deg)
        self._refresh_mode()
        return CorrectionDecision(
            True,
            "consistent",
            predicted_position,
            predicted_yaw,
            coarse_position,
            coarse_yaw,
            math.hypot(state.pose.x - old_pose.x, state.pose.y - old_pose.y),
            abs(angle_difference_deg(state.pose.yaw_deg, old_pose.yaw_deg)),
        )

    def _refresh_mode(self) -> None:
        state = self.state
        if (
            state.position_sigma_m >= self.config.stop_position_sigma_m
            or state.yaw_sigma_deg >= self.config.stop_yaw_sigma_deg
        ):
            state.mode = "STOP"
        elif (
            state.position_sigma_m >= self.config.relocalize_position_sigma_m
            or state.yaw_sigma_deg >= self.config.relocalize_yaw_sigma_deg
        ):
            state.mode = "RELOCALIZE"
        elif (
            state.position_sigma_m >= self.config.cautious_position_sigma_m
            or state.yaw_sigma_deg >= self.config.cautious_yaw_sigma_deg
        ):
            state.mode = "CAUTIOUS"
        else:
            state.mode = "TRACK"
