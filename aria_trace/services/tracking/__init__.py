"""Tracking-state prediction, gating, and fusion services."""

from .pose_fusion import (
    CorrectionDecision,
    FusionConfig,
    FusionState,
    Pose2D,
    PoseFusionGate,
    angle_difference_deg,
    wrap_angle_deg,
)

__all__ = [
    "CorrectionDecision",
    "FusionConfig",
    "FusionState",
    "Pose2D",
    "PoseFusionGate",
    "angle_difference_deg",
    "wrap_angle_deg",
]
