"""Single-source IRIS configuration values and immutable resolved plans.

The objects in this module deliberately separate four kinds of value:

* defaults are product policy owned by one domain;
* requests are caller intent;
* profile facts are immutable calibration observations;
* resolved plans are the only values consumed by runtime components.

Serialized dictionaries remain useful audit snapshots, but they are never the
authority from which another component independently chooses defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple


ADAPTER_MODES = ("full", "minimap", "dual")
NORMALIZATION_MODES = ("auto", "dense_remap", "homography", "none")
COLOR_ORDERS = ("RGB", "BGR")
COLOR_POLICIES = ("auto", "rig_locked", "game_matched", "unadjusted")
ROI_POLICIES = ("auto", "full_phone", "minimap_only")
MASK_POLICIES = ("none", "minimap_circle")
ORIENTATION_BEHAVIORS = ("as_is", "projection", "image")
FRAME_RATE_POLICIES = ("calibrated", "exact")
QUARTER_TURN_DEGREES = (0, 90, 180, 270)
TARGET_PRESENTERS = ("native_app", "owned_http", "legacy_gallery")
PANEL_SCALE_MODES = ("auto", "adb", "hik_charuco")
SHUTTER_MULTIPLIERS = (2, 3)
EXPOSURE_PERIOD_COUNTS = (1, 2, 3)
DISTORTION_CORRECTION_MODES = ("off", "guided")
FINAL_BENCHMARK_MODES = ("auto", "full", "reduced", "skip")
ANDROID_CAPTURE_MODES = ("scrcpy", "adb-screenshot")
ACQUISITION_MODES = ("zigzag", "micro-movement", "cursor-orbit")


@dataclass(frozen=True)
class AdapterDefaults:
    """Built-in adapter request policy; the only owner of these defaults."""

    purpose: str = "application"
    mode: str = "full"
    normalization: str = "auto"
    color_order: str = "RGB"
    color_policy: str = "auto"
    roi_policy: str = "auto"
    mask_policy: str = "none"
    minimap_margin_px: int = 6
    orientation_behavior: str = "projection"
    rotate_degrees_clockwise: int = 0
    frame_rate_policy: str = "calibrated"
    frame_rate: Optional[float] = None
    platform: str = "android"
    rectify: bool = True
    best_effort_initialization: bool = False
    persist_initialization_recovery: bool = False


ADAPTER_DEFAULTS = AdapterDefaults()


@dataclass(frozen=True)
class ResolvedAdapterPlan:
    """Final adapter behavior resolved once before the camera is opened."""

    mode: str
    normalization: str
    rectify: bool
    color_order: str
    color_policy: str
    roi_policy: str
    mask_policy: str
    minimap_margin_px: int
    orientation_behavior: str
    profile_game_upright_quarter_turns_clockwise: int
    manual_rotate_degrees_clockwise: int
    game_upright_quarter_turns_clockwise: int
    initialization_surface_quarter_turns_clockwise_from_natural: Optional[int]
    game_model: Mapping[str, Any]
    registry_reads_per_frame: int = 0
    phone_operations: str = "none"

    @classmethod
    def create(
        cls,
        *,
        mode: str,
        normalization: str,
        color_order: str,
        color_policy: str,
        roi_policy: str,
        mask_policy: str,
        minimap_margin_px: int,
        orientation_behavior: str,
        profile_game_upright_quarter_turns_clockwise: int,
        manual_rotate_degrees_clockwise: int,
        initialization_surface_quarter_turns_clockwise_from_natural: Optional[int],
        game_model: Mapping[str, Any],
        phone_operations: str = "none",
    ) -> "ResolvedAdapterPlan":
        behavior = str(orientation_behavior).lower().replace("-", "_")
        if mode not in ADAPTER_MODES:
            raise ValueError("Adapter mode must be full, minimap, or dual")
        if normalization not in NORMALIZATION_MODES:
            raise ValueError("Unsupported normalization mode")
        if color_order not in COLOR_ORDERS:
            raise ValueError("Color order must be RGB or BGR")
        if color_policy not in COLOR_POLICIES:
            raise ValueError("Unsupported color policy")
        if roi_policy not in ROI_POLICIES:
            raise ValueError("Unsupported ROI policy")
        if mask_policy not in MASK_POLICIES:
            raise ValueError("Unsupported mask policy")
        if behavior not in ORIENTATION_BEHAVIORS:
            raise ValueError("Unsupported orientation behavior")
        rotate = int(manual_rotate_degrees_clockwise)
        if rotate not in QUARTER_TURN_DEGREES:
            raise ValueError("Adapter rotation must be 0, 90, 180, or 270")
        profile_turns = int(profile_game_upright_quarter_turns_clockwise) % 4
        manual_turns = rotate // 90
        effective_turns = manual_turns
        if behavior == "projection":
            effective_turns = (profile_turns + manual_turns) % 4
        surface_turns = (
            int(initialization_surface_quarter_turns_clockwise_from_natural) % 4
            if initialization_surface_quarter_turns_clockwise_from_natural is not None
            else None
        )
        return cls(
            mode=mode,
            normalization=normalization,
            rectify=normalization != "none",
            color_order=color_order,
            color_policy=color_policy,
            roi_policy=roi_policy,
            mask_policy=mask_policy,
            minimap_margin_px=int(minimap_margin_px),
            orientation_behavior=behavior,
            profile_game_upright_quarter_turns_clockwise=profile_turns,
            manual_rotate_degrees_clockwise=rotate,
            game_upright_quarter_turns_clockwise=effective_turns,
            initialization_surface_quarter_turns_clockwise_from_natural=surface_turns,
            game_model=dict(game_model),
            phone_operations=str(phone_operations),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "normalization": self.normalization,
            "rectify": bool(self.rectify),
            "color_order": self.color_order,
            "color_policy": self.color_policy,
            "roi_policy": self.roi_policy,
            "mask_policy": self.mask_policy,
            "minimap_margin_px": int(self.minimap_margin_px),
            "orientation_behavior": self.orientation_behavior,
            "profile_game_upright_quarter_turns_clockwise": int(
                self.profile_game_upright_quarter_turns_clockwise
            ),
            "manual_rotate_degrees_clockwise": int(
                self.manual_rotate_degrees_clockwise
            ),
            "game_upright_quarter_turns_clockwise": int(
                self.game_upright_quarter_turns_clockwise
            ),
            "initialization_surface_quarter_turns_clockwise_from_natural": (
                self.initialization_surface_quarter_turns_clockwise_from_natural
            ),
            "game_model": dict(self.game_model),
            "registry_reads_per_frame": int(self.registry_reads_per_frame),
            "phone_operations": self.phone_operations,
        }


@dataclass(frozen=True)
class RigCalibrationDefaults:
    """Built-in HIK rig-calibration request policy."""

    camera_width_px: int = 2448
    camera_height_px: int = 2048
    camera_fps: float = 30.0
    target_port: int = 0
    target_presenter: str = "native_app"
    panel_scale_mode: str = "auto"
    operation_timeout_seconds: float = 8.0
    maximum_shutter_multiplier: int = 2
    maximum_exposure_periods: int = 1
    maximum_auto_gain_db: float = 12.0
    exposure_noise_frames: int = 4
    geometry_frames: int = 12
    visible_screen_margin_px: int = 8
    settle_frames: int = 3
    distortion_correction: str = "off"
    distortion_view_count: int = 8
    distortion_min_relative_p95_improvement: float = 0.05
    final_benchmark_mode: str = "auto"


RIG_CALIBRATION_DEFAULTS = RigCalibrationDefaults()


@dataclass(frozen=True)
class ZigzagPlanDefaults:
    """Direct long-swipe plan defaults after display geometry is resolved."""

    move_count: int = 12
    step_seconds: float = 0.12
    endpoint_hold_seconds: float = 0.10
    settle_seconds: float = 1.0
    reset_seconds: float = 0.10
    move_sample_hz: float = 22.0


ZIGZAG_PLAN_DEFAULTS = ZigzagPlanDefaults()


@dataclass(frozen=True)
class CursorOrbitPlanDefaults:
    """Direct micro-movement plan defaults after display geometry is resolved."""

    direction_count: int = 12
    repeats: int = 2
    step_seconds: float = 0.12
    settle_seconds: float = 1.0
    reset_seconds: float = 0.18
    move_sample_hz: float = 30.0


CURSOR_ORBIT_PLAN_DEFAULTS = CursorOrbitPlanDefaults()


@dataclass(frozen=True)
class AcquisitionDefaults:
    """Built-in game-acquisition policy in logical display coordinates."""

    camera_width_px: int = 2448
    camera_height_px: int = 2048
    camera_fps: float = 30.0
    android_capture: str = "scrcpy"
    screenshot_settle_seconds: float = 0.35
    capture_mode: str = "zigzag"
    sample_count: int = ZIGZAG_PLAN_DEFAULTS.move_count
    horizontal_swipe_fraction: float = 0.10
    vertical_swipe_fraction: float = 0.20
    look_anchor_x_fraction: float = 0.72
    look_anchor_y_fraction: float = 0.50
    swipe_travel_seconds: float = ZIGZAG_PLAN_DEFAULTS.step_seconds
    endpoint_hold_seconds: float = ZIGZAG_PLAN_DEFAULTS.endpoint_hold_seconds
    reset_seconds: float = ZIGZAG_PLAN_DEFAULTS.reset_seconds
    settle_seconds: float = 1.5
    tail_seconds: float = 1.5
    zigzag_move_sample_hz: float = ZIGZAG_PLAN_DEFAULTS.move_sample_hz
    micro_movement_radius_fraction: float = 0.06
    micro_movement_pulse_seconds: float = CURSOR_ORBIT_PLAN_DEFAULTS.step_seconds
    micro_movement_directions: int = CURSOR_ORBIT_PLAN_DEFAULTS.direction_count
    micro_movement_repeats: int = CURSOR_ORBIT_PLAN_DEFAULTS.repeats
    joystick_center_x_fraction: float = 0.18
    joystick_center_y_fraction: float = 0.78
    micro_movement_sample_hz: float = CURSOR_ORBIT_PLAN_DEFAULTS.move_sample_hz


ACQUISITION_DEFAULTS = AcquisitionDefaults()


__all__ = [
    "ACQUISITION_DEFAULTS",
    "ACQUISITION_MODES",
    "ADAPTER_DEFAULTS",
    "ADAPTER_MODES",
    "ANDROID_CAPTURE_MODES",
    "COLOR_ORDERS",
    "COLOR_POLICIES",
    "CURSOR_ORBIT_PLAN_DEFAULTS",
    "DISTORTION_CORRECTION_MODES",
    "EXPOSURE_PERIOD_COUNTS",
    "FINAL_BENCHMARK_MODES",
    "FRAME_RATE_POLICIES",
    "MASK_POLICIES",
    "NORMALIZATION_MODES",
    "ORIENTATION_BEHAVIORS",
    "PANEL_SCALE_MODES",
    "QUARTER_TURN_DEGREES",
    "RIG_CALIBRATION_DEFAULTS",
    "ROI_POLICIES",
    "SHUTTER_MULTIPLIERS",
    "TARGET_PRESENTERS",
    "ZIGZAG_PLAN_DEFAULTS",
    "AcquisitionDefaults",
    "AdapterDefaults",
    "ResolvedAdapterPlan",
    "RigCalibrationDefaults",
    "CursorOrbitPlanDefaults",
    "ZigzagPlanDefaults",
]
