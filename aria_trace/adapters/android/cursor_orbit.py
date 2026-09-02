"""Balanced short joystick pulses for cursor-center calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

from aria_trace.adapters.android.zigzag import AndroidZigzagInputSource


@dataclass(frozen=True)
class CursorOrbitTouchPlan:
    """Pulse the movement joystick in opposite direction pairs.

    Every gesture starts at the same joystick center and releases after one
    short radial movement. Opposite consecutive pulses minimize accumulated
    character displacement, but the exact game response remains observable
    session data rather than a zero-motion guarantee.
    """

    center_xy: Sequence[int]
    radius_px: int
    direction_count: int = 12
    repeats: int = 2
    step_seconds: float = 0.12
    settle_seconds: float = 1.0
    reset_seconds: float = 0.18
    move_sample_hz: float = 30.0

    @property
    def start_xy(self) -> Sequence[int]:
        return self.center_xy

    def _direction_order(self) -> List[int]:
        count = int(self.direction_count)
        if count < 4 or count % 2:
            raise ValueError("Cursor orbit direction count must be an even value >= 4")
        if int(self.repeats) < 1:
            raise ValueError("Cursor orbit repeats must be positive")
        half = count // 2
        paired = [value for index in range(half) for value in (index, index + half)]
        return paired * int(self.repeats)

    def strokes(self) -> List[dict]:
        center_x, center_y = map(int, self.center_xy)
        radius = int(self.radius_px)
        if min(center_x, center_y) < 0:
            raise ValueError("Cursor orbit coordinates cannot be negative")
        if radius <= 0:
            raise ValueError("Cursor orbit radius must be positive")
        if self.step_seconds < 0.03:
            raise ValueError("Cursor orbit pulse duration must be at least 30 ms")
        if self.settle_seconds < 0.0 or self.reset_seconds < 0.0:
            raise ValueError("Cursor orbit timing cannot be negative")
        if self.move_sample_hz < 10.0 or self.move_sample_hz > 120.0:
            raise ValueError("Cursor orbit MOVE sample rate must be in 10..120 Hz")
        result = []
        for stroke_index, direction_index in enumerate(self._direction_order()):
            angle = 2.0 * math.pi * direction_index / int(self.direction_count)
            end = [
                int(round(center_x + radius * math.cos(angle))),
                int(round(center_y + radius * math.sin(angle))),
            ]
            result.append(
                {
                    "index": stroke_index,
                    "direction_index": direction_index,
                    "direction_angle_screen_deg": float(math.degrees(angle)),
                    "start_xy": [center_x, center_y],
                    "end_xy": end,
                    "opposite_pair_index": stroke_index // 2,
                }
            )
        return result

    @property
    def move_samples_per_stroke(self) -> int:
        return max(2, int(math.ceil(self.step_seconds * self.move_sample_hz)))

    def sampled_strokes(self) -> List[dict]:
        sample_count = self.move_samples_per_stroke
        sampled = []
        for stroke in self.strokes():
            start = stroke["start_xy"]
            end = stroke["end_xy"]
            moves = []
            for sample_index in range(sample_count):
                fraction = float(sample_index + 1) / float(sample_count)
                moves.append(
                    [
                        int(round(start[axis] + (end[axis] - start[axis]) * fraction))
                        for axis in (0, 1)
                    ]
                )
            expanded = dict(stroke)
            expanded["move_points_xy"] = moves
            sampled.append(expanded)
        return sampled

    @property
    def duration_seconds(self) -> float:
        count = len(self.strokes())
        return (
            self.settle_seconds
            + count * self.step_seconds
            + max(0, count - 1) * self.reset_seconds
        )

    def as_dict(self) -> dict:
        return {
            "plan_kind": "balanced_micro_movement",
            "center_xy": list(map(int, self.center_xy)),
            "radius_px": int(self.radius_px),
            "direction_count": int(self.direction_count),
            "repeats": int(self.repeats),
            "step_seconds": float(self.step_seconds),
            "settle_seconds": float(self.settle_seconds),
            "reset_seconds": float(self.reset_seconds),
            "move_sample_hz": float(self.move_sample_hz),
            "move_samples_per_stroke": int(self.move_samples_per_stroke),
            "strokes": self.strokes(),
            "displacement_policy": (
                "opposite consecutive joystick pulses minimize accumulated movement; "
                "zero world displacement is not assumed"
            ),
        }


class AndroidCursorOrbitInputSource(AndroidZigzagInputSource):
    """Legacy name for micro-movement touches through the verified transport."""

    source_id = "android_cursor_orbit_control"
    touch_kind = "cursor_orbit_touch"
    error_kind = "cursor_orbit_control_error"
    thread_name = "android-cursor-orbit"
    use_high_level_swipe = False


class AndroidMicroMovementInputSource(AndroidCursorOrbitInputSource):
    """Issue short balanced movements without assuming how the cursor responds."""

    source_id = "android_micro_movement_control"
    touch_kind = "micro_movement_touch"
    error_kind = "micro_movement_control_error"
    thread_name = "android-micro-movement"


__all__ = [
    "AndroidCursorOrbitInputSource",
    "AndroidMicroMovementInputSource",
    "CursorOrbitTouchPlan",
]
