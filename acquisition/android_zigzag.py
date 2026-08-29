"""Deterministic Android camera-view zigzag control for mini-map isolation."""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .models import InputPacket
from .sources import InputSource


@dataclass(frozen=True)
class ZigzagTouchPlan:
    """Long diagonal swipes with horizon-returning pitch changes.

    Every stroke moves horizontally in the same direction so yaw accumulates.
    Vertical commands are ``up, down, down, up, ...`` so the commanded pitch
    returns to the horizon after every pair. The finger is lifted and
    re-anchored between strokes, keeping every diagonal movement long.
    """

    start_xy: Sequence[int]
    end_x: int
    vertical_amplitude_px: int
    move_count: int = 20
    step_seconds: float = 0.35
    settle_seconds: float = 1.0
    reset_seconds: float = 0.10

    def strokes(self) -> List[dict]:
        start_x, horizon_y = map(int, self.start_xy)
        end_x = int(self.end_x)
        count = int(self.move_count)
        amplitude = int(self.vertical_amplitude_px)
        if min(start_x, horizon_y, end_x) < 0:
            raise ValueError("Zigzag touch coordinates cannot be negative")
        if start_x == end_x:
            raise ValueError("Zigzag needs non-zero horizontal motion")
        if amplitude <= 0:
            raise ValueError("Zigzag vertical amplitude must be positive")
        if count < 4:
            raise ValueError("Zigzag needs at least four strokes")
        if count % 4:
            raise ValueError("Zigzag stroke count must be a multiple of four")
        if self.step_seconds < 0.03:
            raise ValueError("Zigzag stroke duration must be at least 30 ms")
        if self.settle_seconds < 0.0:
            raise ValueError("Zigzag settle duration cannot be negative")
        if self.reset_seconds < 0.0:
            raise ValueError("Zigzag reset duration cannot be negative")
        if abs(end_x - start_x) != amplitude:
            raise ValueError("Zigzag strokes must be 45 degrees in display pixels")

        strokes = []
        signs = (-1, 1, 1, -1)
        for index in range(count):
            sign = signs[index % len(signs)]
            end_y = horizon_y + sign * amplitude
            if end_y < 0:
                raise ValueError("Zigzag sky endpoint is above the display")
            strokes.append(
                {
                    "index": int(index),
                    "direction": "up" if sign < 0 else "down",
                    "start_xy": [int(start_x), int(horizon_y)],
                    "end_xy": [int(end_x), int(end_y)],
                }
            )
        return strokes

    def points(self) -> List[List[int]]:
        """Return stroke endpoints for geometry-only callers."""
        return [stroke["end_xy"] for stroke in self.strokes()]

    @property
    def duration_seconds(self) -> float:
        count = len(self.strokes())
        return (
            float(self.settle_seconds)
            + count * float(self.step_seconds)
            + max(0, count - 1) * float(self.reset_seconds)
        )

    def as_dict(self) -> dict:
        return {
            "start_xy": list(map(int, self.start_xy)),
            "end_x": int(self.end_x),
            "vertical_amplitude_px": int(self.vertical_amplitude_px),
            "move_count": int(self.move_count),
            "step_seconds": float(self.step_seconds),
            "settle_seconds": float(self.settle_seconds),
            "reset_seconds": float(self.reset_seconds),
            "strokes": self.strokes(),
            "up_stroke_count": sum(
                stroke["direction"] == "up" for stroke in self.strokes()
            ),
            "down_stroke_count": sum(
                stroke["direction"] == "down" for stroke in self.strokes()
            ),
            "vertical_command_pattern": ["up", "down", "down", "up"],
        }


class AndroidZigzagInputSource(InputSource):
    """Issue and record the exact ADB touch events used for calibration."""

    source_id = "android_zigzag_control"

    def __init__(
        self,
        adb: Path,
        serial: str,
        plan: ZigzagTouchPlan,
        ready_event: Optional[threading.Event] = None,
        controller=None,
    ) -> None:
        self.adb = Path(adb)
        self.serial = str(serial)
        self.plan = plan
        self.ready_event = ready_event
        self.controller = controller
        self._stop = threading.Event()
        self._completed = threading.Event()
        self._thread = None
        self._emit = None
        self._error = None
        self._events_issued = 0
        self._gesture_completed = False

    def _motion(self, action: str, x: int, y: int) -> None:
        if self.controller is not None:
            self.controller.inject_touch(action, [x, y])
            return
        subprocess.check_call(
            [
                str(self.adb),
                "-s",
                self.serial,
                "shell",
                "input",
                "touchscreen",
                "motionevent",
                action,
                str(int(x)),
                str(int(y)),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def _record(self, kind: str, action: str, point: Sequence[int], index: int) -> None:
        host_time_ns = time.perf_counter_ns()
        self._events_issued += 1
        self._emit(
            InputPacket(
                self.source_id,
                kind,
                host_time_ns,
                {
                    "action": action,
                    "point_xy": list(map(int, point)),
                    "point_index": int(index),
                    "plan": self.plan.as_dict() if index == -1 else None,
                },
            )
        )

    def _run(self) -> None:
        down = False
        last = list(map(int, self.plan.start_xy))
        completed_strokes = 0
        strokes = self.plan.strokes()
        try:
            if self.ready_event is not None:
                while not self._stop.is_set() and not self.ready_event.wait(0.1):
                    pass
            if self._stop.wait(float(self.plan.settle_seconds)):
                return
            for index, stroke in enumerate(strokes):
                last = stroke["start_xy"]
                self._motion("DOWN", last[0], last[1])
                down = True
                self._record(
                    "zigzag_touch", "DOWN", last, -1 if index == 0 else index
                )
                if self._stop.wait(float(self.plan.step_seconds)):
                    break
                last = stroke["end_xy"]
                self._motion("MOVE", last[0], last[1])
                self._record("zigzag_touch", "MOVE", last, index)
                self._motion("UP", last[0], last[1])
                down = False
                self._record("zigzag_touch", "UP", last, index)
                completed_strokes += 1
                if index + 1 < len(strokes) and self._stop.wait(
                    float(self.plan.reset_seconds)
                ):
                    break
        except Exception as exc:
            self._error = "{}: {}".format(type(exc).__name__, exc)
            if self._emit is not None:
                self._emit(
                    InputPacket(
                        self.source_id,
                        "zigzag_control_error",
                        time.perf_counter_ns(),
                        {"error": self._error},
                    )
                )
        finally:
            if down:
                try:
                    self._motion("UP", last[0], last[1])
                    self._record("zigzag_touch", "UP", last, completed_strokes)
                except Exception as exc:
                    if self._error is None:
                        self._error = "{}: {}".format(type(exc).__name__, exc)
            self._gesture_completed = (
                self._error is None and completed_strokes == len(strokes)
            )
            self._completed.set()

    def start(self, emit: Callable[[InputPacket], None]) -> None:
        if self._thread is not None:
            raise RuntimeError("Android zigzag control is already started")
        self._emit = emit
        # Validate every stroke before issuing the first DOWN.
        self.plan.strokes()
        if self.controller is not None:
            self.controller.open()
        self._thread = threading.Thread(
            target=self._run, name="android-minimap-zigzag", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=6)
        if self.controller is not None:
            self.controller.close()

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def completed(self) -> bool:
        return self._gesture_completed

    @property
    def expected_event_count(self) -> int:
        return len(self.plan.strokes()) * 3

    @property
    def events_issued(self) -> int:
        return int(self._events_issued)

    def wait_completed(self, timeout: Optional[float] = None) -> bool:
        return self._completed.wait(timeout)

    def describe(self) -> Dict[str, object]:
        return {
            "type": type(self).__name__,
            "source_id": self.source_id,
            "adb": str(self.adb.resolve()),
            "serial": self.serial,
            "plan": self.plan.as_dict(),
            "events_issued": int(self._events_issued),
            "expected_events": int(self.expected_event_count),
            "gesture_completed": bool(self._gesture_completed),
            "error": self._error,
            "controller": (
                self.controller.describe() if self.controller is not None else {
                    "type": "adb_shell_input",
                    "transport": "one_process_per_event",
                }
            ),
        }
