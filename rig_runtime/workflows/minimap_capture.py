"""Record synchronized Android/rig-normalized HIK game frames during a zigzag."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from rig_runtime.adapters.android.capture import (
    AndroidRoiFrameSource,
    AndroidRoiSpec,
    ScrcpyCaptureHub,
    find_scrcpy_server,
)
from rig_runtime.adapters.android.spaces import image_space_from_surface
from rig_runtime.adapters.android.game_launcher import (
    foreground_component,
    launch_android_game,
)
from rig_runtime.adapters.android.zigzag import AndroidZigzagInputSource, ZigzagTouchPlan
from rig_runtime.adapters.android.cursor_orbit import (
    AndroidCursorOrbitInputSource,
    AndroidMicroMovementInputSource,
    CursorOrbitTouchPlan,
)
from rig_runtime.services.calibration.rig.dual_source_spaces import (
    write_android_source_space_yaml,
    write_dual_source_space_yaml,
)
from rig_runtime.services.calibration.rig.cross_source import (
    GameCrossSourceEvidenceRecorder,
    match_game_camera_orientation,
    orient_hik_source_from_first_adb_frame,
)
from rig_runtime.adapters.hik.capture import CalibratedHikFrameSource
from rig_runtime.adapters.filesystem.profile_registry import ProfileContext, ProfileRegistry
from rig_runtime.adapters.filesystem.session import SessionWriter
from rig_runtime.domain.packets import FramePacket
from rig_runtime.workflows.recording import AcquisitionRecorder
from rig_runtime.adapters.android.phone import (
    AdbPhoneSession,
    connected_adb_devices,
    resolve_adb_executable,
)
from rig_runtime.adapters.hik.driver import HikMvsCameraAdapter, RectifiedHikCamera
from rig_runtime.adapters.android.control import ScrcpyTouchController
from rig_runtime.adapters.sources import AdbClockMapper


def _select_phone(adb: Path, configured: Optional[str]) -> str:
    if configured:
        return str(configured)
    devices = list(connected_adb_devices(adb))
    if len(devices) == 1:
        return devices[0]
    raise RuntimeError(
        "Pass --phone-serial when the Android phone cannot be selected uniquely"
    )


def _phone_surface(adb: Path, serial: str) -> dict:
    phone = AdbPhoneSession(serial, adb_executable=adb)
    return phone.capture_surface()


def _resolve_rig_calibration(path: Path) -> Path:
    """Resolve either a calibration bundle directory or its JSON config."""

    value = Path(path)
    if value.is_dir():
        value = value / "hik_camera_calibration.json"
    if not value.is_file():
        raise RuntimeError("Rig calibration does not exist: {}".format(value))
    return value


def _keyguard_showing(phone: AdbPhoneSession) -> Optional[bool]:
    """Return Android's keyguard state without inferring it from brightness."""

    try:
        text = phone.shell("dumpsys", "window", "policy")
    except RuntimeError:
        return None
    true_patterns = (
        r"\bisKeyguardShowing\s*[=:]\s*true\b",
        r"\bmShowingLockscreen\s*[=:]\s*true\b",
        r"\bshowing\s*[=:]\s*true\b.*?\bKeyguard",
    )
    false_patterns = (
        r"\bisKeyguardShowing\s*[=:]\s*false\b",
        r"\bmShowingLockscreen\s*[=:]\s*false\b",
        r"\bshowing\s*[=:]\s*false\b.*?\bKeyguard",
    )
    if any(re.search(pattern, text, re.IGNORECASE | re.DOTALL) for pattern in true_patterns):
        return True
    if any(re.search(pattern, text, re.IGNORECASE | re.DOTALL) for pattern in false_patterns):
        return False
    return None


def _wake_phone_for_preparation(
    phone: AdbPhoneSession,
    logical_size_px: Sequence[int],
    *,
    sleeper=time.sleep,
) -> dict:
    """Wake/unlock electronically so no physical rig control is touched."""

    width, height = map(int, logical_size_px)
    result = {
        "method": "adb_non_toggling_wakeup",
        "physical_power_button_used": False,
        "actions": [],
    }
    phone.shell("input", "keyevent", "KEYCODE_WAKEUP")
    result["actions"].append("KEYCODE_WAKEUP")
    result["display"] = phone.ensure_display_on(timeout_seconds=8.0)
    phone.shell("wm", "dismiss-keyguard")
    result["actions"].append("wm_dismiss_keyguard")
    sleeper(0.5)
    keyguard = _keyguard_showing(phone)
    result["keyguard_before_remote_swipe"] = keyguard
    if keyguard is True:
        center_x = width // 2
        phone.shell(
            "input",
            "swipe",
            str(center_x),
            str(round(height * 0.72)),
            str(center_x),
            str(round(height * 0.24)),
            "350",
        )
        result["actions"].append("keyguard_upward_swipe")
        sleeper(0.5)
        phone.shell("wm", "dismiss-keyguard")
    result["keyguard_after"] = _keyguard_showing(phone)
    return result


def _select_camera(adapter: HikMvsCameraAdapter, configured: Optional[str]):
    devices = list(adapter.devices(probe=True))
    if configured:
        selected = next(
            (item for item in devices if item.device_id == str(configured)), None
        )
        if selected is None:
            raise RuntimeError("Configured HIK camera was not found: {}".format(configured))
        return selected
    if len(devices) == 1:
        return devices[0]
    if not devices:
        raise RuntimeError("No connected HIK camera was found")
    raise RuntimeError("Pass --camera-id when multiple HIK cameras are connected")


def _hik_fallback_allowed(exc: Exception) -> bool:
    """Recognize hardware absence/ownership failures, not bad calibration."""

    message = "{}: {}".format(type(exc).__name__, exc).lower()
    return any(
        marker in message
        for marker in (
            "no connected hik camera",
            "no hik camera",
            "camera was not found",
            "configured hik camera was not found",
            "open camera failed",
            "create camera handle failed",
            "access denied",
            "permission denied",
            "device is busy",
            "resource busy",
            "mvs runtime is unavailable",
        )
    )


def _game_booster_lock_showing(phone: AdbPhoneSession) -> bool:
    """Detect Samsung's game touch-protection overlay, not Android keyguard."""

    try:
        text = str(phone.shell("dumpsys", "window", "windows"))
    except RuntimeError:
        return False
    match = re.search(
        r"Window #[^\n]*GameBooster Lock Screen[^\n]*\n"
        r"(?P<window>.*?)(?=\n\s*Window #|\Z)",
        text,
        re.DOTALL,
    )
    if match is None:
        return False
    window = match.group("window")
    return bool(
        re.search(r"\bisOnScreen\s*=\s*true\b", window)
        and re.search(r"\bisVisible\s*=\s*true\b", window)
    )


def _dismiss_game_booster_lock(
    phone: AdbPhoneSession,
    adb: Path,
    server: Optional[Path],
    serial: str,
    screen_size_px: Sequence[int],
) -> dict:
    """Dismiss Samsung Game Booster's visible lock using its center drag."""

    width, height = map(int, screen_size_px)
    result = {"detected": False, "dismissed": False, "attempts": 0}
    phone.shell("input", "keyevent", "KEYCODE_WAKEUP")
    phone.ensure_display_on(timeout_seconds=8.0)
    for attempt in range(2):
        if not _game_booster_lock_showing(phone):
            result["dismissed"] = bool(result["detected"])
            return result
        result["detected"] = True
        result["attempts"] = attempt + 1
        points = [
            [round(width * 0.50), round(height * 0.58)],
            [round(width * 0.50), round(height * 0.46)],
            [round(width * 0.50), round(height * 0.33)],
            [round(width * 0.50), round(height * 0.20)],
        ]
        if server is None:
            phone.shell(
                "input",
                "swipe",
                str(points[0][0]),
                str(points[0][1]),
                str(points[-1][0]),
                str(points[-1][1]),
                "480",
            )
        else:
            with ScrcpyTouchController(
                adb, server, serial, [width, height]
            ) as controller:
                controller.inject_touch("DOWN", points[0])
                for point in points[1:]:
                    time.sleep(0.12)
                    controller.inject_touch("MOVE", point)
                time.sleep(0.12)
                controller.inject_touch("UP", points[-1])
        time.sleep(0.75)
    if _game_booster_lock_showing(phone):
        raise RuntimeError(
            "Samsung Game Booster touch protection is still covering the game"
        )
    result["dismissed"] = True
    return result


def _positive_pixel_distance(value: str) -> int:
    distance = int(value)
    if distance <= 0:
        raise argparse.ArgumentTypeError("swipe distance must be a positive integer")
    return distance


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Record Android and, when available, rig-normalized HIK images during "
            "one selected acquisition pattern. Android capture may be continuous "
            "scrcpy or settled ADB screenshots. Cursor response is not inferred at "
            "capture time. No calibration is run or published."
        )
    )
    value.add_argument(
        "--game-id",
        help=(
            "optional game identity used for known-game launch assistance and "
            "session metadata; omit it to capture the currently prepared game"
        ),
    )
    value.add_argument("--android-package")
    value.add_argument(
        "--no-launch-game",
        action="store_true",
        help="wake the phone but leave game launch to the user",
    )
    value.add_argument("--camera-id")
    value.add_argument("--camera-width", type=int, default=2448)
    value.add_argument("--camera-height", type=int, default=2048)
    value.add_argument("--camera-fps", type=float, default=30.0)
    value.add_argument("--mvs-python-path")
    value.add_argument("--profile-root", type=Path)
    value.add_argument("--phone-serial")
    value.add_argument("--adb")
    value.add_argument("--scrcpy-server", type=Path)
    value.add_argument(
        "--ffmpeg",
        type=Path,
        help=(
            "FFmpeg executable used only by continuous scrcpy capture; "
            "ADB-screenshot capture does not locate or execute FFmpeg"
        ),
    )
    value.add_argument(
        "--android-capture",
        choices=("scrcpy", "adb-screenshot"),
        default="scrcpy",
        help=(
            "Android image transport: continuous scrcpy video or one lossless "
            "ADB screencap after each settled swipe (default: scrcpy)"
        ),
    )
    value.add_argument(
        "--screenshot-settle-seconds",
        type=float,
        default=0.35,
        help=(
            "ADB-screenshot mode delay after touch UP before capturing the "
            "settled game image (default: 0.35)"
        ),
    )
    value.add_argument(
        "--output-root", type=Path, default=Path("sessions") / "calibration"
    )
    value.add_argument("--moves", type=int, default=12)
    value.add_argument(
        "--capture-mode",
        choices=("zigzag", "micro-movement", "cursor-orbit"),
        default="zigzag",
        help=(
            "zigzag uses long camera-look strokes; micro-movement uses balanced "
            "short movement-joystick pulses. Cursor behavior is interpreted later "
            "from the active game model. cursor-orbit is a legacy alias."
        ),
    )
    value.add_argument(
        "--horizontal-swipe-distance-px",
        type=_positive_pixel_distance,
        metavar="PX",
        help=(
            "horizontal distance of every diagonal swipe; default is 10%% "
            "of the current game-display width (240 px at 2400x1080)"
        ),
    )
    value.add_argument(
        "--vertical-swipe-distance-px",
        type=_positive_pixel_distance,
        metavar="PX",
        help=(
            "vertical distance of every diagonal swipe; default is 20%% "
            "of the landscape display height (216 px at 2400x1080)"
        ),
    )
    value.add_argument(
        "--travel-seconds",
        "--step-seconds",
        dest="step_seconds",
        type=float,
        default=0.12,
        help=(
            "time spent moving from the swipe start to its endpoint; "
            "--step-seconds is retained as a compatibility alias (default: 0.12)"
        ),
    )
    value.add_argument(
        "--endpoint-hold-seconds",
        type=float,
        default=0.10,
        help=(
            "time to keep the finger DOWN at the swipe endpoint before UP "
            "(default: 0.10)"
        ),
    )
    value.add_argument("--reset-seconds", type=float, default=0.10)
    value.add_argument("--settle-seconds", type=float, default=1.5)
    value.add_argument("--tail-seconds", type=float, default=1.5)
    value.add_argument(
        "--micro-movement-radius-px",
        "--cursor-orbit-radius-px",
        dest="cursor_orbit_radius_px",
        type=_positive_pixel_distance,
        help="micro-movement joystick pulse radius; default is 6%% of display height",
    )
    value.add_argument(
        "--micro-movement-pulse-seconds",
        "--cursor-pulse-seconds",
        dest="cursor_pulse_seconds",
        type=float,
        default=0.12,
        help="duration of each short micro-movement pulse (default: 0.12)",
    )
    value.add_argument(
        "--micro-movement-directions",
        "--cursor-orbit-directions",
        dest="cursor_orbit_directions",
        type=int,
        default=12,
    )
    value.add_argument(
        "--micro-movement-repeats",
        "--cursor-orbit-repeats",
        dest="cursor_orbit_repeats",
        type=int,
        default=2,
    )
    value.add_argument(
        "--joystick-center-x-fraction", type=float, default=0.18
    )
    value.add_argument(
        "--joystick-center-y-fraction", type=float, default=0.78
    )
    value.add_argument("--yes", action="store_true")
    value.add_argument(
        "--diagnostic-rig-calibration-override",
        type=Path,
        help=(
            "explicit calibration for diagnostics; production resolves the active "
            "rig profile automatically"
        ),
    )
    value.add_argument(
        "--require-hik",
        action="store_true",
        help="fail instead of falling back when the HIK camera is absent or occupied",
    )
    value.add_argument(
        "--control-only",
        action="store_true",
        help="run only the Android camera zigzag; no capture or calibration",
    )
    return value


def _build_zigzag_plan(arguments, width: int, height: int) -> ZigzagTouchPlan:
    horizontal_distance = getattr(arguments, "horizontal_swipe_distance_px", None)
    if horizontal_distance is None:
        horizontal_distance = round(width * 0.10)
    vertical_distance = getattr(arguments, "vertical_swipe_distance_px", None)
    if vertical_distance is None:
        vertical_distance = round(height * 0.20)
    horizontal_distance = int(horizontal_distance)
    vertical_distance = int(vertical_distance)
    if horizontal_distance <= 0:
        raise ValueError("Horizontal swipe distance must be positive")
    if vertical_distance <= 0:
        raise ValueError("Vertical swipe distance must be positive")
    # The game receives Android input in its current logical display raster.
    # Anchor on the right-center look-control surface, away from the movement
    # joystick and common edge gestures.
    start_x = round(width * 0.72)
    horizon_y = round(height * 0.50)
    upper_y = horizon_y - round(vertical_distance / 2.0)
    lower_y = upper_y + vertical_distance
    if start_x - horizontal_distance < 0:
        raise ValueError(
            "Horizontal swipe distance {} px exceeds the available {} px".format(
                horizontal_distance, start_x
            )
        )
    if upper_y < 0 or lower_y >= height:
        raise ValueError(
            "Vertical swipe distance {} px does not fit display height {} px".format(
                vertical_distance, height
            )
        )
    plan = ZigzagTouchPlan(
        start_xy=[start_x, horizon_y],
        end_x=start_x - horizontal_distance,
        vertical_amplitude_px=vertical_distance,
        move_count=arguments.moves,
        step_seconds=arguments.step_seconds,
        endpoint_hold_seconds=arguments.endpoint_hold_seconds,
        settle_seconds=arguments.settle_seconds,
        reset_seconds=arguments.reset_seconds,
    )
    plan.strokes()
    return plan


def _build_cursor_orbit_plan(
    arguments, width: int, height: int
) -> CursorOrbitTouchPlan:
    center_x_fraction = float(arguments.joystick_center_x_fraction)
    center_y_fraction = float(arguments.joystick_center_y_fraction)
    if not 0.05 <= center_x_fraction <= 0.45:
        raise ValueError("Joystick center X fraction must be in 0.05..0.45")
    if not 0.50 <= center_y_fraction <= 0.95:
        raise ValueError("Joystick center Y fraction must be in 0.50..0.95")
    center = [
        int(round(width * center_x_fraction)),
        int(round(height * center_y_fraction)),
    ]
    radius = int(
        arguments.cursor_orbit_radius_px
        if arguments.cursor_orbit_radius_px is not None
        else round(height * 0.06)
    )
    if (
        center[0] - radius < 0
        or center[0] + radius >= width
        or center[1] - radius < 0
        or center[1] + radius >= height
    ):
        raise ValueError("Cursor orbit does not fit inside the Android display")
    plan = CursorOrbitTouchPlan(
        center_xy=center,
        radius_px=radius,
        direction_count=int(arguments.cursor_orbit_directions),
        repeats=int(arguments.cursor_orbit_repeats),
        step_seconds=float(arguments.cursor_pulse_seconds),
        settle_seconds=float(arguments.settle_seconds),
        reset_seconds=float(arguments.reset_seconds),
    )
    plan.sampled_strokes()
    return plan


def _build_control_plan(arguments, width: int, height: int):
    if arguments.capture_mode in ("micro-movement", "cursor-orbit"):
        return _build_cursor_orbit_plan(arguments, width, height)
    return _build_zigzag_plan(arguments, width, height)


def _input_source(arguments, adb: Path, serial: str, plan, controller=None):
    if arguments.capture_mode == "micro-movement":
        source = AndroidMicroMovementInputSource
    elif arguments.capture_mode == "cursor-orbit":
        source = AndroidCursorOrbitInputSource
    else:
        source = AndroidZigzagInputSource
    return source(adb, serial, plan, controller=controller)


def _launch_or_defer_game(phone: AdbPhoneSession, arguments) -> dict:
    """Launch an explicitly identified game or preserve the current foreground app."""

    if arguments.no_launch_game or not (
        arguments.game_id or arguments.android_package
    ):
        return {
            "game_id": arguments.game_id,
            "status": (
                "disabled_by_user"
                if arguments.no_launch_game
                else "manual_current_game"
            ),
            "package": arguments.android_package,
            "calibration_controls_changed": False,
            "game_input_injected": False,
        }
    result = launch_android_game(
        phone,
        arguments.game_id or arguments.android_package,
        explicit_package=arguments.android_package,
    )
    result["game_id"] = arguments.game_id
    return result


def _record_foreground_at_capture(
    phone: AdbPhoneSession, game_launch: dict
) -> dict:
    """Record the actual resumed package immediately before acquisition."""

    result = dict(game_launch)
    try:
        component = foreground_component(phone)
    except Exception as exc:
        result["foreground_probe_error"] = "{}: {}".format(
            type(exc).__name__, exc
        )
        component = None
    package = component.split("/", 1)[0] if component else None
    expected = str(result.get("package") or "").strip() or None
    result["foreground_at_capture"] = component
    result["foreground_package_at_capture"] = package
    if package and expected is None:
        result["package"] = package
        result["package_source"] = "adb_foreground_at_capture"
    elif package and expected != package:
        result["package_mismatch_warning"] = (
            "Expected package {!r}, but {!r} was foreground when acquisition "
            "started".format(expected, package)
        )
    return result


def _canonical_game_facts(
    game_launch: dict, surface: dict, game_id: Optional[str] = None
) -> dict:
    """Persist the foreground game and Android rotation-0 relation explicitly."""

    logical_size = [int(value) for value in surface.get("logical_size_px") or []]
    natural_size = [int(value) for value in surface.get("natural_size_px") or []]
    turns = int(surface.get("quarter_turns_clockwise_from_natural", 0)) % 4
    usb_edges = ("bottom", "right", "top", "left")
    package = (
        game_launch.get("foreground_package_at_capture")
        or game_launch.get("package")
    )
    return {
        "game_id": game_id or game_launch.get("game_id"),
        "package": package,
        "foreground_component_at_capture": game_launch.get(
            "foreground_at_capture"
        ),
        "game_orientation": (
            "landscape"
            if len(logical_size) == 2 and logical_size[0] > logical_size[1]
            else "portrait"
        ),
        "phone_pose_at_capture": {"usb_edge": usb_edges[turns]},
        "surface_quarter_turns_clockwise_from_phone_natural": turns,
        "logical_size_px": logical_size,
        "natural_size_px": natural_size,
        "canonical_space": "phone_natural_rotation_0",
        "source": "adb_foreground_and_surface_at_capture",
    }


def _session_game_label(game_id: Optional[str], android_package: Optional[str]) -> str:
    value = game_id or android_package or "unidentified-game"
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip()).strip("-.")
    return cleaned or "unidentified-game"


def _run_control_only(arguments) -> int:
    adb = resolve_adb_executable(arguments.adb)
    serial = _select_phone(adb, arguments.phone_serial)
    server = (
        find_scrcpy_server(arguments.scrcpy_server)
        if arguments.android_capture == "scrcpy"
        else None
    )
    wake_surface = _phone_surface(adb, serial)
    phone = AdbPhoneSession(serial, adb_executable=adb)
    preparation = _wake_phone_for_preparation(
        phone, wake_surface["logical_size_px"]
    )
    game_launch = _launch_or_defer_game(phone, arguments)
    time.sleep(0.75)
    surface = _phone_surface(adb, serial)
    width, height = surface["logical_size_px"]
    if width <= height:
        raise RuntimeError(
            "The game is not landscape yet ({}x{}); prepare it and retry".format(
                width, height
            )
        )
    plan = _build_control_plan(arguments, width, height)
    preparation["game_booster_before_prompt"] = _dismiss_game_booster_lock(
        phone, adb, server, serial, [width, height]
    )

    print("Phone: {} ({}x{})".format(serial, width, height))
    if game_launch.get("package"):
        print("Game: {} ({})".format(game_launch["package"], game_launch["status"]))
    print("Control-only test: no camera, recording, calibration, or profile output.")
    if preparation.get("keyguard_after") is True:
        print("Unlock the phone before continuing.")
    if preparation["game_booster_before_prompt"]["dismissed"]:
        print("Samsung Game Booster touch protection dismissed.")
    if not arguments.yes:
        input(
            "Confirm the target game is visible in touchscreen mode, then press Enter "
            "to run {}: ".format(arguments.capture_mode)
        )
    preparation["game_booster_before_control"] = _dismiss_game_booster_lock(
        phone, adb, server, serial, [width, height]
    )
    if arguments.capture_mode in ("micro-movement", "cursor-orbit"):
        print(
            "Running {} balanced short joystick pulses...".format(
                len(plan.strokes())
            )
        )
    else:
        print(
            "Running {} long strokes ({} up, {} down)...".format(
                len(plan.strokes()),
                sum(stroke["direction"] == "up" for stroke in plan.strokes()),
                sum(stroke["direction"] == "down" for stroke in plan.strokes()),
            )
        )

    controller = (
        ScrcpyTouchController(adb, server, serial, [width, height])
        if server is not None
        else None
    )
    control = _input_source(arguments, adb, serial, plan, controller=controller)
    packets = []
    try:
        control.start(packets.append)
        timeout = max(20.0, plan.duration_seconds + 5.0)
        if not control.wait_completed(timeout):
            raise RuntimeError(
                "{} control did not complete within {:.1f} seconds".format(
                    arguments.capture_mode,
                    timeout
                )
            )
    finally:
        control.stop()
    if control.error:
        raise RuntimeError(
            "Android {} control failed: {}".format(
                arguments.capture_mode, control.error
            )
        )
    if not control.completed or control.events_issued != control.expected_event_count:
        raise RuntimeError(
            "Android {} control was incomplete: {}/{} events".format(
                arguments.capture_mode,
                control.events_issued, control.expected_event_count
            )
        )
    unit = (
        "swipes"
        if arguments.capture_mode == "zigzag"
        else "touch events"
    )
    print(
        "{} control completed: {}/{} {}.".format(
            arguments.capture_mode,
            control.events_issued,
            control.expected_event_count,
            unit,
        )
    )
    return 0


class _AdbSettledScreenshotSource:
    """Describe settled screenshots triggered after each completed control."""

    stream_id = "android_phone"

    def __init__(
        self,
        adb: Path,
        serial: str,
        settle_seconds: float,
        capture_mode: str = "zigzag",
    ) -> None:
        self.adb = Path(adb)
        self.serial = str(serial)
        self.settle_seconds = float(settle_seconds)
        self.capture_mode = str(capture_mode)
        self.capture_count = 0

    def describe(self) -> dict:
        completion_action = (
            "SWIPE" if self.capture_mode == "zigzag" else "UP"
        )
        return {
            "type": type(self).__name__,
            "stream_id": self.stream_id,
            "transport": "adb_exec_out_screencap_png",
            "scrcpy_used": False,
            "trigger": "after_each_{}_touch_{}".format(
                self.capture_mode.replace("-", "_"), completion_action
            ),
            "settle_seconds_after_control": self.settle_seconds,
            "lossless_pngs_retained": True,
            "preferred_frame_storage": "image_series",
            "preferred_image_format": "png",
            "external_ffmpeg_required": False,
            "capture_count": int(self.capture_count),
            "adb": str(self.adb.resolve()),
            "serial": self.serial,
        }


def _capture_adb_screenshot_packet(
    adb: Path,
    serial: str,
    stroke_index: int,
    trigger: str = "settled_after_zigzag_touch_UP",
) -> tuple[FramePacket, bytes]:
    """Capture one lossless Android screenshot with bounded host-time metadata."""

    request_time_ns = time.perf_counter_ns()
    png = subprocess.check_output(
        [str(adb), "-s", str(serial), "exec-out", "screencap", "-p"],
        stderr=subprocess.DEVNULL,
        timeout=12,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    receive_time_ns = time.perf_counter_ns()
    image = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise RuntimeError(
            "ADB screencap after swipe {} was not a decodable PNG".format(
                stroke_index
            )
        )
    capture_time_ns = (request_time_ns + receive_time_ns) // 2
    return (
        FramePacket(
            "android_phone",
            image,
            capture_time_ns,
            receive_time_ns,
            metadata={
                "source": "android_adb_screencap",
                "coordinate_space": "android_logical_display_pixels",
                "transport": "adb_exec_out_screencap_png",
                "trigger": str(trigger),
                "stroke_index": int(stroke_index),
                "request_time_ns": int(request_time_ns),
                "timestamp_uncertainty_ns": int(
                    (receive_time_ns - request_time_ns) // 2
                ),
                "lossless_png_sha256": hashlib.sha256(png).hexdigest(),
            },
        ),
        png,
    )


def _record_adb_screenshot_zigzag(
    pending_path: Path,
    *,
    adb: Path,
    serial: str,
    plan: ZigzagTouchPlan,
    screenshot_settle_seconds: float,
    surface: dict,
    selected_camera,
    hik_adapter,
    rig_calibration: Optional[Path],
    calibration_revision: Optional[str],
    calibration_selection: Optional[str],
    require_hik: bool,
    hik_fallback_reason: Optional[str],
    game_id: Optional[str],
    preparation: dict,
    game_launch: dict,
    capture_mode: str = "zigzag",
) -> tuple[dict, object, Optional[object]]:
    """Record settled swipe endpoints without starting any scrcpy component."""

    if screenshot_settle_seconds < 0.0:
        raise ValueError("Screenshot settle duration cannot be negative")
    input_packets = []
    captured_pairs = []
    orientation_match = None
    orientation_images = None
    output_image_turns = None
    aligned_surface = dict(surface)
    aligned_surface["source"] = "adb_surface_orientation_at_capture"
    hik = None

    if selected_camera is not None:
        hik_reader = RectifiedHikCamera(rig_calibration, adapter=hik_adapter)
        hik = CalibratedHikFrameSource(
            rig_calibration,
            "hik_phone",
            rectify=True,
            output_quarter_turns_clockwise=0,
            reader=hik_reader,
        )
        try:
            hik.start()
            # Claim and warm the stream before the first settled endpoint. This
            # frame is diagnostic warm-up only and is not persisted.
            hik.read()
        except Exception as exc:
            try:
                hik.stop()
            except Exception:
                pass
            if require_hik or not _hik_fallback_allowed(exc):
                raise
            hik_fallback_reason = "{}: {}".format(type(exc).__name__, exc)
            hik = None
            selected_camera = None
            print(
                "HIK camera unavailable or occupied; settled ADB screenshots "
                "remain usable for mini-map calibration: {}".format(
                    hik_fallback_reason
                ),
                flush=True,
            )

    if capture_mode == "micro-movement":
        touch_kind = "micro_movement_touch"
        control_class = AndroidMicroMovementInputSource
    elif capture_mode == "cursor-orbit":
        touch_kind = "cursor_orbit_touch"
        control_class = AndroidCursorOrbitInputSource
    else:
        touch_kind = "zigzag_touch"
        control_class = AndroidZigzagInputSource
    control = control_class(adb, serial, plan, controller=None)
    screenshot_source = _AdbSettledScreenshotSource(
        adb, serial, screenshot_settle_seconds, capture_mode
    )

    def receive_input(packet) -> None:
        nonlocal orientation_match, orientation_images, output_image_turns
        nonlocal aligned_surface
        input_packets.append(packet)
        if packet.kind != touch_kind or packet.payload.get("action") not in (
            "UP",
            "SWIPE",
        ):
            return
        stroke_index = int(packet.payload.get("point_index", len(captured_pairs)))
        if screenshot_settle_seconds:
            time.sleep(float(screenshot_settle_seconds))
        trigger = "settled_after_{}_{}".format(
            touch_kind, packet.payload.get("action")
        )
        if capture_mode in ("micro-movement", "cursor-orbit"):
            adb_packet, _raw_png = _capture_adb_screenshot_packet(
                adb, serial, stroke_index, trigger=trigger
            )
        else:
            adb_packet, _raw_png = _capture_adb_screenshot_packet(
                adb, serial, stroke_index
            )
        adb_height, adb_width = adb_packet.image.shape[:2]
        adb_packet.metadata["image_space"] = image_space_from_surface(
            surface,
            source_size_px=[adb_width, adb_height],
            roi_xywh=[0, 0, adb_width, adb_height],
            stored_size_px=[adb_width, adb_height],
        )
        adb_packet.metadata["trigger_input_up_host_time_ns"] = int(
            packet.host_time_ns
        )
        adb_packet.metadata["settled_after_up_ms"] = (
            int(adb_packet.host_capture_time_ns) - int(packet.host_time_ns)
        ) / 1.0e6
        hik_packet = None
        if hik is not None:
            if orientation_match is None:
                calibration_display_packet = hik.read()
                if calibration_display_packet is None:
                    raise RuntimeError(
                        "HIK stream ended before orientation could be measured"
                    )
                calibration_display_image = hik.alignment_evidence_image(
                    calibration_display_packet
                )
                orientation_match, orientation_images = (
                    match_game_camera_orientation(
                        adb_packet.image,
                        calibration_display_image,
                        rig_calibration,
                        android_reported_quarter_turns=surface[
                            "quarter_turns_clockwise_from_natural"
                        ],
                    )
                )
                orientation_match["first_frame_pair_delta_ms"] = abs(
                    int(calibration_display_packet.host_capture_time_ns)
                    - int(adb_packet.host_capture_time_ns)
                ) / 1.0e6
                output_image_turns = int(
                    orientation_match[
                        "selected_camera_adapter_image_quarter_turns_clockwise_from_calibration_display"
                    ]
                )
                hik.set_output_orientation(
                    output_image_turns,
                    {
                        "status": orientation_match["status"],
                        "selection_basis": orientation_match["selection_basis"],
                        "selected_confidence": orientation_match[
                            "selected_confidence"
                        ],
                        "confidence_margin": orientation_match[
                            "confidence_margin"
                        ],
                    },
                )
                selected_surface_turns = int(
                    orientation_match[
                        "selected_adb_surface_quarter_turns_clockwise_from_phone_natural"
                    ]
                )
                aligned_surface = {
                    **dict(surface),
                    "quarter_turns_clockwise_from_natural": selected_surface_turns,
                    "degrees_clockwise_from_natural": selected_surface_turns * 90,
                    "source": "first_game_adb_and_hik_image_evidence",
                    "android_reported_quarter_turns_clockwise_from_natural": surface[
                        "quarter_turns_clockwise_from_natural"
                    ],
                    "orientation_evidence": (
                        "cross_source_check/orientation_match/summary.json"
                    ),
                }
            hik_packet = hik.read()
            if hik_packet is None:
                raise RuntimeError(
                    "HIK stream ended after settled swipe {}".format(stroke_index)
                )
            hik_packet.metadata["trigger"] = trigger
            hik_packet.metadata["stroke_index"] = stroke_index
            hik_packet.metadata["trigger_input_up_host_time_ns"] = int(
                packet.host_time_ns
            )
            hik_packet.metadata["paired_adb_capture_time_ns"] = int(
                adb_packet.host_capture_time_ns
            )
        game_context = _canonical_game_facts(
            game_launch, aligned_surface, game_id
        )
        adb_packet.metadata["game_context"] = game_context
        if hik_packet is not None:
            hik_packet.metadata["game_context"] = dict(game_context)
        adb_packet.metadata["paired_hik_capture_time_ns"] = (
            int(hik_packet.host_capture_time_ns)
            if hik_packet is not None
            else None
        )
        captured_pairs.append((adb_packet, hik_packet))
        screenshot_source.capture_count += 1

    try:
        control.start(receive_input)
        timeout = max(
            30.0,
            float(plan.duration_seconds)
            + len(plan.strokes()) * (float(screenshot_settle_seconds) + 6.0)
            + 20.0,
        )
        if not control.wait_completed(timeout):
            raise RuntimeError(
                "ADB screenshot zigzag did not complete within {:.1f} seconds"
                .format(timeout)
            )
    finally:
        control.stop()
        if hik is not None:
            hik.stop()

    if control.error:
        raise RuntimeError(
            "Android {} control failed: {}".format(capture_mode, control.error)
        )
    if not control.completed or control.events_issued != control.expected_event_count:
        raise RuntimeError(
            "Android {} control was incomplete: {}/{} events".format(
                capture_mode, control.events_issued, control.expected_event_count
            )
        )
    if len(captured_pairs) != len(plan.strokes()):
        raise RuntimeError(
            "Settled ADB screenshot count {} does not match {} completed swipes"
            .format(len(captured_pairs), len(plan.strokes()))
        )

    frame_processors = []
    if hik is not None:
        frame_processors.append(
            GameCrossSourceEvidenceRecorder(
                rig_calibration,
                aligned_surface,
                sample_period_seconds=0.0,
                orientation_match=orientation_match,
                orientation_evidence_images=orientation_images,
            )
        )
    frame_sources = [screenshot_source] + ([hik] if hik is not None else [])
    hik_context = (
        {
            "mode": "rig_rectified_phone_view_settled_snapshots",
            "status": "active",
            "camera_id": selected_camera.device_id,
            "rig_calibration_used": True,
            "rig_calibration": str(rig_calibration.resolve()),
            "rig_profile_revision": calibration_revision,
            "calibration_selection": calibration_selection,
            "stream_id": "hik_phone",
            "output_image_quarter_turns_clockwise_from_calibration_display": (
                output_image_turns
            ),
            "output_orientation_selection": {
                "basis": orientation_match["selection_basis"],
                "status": orientation_match["status"],
                "confidence": orientation_match["selected_confidence"],
                "confidence_margin": orientation_match["confidence_margin"],
                "evidence": "cross_source_check/orientation_match/summary.json",
            },
        }
        if hik is not None
        else {
            "mode": "android_only",
            "status": "not_requested" if rig_calibration is None else "fallback",
            "reason": hik_fallback_reason,
            "rig_calibration_used": False,
        }
    )
    context = {
        "capture_kind": (
            "micro_movement_game_calibration_source_data"
            if capture_mode == "micro-movement"
            else "cursor_orbit_game_calibration_source_data"
            if capture_mode == "cursor-orbit"
            else "zigzag_minimap_source_data"
        ),
        "capture_schedule": "settled_swipe_endpoint_screenshots",
        "android_capture": {
            "transport": "adb_exec_out_screencap_png",
            "scrcpy_used": False,
            "external_ffmpeg_used": False,
            "trigger": "after_each_{}_{}".format(
                touch_kind,
                "SWIPE" if capture_mode == "zigzag" else "UP",
            ),
            "settle_seconds_after_control": float(screenshot_settle_seconds),
            "lossless_png_directory": "frames/android_phone",
            "lossless_png_space_metadata": "frames.jsonl#metadata.image_space",
            "adb_frame_storage": "lossless_png_image_series",
            "hik_frame_storage": (
                "opencv_mjpeg_video" if hik is not None else "not_present"
            ),
        },
        "game_id": game_id,
        "image_sources": ["android_adb_screencap"]
        + (["hik_mvs_rig_rectified"] if hik is not None else []),
        "hik_capture": hik_context,
        "phone_surface_orientation": aligned_surface,
        "game_facts": _canonical_game_facts(
            game_launch, aligned_surface, game_id
        ),
        "phone_preparation": preparation,
        "game_launch": game_launch,
        (
            "micro_movement_plan"
            if capture_mode == "micro-movement"
            else "cursor_orbit_plan"
            if capture_mode == "cursor-orbit"
            else "zigzag_plan"
        ): plan.as_dict(),
        "calibration_compatibility": {
            "minimap": "compatible_android_phone_session",
            "game_color": (
                "compatible_synchronized_adb_hik_pairs"
                if hik is not None
                else "requires_hik_pairs_not_present"
            ),
        },
        "calibration_status": "not_run",
    }
    writer = SessionWriter(
        pending_path,
        frame_sources,
        [control],
        # Only the optional HIK stream is encoded as MJPEG. The ADB source
        # declares image-series storage and therefore never creates a video.
        video_encoding="mjpeg",
        video_fps=max(1.0, min(30.0, float(len(captured_pairs)))),
        video_crf=16,
        ffmpeg=None,
        session_context=context,
    )
    all_times = [packet.host_time_ns for packet in input_packets]
    all_times.extend(
        packet.host_capture_time_ns
        for pair in captured_pairs
        for packet in pair
        if packet is not None
    )
    try:
        writer.rebase_origin(min(all_times))
        if frame_processors:
            writer.attach_frame_processors(frame_processors)
        for packet in input_packets:
            writer.write_input(packet)
        for adb_packet, hik_packet in captured_pairs:
            writer.write_frame(adb_packet)
            if hik_packet is not None:
                writer.write_frame(hik_packet)
        writer.manifest["frame_sources"] = [
            source.describe() for source in frame_sources
        ]
        writer.manifest["input_sources"] = [control.describe()]
        writer.close(status="complete")
    except Exception as exc:
        writer.close(
            status="incomplete",
            error="{}: {}".format(type(exc).__name__, exc),
        )
        raise

    manifest = json.loads(
        (pending_path / "manifest.json").read_text(encoding="utf-8")
    )
    if hik is not None:
        write_dual_source_space_yaml(
            pending_path, rig_calibration, aligned_surface, manifest
        )
    else:
        write_android_source_space_yaml(
            pending_path, aligned_surface, manifest
        )
    manifest["coordinate_spaces"] = "coordinate_spaces.yaml"
    manifest_path = pending_path / "manifest.json"
    temporary_manifest = pending_path / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return manifest, control, hik


def main(argv: Optional[Sequence[str]] = None) -> int:
    argument_parser = parser()
    arguments = argument_parser.parse_args(argv)
    if (
        arguments.android_capture == "adb-screenshot"
        and arguments.screenshot_settle_seconds < 0.0
    ):
        argument_parser.error("--screenshot-settle-seconds cannot be negative")
    from rig_runtime.adapters.filesystem.system_configuration import (
        load_system_configuration,
    )

    settings = load_system_configuration(arguments.profile_root)
    arguments.camera_id = arguments.camera_id or settings["devices"].get(
        "camera_id"
    )
    arguments.phone_serial = arguments.phone_serial or settings["devices"].get(
        "phone_id"
    )
    arguments.game_id = arguments.game_id or settings["game"].get("game_id")
    arguments.adb = arguments.adb or settings["tools"].get("adb")
    arguments.mvs_python_path = (
        arguments.mvs_python_path or settings["tools"].get("mvs_python_path")
    )
    if arguments.control_only:
        return _run_control_only(arguments)
    adb = resolve_adb_executable(arguments.adb)
    serial = _select_phone(adb, arguments.phone_serial)
    rig_calibration = None
    rig_config = None
    calibration_revision = None
    hik_adapter = None
    selected_camera = None
    hik_fallback_reason = None
    try:
        hik_adapter = HikMvsCameraAdapter(
            sdk_python_path=arguments.mvs_python_path
        )
        selected_camera = _select_camera(hik_adapter, arguments.camera_id)
    except Exception as exc:
        if arguments.require_hik or not _hik_fallback_allowed(exc):
            raise
        hik_fallback_reason = "{}: {}".format(type(exc).__name__, exc)
    if selected_camera is not None:
        if arguments.diagnostic_rig_calibration_override is not None:
            rig_calibration = _resolve_rig_calibration(
                arguments.diagnostic_rig_calibration_override
            )
            calibration_selection = "diagnostic_explicit_path"
        else:
            registry = ProfileRegistry(arguments.profile_root)
            rig_profile = registry.resolve(
                "rig",
                ProfileContext(
                    camera_id=str(selected_camera.device_id),
                    phone_id=serial,
                ),
            )
            rig_calibration = registry.runtime_file(
                rig_profile, "hik_camera_calibration"
            ).resolve()
            calibration_revision = rig_profile["revision_id"]
            calibration_selection = "active_profile_registry"
        rig_config = json.loads(rig_calibration.read_text(encoding="utf-8"))
        if selected_camera is not None:
            calibrated_camera_id = str(rig_config["camera"]["device_id"])
            if str(selected_camera.device_id) != calibrated_camera_id:
                raise RuntimeError(
                    "Rig calibration is for HIK camera {}, but {} was selected".format(
                        calibrated_camera_id, selected_camera.device_id
                    )
                )
    calibrated_phone_id = str(
        ((rig_config or {}).get("phone") or {}).get("serial") or ""
    )
    if calibrated_phone_id and str(serial) != calibrated_phone_id:
        raise RuntimeError(
            "Rig calibration is for Android phone {}, but {} was selected".format(
                calibrated_phone_id, serial
            )
        )
    server = (
        find_scrcpy_server(arguments.scrcpy_server)
        if arguments.android_capture == "scrcpy"
        else None
    )
    wake_surface = _phone_surface(adb, serial)
    wake_width, wake_height = wake_surface["logical_size_px"]
    phone = AdbPhoneSession(serial, adb_executable=adb)
    preparation = _wake_phone_for_preparation(phone, [wake_width, wake_height])
    game_launch = _launch_or_defer_game(phone, arguments)
    # Game launch may rotate the Android logical display. Control coordinates
    # must be probed after that transition, not inherited from the launcher.
    time.sleep(0.75)
    surface = _phone_surface(adb, serial)
    width, height = surface["logical_size_px"]
    preparation["game_booster_before_prompt"] = _dismiss_game_booster_lock(
        phone, adb, server, serial, [width, height]
    )
    plan = _build_control_plan(arguments, width, height)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_path = arguments.output_root / "{}-{}-{}".format(
        _session_game_label(arguments.game_id, arguments.android_package),
        timestamp,
        arguments.capture_mode,
    )
    pending_path = session_path.with_name(session_path.name + ".pending")
    if selected_camera is not None:
        print("HIK camera: {} ({})".format(selected_camera.label, selected_camera.device_id))
        print("Rig calibration: {} ({})".format(rig_calibration, calibration_selection))
    else:
        print("HIK camera unavailable; recording Android only: {}".format(hik_fallback_reason))
    print("Phone: {} ({}x{})".format(serial, width, height))
    print("Phone display awakened electronically through ADB; the physical power button is not used.")
    if preparation.get("keyguard_after") is True:
        print("The credential keyguard is still present; unlock it without moving the rig.")
    if preparation["game_booster_before_prompt"]["dismissed"]:
        print("Samsung Game Booster touch protection dismissed.")
    if game_launch.get("package"):
        print(
            "Game: {} ({})".format(
                game_launch["package"], game_launch["status"]
            )
        )
    elif game_launch["status"] in ("manual_unknown_game", "manual_current_game"):
        print("No game was launched automatically; prepare any target game before confirming readiness.")
    print("Output session: {}".format(session_path.resolve()))
    if arguments.android_capture == "adb-screenshot":
        print(
            "Android capture: ADB screencap after each settled swipe; scrcpy "
            "will not be started."
        )
    print("This command records data only; it does not calibrate or publish profiles.")
    if not arguments.yes:
        input("Prepare the game, then press Enter when its unobstructed view is ready: ")
    preparation["game_booster_before_control"] = _dismiss_game_booster_lock(
        phone, adb, server, serial, [width, height]
    )
    # The human checkpoint or app launch can change both the foreground package
    # and logical orientation. The capture contract must describe this final
    # state, and control coordinates must be rebuilt for it.
    time.sleep(0.25)
    game_launch = _record_foreground_at_capture(phone, game_launch)
    surface = _phone_surface(adb, serial)
    ready_width, ready_height = surface["logical_size_px"]
    if [ready_width, ready_height] != [width, height]:
        width, height = ready_width, ready_height
        plan = _build_control_plan(arguments, width, height)
        print(
            "Game-ready Android surface changed to {}x{}; swipe coordinates "
            "were rebuilt from the fresh surface.".format(width, height),
            flush=True,
        )
    if game_launch.get("foreground_package_at_capture"):
        print(
            "Foreground package at acquisition: {}".format(
                game_launch["foreground_package_at_capture"]
            ),
            flush=True,
        )
    if game_launch.get("package_mismatch_warning"):
        print(
            "Warning: {}".format(game_launch["package_mismatch_warning"]),
            flush=True,
        )

    if arguments.android_capture == "adb-screenshot":
        try:
            manifest, control, settled_hik = _record_adb_screenshot_zigzag(
                pending_path,
                adb=adb,
                serial=serial,
                plan=plan,
                screenshot_settle_seconds=arguments.screenshot_settle_seconds,
                surface=surface,
                selected_camera=selected_camera,
                hik_adapter=hik_adapter,
                rig_calibration=rig_calibration,
                calibration_revision=calibration_revision,
                calibration_selection=(
                    calibration_selection if selected_camera is not None else None
                ),
                require_hik=arguments.require_hik,
                hik_fallback_reason=hik_fallback_reason,
                game_id=arguments.game_id,
                preparation=preparation,
                game_launch=game_launch,
                capture_mode=arguments.capture_mode,
            )
            counts = manifest.get("frame_counts") or {}
            expected_swipes = len(plan.strokes())
            if (
                manifest.get("status") != "complete"
                or int(counts.get("android_phone", 0)) != expected_swipes
                or (
                    settled_hik is not None
                    and int(counts.get("hik_phone", 0)) != expected_swipes
                )
            ):
                raise RuntimeError(
                    "Settled screenshot recording did not preserve one frame "
                    "per completed swipe"
                )
            pending_path.replace(session_path)
        except Exception:
            if pending_path.is_dir():
                shutil.rmtree(str(pending_path))
            raise
        print(
            "Captured {} settled {} endpoints with {}: {}".format(
                len(plan.strokes()),
                arguments.capture_mode,
                "ADB screencap and HIK"
                if settled_hik is not None
                else "ADB screencap only",
                session_path.resolve(),
            )
        )
        return 0

    clock = AdbClockMapper(adb, serial)
    hub = ScrcpyCaptureHub(
        adb,
        server,
        serial=serial,
        ffmpeg=arguments.ffmpeg,
        clock=clock,
        max_fps=60.0,
    )
    android = AndroidRoiFrameSource(
        hub,
        AndroidRoiSpec("android_phone", 0, 0, 0, 0),
        image_space_context=surface,
    )
    hik = None
    orientation_match = None
    output_image_turns = None
    frame_processors = []
    aligned_surface = dict(surface)
    aligned_surface["source"] = "adb_surface_orientation_at_capture"
    if selected_camera is not None:
        hik_reader = RectifiedHikCamera(rig_calibration, adapter=hik_adapter)
        try:
            # Claim the camera before starting scrcpy. Absence or exclusive
            # ownership failure is the only automatic fallback boundary.
            hik_reader.open()
        except Exception as exc:
            try:
                hik_reader.release()
            except Exception:
                pass
            if arguments.require_hik or not _hik_fallback_allowed(exc):
                raise
            hik_fallback_reason = "{}: {}".format(type(exc).__name__, exc)
            selected_camera = None
            hik_adapter = None
            print(
                "HIK camera unavailable or occupied; continuing with Android only: {}"
                .format(hik_fallback_reason),
                flush=True,
            )
        else:
            hik = CalibratedHikFrameSource(
                rig_calibration,
                "hik_phone",
                rectify=True,
                output_quarter_turns_clockwise=0,
                reader=hik_reader,
            )
            print(
                "Matching HIK orientation from the first game ADB/HIK images...",
                flush=True,
            )
            try:
                orientation_match, orientation_images = (
                    orient_hik_source_from_first_adb_frame(
                        android,
                        hik,
                        rig_calibration,
                        android_reported_quarter_turns=surface[
                            "quarter_turns_clockwise_from_natural"
                        ],
                    )
                )
            except Exception:
                hik.stop()
                android.stop()
                raise
            selected_surface_turns = int(
                orientation_match[
                    "selected_adb_surface_quarter_turns_clockwise_from_phone_natural"
                ]
            )
            output_image_turns = int(
                orientation_match[
                    "selected_camera_adapter_image_quarter_turns_clockwise_from_calibration_display"
                ]
            )
            print(
                "ADB surface: {} degrees clockwise from phone-natural; calibrated "
                "HIK output rotation: {} degrees from rig-calibration display "
                "(image confidence {:.3f}, margin {}).".format(
                    selected_surface_turns * 90,
                    output_image_turns * 90,
                    float(orientation_match["selected_confidence"]),
                    (
                        "n/a"
                        if orientation_match["confidence_margin"] is None
                        else "{:.3f}".format(
                            float(orientation_match["confidence_margin"])
                        )
                    ),
                ),
                flush=True,
            )
            if orientation_match.get("warning"):
                print(
                    "Orientation warning: {}".format(orientation_match["warning"]),
                    flush=True,
                )
            aligned_surface = {
                **dict(surface),
                "quarter_turns_clockwise_from_natural": selected_surface_turns,
                "degrees_clockwise_from_natural": selected_surface_turns * 90,
                "source": "first_game_adb_and_hik_image_evidence",
                "android_reported_quarter_turns_clockwise_from_natural": surface[
                    "quarter_turns_clockwise_from_natural"
                ],
                "orientation_evidence": (
                    "cross_source_check/orientation_match/summary.json"
                ),
            }
            frame_processors.append(
                GameCrossSourceEvidenceRecorder(
                    rig_calibration,
                    aligned_surface,
                    orientation_match=orientation_match,
                    orientation_evidence_images=orientation_images,
                )
            )
    controller = ScrcpyTouchController(adb, server, serial, [width, height])
    game_context = _canonical_game_facts(
        game_launch, aligned_surface, arguments.game_id
    )
    android.set_game_context(game_context)
    if hik is not None:
        hik.set_game_context(game_context)
    control = _input_source(arguments, adb, serial, plan, controller=controller)
    frame_sources = [android] + ([hik] if hik is not None else [])
    image_sources = ["android_scrcpy"] + (
        ["hik_mvs_rig_rectified"] if hik is not None else []
    )
    hik_context = (
        {
            "mode": "rig_rectified_phone_view",
            "status": "active",
            "camera_id": selected_camera.device_id,
            "rig_calibration_used": True,
            "rig_calibration": str(rig_calibration.resolve()),
            "rig_profile_revision": calibration_revision,
            "calibration_selection": calibration_selection,
            "stream_id": "hik_phone",
            "output_image_quarter_turns_clockwise_from_calibration_display": (
                output_image_turns
            ),
            "output_orientation_selection": {
                "basis": orientation_match["selection_basis"],
                "status": orientation_match["status"],
                "confidence": orientation_match["selected_confidence"],
                "confidence_margin": orientation_match["confidence_margin"],
                "evidence": "cross_source_check/orientation_match/summary.json",
            },
            "android_reported_quarter_turns_clockwise_from_natural": surface[
                "quarter_turns_clockwise_from_natural"
            ],
        }
        if hik is not None
        else {
            "mode": "android_only",
            "status": "not_requested" if rig_calibration is None else "fallback",
            "reason": hik_fallback_reason,
            "rig_calibration_used": False,
        }
    )
    recorder = AcquisitionRecorder(
        pending_path,
        frame_sources,
        [control],
        video_fps=30.0,
        video_crf=16,
        frame_processors=frame_processors,
        session_context={
            "capture_kind": (
                "micro_movement_game_calibration_source_data"
                if arguments.capture_mode == "micro-movement"
                else "cursor_orbit_game_calibration_source_data"
                if arguments.capture_mode == "cursor-orbit"
                else "zigzag_minimap_source_data"
            ),
            "game_id": arguments.game_id,
            "image_sources": image_sources,
            "hik_capture": hik_context,
            "phone_surface_orientation": aligned_surface,
            "game_facts": _canonical_game_facts(
                game_launch, aligned_surface, arguments.game_id
            ),
            "phone_preparation": preparation,
            "game_launch": game_launch,
            (
                "micro_movement_plan"
                if arguments.capture_mode == "micro-movement"
                else "cursor_orbit_plan"
                if arguments.capture_mode == "cursor-orbit"
                else "zigzag_plan"
            ): plan.as_dict(),
            "calibration_status": "not_run",
        },
    )
    stop = threading.Event()
    completion_error = []

    def finish_after_control() -> None:
        timeout = max(30.0, plan.duration_seconds + 20.0)
        if not control.wait_completed(timeout):
            completion_error.append(
                "{} control did not complete within {:.1f} seconds".format(
                    arguments.capture_mode, timeout
                )
            )
        elif arguments.tail_seconds > 0:
            time.sleep(float(arguments.tail_seconds))
        stop.set()

    monitor = threading.Thread(
        target=finish_after_control,
        name="zigzag-capture-completion",
        daemon=True,
    )
    monitor.start()
    try:
        recorder.run(external_stop=stop)
        monitor.join(timeout=1)
        if completion_error:
            raise RuntimeError(completion_error[0])
        if control.error:
            raise RuntimeError(
                "Android {} control failed: {}".format(
                    arguments.capture_mode, control.error
                )
            )
        if not control.completed or control.events_issued != control.expected_event_count:
            raise RuntimeError(
                "Android {} control was incomplete: {}/{} events".format(
                    arguments.capture_mode,
                    control.events_issued,
                    control.expected_event_count,
                )
            )
        manifest = json.loads(
            (pending_path / "manifest.json").read_text(encoding="utf-8")
        )
        counts = manifest.get("frame_counts") or {}
        if (
            manifest.get("status") != "complete"
            or int(counts.get("android_phone", 0)) <= 0
            or (hik is not None and int(counts.get("hik_phone", 0)) <= 0)
        ):
            raise RuntimeError(
                "Recording did not complete with the requested frame streams"
            )
        if hik is not None:
            write_dual_source_space_yaml(
                pending_path,
                rig_calibration,
                aligned_surface,
                manifest,
            )
        else:
            write_android_source_space_yaml(
                pending_path,
                aligned_surface,
                manifest,
            )
        manifest["coordinate_spaces"] = "coordinate_spaces.yaml"
        manifest_path = pending_path / "manifest.json"
        temporary_manifest = pending_path / "manifest.json.tmp"
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        temporary_manifest.replace(manifest_path)
        pending_path.replace(session_path)
    except Exception:
        if pending_path.is_dir():
            shutil.rmtree(str(pending_path))
        raise
    unit = "swipe" if arguments.capture_mode == "zigzag" else "event"
    print(
        "Captured {}-{} {} with {}: {}".format(
            control.expected_event_count,
            unit,
            arguments.capture_mode,
            "ADB and HIK" if hik is not None else "ADB only",
            session_path.resolve(),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
