"""Record synchronized Android/native-HIK frames during one controlled zigzag.

Acquisition is always native full-sensor HIK and has no rig dependency.  The
optional post-capture analysis is a separate layer invoked only after the
session has finalized successfully.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

from .android_capture import (
    AndroidRoiFrameSource,
    AndroidRoiSpec,
    ScrcpyCaptureHub,
    find_scrcpy_server,
)
from .android_zigzag import AndroidZigzagInputSource, ZigzagTouchPlan
from .hik_capture import NativeHikFrameSource
from .recorder import AcquisitionRecorder
from .rig_calibration.hik.phone import (
    AdbPhoneSession,
    connected_adb_devices,
    resolve_adb_executable,
)
from .rig_calibration.hik.driver import HikMvsCameraAdapter
from .scrcpy_control import ScrcpyTouchController
from .sources import AdbClockMapper


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
    metrics = phone.metrics()
    quarter_turns = int(metrics.orientation_quarter_turns)
    natural = list(map(int, metrics.natural_screen_size_px))
    logical = list(map(int, metrics.screen_size_px))
    return {
        "quarter_turns_clockwise_from_natural": quarter_turns,
        "degrees_clockwise_from_natural": quarter_turns * 90,
        "logical_size_px": logical,
        "natural_size_px": natural,
        "source": "adb_surface_orientation_at_capture",
    }


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


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Record Android and native full-view HIK streams during one continuous "
            "horizon-returning zigzag. No calibration is run or published."
        )
    )
    value.add_argument("--game-id", default="genshin-impact")
    value.add_argument("--camera-id")
    value.add_argument("--camera-width", type=int, default=2448)
    value.add_argument("--camera-height", type=int, default=2048)
    value.add_argument("--camera-fps", type=float, default=30.0)
    value.add_argument("--mvs-python-path")
    value.add_argument("--phone-serial")
    value.add_argument("--adb")
    value.add_argument("--scrcpy-server", type=Path)
    value.add_argument("--ffmpeg", type=Path)
    value.add_argument(
        "--output-root", type=Path, default=Path("sessions") / "calibration"
    )
    value.add_argument("--moves", type=int, default=24)
    value.add_argument("--step-seconds", type=float, default=0.12)
    value.add_argument("--settle-seconds", type=float, default=1.5)
    value.add_argument("--tail-seconds", type=float, default=1.5)
    value.add_argument("--yes", action="store_true")
    value.add_argument(
        "--analyze",
        action="store_true",
        help="run standalone mini-map analysis after successful capture",
    )
    value.add_argument("--rig-calibration", type=Path)
    value.add_argument("--profiles-root", type=Path, default=Path("profiles"))
    value.add_argument("--calibration-output", type=Path)
    value.add_argument("--android-crop")
    value.add_argument("--hik-crop")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    hik_adapter = HikMvsCameraAdapter(sdk_python_path=arguments.mvs_python_path)
    selected_camera = _select_camera(hik_adapter, arguments.camera_id)
    adb = resolve_adb_executable(arguments.adb)
    serial = _select_phone(adb, arguments.phone_serial)
    server = find_scrcpy_server(arguments.scrcpy_server)
    surface = _phone_surface(adb, serial)
    width, height = surface["logical_size_px"]
    phone = AdbPhoneSession(serial, adb_executable=adb)
    preparation = _wake_phone_for_preparation(phone, [width, height])
    plan = ZigzagTouchPlan(
        start_xy=[round(width * 0.82), round(height * 0.50)],
        end_x=round(width * 0.18),
        vertical_amplitude_px=round(height * 0.16),
        move_count=arguments.moves,
        step_seconds=arguments.step_seconds,
        settle_seconds=arguments.settle_seconds,
    )
    plan.points()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_path = arguments.output_root / "{}-{}-zigzag".format(
        arguments.game_id, timestamp
    )
    pending_path = session_path.with_name(session_path.name + ".pending")
    print("HIK camera: {} ({})".format(selected_camera.label, selected_camera.device_id))
    print("Phone: {} ({}x{})".format(serial, width, height))
    print("Phone display awakened electronically through ADB; the physical power button is not used.")
    if preparation.get("keyguard_after") is True:
        print("The credential keyguard is still present; unlock it without moving the rig.")
    print("Output session: {}".format(session_path.resolve()))
    print("This command records data only; it does not calibrate or publish profiles.")
    if not arguments.yes:
        input("Prepare the game, then press Enter when its unobstructed view is ready: ")

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
        hub, AndroidRoiSpec("android_phone", 0, 0, 0, 0)
    )
    hik = NativeHikFrameSource(
        selected_camera.device_id,
        "hik_full",
        width_px=arguments.camera_width,
        height_px=arguments.camera_height,
        fps=arguments.camera_fps,
        adapter=hik_adapter,
    )
    controller = ScrcpyTouchController(adb, server, serial, [width, height])
    control = AndroidZigzagInputSource(adb, serial, plan, controller=controller)
    recorder = AcquisitionRecorder(
        pending_path,
        [android, hik],
        [control],
        video_fps=30.0,
        video_crf=16,
        session_context={
            "capture_kind": "zigzag_minimap_source_data",
            "game_id": arguments.game_id,
            "image_sources": ["android_scrcpy", "hik_mvs"],
            "hik_capture": {
                "mode": "native_full_sensor",
                "camera_id": selected_camera.device_id,
                "rig_calibration_used": False,
            },
            "phone_surface_orientation": surface,
            "phone_preparation": preparation,
            "zigzag_plan": plan.as_dict(),
            "calibration_status": "not_run",
        },
    )
    stop = threading.Event()
    completion_error = []

    def finish_after_control() -> None:
        timeout = max(30.0, plan.duration_seconds + 20.0)
        if not control.wait_completed(timeout):
            completion_error.append(
                "Zigzag control did not complete within {:.1f} seconds".format(timeout)
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
            raise RuntimeError("Android zigzag control failed: {}".format(control.error))
        if not control.completed or control.events_issued != control.expected_event_count:
            raise RuntimeError(
                "Android zigzag control was incomplete: {}/{} events".format(
                    control.events_issued, control.expected_event_count
                )
            )
        manifest = json.loads(
            (pending_path / "manifest.json").read_text(encoding="utf-8")
        )
        counts = manifest.get("frame_counts") or {}
        if (
            manifest.get("status") != "complete"
            or int(counts.get("android_phone", 0)) <= 0
            or int(counts.get("hik_full", 0)) <= 0
        ):
            raise RuntimeError("Recording did not complete with both frame streams")
        pending_path.replace(session_path)
    except Exception:
        if pending_path.is_dir():
            shutil.rmtree(str(pending_path))
        raise
    print(
        "Captured {}-event zigzag with both streams: {}".format(
            control.expected_event_count, session_path.resolve()
        )
    )
    if arguments.analyze:
        from .automated_minimap_calibration import (
            _parse_crop,
            calibrate_zigzag_session,
        )

        calibration_output = arguments.calibration_output or (
            Path("artifacts")
            / "game-minimap-calibration-{}".format(timestamp)
        )
        result = calibrate_zigzag_session(
            session_path,
            calibration_output,
            profiles_root=arguments.profiles_root,
            rig_calibration=arguments.rig_calibration,
            android_selected_crop_xywh=_parse_crop(arguments.android_crop),
            hik_selected_crop_xywh=_parse_crop(arguments.hik_crop),
        )
        print("Mini-map calibration: {}".format(calibration_output.resolve()))
        print("Phone-game profile: {}".format(result["summary"]["phone_game_profile"]))
        if result["summary"]["rig_game_profile"]:
            print("Rig-game profile: {}".format(result["summary"]["rig_game_profile"]))
        else:
            print("Rig-game profile: skipped (no optional rig calibration supplied)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
