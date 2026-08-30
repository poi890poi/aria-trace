"""Record synchronized Android/rig-normalized HIK game frames during a zigzag."""

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
from .android_game_launcher import launch_android_game
from .android_zigzag import AndroidZigzagInputSource, ZigzagTouchPlan
from .dual_source_spaces import write_dual_source_space_yaml
from .game_cross_source_check import GameCrossSourceEvidenceRecorder
from .hik_capture import CalibratedHikFrameSource
from .recorder import AcquisitionRecorder
from .rig_calibration.hik.phone import (
    AdbPhoneSession,
    connected_adb_devices,
    resolve_adb_executable,
)
from .rig_calibration.hik.driver import HikMvsCameraAdapter, RectifiedHikCamera
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
    server: Path,
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


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Record Android and rig-normalized HIK streams during one continuous "
            "horizon-returning zigzag. No calibration is run or published."
        )
    )
    value.add_argument("--game-id", default="genshin-impact")
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
    value.add_argument("--phone-serial")
    value.add_argument("--adb")
    value.add_argument("--scrcpy-server", type=Path)
    value.add_argument("--ffmpeg", type=Path)
    value.add_argument(
        "--output-root", type=Path, default=Path("sessions") / "calibration"
    )
    value.add_argument("--moves", type=int, default=12)
    value.add_argument("--step-seconds", type=float, default=0.35)
    value.add_argument("--reset-seconds", type=float, default=0.10)
    value.add_argument("--settle-seconds", type=float, default=1.5)
    value.add_argument("--tail-seconds", type=float, default=1.5)
    value.add_argument("--yes", action="store_true")
    value.add_argument(
        "--analyze",
        action="store_true",
        help="run standalone mini-map analysis after successful capture",
    )
    value.add_argument(
        "--rig-calibration",
        type=Path,
        help=(
            "required calibration bundle directory or hik_camera_calibration.json; "
            "the HIK stream is rectified and oriented from this result"
        ),
    )
    value.add_argument("--profiles-root", type=Path, default=Path("profiles"))
    value.add_argument("--calibration-output", type=Path)
    value.add_argument("--android-crop")
    value.add_argument(
        "--hik-crop",
        help=(
            "legacy diagnostic override; automatic analysis derives the current "
            "HIK observation from Android and does not persist this crop"
        ),
    )
    value.add_argument(
        "--android-discovery",
        choices=("relative-prior", "unrestricted", "legacy-hint"),
        default="relative-prior",
    )
    value.add_argument(
        "--android-center-region",
        default="0,0,0.35,0.35",
        help="relative x0,y0,x1,y1 bounds for automatic Android circle centers",
    )
    value.add_argument(
        "--android-radius-fraction",
        default="0.07,0.22",
        help="minimum,maximum radius as fractions of the shorter Android dimension",
    )
    value.add_argument(
        "--android-min-visible",
        type=float,
        default=0.85,
        help="minimum visible circumference fraction for Android candidates",
    )
    value.add_argument(
        "--control-only",
        action="store_true",
        help="run only the Android camera zigzag; no capture or calibration",
    )
    return value


def _build_zigzag_plan(arguments, width: int, height: int) -> ZigzagTouchPlan:
    horizontal_distance = round(height * 0.225)
    vertical_distance = round(height * 0.45)
    # Start on Genshin's unobstructed look-control surface. The right-side
    # action cluster can consume DOWN before the camera handler sees it.
    start_x = round(width * 0.55)
    plan = ZigzagTouchPlan(
        start_xy=[start_x, round(height * 0.50)],
        end_x=start_x - horizontal_distance,
        vertical_amplitude_px=vertical_distance,
        move_count=arguments.moves,
        step_seconds=arguments.step_seconds,
        settle_seconds=arguments.settle_seconds,
        reset_seconds=arguments.reset_seconds,
    )
    plan.sampled_strokes()
    return plan


def _run_control_only(arguments) -> int:
    adb = resolve_adb_executable(arguments.adb)
    serial = _select_phone(adb, arguments.phone_serial)
    server = find_scrcpy_server(arguments.scrcpy_server)
    wake_surface = _phone_surface(adb, serial)
    phone = AdbPhoneSession(serial, adb_executable=adb)
    preparation = _wake_phone_for_preparation(
        phone, wake_surface["logical_size_px"]
    )
    game_launch = (
        {
            "game_id": arguments.game_id,
            "status": "disabled_by_user",
            "package": arguments.android_package,
        }
        if arguments.no_launch_game
        else launch_android_game(
            phone,
            arguments.game_id,
            explicit_package=arguments.android_package,
        )
    )
    time.sleep(0.75)
    surface = _phone_surface(adb, serial)
    width, height = surface["logical_size_px"]
    if width <= height:
        raise RuntimeError(
            "The game is not landscape yet ({}x{}); prepare it and retry".format(
                width, height
            )
        )
    plan = _build_zigzag_plan(arguments, width, height)
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
            "Confirm Genshin is visible in touchscreen mode, then press Enter "
            "to run the zigzag: "
        )
    preparation["game_booster_before_control"] = _dismiss_game_booster_lock(
        phone, adb, server, serial, [width, height]
    )
    print(
        "Running {} long strokes ({} up, {} down)...".format(
            len(plan.strokes()),
            sum(stroke["direction"] == "up" for stroke in plan.strokes()),
            sum(stroke["direction"] == "down" for stroke in plan.strokes()),
        )
    )

    controller = ScrcpyTouchController(adb, server, serial, [width, height])
    control = AndroidZigzagInputSource(adb, serial, plan, controller=controller)
    packets = []
    try:
        control.start(packets.append)
        timeout = max(20.0, plan.duration_seconds + 5.0)
        if not control.wait_completed(timeout):
            raise RuntimeError(
                "Zigzag control did not complete within {:.1f} seconds".format(
                    timeout
                )
            )
    finally:
        control.stop()
    if control.error:
        raise RuntimeError("Android zigzag control failed: {}".format(control.error))
    if not control.completed or control.events_issued != control.expected_event_count:
        raise RuntimeError(
            "Android zigzag control was incomplete: {}/{} events".format(
                control.events_issued, control.expected_event_count
            )
        )
    print(
        "Zigzag control completed: {}/{} touch events.".format(
            control.events_issued, control.expected_event_count
        )
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.control_only:
        return _run_control_only(arguments)
    if arguments.rig_calibration is None:
        parser().error("--rig-calibration is required for dual-source capture")
    rig_calibration = _resolve_rig_calibration(arguments.rig_calibration)
    rig_config = json.loads(rig_calibration.read_text(encoding="utf-8"))
    hik_adapter = HikMvsCameraAdapter(sdk_python_path=arguments.mvs_python_path)
    selected_camera = _select_camera(hik_adapter, arguments.camera_id)
    calibrated_camera_id = str(rig_config["camera"]["device_id"])
    if str(selected_camera.device_id) != calibrated_camera_id:
        raise RuntimeError(
            "Rig calibration is for HIK camera {}, but {} was selected".format(
                calibrated_camera_id, selected_camera.device_id
            )
        )
    adb = resolve_adb_executable(arguments.adb)
    serial = _select_phone(adb, arguments.phone_serial)
    calibrated_phone_id = str((rig_config.get("phone") or {}).get("serial") or "")
    if calibrated_phone_id and str(serial) != calibrated_phone_id:
        raise RuntimeError(
            "Rig calibration is for Android phone {}, but {} was selected".format(
                calibrated_phone_id, serial
            )
        )
    server = find_scrcpy_server(arguments.scrcpy_server)
    wake_surface = _phone_surface(adb, serial)
    wake_width, wake_height = wake_surface["logical_size_px"]
    phone = AdbPhoneSession(serial, adb_executable=adb)
    preparation = _wake_phone_for_preparation(phone, [wake_width, wake_height])
    game_launch = (
        {
            "game_id": arguments.game_id,
            "status": "disabled_by_user",
            "package": arguments.android_package,
            "calibration_controls_changed": False,
            "game_input_injected": False,
        }
        if arguments.no_launch_game
        else launch_android_game(
            phone,
            arguments.game_id,
            explicit_package=arguments.android_package,
        )
    )
    # Game launch may rotate the Android logical display. Control coordinates
    # must be probed after that transition, not inherited from the launcher.
    time.sleep(0.75)
    surface = _phone_surface(adb, serial)
    width, height = surface["logical_size_px"]
    preparation["game_booster_before_prompt"] = _dismiss_game_booster_lock(
        phone, adb, server, serial, [width, height]
    )
    plan = _build_zigzag_plan(arguments, width, height)
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
    if preparation["game_booster_before_prompt"]["dismissed"]:
        print("Samsung Game Booster touch protection dismissed.")
    if game_launch.get("package"):
        print(
            "Game: {} ({})".format(
                game_launch["package"], game_launch["status"]
            )
        )
    elif game_launch["status"] == "manual_unknown_game":
        print("Game package is unknown; launch it manually before confirming readiness.")
    print("Output session: {}".format(session_path.resolve()))
    print("This command records data only; it does not calibrate or publish profiles.")
    if not arguments.yes:
        input("Prepare the game, then press Enter when its unobstructed view is ready: ")
    preparation["game_booster_before_control"] = _dismiss_game_booster_lock(
        phone, adb, server, serial, [width, height]
    )

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
    cross_source_check = GameCrossSourceEvidenceRecorder(
        rig_calibration, surface
    )
    output_image_turns = cross_source_check.geometry[
        "output_image_quarter_turns_clockwise_from_phone_natural"
    ]
    hik = CalibratedHikFrameSource(
        rig_calibration,
        "hik_phone",
        rectify=True,
        output_quarter_turns_clockwise=output_image_turns,
        reader=RectifiedHikCamera(rig_calibration, adapter=hik_adapter),
    )
    controller = ScrcpyTouchController(adb, server, serial, [width, height])
    control = AndroidZigzagInputSource(adb, serial, plan, controller=controller)
    recorder = AcquisitionRecorder(
        pending_path,
        [android, hik],
        [control],
        video_fps=30.0,
        video_crf=16,
        frame_processors=[cross_source_check],
        session_context={
            "capture_kind": "zigzag_minimap_source_data",
            "game_id": arguments.game_id,
            "image_sources": ["android_scrcpy", "hik_mvs_rig_rectified"],
            "hik_capture": {
                "mode": "rig_rectified_phone_view",
                "camera_id": selected_camera.device_id,
                "rig_calibration_used": True,
                "rig_calibration": str(rig_calibration.resolve()),
                "stream_id": "hik_phone",
                "output_image_quarter_turns_clockwise_from_phone_natural": (
                    output_image_turns
                ),
                "android_surface_quarter_turns_clockwise_from_natural": surface[
                    "quarter_turns_clockwise_from_natural"
                ],
            },
            "phone_surface_orientation": surface,
            "phone_preparation": preparation,
            "game_launch": game_launch,
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
            or int(counts.get("hik_phone", 0)) <= 0
        ):
            raise RuntimeError("Recording did not complete with both frame streams")
        write_dual_source_space_yaml(
            pending_path,
            rig_calibration,
            surface,
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
    print(
        "Captured {}-event zigzag with both streams: {}".format(
            control.expected_event_count, session_path.resolve()
        )
    )
    if arguments.analyze:
        from .automated_minimap_calibration import (
            _parse_crop,
            android_discovery_config,
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
            android_discovery=android_discovery_config(
                arguments.android_discovery,
                arguments.android_center_region,
                arguments.android_radius_fraction,
                arguments.android_min_visible,
            ),
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
