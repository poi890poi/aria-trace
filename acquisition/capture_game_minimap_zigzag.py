"""Record synchronized Android/HIK frames during one controlled zigzag.

This module is deliberately limited to acquisition.  It does not select a
mini-map, estimate orientation, create calibration profiles, or publish a
calibration result.
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
from .hik_capture import CalibratedHikFrameSource
from .recorder import AcquisitionRecorder
from .rig_calibration.hik.phone import (
    AdbPhoneSession,
    connected_adb_devices,
    resolve_adb_executable,
)
from .scrcpy_control import ScrcpyTouchController
from .sources import AdbClockMapper


def _calibration_file(value: Path) -> Path:
    path = Path(value)
    if path.is_dir():
        path = path / "hik_camera_calibration.json"
    if not path.is_file():
        raise FileNotFoundError("HIK rig calibration does not exist: {}".format(path))
    return path.resolve()


def _select_phone(adb: Path, configured: Optional[str], rig: dict) -> str:
    if configured:
        return str(configured)
    calibrated = str((rig.get("phone") or {}).get("serial") or "")
    devices = list(connected_adb_devices(adb))
    if calibrated and calibrated in devices:
        return calibrated
    if len(devices) == 1:
        return devices[0]
    raise RuntimeError(
        "Pass --phone-serial when the calibrated phone cannot be selected uniquely"
    )


def _phone_surface(adb: Path, serial: str, rig: dict) -> dict:
    phone = AdbPhoneSession(serial, adb_executable=adb)
    quarter_turns = None
    for command, pattern in (
        (("dumpsys", "input"), r"SurfaceOrientation:\s*([0-3])"),
        (("dumpsys", "display"), r"mCurrentOrientation=([0-3])"),
    ):
        try:
            match = re.search(pattern, phone.shell(*command))
            if match:
                quarter_turns = int(match.group(1))
                break
        except Exception:
            pass
    if quarter_turns is None:
        raise RuntimeError("Android did not report its current surface orientation")
    natural = list(
        map(
            int,
            (rig.get("phone") or {}).get("natural_screen_size_px")
            or (rig.get("phone") or {}).get("screen_size_px"),
        )
    )
    logical = list(natural)
    if quarter_turns % 2:
        logical.reverse()
    return {
        "quarter_turns_clockwise_from_natural": quarter_turns,
        "degrees_clockwise_from_natural": quarter_turns * 90,
        "logical_size_px": logical,
        "natural_size_px": natural,
        "source": "adb_surface_orientation_at_capture",
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Record Android and calibrated HIK streams during one continuous "
            "horizon-returning zigzag. No calibration is run or published."
        )
    )
    value.add_argument("rig_calibration", type=Path)
    value.add_argument("--game-id", default="genshin-impact")
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
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    rig_file = _calibration_file(arguments.rig_calibration)
    rig = json.loads(rig_file.read_text(encoding="utf-8"))
    adb = resolve_adb_executable(arguments.adb)
    serial = _select_phone(adb, arguments.phone_serial, rig)
    server = find_scrcpy_server(arguments.scrcpy_server)
    surface = _phone_surface(adb, serial, rig)
    width, height = surface["logical_size_px"]
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
    print("Phone: {} ({}x{})".format(serial, width, height))
    print("HIK calibration: {}".format(rig_file))
    print("Output session: {}".format(session_path.resolve()))
    print("This command records data only; it does not calibrate or publish profiles.")
    if not arguments.yes:
        input("Confirm the game is awake and unobstructed, then press Enter: ")

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
    hik = CalibratedHikFrameSource(rig_file, "hik_phone", rectify=True)
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
            "rig_calibration": str(rig_file),
            "phone_surface_orientation": surface,
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
        pending_path.replace(session_path)
    except Exception:
        if pending_path.is_dir():
            shutil.rmtree(str(pending_path))
        raise
    print("Captured 26-event zigzag with both streams: {}".format(session_path.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
