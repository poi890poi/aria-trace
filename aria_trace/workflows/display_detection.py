"""Detect compositor-visible but physically dim Android game display and wake it."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from aria_trace.adapters.android.capture import find_scrcpy_server
from aria_trace.adapters.filesystem.profile_registry import ProfileContext, ProfileRegistry
from aria_trace.services.calibration.rig.hik.driver import HikMvsCameraAdapter, RectifiedHikCamera
from aria_trace.services.calibration.rig.hik.phone import AdbPhoneSession, resolve_adb_executable
from aria_trace.adapters.android.control import ScrcpyTouchController


def _calibration_file(value: Path) -> Path:
    path = Path(value)
    if path.is_dir():
        path = path / "hik_camera_calibration.json"
    if not path.is_file():
        raise FileNotFoundError("HIK calibration does not exist: {}".format(path))
    return path.resolve()


def _active_rig_calibration(
    profile_root: Optional[Path],
    *,
    camera_id: Optional[str] = None,
    phone_serial: Optional[str] = None,
    mvs_python_path: Optional[str] = None,
) -> Path:
    selected_camera = camera_id
    if not selected_camera:
        devices = list(
            HikMvsCameraAdapter(sdk_python_path=mvs_python_path).devices(probe=True)
        )
        if not devices:
            raise RuntimeError("No connected HIK camera was found")
        if len(devices) != 1:
            raise RuntimeError(
                "Multiple HIK cameras are connected; pass --camera-id: {}".format(
                    ", ".join(str(device.device_id) for device in devices)
                )
            )
        selected_camera = str(devices[0].device_id)
    registry = ProfileRegistry(profile_root)
    profile = registry.resolve(
        "rig",
        ProfileContext(camera_id=str(selected_camera), phone_id=phone_serial),
    )
    return registry.runtime_file(profile, "hik_camera_calibration").resolve()


def _screenshot(adb: str, serial: str) -> np.ndarray:
    encoded = subprocess.check_output(
        [adb, "-s", serial, "exec-out", "screencap", "-p"], timeout=10
    )
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("ADB returned an invalid screenshot")
    return image


def _frame_stats(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return {
        "mean_dn": float(np.mean(gray)),
        "p95_dn": float(np.percentile(gray, 95.0)),
        "p99_dn": float(np.percentile(gray, 99.0)),
        "p99_5_dn": float(np.percentile(gray, 99.5)),
        "bright_pixel_fraction": float(np.mean(gray >= 200)),
    }


def _camera_sample(camera: RectifiedHikCamera, count: int = 8) -> Tuple[dict, np.ndarray]:
    frames = []
    for _index in range(max(3, int(count))):
        ok, frame = camera.read()
        if not ok or frame is None:
            raise RuntimeError("HIK camera did not return a frame")
        frames.append(frame)
    statistics = [_frame_stats(frame) for frame in frames]
    return (
        {
            key: float(np.median([item[key] for item in statistics]))
            for key in statistics[0]
        },
        frames[-1],
    )


def _foreground(phone: AdbPhoneSession) -> Optional[str]:
    text = phone.shell("dumpsys", "activity", "activities")
    import re

    match = re.search(
        r"(?:mResumedActivity:|topResumedActivity=).*?\s"
        r"([A-Za-z0-9_.]+/[A-Za-z0-9_.$]+)",
        text,
    )
    return match.group(1) if match else None


def _game_display_ready(android: dict, hik: dict) -> bool:
    # The game always contributes bright UI pixels even in a dark scene.  This
    # gate is intentionally independent of the white calibration target.
    compositor_ready = bool(
        android["p99_dn"] >= 80.0
        and android["bright_pixel_fraction"] >= 0.0005
    )
    optical_ready = bool(hik["p99_dn"] >= 60.0)
    return compositor_ready and optical_ready


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Detect a dim Android game using paired ADB/HIK evidence, issue a "
            "non-toggling wake event and, if needed, the Samsung unlock drag."
        )
    )
    value.add_argument("--profile-root", type=Path)
    value.add_argument("--camera-id")
    value.add_argument("--mvs-python-path")
    value.add_argument(
        "--diagnostic-calibration-override",
        type=Path,
        help="explicit file for diagnostics; production uses the active registry profile",
    )
    value.add_argument("--adb")
    value.add_argument("--phone-serial")
    value.add_argument("--scrcpy-server", type=Path)
    value.add_argument("--settle-seconds", type=float, default=1.25)
    value.add_argument("--output-root", type=Path, default=Path("artifacts"))
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    calibration_file = (
        _calibration_file(arguments.diagnostic_calibration_override)
        if arguments.diagnostic_calibration_override is not None
        else _active_rig_calibration(
            arguments.profile_root,
            camera_id=arguments.camera_id,
            phone_serial=arguments.phone_serial,
            mvs_python_path=arguments.mvs_python_path,
        )
    )
    calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
    serial = str(
        arguments.phone_serial or (calibration.get("phone") or {}).get("serial") or ""
    ).strip()
    if not serial:
        raise RuntimeError("No phone serial is saved; pass --phone-serial")
    adb = resolve_adb_executable(arguments.adb)
    scrcpy_server = find_scrcpy_server(arguments.scrcpy_server)
    output = arguments.output_root / "game-display-wake-{}".format(
        datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output.mkdir(parents=True, exist_ok=False)

    phone = AdbPhoneSession(serial, adb_executable=adb)
    camera = RectifiedHikCamera(calibration_file, rectify=False)
    result = {
        "calibration": str(calibration_file),
        "calibration_selection": (
            "diagnostic_explicit_path"
            if arguments.diagnostic_calibration_override is not None
            else "active_profile_registry"
        ),
        "phone_serial": serial,
        "camera_controls": "saved calibration; exposure/gain/WB are not adjusted",
        "android_mutation": (
            "KEYCODE_WAKEUP, then one center unlock drag only if paired evidence "
            "still shows a dim game"
        ),
        "evidence_directory": str(output.resolve()),
    }
    try:
        result["display_state_before"] = phone.display_state()
        result["foreground_before"] = _foreground(phone)
        android = _screenshot(adb, serial)
        result["android_compositor"] = _frame_stats(android)
        cv2.imwrite(str(output / "android_compositor.png"), android)

        camera.open()
        before, before_frame = _camera_sample(camera)
        result["hik_before"] = before
        cv2.imwrite(str(output / "hik_before.png"), before_frame)
        # Some devices apply an inactivity dim layer to the composed game, so
        # both screencap and the real panel become dark while power remains ON.
        # KEYCODE_WAKEUP is non-toggling and safe to probe in either dim form.
        dim_detected = bool(not _game_display_ready(result["android_compositor"], before))
        result["dim_detected"] = dim_detected
        adb_dim = result["android_compositor"]["p99_dn"] < 80.0
        hik_dim = before["p99_dn"] < 60.0
        if adb_dim and hik_dim:
            result["detection_basis"] = "adb_and_optical_game_views_dim"
        elif adb_dim:
            result["detection_basis"] = "adb_game_view_dim"
        elif hik_dim:
            result["detection_basis"] = "optical_game_view_dim"
        else:
            result["detection_basis"] = "paired_game_views_ready"

        if not dim_detected:
            result["status"] = "normal"
        else:
            result["actions"] = ["KEYCODE_WAKEUP"]
            phone.shell("input", "keyevent", "KEYCODE_WAKEUP")
            time.sleep(max(0.25, float(arguments.settle_seconds)))
            android_after = _screenshot(adb, serial)
            after, after_frame = _camera_sample(camera)
            if not _game_display_ready(_frame_stats(android_after), after):
                screen_height, screen_width = android_after.shape[:2]
                with ScrcpyTouchController(
                    adb,
                    scrcpy_server,
                    serial,
                    [screen_width, screen_height],
                ) as controller:
                    start = [round(screen_width * 0.50), round(screen_height * 0.58)]
                    points = [
                        start,
                        [start[0], round(screen_height * 0.46)],
                        [start[0], round(screen_height * 0.33)],
                        [start[0], round(screen_height * 0.20)],
                    ]
                    controller.inject_touch("DOWN", points[0])
                    for point in points[1:]:
                        time.sleep(0.12)
                        controller.inject_touch("MOVE", point)
                    time.sleep(0.12)
                    controller.inject_touch("UP", points[-1])
                    result["unlock_controller"] = controller.describe()
                result["actions"].append("center_unlock_drag")
                time.sleep(max(0.25, float(arguments.settle_seconds)))
                android_after = _screenshot(adb, serial)
                after, after_frame = _camera_sample(camera)
            result["android_compositor_after"] = _frame_stats(android_after)
            result["hik_after"] = after
            result["display_state_after"] = phone.display_state()
            result["foreground_after"] = _foreground(phone)
            cv2.imwrite(str(output / "android_compositor_after.png"), android_after)
            cv2.imwrite(str(output / "hik_after.png"), after_frame)
            foreground_unchanged = bool(
                not result["foreground_before"]
                or result["foreground_before"] == result["foreground_after"]
            )
            result["foreground_unchanged"] = foreground_unchanged
            result["p99_improvement_ratio"] = float(
                after["p99_dn"] / max(before["p99_dn"], 1.0)
            )
            result["status"] = (
                "resolved"
                if _game_display_ready(result["android_compositor_after"], after)
                and foreground_unchanged
                else "unresolved"
            )
    finally:
        camera.release()

    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Display state: {}".format(result.get("display_state_before", "unknown")))
    print(
        "ADB p99: {:.1f}; HIK p99 before: {:.1f}.".format(
            result["android_compositor"]["p99_dn"], result["hik_before"]["p99_dn"]
        )
    )
    if result.get("hik_after"):
        print(
            "ADB p99 after: {:.1f}; HIK p99 after: {:.1f}.".format(
                result["android_compositor_after"]["p99_dn"],
                result["hik_after"]["p99_dn"],
            )
        )
    print("Result: {}".format(result["status"]))
    print("Evidence: {}".format(output.resolve()))
    return 0 if result["status"] in ("normal", "resolved") else 3


if __name__ == "__main__":
    raise SystemExit(main())
