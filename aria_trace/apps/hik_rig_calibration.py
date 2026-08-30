"""Standalone HIK/Android rig calibration command."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Sequence

from .driver import HikMvsCameraAdapter
from .phone import (
    AdbPhoneSession,
    connected_adb_devices,
    resolve_adb_executable,
)
from .workflow import (
    HikCalibrationOptions,
    HikRigCalibrationSession,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Calibrate a HIK MVS camera against one explicitly selected Android phone."
    )
    value.add_argument("--camera-id", help="HIK serial or enumeration index")
    value.add_argument(
        "--phone-serial",
        help="ADB device serial; auto-selects only when exactly one is connected",
    )
    value.add_argument("--output", type=Path, help="New calibration bundle directory")
    value.add_argument("--adb", help="ADB executable; normally auto-detected")
    value.add_argument("--mvs-python-path")
    value.add_argument("--profile-root", type=Path)
    value.add_argument(
        "--no-profile",
        action="store_true",
        help="save the calibration bundle without publishing its active rig profile",
    )
    value.add_argument("--list-cameras", action="store_true")
    value.add_argument("--camera-width", type=int, default=2448)
    value.add_argument("--camera-height", type=int, default=2048)
    value.add_argument("--camera-fps", type=float, default=30.0)
    value.add_argument("--refresh-hz", type=float)
    value.add_argument(
        "--max-shutter-multiplier",
        type=int,
        choices=(2, 3),
        default=2,
        help="Fastest shutter-rate multiple of panel refresh",
    )
    value.add_argument(
        "--max-exposure-periods",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help="Longest HIK auto exposure in complete panel refresh periods (default: 1)",
    )
    value.add_argument(
        "--max-auto-gain-db",
        type=float,
        default=12.0,
        help="HIK one-shot auto-gain upper limit in dB (default: 12)",
    )
    value.add_argument("--exposure-noise-frames", type=int, default=4)
    value.add_argument("--target-port", type=int, default=8765)
    value.add_argument(
        "--display-component",
        help="Android image viewer component; auto-detects the built-in Gallery/Display app",
    )
    value.add_argument("--operation-timeout-seconds", type=float, default=8.0)
    value.add_argument("--geometry-frames", type=int, default=12)
    value.add_argument(
        "--visible-screen-margin-px",
        type=int,
        default=8,
        help=(
            "Outward display-space margin around all camera-visible phone pixels "
            "for normalization and hardware ROI coverage (default: 8)"
        ),
    )
    value.add_argument("--settle-frames", type=int, default=3)
    value.add_argument(
        "--strict-display-screenshot-verification",
        action="store_true",
        help=(
            "testing only: fail when the ADB screenshot probe cannot prove target "
            "presentation; product runs keep it as diagnostic evidence"
        ),
    )
    value.add_argument("--headless", action="store_true", help="Skip interactive focus UI")
    value.add_argument(
        "--test",
        action="store_true",
        help="Run a no-save headless observation test",
    )
    value.add_argument("--save", action="store_true", help="Save without the GUI S hotkey")
    value.add_argument(
        "--grade-data-matrix",
        "--test-data-matrix-decode",
        dest="grade_data_matrix",
        action="store_true",
        help="Run the batched exact-payload Data Matrix decode test",
    )
    value.add_argument(
        "--data-matrix-trials",
        type=int,
        default=40,
        help="Patterns per module size (minimum 20, default 40)",
    )
    value.add_argument("--data-matrix-initial-module-px", type=int, default=1)
    value.add_argument(
        "--complete-grader-plugin",
        help="Optional module:callable implementing an external complete symbol verifier",
    )
    return value


def _select(label: str, rows, description):
    values = list(rows)
    if not values:
        raise RuntimeError("No connected {} was found".format(label))
    if len(values) == 1:
        return values[0]
    print("Multiple {} devices were found:".format(label))
    for index, item in enumerate(values, 1):
        print("  {}) {}".format(index, description(item)))
    while True:
        answer = input("Select {} [1-{}]: ".format(label, len(values))).strip()
        if answer.isdigit() and 1 <= int(answer) <= len(values):
            return values[int(answer) - 1]


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    camera = HikMvsCameraAdapter(sdk_python_path=arguments.mvs_python_path)
    if arguments.list_cameras:
        for device in camera.devices(probe=True):
            print("{}\t{}\t{}".format(device.device_id, device.label, dict(device.metadata)))
        return 0
    if arguments.camera_id is None:
        selected_camera = _select(
            "HIK camera",
            camera.devices(probe=True),
            lambda item: "{} ({})".format(item.label, item.device_id),
        )
        arguments.camera_id = selected_camera.device_id
        print("Selected HIK camera: {}".format(selected_camera.label))
    adb = resolve_adb_executable(arguments.adb)
    if arguments.phone_serial is None:
        arguments.phone_serial = _select(
            "Android phone",
            connected_adb_devices(adb),
            lambda item: item,
        )
        print("Selected Android phone: {}".format(arguments.phone_serial))
    if arguments.output is None:
        arguments.output = Path("artifacts") / "hik-calibration-{}".format(
            time.strftime("%Y%m%d-%H%M%S")
        )
    if arguments.test:
        arguments.headless = True
        arguments.save = False
        print("Mode: observation test; no calibration bundle will be saved.")
    else:
        print("Output: {}".format(arguments.output.resolve()))
    options = HikCalibrationOptions(
        camera_id=arguments.camera_id,
        phone_serial=arguments.phone_serial,
        output_directory=arguments.output,
        camera_width_px=arguments.camera_width,
        camera_height_px=arguments.camera_height,
        camera_fps=arguments.camera_fps,
        target_port=arguments.target_port,
        display_component=arguments.display_component,
        operation_timeout_seconds=arguments.operation_timeout_seconds,
        refresh_hz_override=arguments.refresh_hz,
        maximum_shutter_multiplier=arguments.max_shutter_multiplier,
        maximum_exposure_periods=arguments.max_exposure_periods,
        maximum_auto_gain_db=arguments.max_auto_gain_db,
        exposure_noise_frames=arguments.exposure_noise_frames,
        geometry_frames=arguments.geometry_frames,
        visible_screen_margin_px=arguments.visible_screen_margin_px,
        settle_frames=arguments.settle_frames,
        headless=arguments.headless,
        save_without_prompt=arguments.save,
        grade_data_matrix=arguments.grade_data_matrix,
        data_matrix_trials_per_size=arguments.data_matrix_trials,
        data_matrix_initial_module_px=arguments.data_matrix_initial_module_px,
        complete_grader_plugin=arguments.complete_grader_plugin,
        strict_display_screenshot_verification=(
            arguments.strict_display_screenshot_verification
        ),
    )
    phone = AdbPhoneSession(arguments.phone_serial, adb_executable=adb)
    result = HikRigCalibrationSession(options, camera=camera, phone=phone).run()
    if result is None:
        print("Calibration ended without saving.")
    elif not arguments.no_profile:
        from aria_trace.workflows.profile_management import publish_rig_calibration

        profile = publish_rig_calibration(
            result, profile_root=arguments.profile_root, activate=True
        )
        print(
            "Active rig profile: {} ({})".format(
                profile["revision_id"], profile["publication"]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
