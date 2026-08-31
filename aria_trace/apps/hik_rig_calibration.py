"""Standalone HIK/Android rig calibration command."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Sequence

from aria_trace.adapters.hik.driver import HikMvsCameraAdapter
from aria_trace.adapters.android.phone import (
    AdbPhoneSession,
    connected_adb_devices,
    resolve_adb_executable,
)
from aria_trace.workflows.hik_rig_calibration import (
    HikCalibrationOptions,
    HikRigCalibrationSession,
)
from aria_trace.adapters.filesystem.profile_registry import default_profile_root
from aria_trace.adapters.filesystem.system_configuration import (
    load_system_configuration,
    resolve_rig_repeatability_policy,
)


def _write_standalone_adapter(
    calibration: Path, output: Path, profile_root: Optional[Path]
) -> None:
    from aria_trace.adapters.filesystem.profile_registry import (
        AdapterRequest,
        ProfileRegistry,
        context_from_rig_calibration,
    )
    from aria_trace.workflows.adapter_export import export_resolved_adapter

    calibration = Path(calibration)
    if calibration.is_dir():
        calibration = calibration / "hik_camera_calibration.json"
    document = json.loads(calibration.read_text(encoding="utf-8"))
    result = export_resolved_adapter(
        output,
        registry=ProfileRegistry(profile_root),
        context=context_from_rig_calibration(document),
        request=AdapterRequest(mode="full", color_policy="rig_locked"),
    )
    print("Standalone camera adapter: {}".format(result["output"]))


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
        "--reuse-if-unchanged",
        action="store_true",
        help=(
            "check the active registry rig first and skip full calibration only "
            "when the saved ChArUco snapshot is unchanged"
        ),
    )
    value.add_argument(
        "--reuse-evidence-output",
        type=Path,
        help="precheck evidence directory; defaults beside --output",
    )
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
    settings = load_system_configuration(arguments.profile_root)
    repeatability = resolve_rig_repeatability_policy(settings)
    arguments.camera_id = arguments.camera_id or settings["devices"].get(
        "camera_id"
    )
    arguments.phone_serial = arguments.phone_serial or settings["devices"].get(
        "phone_id"
    )
    arguments.adb = arguments.adb or settings["tools"].get("adb")
    arguments.mvs_python_path = (
        arguments.mvs_python_path or settings["tools"].get("mvs_python_path")
    )
    profile_root = default_profile_root(arguments.profile_root)
    print(
        "Profile root: {} ({})".format(
            profile_root, settings["profile_root_source"]
        )
    )
    print(
        "Rig repeatability policy: {} (reuse correlation >= {:.3f}, MAE <= {:.1f} DN; "
        "save is blocked after displacement > {:.1f} px for {} consecutive frames)."
        .format(
            repeatability["name"],
            repeatability["reuse_minimum_correlation"],
            repeatability["reuse_maximum_mae_dn"],
            repeatability["save_max_displacement_px"],
            repeatability["save_movement_consecutive_frames"],
        )
    )
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
        arguments.output = profile_root / "calibrations" / (
            "hik-calibration-{}".format(time.strftime("%Y%m%d-%H%M%S"))
        )
    if arguments.test:
        arguments.headless = True
        arguments.save = False
        print("Mode: observation test; no calibration bundle will be saved.")
    else:
        print("Output: {}".format(arguments.output.resolve()))
    if arguments.reuse_if_unchanged and not arguments.test:
        from aria_trace.workflows.rig_reuse_precheck import (
            format_reuse_precheck_failure,
            run_active_reuse_precheck,
        )

        precheck_output = arguments.reuse_evidence_output or Path(
            "{}-precheck".format(arguments.output)
        )
        print("Checking active rig profile before full calibration...")
        precheck = run_active_reuse_precheck(
            precheck_output,
            profile_root=profile_root,
            adb=str(adb),
            mvs_python_path=arguments.mvs_python_path,
            camera_id=arguments.camera_id,
            phone_serial=arguments.phone_serial,
            minimum_correlation=repeatability["reuse_minimum_correlation"],
            maximum_mae_dn=repeatability["reuse_maximum_mae_dn"],
            sample_frames=repeatability["reuse_sample_frames"],
        )
        if (
            precheck.get("reusable")
            and precheck.get("camera_adapter_is_calibrated")
        ):
            arguments.output.mkdir(parents=True, exist_ok=False)
            receipt = {
                "schema_version": "1.0",
                "status": "reused",
                "calibration": precheck["calibration"],
                "selection": "active_profile_registry",
                "precheck": str((precheck_output / "precheck.json").resolve()),
                "comparison": precheck.get("comparison"),
            }
            (arguments.output / "reused_calibration.json").write_text(
                json.dumps(receipt, indent=2), encoding="utf-8"
            )
            print(
                "Saved rig calibration is unchanged; full calibration skipped."
            )
            print("Calibration: {}".format(precheck["calibration"]))
            _write_standalone_adapter(
                Path(str(precheck["calibration"])),
                arguments.output / "hikcam_adapter.py",
                profile_root,
            )
            return 0
        print(format_reuse_precheck_failure(precheck))
        print("Full rig calibration will start now.")
        print(
            "Repeatability evidence: {}".format(
                (precheck_output / "precheck.json").resolve()
            )
        )
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
        repeatability_policy=repeatability["name"],
        save_max_displacement_px=repeatability["save_max_displacement_px"],
        save_movement_consecutive_frames=repeatability[
            "save_movement_consecutive_frames"
        ],
    )
    phone = AdbPhoneSession(arguments.phone_serial, adb_executable=adb)
    result = HikRigCalibrationSession(options, camera=camera, phone=phone).run()
    if result is None:
        print("Calibration ended without saving.")
    elif not arguments.no_profile:
        from aria_trace.workflows.profile_management import publish_rig_calibration

        profile = publish_rig_calibration(
            result, profile_root=profile_root, activate=True
        )
        print(
            "Active rig profile: {} ({})".format(
                profile["revision_id"], profile["publication"]
            )
        )
        _write_standalone_adapter(
            Path(result),
            Path(result) / "hikcam_adapter.py",
            profile_root,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
