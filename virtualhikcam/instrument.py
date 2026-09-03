"""Repeatable real-device instrument procedure for the virtual rig driver.

The procedure drives the production rig-calibration entry point through its
public camera-plugin interface.  It contains no calibration shortcut or
virtual-camera branch.  Every stage gets isolated profiles, logs, timings, and
native rig/precheck evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence
from urllib.parse import urlencode

from aria_trace.adapters.rig.devices import CameraConfiguration

from .driver import VirtualHikCameraAdapter


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(temporary), str(path))


def _run_logged(
    name: str,
    command: Sequence[str],
    *,
    output_root: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    stage_root = output_root / "stages" / name
    stage_root.mkdir(parents=True, exist_ok=False)
    log_path = stage_root / "console.log"
    print("\n=== {} ===".format(name))
    print("Command: {}".format(subprocess.list2cmdline(list(command))))
    started = time.perf_counter_ns()
    lines = []
    process = subprocess.Popen(
        list(command),
        cwd=str(Path(__file__).resolve().parents[1]),
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
        lines.append(line)
    exit_code = int(process.wait())
    elapsed_ms = (time.perf_counter_ns() - started) / 1.0e6
    log_path.write_text("".join(lines), encoding="utf-8")
    calibration = stage_root / "calibration"
    precheck = stage_root / "precheck"
    result_kind = "failed"
    if exit_code == 0:
        if (calibration / "reused_calibration.json").is_file():
            result_kind = "reused"
        elif (calibration / "hik_camera_calibration.json").is_file():
            result_kind = "fresh"
        else:
            result_kind = "completed_unknown"
    evidence = [
        str(path.resolve())
        for parent in (calibration, precheck)
        if parent.is_dir()
        for path in parent.rglob("*")
        if path.suffix.lower() in (".png", ".json", ".yaml", ".yml")
    ]
    return {
        "name": name,
        "command": list(command),
        "exit_code": exit_code,
        "elapsed_ms": elapsed_ms,
        "result_kind": result_kind,
        "console_log": str(log_path.resolve()),
        "calibration_output": str(calibration.resolve()),
        "precheck_output": str(precheck.resolve()),
        "evidence": evidence,
    }


def _calibration_command(
    arguments: argparse.Namespace, output_root: Path, name: str, reuse: bool
) -> list[str]:
    stage_root = output_root / "stages" / name
    command = [
        str(Path(sys.executable).resolve()),
        "-B",
        "-m",
        "acquisition.rig_calibration.hik.calibrate",
        "--camera-adapter",
        "virtualhikcam.driver:create_camera_adapter",
        "--camera-id",
        arguments.camera_id,
        "--phone-serial",
        arguments.phone_serial,
        "--profile-root",
        str((output_root / "profiles").resolve()),
        "--output",
        str((stage_root / "calibration").resolve()),
        "--reuse-evidence-output",
        str((stage_root / "precheck").resolve()),
        "--camera-width",
        str(arguments.width),
        "--camera-height",
        str(arguments.height),
        "--camera-fps",
        str(arguments.fps),
        "--headless",
        "--save",
        "--final-benchmark",
        "skip",
    ]
    if reuse:
        command.append("--reuse-if-unchanged")
    if arguments.adb:
        command.extend(("--adb", arguments.adb))
    return command


def _change_pose(
    arguments: argparse.Namespace,
    state_root: Path,
    *,
    reset: bool,
) -> dict[str, Any]:
    adapter = VirtualHikCameraAdapter(state_root=state_root)
    configuration = CameraConfiguration(
        arguments.camera_id,
        arguments.width,
        arguments.height,
        arguments.fps,
        "virtual_hik",
    )
    try:
        adapter.open(configuration)
        before = dict(adapter.simulated_displacement())
        if reset:
            after = dict(adapter.reset_simulated_displacement())
        else:
            after = dict(
                adapter.randomize_simulated_displacement(
                    max_x_px=arguments.max_x_px,
                    max_y_px=arguments.max_y_px,
                    max_rotation_deg=arguments.max_rotation_deg,
                    seed=arguments.seed,
                )
            )
        return {
            "before": before,
            "after": after,
            "state_file": adapter.metadata["state_file"],
            "displacement_map_build_ms": adapter._displacement_map_build_ms,
        }
    finally:
        adapter.close()


def _percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * float(fraction)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _benchmark_stream(
    arguments: argparse.Namespace, state_root: Path
) -> dict[str, Any]:
    adapter = VirtualHikCameraAdapter(state_root=state_root)
    configuration = CameraConfiguration(
        arguments.camera_id,
        arguments.width,
        arguments.height,
        arguments.fps,
        "virtual_hik",
    )
    try:
        adapter.open(configuration)
        samples = [adapter.read() for _ in range(arguments.benchmark_frames)]
        measured = samples[3:]
        processing = [sample.metadata["virtual_processing_ms"] for sample in measured]
        remap = [sample.metadata["displacement_remap_ms"] for sample in measured]
        return {
            "simulated_displacement": dict(adapter.simulated_displacement()),
            "frame_count": len(measured),
            "effective_roi_xywh": list(adapter.active_roi),
            "map_build_ms": adapter._displacement_map_build_ms,
            "map_generation_values": sorted(
                set(
                    int(sample.metadata["displacement_map_generation"])
                    for sample in measured
                )
            ),
            "virtual_processing_ms_p50": statistics.median(processing),
            "virtual_processing_ms_p95": _percentile(processing, 0.95),
            "displacement_remap_ms_p50": statistics.median(remap),
            "displacement_remap_ms_p95": _percentile(remap, 0.95),
        }
    finally:
        adapter.close()


def _adb(arguments: argparse.Namespace) -> str:
    if arguments.adb:
        return arguments.adb
    configured = os.environ.get("ADB")
    if configured:
        return configured
    return "adb"


def _sleep_devices(arguments: argparse.Namespace) -> list[dict[str, Any]]:
    outcomes = []
    source_serial = arguments.camera_id.split("://", 1)[-1].split("/", 1)[0]
    for serial in (source_serial, arguments.phone_serial):
        command = [_adb(arguments), "-s", serial, "shell", "input", "keyevent", "223"]
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=8,
                universal_newlines=True,
            )
            outcomes.append(
                {"serial": serial, "exit_code": completed.returncode, "output": completed.stdout}
            )
        except Exception as exc:
            outcomes.append(
                {"serial": serial, "error": "{}: {}".format(type(exc).__name__, exc)}
            )
    return outcomes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the standard rig-only virtual-camera instrument sequence: "
            "fresh calibration, unchanged reuse, persistent displacement and "
            "recalibration, shifted reuse, canonical reset and recalibration, "
            "then canonical reuse."
        )
    )
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--camera-id")
    identity.add_argument(
        "--camera-source-serial",
        help="Android camera-source serial; preferred for Windows batch use",
    )
    parser.add_argument("--camera-source-id", default="1")
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--bit-rate", type=int, default=12000000)
    parser.add_argument("--phone-serial", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--adb")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-x-px", type=float, default=32.0)
    parser.add_argument("--max-y-px", type=float, default=24.0)
    parser.add_argument("--max-rotation-deg", type=float, default=2.0)
    parser.add_argument("--benchmark-frames", type=int, default=33)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.benchmark_frames < 5:
        raise ValueError("Stream benchmark requires at least five frames")
    if arguments.camera_id is None:
        arguments.camera_id = "virtual-hik://{}/camera/{}?{}".format(
            arguments.camera_source_serial,
            arguments.camera_source_id,
            urlencode(
                [
                    ("width", arguments.width),
                    ("height", arguments.height),
                    ("fps", arguments.fps),
                    ("zoom", arguments.zoom),
                    ("bit_rate", arguments.bit_rate),
                ]
            ),
        )
    repository = Path(__file__).resolve().parents[1]
    output_root = (
        arguments.output
        or repository / "artifacts" / "virtual-rig-instrument-{}".format(_timestamp())
    ).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    state_root = output_root / "virtual-camera-state"
    environment = dict(os.environ)
    environment["IRIS_VIRTUAL_HIK_STATE_ROOT"] = str(state_root)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "scope": "rig_only_virtual_camera_production_interface",
        "output_root": str(output_root),
        "camera_id": arguments.camera_id,
        "phone_serial": arguments.phone_serial,
        "python": str(Path(sys.executable).resolve()),
        "parameters": {
            "width": arguments.width,
            "height": arguments.height,
            "fps": arguments.fps,
            "seed": arguments.seed,
            "max_x_px": arguments.max_x_px,
            "max_y_px": arguments.max_y_px,
            "max_rotation_deg": arguments.max_rotation_deg,
            "benchmark_frames": arguments.benchmark_frames,
        },
        "stages": [],
        "pose_changes": [],
        "stream_benchmarks": [],
        "failures": [],
    }
    manifest_path = output_root / "instrument_run.json"
    _write_json(manifest_path, manifest)
    sequence = (
        ("01_canonical_fresh", False, "fresh"),
        ("02_canonical_reuse", True, "reused"),
    )
    exit_code = 0
    try:
        for name, reuse, expected in sequence:
            result = _run_logged(
                name,
                _calibration_command(arguments, output_root, name, reuse),
                output_root=output_root,
                environment=environment,
            )
            result["expected_result_kind"] = expected
            manifest["stages"].append(result)
            if result["result_kind"] != expected:
                manifest["failures"].append(
                    {"stage": name, "expected": expected, "actual": result["result_kind"]}
                )
                exit_code = 1
                return exit_code
            if name == "01_canonical_fresh":
                benchmark = _benchmark_stream(arguments, state_root)
                benchmark["state"] = "canonical_after_fresh_calibration"
                manifest["stream_benchmarks"].append(benchmark)
            _write_json(manifest_path, manifest)

        changed = _change_pose(arguments, state_root, reset=False)
        changed["operation"] = "seeded_random_displacement"
        manifest["pose_changes"].append(changed)
        benchmark = _benchmark_stream(arguments, state_root)
        benchmark["state"] = "seeded_random_displacement"
        manifest["stream_benchmarks"].append(benchmark)
        _write_json(manifest_path, manifest)

        for name, expected in (
            ("03_shifted_recalibration", "fresh"),
            ("04_shifted_reuse", "reused"),
        ):
            result = _run_logged(
                name,
                _calibration_command(arguments, output_root, name, True),
                output_root=output_root,
                environment=environment,
            )
            result["expected_result_kind"] = expected
            manifest["stages"].append(result)
            if result["result_kind"] != expected:
                manifest["failures"].append(
                    {"stage": name, "expected": expected, "actual": result["result_kind"]}
                )
                exit_code = 1
                return exit_code
            _write_json(manifest_path, manifest)

        reset = _change_pose(arguments, state_root, reset=True)
        reset["operation"] = "reset_canonical"
        manifest["pose_changes"].append(reset)
        benchmark = _benchmark_stream(arguments, state_root)
        benchmark["state"] = "canonical_after_reset"
        manifest["stream_benchmarks"].append(benchmark)
        _write_json(manifest_path, manifest)

        for name, expected in (
            ("05_reset_recalibration", "fresh"),
            ("06_reset_reuse", "reused"),
        ):
            result = _run_logged(
                name,
                _calibration_command(arguments, output_root, name, True),
                output_root=output_root,
                environment=environment,
            )
            result["expected_result_kind"] = expected
            manifest["stages"].append(result)
            if result["result_kind"] != expected:
                manifest["failures"].append(
                    {"stage": name, "expected": expected, "actual": result["result_kind"]}
                )
                exit_code = 1
                return exit_code
            _write_json(manifest_path, manifest)
        return 0
    except Exception as exc:
        manifest["failures"].append(
            {"stage": "instrument", "error": "{}: {}".format(type(exc).__name__, exc)}
        )
        exit_code = 1
        return exit_code
    finally:
        manifest["device_sleep"] = _sleep_devices(arguments)
        manifest["completed_unix_time"] = time.time()
        manifest["exit_code"] = exit_code
        _write_json(manifest_path, manifest)
        print("\nInstrument evidence: {}".format(output_root))


if __name__ == "__main__":
    raise SystemExit(main())
