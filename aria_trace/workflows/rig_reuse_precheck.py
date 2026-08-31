"""Conservative saved-rig reuse check using a repeated ChArUco snapshot."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np

from aria_trace.services.calibration.rig.geometry import CharucoLayout
from aria_trace.adapters.android.hik_display import AdbDisplayTarget
from aria_trace.adapters.hik.driver import HikMvsCameraAdapter, RectifiedHikCamera
from aria_trace.adapters.android.phone import AdbPhoneSession, resolve_adb_executable
from aria_trace.adapters.filesystem.profile_registry import ProfileContext, ProfileRegistry, ProfileResolutionError


DEFAULT_MINIMUM_CORRELATION = 0.985
DEFAULT_MAXIMUM_MAE_DN = 8.0


def resolve_calibration_file(path: Path) -> Path:
    value = Path(path)
    if value.is_dir():
        value = value / "hik_camera_calibration.json"
    return value.resolve()


def discover_active_profile_calibration(
    profile_root: Optional[Path],
    *,
    camera_id: Optional[str] = None,
    phone_serial: Optional[str] = None,
) -> Optional[Path]:
    """Resolve one exact active rig without inspecting calibration artifacts."""

    registry = ProfileRegistry(profile_root)
    try:
        profile = registry.resolve(
            "rig",
            ProfileContext(camera_id=camera_id, phone_id=phone_serial),
        )
    except ProfileResolutionError as exc:
        if str(exc).startswith("No active rig profile"):
            return None
        raise
    return registry.runtime_file(profile, "hik_camera_calibration").resolve()


def compare_calibration_snapshots(
    reference: np.ndarray,
    fresh: np.ndarray,
    *,
    minimum_correlation: float = DEFAULT_MINIMUM_CORRELATION,
    maximum_mae_dn: float = DEFAULT_MAXIMUM_MAE_DN,
) -> Mapping[str, object]:
    """Compare locked-camera images without fitting away physical movement."""

    if reference is None or fresh is None or reference.size == 0 or fresh.size == 0:
        raise ValueError("Reference and fresh rig snapshots are required")
    if reference.shape != fresh.shape:
        return {
            "matches": False,
            "reason": "shape_mismatch",
            "reference_shape": list(reference.shape),
            "fresh_shape": list(fresh.shape),
            "minimum_correlation": float(minimum_correlation),
            "maximum_mae_dn": float(maximum_mae_dn),
        }
    reference_gray = (
        cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        if reference.ndim == 3
        else reference
    ).astype(np.float32)
    fresh_gray = (
        cv2.cvtColor(fresh, cv2.COLOR_BGR2GRAY)
        if fresh.ndim == 3
        else fresh
    ).astype(np.float32)
    reference_std = float(np.std(reference_gray))
    fresh_std = float(np.std(fresh_gray))
    if min(reference_std, fresh_std) < 1.0e-6:
        correlation = 1.0 if np.array_equal(reference_gray, fresh_gray) else 0.0
    else:
        normalized_reference = (
            reference_gray - float(np.mean(reference_gray))
        ) / reference_std
        normalized_fresh = (
            fresh_gray - float(np.mean(fresh_gray))
        ) / fresh_std
        correlation = float(np.mean(normalized_reference * normalized_fresh))
    mae = float(np.mean(np.abs(reference_gray - fresh_gray)))
    matches = bool(
        np.isfinite(correlation)
        and correlation >= float(minimum_correlation)
        and mae <= float(maximum_mae_dn)
    )
    return {
        "matches": matches,
        "reason": "within_thresholds" if matches else "snapshot_changed",
        "correlation": correlation,
        "mean_absolute_error_dn": mae,
        "minimum_correlation": float(minimum_correlation),
        "maximum_mae_dn": float(maximum_mae_dn),
        "reference_shape": list(reference.shape),
        "fresh_shape": list(fresh.shape),
        "method": "locked_imaging_direct_grayscale_correlation_and_mae_no_alignment_fit",
    }


def _saved_layout(config: Mapping[str, object]) -> CharucoLayout:
    phone = config["phone"]
    layout = config["geometry"]["charuco_layout"]
    return CharucoLayout(
        tuple(map(int, phone["screen_size_px"])),
        squares_x=int(layout["squares_x"]),
        squares_y=int(layout["squares_y"]),
        margin_px=tuple(map(int, layout.get("margin_px") or (0, 0))),
    )


def _write_result(output: Path, result: Mapping[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "precheck.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


def run_reuse_precheck(
    calibration: Path,
    output: Path,
    *,
    adb: Optional[str] = None,
    mvs_python_path: Optional[str] = None,
    camera_id: Optional[str] = None,
    phone_serial: Optional[str] = None,
    minimum_correlation: float = DEFAULT_MINIMUM_CORRELATION,
    maximum_mae_dn: float = DEFAULT_MAXIMUM_MAE_DN,
    sample_frames: int = 3,
) -> Mapping[str, object]:
    """Present the saved target and prove the camera/phone pose is unchanged."""

    calibration_file = resolve_calibration_file(calibration)
    config = json.loads(calibration_file.read_text(encoding="utf-8"))
    saved_camera = str(config["camera"]["device_id"])
    saved_phone = str(config["phone"]["serial"])
    result = {
        "schema_version": "1.0",
        "status": "unavailable",
        "reusable": False,
        "calibration": str(calibration_file),
        "camera_id": saved_camera,
        "phone_serial": saved_phone,
        "camera_adapter_is_calibrated": False,
        "generated_unix_time": time.time(),
    }
    if camera_id and str(camera_id) != saved_camera:
        result.update(status="identity_mismatch", reason="camera_id_mismatch")
        _write_result(output, result)
        return result
    if phone_serial and str(phone_serial) != saved_phone:
        result.update(status="identity_mismatch", reason="phone_serial_mismatch")
        _write_result(output, result)
        return result

    adapter = HikMvsCameraAdapter(sdk_python_path=mvs_python_path)
    reader = RectifiedHikCamera(calibration_file, adapter=adapter, rectify=False)
    result["camera_adapter_is_calibrated"] = reader.is_calibrated()
    reference_path = calibration_file.parent / "last_camera_frame.png"
    if not reader.is_calibrated() or not reference_path.is_file():
        result.update(status="incomplete_calibration", reason="adapter_or_snapshot_missing")
        _write_result(output, result)
        return result

    adb_path = resolve_adb_executable(adb)
    phone = AdbPhoneSession(saved_phone, adb_executable=adb_path)
    viewer = str(((config.get("phone") or {}).get("viewer") or {}).get("activity") or "")
    target = AdbDisplayTarget(phone, component=viewer or None)
    orientation = int(config["phone"].get("orientation_quarter_turns", 0))
    reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    fresh_frames = []
    cleanup_warnings = []
    try:
        target.configure_canonical_orientation(orientation)
        phone.wake_and_hold_display(orientation)
        target.start(_saved_layout(config))
        reader.open()
        for _ in range(3):
            ok, _frame = reader.read()
            if not ok:
                raise RuntimeError("HIK camera returned no precheck settle frame")
        for _ in range(max(1, int(sample_frames))):
            ok, frame = reader.read()
            if not ok or frame is None:
                raise RuntimeError("HIK camera returned no precheck comparison frame")
            fresh_frames.append(frame)
        fresh = np.median(np.stack(fresh_frames), axis=0).astype(np.uint8)
        comparison = dict(
            compare_calibration_snapshots(
                reference,
                fresh,
                minimum_correlation=minimum_correlation,
                maximum_mae_dn=maximum_mae_dn,
            )
        )
        output.mkdir(parents=True, exist_ok=False)
        cv2.imwrite(str(output / "reference_snapshot.png"), reference)
        cv2.imwrite(str(output / "fresh_snapshot.png"), fresh)
        if reference.shape == fresh.shape:
            difference = cv2.absdiff(reference, fresh)
            cv2.imwrite(str(output / "absolute_difference.png"), difference)
            cv2.imwrite(
                str(output / "side_by_side_reference_then_fresh.png"),
                np.hstack((reference, fresh)),
            )
        result.update(
            status="reusable" if comparison["matches"] else "rig_moved",
            reusable=bool(comparison["matches"]),
            comparison=comparison,
            evidence={
                "reference": "reference_snapshot.png",
                "fresh": "fresh_snapshot.png",
                "difference": "absolute_difference.png",
                "side_by_side": "side_by_side_reference_then_fresh.png",
            },
        )
    except Exception as exc:
        if not output.exists():
            output.mkdir(parents=True)
        result.update(
            status="unavailable",
            reusable=False,
            reason="{}: {}".format(type(exc).__name__, exc),
        )
    finally:
        try:
            reader.release()
        except Exception as exc:
            cleanup_warnings.append(str(exc))
        try:
            target.stop()
        except Exception as exc:
            cleanup_warnings.append(str(exc))
        try:
            phone.cleanup(turn_display_off=False)
        except Exception as exc:
            cleanup_warnings.append(str(exc))
    if cleanup_warnings:
        result["cleanup_warnings"] = cleanup_warnings
    (output / "precheck.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def run_active_reuse_precheck(
    output: Path,
    *,
    profile_root: Optional[Path] = None,
    adb: Optional[str] = None,
    mvs_python_path: Optional[str] = None,
    camera_id: Optional[str] = None,
    phone_serial: Optional[str] = None,
    minimum_correlation: float = DEFAULT_MINIMUM_CORRELATION,
    maximum_mae_dn: float = DEFAULT_MAXIMUM_MAE_DN,
    sample_frames: int = 3,
) -> Mapping[str, object]:
    """Resolve and check the active rig profile without accepting artifact paths.

    This is the shared product boundary used by both the standalone precheck
    command and the rig calibration application's reuse option.  An unavailable
    check is deliberately non-gating: callers can continue with full
    calibration using the returned status and retained evidence.
    """

    calibration = discover_active_profile_calibration(
        profile_root,
        camera_id=camera_id,
        phone_serial=phone_serial,
    )
    if calibration is None:
        result = {
            "schema_version": "1.0",
            "status": "no_previous_calibration",
            "reusable": False,
            "camera_adapter_is_calibrated": False,
            "calibration_selection": "active_profile_registry",
        }
        _write_result(output, result)
        return result
    try:
        result = dict(
            run_reuse_precheck(
                calibration,
                output,
                adb=adb,
                mvs_python_path=mvs_python_path,
                camera_id=camera_id,
                phone_serial=phone_serial,
                minimum_correlation=minimum_correlation,
                maximum_mae_dn=maximum_mae_dn,
                sample_frames=sample_frames,
            )
        )
    except Exception as exc:
        result = {
            "schema_version": "1.0",
            "status": "unavailable",
            "reusable": False,
            "camera_adapter_is_calibrated": False,
            "calibration": str(calibration),
            "reason": "{}: {}".format(type(exc).__name__, exc),
        }
        if output.exists():
            (output / "precheck.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
        else:
            _write_result(output, result)
    result["calibration_selection"] = "active_profile_registry"
    (output / "precheck.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Reuse a saved HIK rig only when its repeated snapshot is unchanged"
    )
    value.add_argument(
        "--diagnostic-calibration-override",
        type=Path,
        help="explicit diagnostic override; production reuse uses the registry",
    )
    value.add_argument("--profile-root", type=Path)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--camera-id")
    value.add_argument("--phone-serial")
    value.add_argument("--adb")
    value.add_argument("--mvs-python-path")
    value.add_argument("--minimum-correlation", type=float, default=DEFAULT_MINIMUM_CORRELATION)
    value.add_argument("--maximum-mae-dn", type=float, default=DEFAULT_MAXIMUM_MAE_DN)
    value.add_argument("--sample-frames", type=int, default=3)
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.diagnostic_calibration_override:
        calibration = resolve_calibration_file(
            arguments.diagnostic_calibration_override
        )
        try:
            result = dict(
                run_reuse_precheck(
                    calibration,
                    arguments.output,
                    adb=arguments.adb,
                    mvs_python_path=arguments.mvs_python_path,
                    camera_id=arguments.camera_id,
                    phone_serial=arguments.phone_serial,
                    minimum_correlation=arguments.minimum_correlation,
                    maximum_mae_dn=arguments.maximum_mae_dn,
                    sample_frames=arguments.sample_frames,
                )
            )
            result["calibration_selection"] = "diagnostic_explicit_path"
            (arguments.output / "precheck.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            _write_result(
                arguments.output,
                {
                    "schema_version": "1.0",
                    "status": "unavailable",
                    "reusable": False,
                    "calibration": str(calibration),
                    "reason": "{}: {}".format(type(exc).__name__, exc),
                },
            )
            print("Rig precheck unavailable: {}".format(exc))
            return 0
    else:
        result = run_active_reuse_precheck(
            arguments.output,
            profile_root=arguments.profile_root,
            adb=arguments.adb,
            mvs_python_path=arguments.mvs_python_path,
            camera_id=arguments.camera_id,
            phone_serial=arguments.phone_serial,
            minimum_correlation=arguments.minimum_correlation,
            maximum_mae_dn=arguments.maximum_mae_dn,
            sample_frames=arguments.sample_frames,
        )
        if result["status"] == "no_previous_calibration":
            print("Rig precheck: no active registry calibration is available.")
            return 0
    comparison = result.get("comparison") or {}
    print(
        "Rig precheck: {}{}".format(
            result["status"],
            " (correlation {:.6f}, MAE {:.3f} DN)".format(
                float(comparison["correlation"]),
                float(comparison["mean_absolute_error_dn"]),
            )
            if comparison
            else "",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
