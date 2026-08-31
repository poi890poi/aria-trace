"""Saved-rig displacement check using full-sensor ChArUco geometry."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np

from aria_trace.services.calibration.rig.geometry import (
    CharucoLayout,
    detect_charuco_correspondences,
)
from aria_trace.adapters.rig.devices import CameraConfiguration
from aria_trace.adapters.android.hik_display import AdbDisplayTarget
from aria_trace.adapters.hik.driver import HikMvsCameraAdapter, RectifiedHikCamera
from aria_trace.adapters.android.phone import AdbPhoneSession, resolve_adb_executable
from aria_trace.adapters.filesystem.profile_registry import ProfileContext, ProfileRegistry, ProfileResolutionError
from aria_trace.adapters.filesystem.system_configuration import (
    DEFAULT_RIG_REPEATABILITY_POLICY,
    RIG_REPEATABILITY_POLICIES,
)


_DEFAULT_REPEATABILITY = RIG_REPEATABILITY_POLICIES[
    DEFAULT_RIG_REPEATABILITY_POLICY
]
DEFAULT_MAXIMUM_DISPLACEMENT_PX = float(
    _DEFAULT_REPEATABILITY["reuse_max_displacement_px"]
)


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


def compare_charuco_alignment(
    observations: Sequence[Mapping[str, object]],
    screen_to_full_sensor_camera_3x3: Sequence[Sequence[float]],
    *,
    maximum_displacement_px: float = DEFAULT_MAXIMUM_DISPLACEMENT_PX,
) -> Mapping[str, object]:
    """Compare detected board corners with saved full-sensor coordinates.

    Only ChArUco geometry participates. Pixel intensity, color, background,
    and environmental lighting are intentionally absent from the gate.
    """

    matrix = np.asarray(screen_to_full_sensor_camera_3x3, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Saved screen-to-camera matrix must be a finite 3x3 matrix")
    if maximum_displacement_px <= 0:
        raise ValueError("Maximum ChArUco displacement must be positive")
    frame_metrics = []
    all_errors = []
    for frame_index, observation in enumerate(observations):
        current = np.asarray(observation["camera_points_xy"], dtype=np.float64)
        screen = np.asarray(observation["screen_points_xy"], dtype=np.float64)
        if current.ndim != 2 or current.shape[1:] != (2,) or current.shape != screen.shape:
            raise ValueError("ChArUco camera/screen points must be matching Nx2 arrays")
        expected = cv2.perspectiveTransform(
            screen.reshape((-1, 1, 2)), matrix
        ).reshape((-1, 2))
        errors = np.linalg.norm(current - expected, axis=1)
        all_errors.append(errors)
        frame_metrics.append(
            {
                "frame_index": int(frame_index),
                "corner_count": int(len(errors)),
                "median_displacement_px": float(np.median(errors)),
                "p95_displacement_px": float(np.percentile(errors, 95)),
                "maximum_displacement_px": float(np.max(errors)),
            }
        )
    if not all_errors:
        raise ValueError("At least one ChArUco observation is required")
    errors = np.concatenate(all_errors)
    # A median across frame-level p95 values rejects persistent displacement
    # while tolerating one noisy detector frame in the relaxed default policy.
    alignment_p95 = float(
        np.median([row["p95_displacement_px"] for row in frame_metrics])
    )
    matches = bool(
        np.isfinite(alignment_p95)
        and alignment_p95 <= float(maximum_displacement_px)
    )
    return {
        "matches": matches,
        "reason": "within_alignment_limit" if matches else "charuco_board_displaced",
        "frame_count": int(len(frame_metrics)),
        "corner_observation_count": int(len(errors)),
        "median_displacement_px": float(np.median(errors)),
        "p95_displacement_px": alignment_p95,
        "maximum_observed_displacement_px": float(np.max(errors)),
        "maximum_allowed_displacement_px": float(maximum_displacement_px),
        "frame_metrics": frame_metrics,
        "method": "full_sensor_saved_projection_vs_detected_charuco_corners_no_pixel_matching",
        "lighting_invariant": True,
    }


def format_reuse_precheck_failure(result: Mapping[str, object]) -> str:
    """Explain why an active rig cannot be reused in operator-facing terms."""

    status = str(result.get("status") or "unknown")
    if status == "rig_moved":
        comparison = dict(result.get("comparison") or {})
        displacement = float(
            comparison.get("p95_displacement_px", float("nan"))
        )
        maximum = float(
            comparison.get(
                "maximum_allowed_displacement_px",
                DEFAULT_MAXIMUM_DISPLACEMENT_PX,
            )
        )
        return (
            "Rig reuse check detected ChArUco board displacement.\n"
            "  Corner alignment p95: {:.3f} full-sensor px; allowed <= {:.3f} px [{}]\n"
            "The camera or phone position may have changed. Lighting and pixel "
            "brightness are not part of this check."
        ).format(
            displacement,
            maximum,
            "PASS" if displacement <= maximum else "FAIL",
        )
    if status == "no_previous_calibration":
        return (
            "Rig reuse was skipped: no active rig profile exists for the selected "
            "camera and phone."
        )
    if status == "identity_mismatch":
        mismatch = str(result.get("reason") or "device_identity_mismatch")
        return (
            "Rig reuse check failed: the selected device identity does not match "
            "the saved calibration ({})."
        ).format(mismatch)
    if status == "incomplete_calibration":
        return (
            "Rig reuse check failed: the saved rig profile is incomplete; its "
            "camera adapter or saved ChArUco geometry is missing."
        )
    if status == "unavailable":
        return "Rig reuse check could not run: {}".format(
            result.get("reason") or "no diagnostic reason was reported"
        )
    if status == "reusable" and not result.get("camera_adapter_is_calibrated"):
        return (
            "Rig reuse check passed ChArUco alignment, but the saved camera "
            "adapter reports incomplete calibration data."
        )
    return "Rig reuse check did not pass (status: {}).".format(status)


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


def _apply_saved_imaging(
    adapter: HikMvsCameraAdapter, config: Mapping[str, object]
) -> None:
    imaging = config["imaging"]
    if imaging.get("black_level") is not None:
        adapter.set_black_level(int(imaging["black_level"]))
    adapter.set_manual_imaging(imaging["exposure_us"], imaging["gain"])
    white_balance = imaging["white_balance"]
    adapter.set_white_balance(
        white_balance["ratio_red"],
        white_balance["ratio_green"],
        white_balance["ratio_blue"],
    )


def _alignment_overlay(
    frame: np.ndarray,
    observation: Mapping[str, object],
    screen_to_full_sensor_camera_3x3: Sequence[Sequence[float]],
) -> np.ndarray:
    """Draw saved expected and fresh detected corners in full-sensor space."""

    result = frame.copy()
    current = np.asarray(observation["camera_points_xy"], dtype=np.float64)
    screen = np.asarray(observation["screen_points_xy"], dtype=np.float64)
    matrix = np.asarray(screen_to_full_sensor_camera_3x3, dtype=np.float64)
    expected = cv2.perspectiveTransform(
        screen.reshape((-1, 1, 2)), matrix
    ).reshape((-1, 2))
    for saved_xy, fresh_xy in zip(expected, current):
        saved = tuple(np.rint(saved_xy).astype(int))
        fresh = tuple(np.rint(fresh_xy).astype(int))
        cv2.line(result, saved, fresh, (0, 210, 255), 1, cv2.LINE_AA)
        cv2.circle(result, saved, 4, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.drawMarker(
            result,
            fresh,
            (255, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=7,
            thickness=1,
        )
    cv2.putText(
        result,
        "green=saved  magenta=fresh",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def run_reuse_precheck(
    calibration: Path,
    output: Path,
    *,
    adb: Optional[str] = None,
    mvs_python_path: Optional[str] = None,
    camera_id: Optional[str] = None,
    phone_serial: Optional[str] = None,
    maximum_displacement_px: float = DEFAULT_MAXIMUM_DISPLACEMENT_PX,
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

    validator = RectifiedHikCamera(calibration_file, rectify=False)
    result["camera_adapter_is_calibrated"] = validator.is_calibrated()
    geometry = config.get("geometry") or {}
    screen_to_camera = geometry.get("screen_to_full_sensor_camera_3x3")
    if not validator.is_calibrated() or screen_to_camera is None:
        result.update(status="incomplete_calibration", reason="adapter_or_geometry_missing")
        _write_result(output, result)
        return result

    adapter = HikMvsCameraAdapter(sdk_python_path=mvs_python_path)

    adb_path = resolve_adb_executable(adb)
    phone = AdbPhoneSession(saved_phone, adb_executable=adb_path)
    viewer = str(((config.get("phone") or {}).get("viewer") or {}).get("activity") or "")
    target = AdbDisplayTarget(phone, component=viewer or None)
    orientation = int(config["phone"].get("orientation_quarter_turns", 0))
    full_mode = config["camera"]["full_sensor_mode"]
    observations = []
    evidence_frames = []
    detection_failures = []
    cleanup_warnings = []
    try:
        target.configure_canonical_orientation(orientation)
        phone.wake_and_hold_display(orientation)
        target.start(_saved_layout(config))
        adapter.open(
            CameraConfiguration(
                device_id=saved_camera,
                width_px=int(full_mode["width_px"]),
                height_px=int(full_mode["height_px"]),
                fps=float(full_mode["fps"]),
                backend="hik_mvs",
            )
        )
        full_roi = list(adapter.reset_full_sensor_roi())
        _apply_saved_imaging(adapter, config)
        for _ in range(3):
            adapter.read()
        for frame_index in range(max(1, int(sample_frames))):
            sample = adapter.read()
            try:
                detected = detect_charuco_correspondences(
                    sample.image, _saved_layout(config)
                )
            except RuntimeError as exc:
                detection_failures.append(
                    {"frame_index": int(frame_index), "error": str(exc)}
                )
                continue
            observations.append(detected)
            evidence_frames.append((sample.image.copy(), dict(sample.metadata)))
        if not observations:
            raise RuntimeError(
                "ChArUco board was not detected in any full-sensor precheck frame"
            )
        comparison = dict(
            compare_charuco_alignment(
                observations,
                screen_to_camera,
                maximum_displacement_px=maximum_displacement_px,
            )
        )
        output.mkdir(parents=True, exist_ok=False)
        representative_index = int(
            np.argsort(
                [row["p95_displacement_px"] for row in comparison["frame_metrics"]]
            )[len(observations) // 2]
        )
        fresh, fresh_metadata = evidence_frames[representative_index]
        overlay = _alignment_overlay(
            fresh, observations[representative_index], screen_to_camera
        )
        if not cv2.imwrite(str(output / "fresh_full_sensor_frame.png"), fresh):
            raise OSError("Could not save full-sensor precheck frame")
        if not cv2.imwrite(str(output / "charuco_alignment_overlay.png"), overlay):
            raise OSError("Could not save ChArUco alignment overlay")
        result.update(
            status="reusable" if comparison["matches"] else "rig_moved",
            reusable=bool(comparison["matches"]),
            comparison=comparison,
            detection_failures=detection_failures,
            image_space=fresh_metadata.get("image_space"),
            effective_full_sensor_roi_xywh=full_roi,
            evidence={
                "fresh_full_sensor_frame": "fresh_full_sensor_frame.png",
                "charuco_alignment_overlay": "charuco_alignment_overlay.png",
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
            adapter.close()
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
    maximum_displacement_px: float = DEFAULT_MAXIMUM_DISPLACEMENT_PX,
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
                maximum_displacement_px=maximum_displacement_px,
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
        description="Reuse a saved HIK rig only when full-sensor ChArUco alignment is unchanged"
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
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    from aria_trace.adapters.filesystem.system_configuration import (
        load_system_configuration,
        resolve_rig_repeatability_policy,
    )

    repeatability = resolve_rig_repeatability_policy(
        load_system_configuration(arguments.profile_root)
    )
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
                    maximum_displacement_px=repeatability[
                        "reuse_max_displacement_px"
                    ],
                    sample_frames=repeatability["reuse_sample_frames"],
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
            maximum_displacement_px=repeatability["reuse_max_displacement_px"],
            sample_frames=repeatability["reuse_sample_frames"],
        )
        if result["status"] == "no_previous_calibration":
            print(format_reuse_precheck_failure(result))
            print(
                "Precheck evidence: {}".format(
                    (arguments.output / "precheck.json").resolve()
                )
            )
            return 0
    comparison = result.get("comparison") or {}
    if result.get("reusable") and result.get("camera_adapter_is_calibrated"):
        print(
            "Rig precheck: reusable (ChArUco alignment p95 {:.3f} px; "
            "allowed {:.3f} px)".format(
                float(comparison["p95_displacement_px"]),
                float(comparison["maximum_allowed_displacement_px"]),
            )
        )
    else:
        print(format_reuse_precheck_failure(result))
    print(
        "Precheck evidence: {}".format(
            (arguments.output / "precheck.json").resolve()
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
