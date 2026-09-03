"""Capture the standard ADB/full-camera/rectified rig review evidence."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional, Sequence

import cv2

from rig_runtime.adapters.android.phone import (
    probe_android_capture_surface,
    resolve_adb_executable,
)
from rig_runtime.adapters.filesystem.system_configuration import (
    load_system_configuration,
)
from rig_runtime.adapters.hik.capture import CalibratedHikFrameSource
from rig_runtime.adapters.hik.driver import HikMvsCameraAdapter, RectifiedHikCamera
from rig_runtime.adapters.rig.devices import (
    CameraAdapter,
    CameraConfiguration,
    create_camera_adapter,
)
from rig_runtime.adapters.sources import AdbScreenshotFrameSource
from rig_runtime.domain.packets import FramePacket
from rig_runtime.evidence.rig_spatial import (
    ExpandedRigReview,
    StandardizedRigComparison,
    expanded_rig_camera_review_from_calibration,
    standardized_rig_comparison,
    validated_rig_sample,
)
from rig_runtime.services.calibration.rig.contracts import FrameSample
from rig_runtime.services.calibration.rig.cross_source import (
    match_game_camera_orientation,
)
from rig_runtime.workflows.rig_reuse_precheck import (
    discover_active_profile_calibration,
)


@dataclass(frozen=True)
class RigEvidenceCapture:
    orientation: Mapping[str, object]
    rectified_packet: FramePacket
    full_camera_sample: FrameSample
    full_camera_review: ExpandedRigReview
    comparison: StandardizedRigComparison
    evidence: Mapping[str, str]
    evidence_spaces: Mapping[str, Mapping[str, object]]
    timing: Mapping[str, float]


def _save_image(output: Path, name: str, image) -> str:
    path = Path(output) / str(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError("Could not save {}".format(path))
    return str(name).replace("\\", "/")


def _capture_adb_frame(adb: Path, serial: str, surface: Mapping[str, object]):
    source = AdbScreenshotFrameSource(
        adb,
        serial,
        "android_phone",
        fps=1.0,
        image_space_context=surface,
    )
    try:
        source.start()
        packet = source.read()
    finally:
        source.stop()
    if packet is None:
        raise RuntimeError("ADB returned no review screenshot")
    return packet


def _apply_saved_imaging(adapter: CameraAdapter, config: Mapping[str, object]) -> None:
    imaging = dict(config["imaging"])
    black_level = imaging.get("black_level")
    if black_level is not None:
        setter = getattr(adapter, "set_black_level", None)
        if callable(setter):
            setter(int(black_level))
    manual = getattr(adapter, "set_manual_imaging", None)
    if not callable(manual):
        raise TypeError("HIK-compatible adapter lacks set_manual_imaging")
    manual(float(imaging["exposure_us"]), float(imaging["gain"]))
    white_balance = dict(imaging["white_balance"])
    balance = getattr(adapter, "set_white_balance", None)
    if not callable(balance):
        raise TypeError("HIK-compatible adapter lacks set_white_balance")
    balance(
        int(white_balance["ratio_red"]),
        int(white_balance["ratio_green"]),
        int(white_balance["ratio_blue"]),
    )


def capture_full_camera_sample(
    adapter: CameraAdapter,
    config: Mapping[str, object],
    *,
    settle_frames: int = 3,
) -> FrameSample:
    """Acquire one complete sensor frame with saved manual imaging controls."""

    camera = dict(config["camera"])
    mode = dict(camera["full_sensor_mode"])
    full_size = [int(mode["width_px"]), int(mode["height_px"])]
    adapter.open(
        CameraConfiguration(
            device_id=str(camera["device_id"]),
            width_px=full_size[0],
            height_px=full_size[1],
            fps=float(mode["fps"]),
            backend=str(camera.get("adapter_id") or "hik_mvs"),
        )
    )
    try:
        reset = getattr(adapter, "reset_full_sensor_roi", None)
        effective = (
            list(reset())
            if callable(reset)
            else list(adapter.set_roi([0, 0] + full_size))
        )
        if list(map(int, effective)) != [0, 0] + full_size:
            raise RuntimeError(
                "Camera did not expose the complete calibrated sensor: {}"
                .format(effective)
            )
        _apply_saved_imaging(adapter, config)
        sample = None
        for _index in range(max(1, int(settle_frames))):
            sample = adapter.read()
        if sample is None:
            raise RuntimeError("HIK returned no full-camera review frame")
        return validated_rig_sample(sample, parent_size_px=full_size, copy_image=True)
    finally:
        adapter.close()


def capture_standardized_rig_evidence(
    adb_packet: FramePacket,
    surface: Mapping[str, object],
    calibration_file: Path,
    adapter: CameraAdapter,
    output: Path,
    *,
    write_orientation_candidates: bool = False,
) -> RigEvidenceCapture:
    """Acquire and persist one traceable three-space rig comparison."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    calibration_file = Path(calibration_file).resolve()
    config = json.loads(calibration_file.read_text(encoding="utf-8"))
    started_ns = time.perf_counter_ns()
    full_sample = capture_full_camera_sample(adapter, config)
    full_capture_done_ns = time.perf_counter_ns()
    full_review = expanded_rig_camera_review_from_calibration(
        full_sample,
        config,
        title="FULL CAMERA WITH SAVED PHONE PROJECTION",
    )

    reader = RectifiedHikCamera(calibration_file, adapter=adapter, rectify=True)
    hik = CalibratedHikFrameSource(
        calibration_file,
        "hik_phone",
        rectify=True,
        output_quarter_turns_clockwise=0,
        reader=reader,
    )
    orientation_images = {}
    try:
        hik.start()
        initial = hik.read()
        if initial is None:
            raise RuntimeError("HIK returned no initial rectified review frame")
        calibration_display = hik.alignment_evidence_image(initial)
        orientation, orientation_images = match_game_camera_orientation(
            adb_packet.image,
            calibration_display,
            calibration_file,
            android_reported_quarter_turns=int(
                surface["quarter_turns_clockwise_from_natural"]
            ),
        )
        selected_turns = int(
            orientation[
                "selected_camera_adapter_image_quarter_turns_clockwise_from_calibration_display"
            ]
        )
        hik.set_output_orientation(
            selected_turns,
            {
                "status": orientation.get("status"),
                "selection_basis": orientation.get("selection_basis"),
                "selected_confidence": orientation.get("selected_confidence"),
                "confidence_margin": orientation.get("confidence_margin"),
            },
        )
        rectified_packet = hik.read()
        if rectified_packet is None:
            raise RuntimeError("HIK returned no oriented rectified review frame")
    finally:
        hik.stop()
    rectified_done_ns = time.perf_counter_ns()

    comparison = standardized_rig_comparison(
        adb_packet.image,
        dict(adb_packet.metadata.get("image_space") or {}),
        full_review,
        rectified_packet.image,
        dict(rectified_packet.metadata.get("image_space") or {}),
    )
    evidence = {
        "adb_view": _save_image(output, "adb_view.png", adb_packet.image),
        "full_camera_view": _save_image(
            output, "full_camera_view.png", full_sample.image
        ),
        "full_camera_expanded_review": _save_image(
            output, "full_camera_expanded_review.png", full_review.image
        ),
        "rectified_view": _save_image(
            output, "rectified_view.png", rectified_packet.image
        ),
        "adb_full_camera_rectified_comparison": _save_image(
            output,
            "adb_full_camera_rectified_comparison.png",
            comparison.image,
        ),
    }
    evidence_spaces = {
        evidence["adb_view"]: dict(adb_packet.metadata.get("image_space") or {}),
        evidence["full_camera_view"]: dict(
            full_sample.metadata.get("image_space") or {}
        ),
        evidence["full_camera_expanded_review"]: dict(full_review.image_space),
        evidence["rectified_view"]: dict(
            rectified_packet.metadata.get("image_space") or {}
        ),
        evidence["adb_full_camera_rectified_comparison"]: dict(
            comparison.image_space
        ),
    }
    if write_orientation_candidates:
        for name, image in orientation_images.items():
            key = "orientation/{}".format(name.rsplit(".", 1)[0])
            evidence[key] = _save_image(
                output, "orientation/{}".format(name), image
            )
    return RigEvidenceCapture(
        orientation=dict(orientation),
        rectified_packet=rectified_packet,
        full_camera_sample=full_sample,
        full_camera_review=full_review,
        comparison=comparison,
        evidence=evidence,
        evidence_spaces=evidence_spaces,
        timing={
            "full_camera_capture_ms": (
                full_capture_done_ns - started_ns
            ) / 1.0e6,
            "rectified_capture_and_orientation_ms": (
                rectified_done_ns - full_capture_done_ns
            ) / 1.0e6,
            "capture_separation_ms": (
                int(rectified_packet.host_capture_time_ns)
                - int(full_sample.receive_time_ns or full_sample.time_ns)
            ) / 1.0e6,
        },
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Capture the standard ADB/full-camera/rectified rig evidence set. "
            "The command does not launch apps, send input, or recalibrate."
        )
    )
    value.add_argument("--profile-root", type=Path)
    value.add_argument("--camera-id")
    value.add_argument("--phone-serial")
    value.add_argument("--adb")
    value.add_argument("--mvs-python-path")
    value.add_argument("--diagnostic-calibration-override", type=Path)
    value.add_argument("--camera-adapter")
    value.add_argument("--output", type=Path)
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    output = arguments.output or (
        Path("artifacts")
        / "rig-evidence-review-{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    )
    output = Path(output)
    try:
        output.mkdir(parents=True, exist_ok=False)
        settings = load_system_configuration(arguments.profile_root)
        profile_root = Path(settings["effective_profile_root"])
        phone_serial = arguments.phone_serial or settings["devices"].get("phone_id")
        camera_id = arguments.camera_id or settings["devices"].get("camera_id")
        adb_path = Path(
            resolve_adb_executable(arguments.adb or settings["tools"].get("adb"))
        )
        phone_serial, surface = probe_android_capture_surface(
            str(adb_path), phone_serial
        )
        calibration_file = (
            Path(arguments.diagnostic_calibration_override).resolve()
            if arguments.diagnostic_calibration_override is not None
            else discover_active_profile_calibration(
                profile_root, camera_id=camera_id, phone_serial=phone_serial
            )
        )
        if calibration_file is None:
            raise RuntimeError("No active rig calibration is available")
        adb_packet = _capture_adb_frame(adb_path, phone_serial, surface)
        adapter = (
            create_camera_adapter(arguments.camera_adapter)
            if arguments.camera_adapter
            else HikMvsCameraAdapter(
                sdk_python_path=(
                    arguments.mvs_python_path
                    or settings["tools"].get("mvs_python_path")
                )
            )
        )
        captured = capture_standardized_rig_evidence(
            adb_packet, surface, calibration_file, adapter, output
        )
        result = {
            "schema_version": "1.0",
            "status": "captured",
            "operation": {
                "app_launch_or_input": "none",
                "rig_recalibration": False,
                "phone_display_power_change": "none",
            },
            "surface": dict(surface),
            "orientation": dict(captured.orientation),
            "capture": {
                "adb_image_space": adb_packet.metadata.get("image_space"),
                "full_camera_image_space": captured.full_camera_sample.metadata.get(
                    "image_space"
                ),
                "rectified_image_space": captured.rectified_packet.metadata.get(
                    "image_space"
                ),
                "expanded_full_camera_review_space": dict(
                    captured.full_camera_review.image_space
                ),
                "standardized_comparison_space": dict(
                    captured.comparison.image_space
                ),
                "rig_calibration": str(Path(calibration_file).resolve()),
            },
            "evidence": dict(captured.evidence),
            "evidence_spaces": dict(captured.evidence_spaces),
            "timing": dict(captured.timing),
        }
        (output / "result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        result = {
            "schema_version": "1.0",
            "status": "unavailable",
            "error": "{}: {}".format(type(exc).__name__, exc),
        }
        (output / "result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    print(json.dumps({**result, "output": str(output.resolve())}, indent=2))
    return 0 if result.get("status") == "captured" else 2


if __name__ == "__main__":
    raise SystemExit(main())
