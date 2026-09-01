"""Standalone game/app geometry and orientation repeatability check."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from aria_trace.adapters.android.spaces import natural_to_logical_matrix
from aria_trace.adapters.android.game_launcher import foreground_component
from aria_trace.adapters.android.phone import (
    AdbPhoneSession,
    probe_android_capture_surface,
    resolve_adb_executable,
)
from aria_trace.adapters.filesystem.profile_registry import (
    ProfileContext,
    ProfileRegistry,
)
from aria_trace.adapters.filesystem.system_configuration import (
    load_system_configuration,
)
from aria_trace.adapters.hik.capture import CalibratedHikFrameSource
from aria_trace.adapters.hik.driver import HikMvsCameraAdapter, RectifiedHikCamera
from aria_trace.adapters.rig.devices import create_camera_adapter
from aria_trace.adapters.sources import AdbScreenshotFrameSource
from aria_trace.services.calibration.game_repeatability import (
    compare_thresholded_app_geometry,
    evaluate_minimap_static_geometry,
)
from aria_trace.services.calibration.rig.cross_source import (
    match_game_camera_orientation,
    natural_crop_to_logical,
)
from aria_trace.workflows.rig_reuse_precheck import (
    discover_active_profile_calibration,
)


def _component_package(component: Optional[str]) -> Optional[str]:
    return str(component).split("/", 1)[0] if component else None


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
        raise RuntimeError("ADB returned no game screenshot")
    return packet


def _profile_for_current_game(
    registry: ProfileRegistry,
    context: ProfileContext,
    *,
    revision_id: Optional[str],
) -> Mapping[str, object]:
    if revision_id:
        return registry.resolve_revision(
            revision_id, context, expected_kind="phone_game"
        )
    candidates = []
    requested_panel = context.panel_display.get("natural_panel_px")
    for row in registry.list_revisions(kind="phone_game", active_only=True):
        if context.game_id and row.get("game_id") != context.game_id:
            continue
        profile = registry.revision(str(row["revision_id"]))
        stored = ProfileContext.from_dict(profile.get("context") or {})
        stored_panel = stored.panel_display.get("natural_panel_px")
        panel_matches = (
            requested_panel is None
            or stored_panel is None
            or list(stored_panel) == list(requested_panel)
        )
        if panel_matches:
            candidates.append(profile)
    if not candidates:
        raise RuntimeError(
            "No active phone-game profile matches game {!r} and panel {}".format(
                context.game_id, requested_panel
            )
        )
    selected = candidates[0]
    selected["resolution"] = {
        "selection": "newest_active_same_game_and_panel_ignoring_runtime_rotation",
        "candidate_count": len(candidates),
    }
    return selected


def _logical_profile_crop(
    profile: Mapping[str, object], surface: Mapping[str, object]
) -> Tuple[list, Mapping[str, object]]:
    payload = dict(profile.get("payload") or profile)
    canonical = payload.get("canonical_phone_crop_xywh")
    stored_surface = dict(payload.get("phone_surface_orientation") or {})
    current_turns = int(surface["quarter_turns_clockwise_from_natural"]) % 4
    if canonical is not None:
        crop = natural_crop_to_logical(
            canonical,
            surface["natural_size_px"],
            current_turns,
        )
    else:
        crop = payload.get("android_logical_crop_xywh")
        if crop is None:
            raise RuntimeError("Phone-game profile has no mini-map crop")
        stored_turns = int(
            stored_surface.get("quarter_turns_clockwise_from_natural", current_turns)
        ) % 4
        if stored_turns != current_turns:
            raise RuntimeError(
                "Phone-game profile lacks a canonical mini-map crop for the "
                "current Android surface rotation"
            )
    boundary = payload.get("outer_boundary")
    if not isinstance(boundary, Mapping):
        raise RuntimeError("Phone-game profile has no calibrated mini-map boundary")
    crop = list(map(int, crop))
    stored_turns = int(
        stored_surface.get("quarter_turns_clockwise_from_natural", current_turns)
    ) % 4
    natural_size = surface["natural_size_px"]
    stored_to_natural = np.linalg.inv(
        natural_to_logical_matrix(natural_size, stored_turns)
    )
    natural_to_current = natural_to_logical_matrix(natural_size, current_turns)
    stored_center = np.asarray(
        [float(boundary["center_x"]), float(boundary["center_y"]), 1.0],
        dtype=np.float64,
    )
    current_center = natural_to_current.dot(stored_to_natural).dot(stored_center)
    local_boundary = dict(boundary)
    local_boundary.update(
        center_x=float(current_center[0]) - float(crop[0]),
        center_y=float(current_center[1]) - float(crop[1]),
        coordinate_space="current_minimap_crop_pixels",
        source_coordinate_space="profile_android_logical_display_pixels",
    )
    return crop, local_boundary


def _save_image(output: Path, name: str, image) -> str:
    path = output / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError("Could not save {}".format(path))
    return name.replace("\\", "/")


def apply_orientation_result(hik_source, orientation: Mapping[str, object]) -> int:
    """Apply a checked app-up quarter-turn to an integrator-owned HIK source."""

    turns = int(
        orientation[
            "selected_camera_adapter_image_quarter_turns_clockwise_from_calibration_display"
        ]
    ) % 4
    setter = getattr(hik_source, "set_output_orientation", None)
    if not callable(setter):
        raise TypeError(
            "HIK source does not expose set_output_orientation; use "
            "CalibratedHikFrameSource or consume the structured quarter-turn"
        )
    setter(
        turns,
        {
            "status": orientation.get("status"),
            "selection_basis": orientation.get("selection_basis"),
            "selected_confidence": orientation.get("selected_confidence"),
            "confidence_margin": orientation.get("confidence_margin"),
        },
    )
    return turns


def _camera_candidate_agrees_with_android_surface(
    selected_candidate_turns: int, android_surface_turns: int
) -> bool:
    """Compare inverse descriptions of the same physical panel rotation."""

    # Android reports the clockwise rotation applied to the logical surface.
    # The image matcher reports the clockwise adapter-output rotation required
    # to undo it and make app-up camera-up.
    return (int(selected_candidate_turns) + int(android_surface_turns)) % 4 == 0


def _reference_from_result(path: Path):
    value = Path(path)
    result_path = value / "result.json" if value.is_dir() else value
    document = json.loads(result_path.read_text(encoding="utf-8"))
    image_path = result_path.parent / str(
        (document.get("evidence") or {}).get("adb_current")
    )
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Diagnostic reference image is missing: {}".format(image_path))
    image_space = (document.get("capture") or {}).get("adb_image_space")
    if not image_space:
        raise RuntimeError("Diagnostic reference has no ADB image-space metadata")
    return image, image_space, result_path


def run_game_repeatability_check(
    output: Path,
    *,
    profile_root: Optional[Path] = None,
    game_id: Optional[str] = None,
    phone_game_revision: Optional[str] = None,
    camera_id: Optional[str] = None,
    phone_serial: Optional[str] = None,
    adb: Optional[str] = None,
    mvs_python_path: Optional[str] = None,
    expected_package: Optional[str] = None,
    adb_only: bool = False,
    create_diagnostic_reference: bool = False,
    diagnostic_reference_result: Optional[Path] = None,
    rig_calibration: Optional[Path] = None,
    camera_adapter: Optional[str] = None,
) -> Mapping[str, object]:
    """Check one running app without launching it, touching it, or changing display state."""

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    settings = load_system_configuration(profile_root)
    profile_root = Path(settings["effective_profile_root"])
    game_id = game_id or settings["game"].get("game_id")
    camera_id = camera_id or settings["devices"].get("camera_id")
    phone_serial = phone_serial or settings["devices"].get("phone_id")
    adb_path = Path(resolve_adb_executable(adb or settings["tools"].get("adb")))
    mvs_python_path = mvs_python_path or settings["tools"].get("mvs_python_path")
    phone_serial, surface = probe_android_capture_surface(
        str(adb_path), phone_serial
    )
    phone = AdbPhoneSession(phone_serial, adb_executable=str(adb_path))
    orientation_settings = phone.orientation_settings()
    foreground = foreground_component(phone)
    foreground_package = _component_package(foreground)
    adb_packet = _capture_adb_frame(adb_path, phone_serial, surface)
    adb_image = adb_packet.image
    evidence = {
        "adb_current": _save_image(output, "adb_current.png", adb_image)
    }
    context = ProfileContext(
        game_id=game_id,
        platform="android",
        package=foreground_package,
        camera_id=camera_id,
        phone_id=phone_serial,
        panel_display={"natural_panel_px": surface["natural_size_px"]},
        game_display={
            "logical_frame_px": surface["logical_size_px"],
            "rotation_quarter_turns": surface[
                "quarter_turns_clockwise_from_natural"
            ],
        },
    )
    profile = None
    if create_diagnostic_reference and diagnostic_reference_result is not None:
        raise ValueError(
            "Choose either diagnostic reference creation or comparison, not both"
        )
    if create_diagnostic_reference:
        geometry = {
            "method": "fixed_threshold_full_app_diagnostic",
            "status": "reference_created",
            "source": "diagnostic_full_app_fixed_threshold_reference",
            "matches": True,
        }
        geometry_images = {}
    elif diagnostic_reference_result is None:
        if not game_id and not phone_game_revision:
            raise RuntimeError(
                "A game ID or explicit phone-game revision is required outside "
                "diagnostic reference mode"
            )
        profile = _profile_for_current_game(
            ProfileRegistry(profile_root),
            context,
            revision_id=phone_game_revision,
        )
        crop, boundary = _logical_profile_crop(profile, surface)
        geometry, geometry_images = evaluate_minimap_static_geometry(
            adb_image, crop, boundary
        )
        geometry["source"] = "phone_game_profile_static_minimap_boundary"
    else:
        reference, reference_space, reference_result = _reference_from_result(
            diagnostic_reference_result
        )
        geometry, geometry_images = compare_thresholded_app_geometry(
            reference, adb_image
        )
        geometry.update(
            source="diagnostic_full_app_fixed_threshold_reference",
            reference_result=str(reference_result.resolve()),
            reference_image_space=reference_space,
        )
    for name, image in geometry_images.items():
        evidence[name.rsplit(".", 1)[0]] = _save_image(output, name, image)

    profile_context = (
        ProfileContext.from_dict(profile.get("context") or {}) if profile else None
    )
    profile_package = profile_context.package if profile_context else None
    expected_package = expected_package or profile_package
    app_match = (
        foreground_package == expected_package if expected_package else None
    )

    camera_orientation = {
        "status": "not_requested",
        "selected_camera_adapter_image_quarter_turns_clockwise_from_calibration_display": None,
    }
    hik_packet = None
    calibration_file = None
    if not adb_only:
        calibration_file = (
            Path(rig_calibration).resolve()
            if rig_calibration is not None
            else discover_active_profile_calibration(
                profile_root, camera_id=camera_id, phone_serial=phone_serial
            )
        )
        if calibration_file is None:
            raise RuntimeError("No active rig profile is available for camera orientation")
        adapter = (
            create_camera_adapter(camera_adapter)
            if camera_adapter
            else HikMvsCameraAdapter(sdk_python_path=mvs_python_path)
        )
        reader = RectifiedHikCamera(calibration_file, adapter=adapter, rectify=True)
        hik = CalibratedHikFrameSource(
            calibration_file,
            "hik_phone",
            rectify=True,
            output_quarter_turns_clockwise=0,
            reader=reader,
        )
        try:
            hik.start()
            initial = hik.read()
            if initial is None:
                raise RuntimeError("HIK returned no orientation evidence frame")
            calibration_display = hik.alignment_evidence_image(initial)
            camera_orientation, orientation_images = match_game_camera_orientation(
                adb_image,
                calibration_display,
                calibration_file,
                android_reported_quarter_turns=int(
                    surface["quarter_turns_clockwise_from_natural"]
                ),
            )
            apply_orientation_result(hik, camera_orientation)
            hik_packet = hik.read()
            if hik_packet is None:
                raise RuntimeError("HIK returned no oriented verification frame")
            evidence["hik_oriented"] = _save_image(
                output, "hik_oriented.png", hik_packet.image
            )
            for name, image in orientation_images.items():
                evidence["orientation/{}".format(name.rsplit(".", 1)[0])] = (
                    _save_image(output, "orientation/{}".format(name), image)
                )
        finally:
            hik.stop()

    orientation_agrees = (
        None
        if adb_only
        else _camera_candidate_agrees_with_android_surface(
            int(
                camera_orientation[
                    "selected_adb_surface_quarter_turns_clockwise_from_phone_natural"
                ]
            ),
            int(surface["quarter_turns_clockwise_from_natural"]),
        )
    )
    matches = bool(geometry.get("matches")) and app_match is not False
    if not adb_only:
        matches = matches and str(camera_orientation.get("status", "")).startswith(
            "selected"
        ) and bool(orientation_agrees)
    result = {
        "schema_version": "1.0",
        "status": "match" if matches else "mismatch",
        "matches": matches,
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "operation": {
            "phone_operations": "read_only_adb_queries_and_screencap",
            "app_launch_or_input": "none",
            "rig_recalibration": False,
        },
        "application": {
            "foreground_component": foreground,
            "foreground_package": foreground_package,
            "expected_package": expected_package,
            "matches_expected": app_match,
        },
        "orientation": {
            "settings": orientation_settings,
            "surface": surface,
            "camera_image": camera_orientation,
            "camera_and_surface_agree": orientation_agrees,
            "adapter_action": {
                "method": "CalibratedHikFrameSource.set_output_orientation",
                "python_api": (
                    "aria_trace.workflows.game_repeatability."
                    "apply_orientation_result(source, result['orientation']['camera_image'])"
                ),
                "quarter_turns_clockwise_from_calibration_display": (
                    camera_orientation.get(
                        "selected_camera_adapter_image_quarter_turns_clockwise_from_calibration_display"
                    )
                ),
                "full_rig_calibration_required": False,
            },
        },
        "geometry": geometry,
        "capture": {
            "adb_image_space": adb_packet.metadata.get("image_space"),
            "hik_image_space": (
                hik_packet.metadata.get("image_space")
                if hik_packet is not None
                else None
            ),
            "rig_calibration": (
                str(calibration_file.resolve()) if calibration_file else None
            ),
        },
        "profile": (
            {
                "revision_id": profile.get("revision_id"),
                "resolution": profile.get("resolution"),
            }
            if profile
            else None
        ),
        "evidence": evidence,
        "timing": {"completed_unix_time": time.time()},
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Check prepared-game static geometry, foreground app, Android rotation, "
            "and HIK image orientation without recalibrating or operating the phone."
        )
    )
    value.add_argument("--profile-root", type=Path)
    value.add_argument("--game-id")
    value.add_argument("--phone-game-revision")
    value.add_argument("--camera-id")
    value.add_argument("--phone-serial")
    value.add_argument("--adb")
    value.add_argument("--mvs-python-path")
    value.add_argument("--expected-package")
    value.add_argument("--adb-only", action="store_true")
    value.add_argument("--create-diagnostic-reference", action="store_true")
    value.add_argument("--diagnostic-reference-result", type=Path)
    value.add_argument("--diagnostic-rig-calibration", type=Path)
    value.add_argument(
        "--camera-adapter",
        help="optional module:function CameraAdapter factory for a non-HIK rig",
    )
    value.add_argument("--output", type=Path)
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    output = arguments.output or (
        Path("artifacts")
        / "game-repeatability-{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    )
    try:
        result = run_game_repeatability_check(
            output,
            profile_root=arguments.profile_root,
            game_id=arguments.game_id,
            phone_game_revision=arguments.phone_game_revision,
            camera_id=arguments.camera_id,
            phone_serial=arguments.phone_serial,
            adb=arguments.adb,
            mvs_python_path=arguments.mvs_python_path,
            expected_package=arguments.expected_package,
            adb_only=arguments.adb_only,
            create_diagnostic_reference=arguments.create_diagnostic_reference,
            diagnostic_reference_result=arguments.diagnostic_reference_result,
            rig_calibration=arguments.diagnostic_rig_calibration,
            camera_adapter=arguments.camera_adapter,
        )
    except Exception as exc:
        output.mkdir(parents=True, exist_ok=True)
        result = {
            "schema_version": "1.0",
            "status": "unavailable",
            "matches": False,
            "error": "{}: {}".format(type(exc).__name__, exc),
        }
        (output / "result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
    print(json.dumps({**result, "output": str(output.resolve())}, indent=2))
    return 0 if result.get("status") == "match" else 2


if __name__ == "__main__":
    raise SystemExit(main())
