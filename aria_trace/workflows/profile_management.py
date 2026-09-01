"""Publish, activate, and resolve production calibration profiles."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from aria_trace.adapters.filesystem.profile_registry import (
    AdapterRequest,
    PROFILE_KINDS,
    ProfileContext,
    ProfileRegistry,
    context_from_rig_calibration,
)
from aria_trace.adapters.hik.game_camera import _source_crop_to_canonical_phone
from aria_trace.adapters.android.spaces import natural_to_logical_matrix
from aria_trace.domain.spatial import (
    normalize_legacy_geometry,
    raster_space,
    require_spatial_geometry,
    transform_circle_similarity,
    transform_point,
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rig_calibration_file(path: Path) -> Path:
    value = Path(path)
    if value.is_dir():
        value = value / "hik_camera_calibration.json"
    value = value.resolve()
    if not value.is_file():
        raise FileNotFoundError("Rig calibration does not exist: {}".format(value))
    return value


def publish_rig_calibration(
    calibration: Path,
    *,
    registry: Optional[ProfileRegistry] = None,
    profile_root: Optional[Path] = None,
    activate: bool = True,
) -> Dict[str, Any]:
    calibration_file = _rig_calibration_file(calibration)
    document = _load_json(calibration_file)
    context = context_from_rig_calibration(document)
    runtime_files = {"hik_camera_calibration": calibration_file}
    for logical_name, filename in (
        ("hik_camera_calibration_yaml", "hik_camera_calibration.yaml"),
        ("rectification_maps", str((document.get("normalization") or {}).get("dense_map_file") or "")),
        ("last_camera_frame", "last_camera_frame.png"),
    ):
        if not filename:
            continue
        candidate = calibration_file.parent / filename
        if candidate.is_file():
            runtime_files[logical_name] = candidate
    store = registry or ProfileRegistry(profile_root)
    payload = {
        "profile_kind": "rig",
        "calibration_file": "hik_camera_calibration",
        "camera_id": context.camera_id,
        "phone_id": context.phone_id,
        "normalization_output_size_px": (document.get("normalization") or {}).get(
            "output_size_px"
        ),
        "capabilities": {
            "modes": ["full", "minimap", "dual"],
            "normalization": ["dense_remap", "homography", "none"],
            "locked_imaging": True,
        },
    }
    return store.publish(
        "rig",
        context,
        payload,
        runtime_files=runtime_files,
        provenance={"source_calibration": str(calibration_file)},
        review_state="accepted" if activate else "review_required",
        activate=activate,
    )


def _session_context(summary: Mapping[str, Any]) -> tuple[Path, Dict[str, Any], Dict[str, Any]]:
    session_path = Path(summary["provenance"]["session_path"]).resolve()
    manifest_path = session_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Localization session manifest is missing: {}".format(manifest_path))
    manifest = _load_json(manifest_path)
    context = dict(manifest.get("context") or {})
    return session_path, manifest, context


def _phone_id_from_manifest(manifest: Mapping[str, Any]) -> str:
    for source in manifest.get("frame_sources") or []:
        shared = source.get("shared_capture") or {}
        if source.get("stream_id") == "android_phone" and shared.get("serial"):
            return str(shared["serial"])
    raise ValueError("Session does not identify its Android phone")


def _logical_minimap_crop(summary: Mapping[str, Any]) -> list[int]:
    if summary.get("crop_xywh") is not None:
        return [int(value) for value in summary["crop_xywh"]]
    android = summary["android"]
    width, height = map(int, android["frame_size_px"])
    logical_space = raster_space(
        "android_logical_display_pixels", [width, height]
    )
    boundary = android["outer_boundary"]
    if "space" not in boundary:
        boundary = normalize_legacy_geometry(boundary, "circle", logical_space)
    else:
        boundary = require_spatial_geometry(boundary, "circle")
        boundary_space = boundary["space"]
        if boundary_space["space_id"] == logical_space["space_id"]:
            if boundary_space["size_px"] != [width, height]:
                raise ValueError("Mini-map boundary logical raster size is incompatible")
        elif boundary_space.get("parent_space_id") == logical_space["space_id"]:
            boundary = transform_circle_similarity(
                boundary,
                boundary_space["local_to_parent_3x3"],
                logical_space,
            )
        else:
            raise ValueError(
                "Cannot derive a logical mini-map crop from boundary space {!r}".format(
                    boundary_space["space_id"]
                )
            )
    center_x = float(boundary["center_x"])
    center_y = float(boundary["center_y"])
    radius = float(boundary["radius"])
    left = max(0, int(math.floor(center_x - radius)))
    top = max(0, int(math.floor(center_y - radius)))
    right = min(width, int(math.ceil(center_x + radius)))
    bottom = min(height, int(math.ceil(center_y + radius)))
    if right <= left or bottom <= top:
        raise ValueError("Localized mini-map does not produce a valid display crop")
    return [left, top, right - left, bottom - top]


def _geometry_to_canonical_phone(
    value: Mapping[str, Any],
    geometry_type: str,
    *,
    logical_size_px: Sequence[int],
    natural_size_px: Sequence[int],
    logical_crop_xywh: Sequence[int],
    quarter_turns_clockwise_from_natural: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Normalize source geometry and explicitly convert it to panel space."""

    logical_size = [int(item) for item in logical_size_px]
    natural_size = [int(item) for item in natural_size_px]
    crop = [int(item) for item in logical_crop_xywh]
    logical_space = raster_space("android_logical_display_pixels", logical_size)
    natural_space = raster_space(
        "android_phone_natural_display_pixels", natural_size
    )
    source = dict(value)
    if "space" not in source:
        source = normalize_legacy_geometry(source, geometry_type, logical_space)
    else:
        source = require_spatial_geometry(source, geometry_type)

    source_space = source["space"]
    source_id = source_space["space_id"]
    source_size = list(map(int, source_space["size_px"]))
    logical_to_natural = np.linalg.inv(
        natural_to_logical_matrix(
            natural_size, int(quarter_turns_clockwise_from_natural) % 4
        )
    )
    if source_id == natural_space["space_id"]:
        if source_size != natural_size:
            raise ValueError("Natural-panel geometry has an incompatible raster size")
        return source, source
    if source_id in (
        logical_space["space_id"],
        "profile_android_logical_display_pixels",
    ):
        if source_size != logical_size:
            raise ValueError("Android logical geometry has an incompatible raster size")
        source_to_natural = logical_to_natural
    elif source_size == crop[2:]:
        crop_to_logical = np.asarray(
            [[1.0, 0.0, crop[0]], [0.0, 1.0, crop[1]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        source_to_natural = logical_to_natural.dot(crop_to_logical)
    else:
        raise ValueError(
            "Cannot convert {} geometry from space {!r} size {} to canonical "
            "phone space".format(geometry_type, source_id, source_size)
        )
    if geometry_type == "circle":
        canonical = transform_circle_similarity(
            source, source_to_natural, natural_space
        )
    elif geometry_type == "point":
        canonical = transform_point(source, source_to_natural, natural_space)
    else:
        raise ValueError("Unsupported profile geometry {}".format(geometry_type))
    return source, canonical


def _profile_context_from_localization(
    summary: Mapping[str, Any], manifest: Mapping[str, Any], rig_context: Optional[ProfileContext]
) -> ProfileContext:
    capture_context = manifest.get("context") or {}
    surface = capture_context.get("phone_surface_orientation") or {}
    launch = capture_context.get("game_launch") or {}
    phone_id = _phone_id_from_manifest(manifest)
    android_result = summary.get("android") or {}
    logical = (
        surface.get("logical_size_px")
        or android_result.get("frame_size_px")
        or (summary.get("coordinate_space") or {}).get("frame_size_px")
    )
    if not logical:
        raise ValueError("Localization result does not declare its display dimensions")
    natural = surface.get("natural_size_px")
    if natural is None:
        turns = int(surface.get("quarter_turns_clockwise_from_natural", 0)) % 4
        natural = [logical[1], logical[0]] if turns % 2 else list(logical)
    panel = dict(rig_context.panel_display) if rig_context is not None else {
        "natural_panel_px": natural,
        "logical_frame_px": natural,
    }
    return ProfileContext(
        game_id=str(capture_context.get("game_id") or launch.get("game_id") or ""),
        platform="android",
        package=launch.get("package"),
        camera_adapter=(rig_context.camera_adapter if rig_context else "hik_mvs"),
        camera_id=(rig_context.camera_id if rig_context else None),
        phone_id=phone_id,
        phone_model=(rig_context.phone_model if rig_context else None),
        panel_display=panel,
        game_display={
            "natural_panel_px": natural,
            "logical_frame_px": logical,
            "game_viewport_xywh": [0, 0, int(logical[0]), int(logical[1])],
            "rotation_quarter_turns": int(
                surface.get("quarter_turns_clockwise_from_natural", 0)
            ),
            "ui_layout_id": "default",
        },
    )


def publish_minimap_profiles(
    localization_summary: Path,
    *,
    registry: Optional[ProfileRegistry] = None,
    profile_root: Optional[Path] = None,
    rig_calibration: Optional[Path] = None,
    activate: bool = False,
) -> Dict[str, Any]:
    summary_path = Path(localization_summary)
    if summary_path.is_dir():
        summary_path = summary_path / "localization_summary.json"
    summary_path = summary_path.resolve()
    summary = _load_json(summary_path)
    if summary.get("status") not in ("review_required", "accepted", "complete"):
        raise ValueError("Localization result is not publishable: {}".format(summary.get("status")))
    session_path, manifest, capture_context = _session_context(summary)
    selected_rig = rig_calibration
    if selected_rig is None:
        selected_rig = ((capture_context.get("hik_capture") or {}).get("rig_calibration"))
    store = registry or ProfileRegistry(profile_root)
    rig_profile = rig_context = None
    if selected_rig:
        rig_file = _rig_calibration_file(Path(selected_rig))
        rig_document = _load_json(rig_file)
        rig_context = context_from_rig_calibration(rig_document)
        rig_profile = publish_rig_calibration(
            rig_file, registry=store, activate=True
        )
    context = _profile_context_from_localization(summary, manifest, rig_context)
    logical_crop = _logical_minimap_crop(summary)
    surface = capture_context.get("phone_surface_orientation") or {}
    canonical_crop = summary.get("canonical_phone_crop_xywh")
    if canonical_crop is None:
        canonical_crop = _source_crop_to_canonical_phone(
            {
                "crop_xywh": logical_crop,
                "image_source": "android_scrcpy",
                "source_space": {"origin_in_canonical_phone_xy": [0, 0]},
                "phone_surface_orientation": surface,
            }
        )
    canonical_crop = [int(value) for value in canonical_crop]
    android_result = summary.get("android") or {}
    source_boundary = summary.get("source_boundary") or android_result.get(
        "outer_boundary"
    )
    fitted_boundary = summary.get("outer_boundary") or android_result.get(
        "outer_boundary"
    )
    if not isinstance(fitted_boundary, Mapping):
        raise ValueError("Localization result has no mini-map boundary geometry")
    logical_size = context.game_display["logical_frame_px"]
    natural_size = context.game_display["natural_panel_px"]
    turns = int(context.game_display["rotation_quarter_turns"])
    fitted_boundary, canonical_boundary = _geometry_to_canonical_phone(
        fitted_boundary,
        "circle",
        logical_size_px=logical_size,
        natural_size_px=natural_size,
        logical_crop_xywh=logical_crop,
        quarter_turns_clockwise_from_natural=turns,
    )
    if isinstance(source_boundary, Mapping):
        source_boundary, _ = _geometry_to_canonical_phone(
            source_boundary,
            "circle",
            logical_size_px=logical_size,
            natural_size_px=natural_size,
            logical_crop_xywh=logical_crop,
            quarter_turns_clockwise_from_natural=turns,
        )
    rotation_center = summary.get("rotation_center") or android_result.get(
        "rotation_center"
    )
    canonical_rotation_center = None
    if isinstance(rotation_center, Mapping):
        rotation_center, canonical_rotation_center = _geometry_to_canonical_phone(
            rotation_center,
            "point",
            logical_size_px=logical_size,
            natural_size_px=natural_size,
            logical_crop_xywh=logical_crop,
            quarter_turns_clockwise_from_natural=turns,
        )
    evidence = summary.get("evidence") or android_result.get(
        "verified_backend_evidence"
    ) or []
    phone_payload = {
        "profile_kind": "phone_game",
        "coordinate_space": "phone_natural_display_pixels",
        "canonical_phone_crop_xywh": canonical_crop,
        "android_logical_crop_xywh": logical_crop,
        "phone_surface_orientation": surface,
        "outer_boundary": canonical_boundary,
        "rotation_center": canonical_rotation_center,
        "source_boundary": source_boundary,
        "shift_estimation_mask": android_result.get("shift_estimation_mask"),
        "capabilities": {"adb_minimap": True, "camera_independent": True},
    }
    portable_files = {}
    shift_mask_value = android_result.get("shift_estimation_mask")
    if shift_mask_value:
        shift_mask_path = Path(str(shift_mask_value))
        if not shift_mask_path.is_absolute():
            shift_mask_path = summary_path.parent / shift_mask_path
        if shift_mask_path.is_file():
            portable_files["shift_estimation_mask"] = shift_mask_path
            phone_payload["shift_estimation_mask_runtime_file"] = (
                "shift_estimation_mask"
            )
    for index, evidence_value in enumerate(evidence):
        evidence_path = Path(str(evidence_value))
        if not evidence_path.is_absolute():
            evidence_path = summary_path.parent / evidence_path
        if evidence_path.is_file():
            portable_files["review_evidence_{:03d}".format(index)] = evidence_path
    phone_profile = store.publish(
        "phone_game",
        context,
        phone_payload,
        runtime_files=portable_files,
        provenance={
            "localization_summary": str(summary_path),
            "session_path": str(session_path),
            "evidence": evidence,
        },
        review_state="accepted" if activate else "review_required",
        activate=activate,
    )
    result = {
        "rig": rig_profile,
        "phone_game": phone_profile,
        "rig_game": None,
        "rig_game_color": None,
    }
    if rig_profile is not None:
        rig_game_payload = {
            "profile_kind": "rig_game",
            "canonical_phone_crop_xywh": canonical_crop,
            "composition_rule": (
                "Apply the exact rig revision, then use the exact phone-game "
                "crop. No optical geometry is fitted in this profile."
            ),
            "session_hik_observation": summary.get("rig_observation")
            or summary.get("hik_session_observation"),
            "capabilities": {
                "modes": ["minimap", "dual"],
            },
        }
        result["rig_game"] = store.publish(
            "rig_game",
            context,
            rig_game_payload,
            dependencies={
                "rig": rig_profile["revision_id"],
                "phone_game": phone_profile["revision_id"],
            },
            provenance={
                "localization_summary": str(summary_path),
                "session_path": str(session_path),
                "cross_source_registration": summary.get("cross_source_registration"),
            },
            review_state="accepted" if activate else "review_required",
            activate=activate,
        )
        conversion = summary.get("hik_bayer_conversion")
        if isinstance(conversion, Mapping):
            result["rig_game_color"] = store.publish(
                "rig_game_color",
                context,
                {
                    "profile_kind": "rig_game_color",
                    "hik_bayer_conversion": conversion,
                    "capabilities": {
                        "game_matched_color": conversion.get("status") == "selected",
                        "runtime_frame_passes": 0,
                    },
                },
                dependencies={"rig": rig_profile["revision_id"]},
                provenance={
                    "localization_summary": str(summary_path),
                    "session_path": str(session_path),
                    "producer": "legacy_combined_localization_result",
                },
                review_state="accepted" if activate else "review_required",
                activate=activate,
            )
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Manage production calibration profiles")
    value.add_argument("--profile-root", type=Path)
    subcommands = value.add_subparsers(dest="command", required=True)
    rig = subcommands.add_parser("publish-rig")
    rig.add_argument("calibration", type=Path)
    rig.add_argument("--candidate", action="store_true")
    minimap = subcommands.add_parser("publish-minimap")
    minimap.add_argument("localization", type=Path)
    minimap.add_argument("--rig-calibration", type=Path)
    minimap.add_argument("--activate", action="store_true")
    activate = subcommands.add_parser("activate")
    activate.add_argument("revision_id")
    list_profiles = subcommands.add_parser("list")
    list_profiles.add_argument("--kind", choices=PROFILE_KINDS)
    list_profiles.add_argument("--active-only", action="store_true")
    show = subcommands.add_parser("show")
    show.add_argument("revision_id")
    resolve = subcommands.add_parser("resolve")
    resolve.add_argument("--game-id")
    resolve.add_argument("--camera-id")
    resolve.add_argument("--phone-id")
    resolve.add_argument("--mode", choices=("full", "minimap", "dual"), default="full")
    resolve.add_argument("--normalization", default="auto")
    resolve.add_argument("--color-order", default="RGB")
    resolve.add_argument(
        "--color-policy",
        choices=("auto", "rig_locked", "game_matched", "unadjusted"),
        default="auto",
    )
    export = subcommands.add_parser(
        "export-adapter",
        help="write one registry-resolved adapter with embedded calibration data",
    )
    export.add_argument("output", type=Path)
    export.add_argument("--game-id")
    export.add_argument("--camera-id")
    export.add_argument("--phone-id")
    export.add_argument("--mode", choices=("full", "minimap", "dual"), default="full")
    export.add_argument("--normalization", default="auto")
    export.add_argument("--color-order", default="RGB")
    export.add_argument(
        "--color-policy",
        choices=("auto", "rig_locked", "game_matched", "unadjusted"),
        default="auto",
    )
    portable_export = subcommands.add_parser(
        "export-portable",
        help="export one camera-independent phone-game revision",
    )
    portable_export.add_argument("revision_id")
    portable_export.add_argument("output", type=Path)
    portable_import = subcommands.add_parser(
        "import-portable",
        help="import portable panel/game data as review-first local candidates",
    )
    portable_import.add_argument("package", type=Path)
    portable_import.add_argument("--camera-id")
    portable_import.add_argument("--phone-id")
    portable_import.add_argument("--game-id")
    portable_import.add_argument("--package-id")
    portable_import.add_argument("--game-version")
    portable_import.add_argument("--panel-size", type=int, nargs=2, metavar=("W", "H"))
    portable_import.add_argument("--game-size", type=int, nargs=2, metavar=("W", "H"))
    portable_import.add_argument(
        "--activate",
        action="store_true",
        help="explicitly activate the imported and locally composed revisions",
    )
    portable_import.add_argument("--no-compose-rig", action="store_true")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    registry = ProfileRegistry(arguments.profile_root)
    if arguments.command == "publish-rig":
        result = publish_rig_calibration(
            arguments.calibration, registry=registry, activate=not arguments.candidate
        )
        print("Rig profile: {} ({})".format(result["revision_id"], result["publication"]))
        return 0
    if arguments.command == "publish-minimap":
        result = publish_minimap_profiles(
            arguments.localization,
            registry=registry,
            rig_calibration=arguments.rig_calibration,
            activate=arguments.activate,
        )
        print("Phone-game profile: {}".format(result["phone_game"]["revision_id"]))
        if result["rig_game"] is not None:
            print("Rig-game profile: {}".format(result["rig_game"]["revision_id"]))
        if result["rig_game_color"] is not None:
            print(
                "Rig-game color profile: {}".format(
                    result["rig_game_color"]["revision_id"]
                )
            )
        if not arguments.activate:
            print("Profiles are review-required candidates; activate after evidence review.")
        return 0
    if arguments.command == "activate":
        result = registry.activate(arguments.revision_id)
        print("Activated profile: {}".format(result["revision_id"]))
        return 0
    if arguments.command == "list":
        print(
            json.dumps(
                registry.list_revisions(
                    kind=arguments.kind, active_only=arguments.active_only
                ),
                indent=2,
            )
        )
        return 0
    if arguments.command == "show":
        print(json.dumps(registry.revision(arguments.revision_id), indent=2))
        return 0
    if arguments.command == "export-portable":
        from aria_trace.workflows.portable_profiles import export_portable_profile

        result = export_portable_profile(
            arguments.revision_id, arguments.output, registry=registry
        )
        print("Portable calibration: {}".format(result["output"]))
        print("Source revision: {}".format(result["source_revision"]))
        return 0
    from aria_trace.adapters.filesystem.system_configuration import (
        load_system_configuration,
    )

    settings = load_system_configuration(arguments.profile_root)
    if arguments.command == "import-portable":
        from aria_trace.workflows.portable_profiles import import_portable_profile

        requested = ProfileContext(
            game_id=arguments.game_id or settings["game"].get("game_id"),
            package=arguments.package_id,
            game_version=arguments.game_version,
            camera_id=arguments.camera_id or settings["devices"].get("camera_id"),
            phone_id=arguments.phone_id or settings["devices"].get("phone_id"),
            panel_display=(
                {"natural_panel_px": arguments.panel_size}
                if arguments.panel_size
                else {}
            ),
            game_display=(
                {
                    "natural_panel_px": arguments.panel_size or arguments.game_size,
                    "logical_frame_px": arguments.game_size,
                    "game_viewport_xywh": [
                        0,
                        0,
                        int(arguments.game_size[0]),
                        int(arguments.game_size[1]),
                    ],
                }
                if arguments.game_size
                else {}
            ),
        )
        result = import_portable_profile(
            arguments.package,
            registry=registry,
            requested_context=requested,
            activate=arguments.activate,
            compose_local_rig=not arguments.no_compose_rig,
        )
        print(
            "Imported portable profile: {}".format(
                result["portable_profile"]["revision_id"]
            )
        )
        if result["rig_game"] is not None:
            print("Local rig-game profile: {}".format(result["rig_game"]["revision_id"]))
        for message in result["warnings"]:
            print("Warning: {}".format(message))
        if result["local_fit_required"]:
            print(
                "Warning: portable ADB color reference imported; local HIK color "
                "fitting is still required"
            )
        if not arguments.activate:
            print("Imported revisions are review-required candidates; use --activate after review.")
        return 0
    context = ProfileContext(
        game_id=arguments.game_id or settings["game"].get("game_id"),
        camera_id=arguments.camera_id or settings["devices"].get("camera_id"),
        phone_id=arguments.phone_id or settings["devices"].get("phone_id"),
    )
    request = AdapterRequest(
        mode=arguments.mode,
        normalization=arguments.normalization,
        color_order=arguments.color_order,
        color_policy=arguments.color_policy,
    )
    if arguments.command == "export-adapter":
        from aria_trace.workflows.adapter_export import export_resolved_adapter

        result = export_resolved_adapter(
            arguments.output,
            registry=registry,
            context=context,
            request=request,
        )
        print("Standalone camera adapter: {}".format(result["output"]))
        print("Embedded profiles: {}".format(result["profile_revisions"]))
        return 0
    print(json.dumps(registry.resolve_adapter(context, request), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
