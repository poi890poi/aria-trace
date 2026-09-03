"""Publish, activate, and resolve production calibration profiles."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from rig_runtime.adapters.filesystem.profile_registry import (
    AdapterRequest,
    PROFILE_KINDS,
    ProfileContext,
    ProfileRegistry,
    ProfileResolutionError,
    context_from_rig_calibration,
)
from rig_runtime.adapters.hik.game_camera import _source_crop_to_canonical_phone
from rig_runtime.adapters.android.spaces import natural_to_logical_matrix
from rig_runtime.domain.spatial import (
    normalize_legacy_geometry,
    raster_space,
    require_spatial_geometry,
    transform_circle_similarity,
    transform_point,
)


DEFAULT_GAME_MODEL = {
    "schema_version": "1.0",
    "cursor_follows": "character",
    "cursor_behavior_by_acquisition": {
        "zigzag": "static",
        "micro_movement": "rotating",
    },
    "minimap_orientation": "unspecified",
    "source": "iris_default",
}
CURSOR_FOLLOWS_VALUES = ("character", "camera")
MINIMAP_ORIENTATION_VALUES = ("unspecified", "rotating", "north_up")


def cursor_behavior_by_acquisition(cursor_follows: str) -> Dict[str, str]:
    """Map physical acquisition motion to the cursor response it should expose."""

    selected = str(cursor_follows)
    if selected == "character":
        return {"zigzag": "static", "micro_movement": "rotating"}
    if selected == "camera":
        return {"zigzag": "rotating", "micro_movement": "static"}
    raise ValueError("cursor_follows must be character or camera")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_game_model(
    registry: ProfileRegistry,
    game_id: str,
    *,
    platform: str = "android",
    package: Optional[str] = None,
    game_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve declared game behavior or return the documented safe default."""

    context = ProfileContext(
        game_id=game_id,
        platform=platform,
        package=package,
        game_version=game_version,
    )
    try:
        profile = registry.resolve("game_model", context)
    except ProfileResolutionError:
        return {**DEFAULT_GAME_MODEL, "game_id": context.game_id, "revision_id": None}
    payload = {**DEFAULT_GAME_MODEL, **dict(profile.get("payload") or {})}
    payload["cursor_behavior_by_acquisition"] = cursor_behavior_by_acquisition(
        str(payload["cursor_follows"])
    )
    payload.update(
        game_id=context.game_id,
        revision_id=profile["revision_id"],
        source="active_game_model_profile",
    )
    return payload


def publish_game_model(
    registry: ProfileRegistry,
    game_id: str,
    *,
    cursor_follows: Optional[str] = None,
    minimap_orientation: Optional[str] = None,
    platform: str = "android",
    package: Optional[str] = None,
    game_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Publish one portable game-ID-scoped behavior model."""

    current = resolve_game_model(
        registry,
        game_id,
        platform=platform,
        package=package,
        game_version=game_version,
    )
    selected_cursor = str(cursor_follows or current["cursor_follows"])
    selected_map = str(minimap_orientation or current["minimap_orientation"])
    if selected_cursor not in CURSOR_FOLLOWS_VALUES:
        raise ValueError("cursor_follows must be character or camera")
    if selected_map not in MINIMAP_ORIENTATION_VALUES:
        raise ValueError(
            "minimap_orientation must be unspecified, rotating, or north_up"
        )
    context = ProfileContext(
        game_id=game_id,
        platform=platform,
        package=package,
        game_version=game_version,
    )
    return registry.publish(
        "game_model",
        context,
        {
            "schema_version": "1.0",
            "profile_kind": "game_model",
            "cursor_follows": selected_cursor,
            "minimap_orientation": selected_map,
            "cursor_behavior_by_acquisition": cursor_behavior_by_acquisition(
                selected_cursor
            ),
        },
        provenance={"configured_by": "iris_tools profiles configure-game"},
        review_state="accepted",
        activate=True,
    )


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
        (
            "valid_screen_mask",
            str(
                (document.get("normalization") or {}).get("valid_mask_file")
                or "valid_screen_mask.png"
            ),
        ),
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
    profile = store.publish(
        "rig",
        context,
        payload,
        runtime_files=runtime_files,
        provenance={"source_calibration": str(calibration_file)},
        review_state="accepted" if activate else "review_required",
        activate=activate,
    )
    reconciliation = (
        reconcile_active_rig_dependents(profile, registry=store, activate=True)
        if activate
        else {
            "recomposed": {"rig_game": [], "rig_game_orientation": []},
            "requires_fresh_evidence": {"rig_game_color": []},
        }
    )
    profile["rig_dependent_reconciliation"] = reconciliation
    # Preserve the established result keys for callers while the structured
    # reconciliation report becomes the single owner of derived-profile state.
    profile["recomposed_rig_game_profiles"] = reconciliation["recomposed"][
        "rig_game"
    ]
    profile["recomposed_rig_game_orientation_profiles"] = reconciliation[
        "recomposed"
    ]["rig_game_orientation"]
    return profile


def _rig_game_payload_from_phone_game(
    phone_profile: Mapping[str, Any],
) -> Dict[str, Any]:
    phone_payload = dict(phone_profile.get("payload") or {})
    return {
        "profile_kind": "rig_game",
        "canonical_phone_crop_xywh": phone_payload.get(
            "canonical_phone_crop_xywh"
        ),
        "outer_boundary": phone_payload.get("outer_boundary"),
        "rotation_center": phone_payload.get("rotation_center"),
        "cursor_geometry": phone_payload.get("cursor_geometry"),
        "composition_rule": (
            "Apply the exact rig revision, then use portable phone-game "
            "geometry in canonical phone-panel coordinates. No game images "
            "or optical geometry are fitted while composing this profile."
        ),
        "capabilities": {
            "modes": ["minimap", "dual"],
            "recomposable_after_rig_displacement": True,
        },
    }


def _rig_phone_game_context(
    rig_context: ProfileContext,
    phone_context: ProfileContext,
) -> ProfileContext:
    return ProfileContext(
        platform=phone_context.platform,
        camera_id=rig_context.camera_id,
        phone_id=rig_context.phone_id,
        phone_model=rig_context.phone_model,
        panel_display=dict(rig_context.panel_display),
        game_id=phone_context.game_id,
        package=phone_context.package,
        game_version=phone_context.game_version,
        game_display=dict(phone_context.game_display),
    )


def _portable_panel_geometry_matches(
    rig_context: ProfileContext,
    phone_context: ProfileContext,
) -> bool:
    """Match portable game geometry by physical panel raster, not all metrics."""

    rig_size = rig_context.panel_display.get("natural_panel_px")
    phone_size = phone_context.panel_display.get("natural_panel_px")
    if rig_size is not None and phone_size is not None:
        return list(map(int, rig_size)) == list(map(int, phone_size))
    return rig_context.panel_signature == phone_context.panel_signature


def _rig_calibration_display_turns(
    rig_profile: Mapping[str, Any], registry: ProfileRegistry
) -> int:
    """Return the rig-normalized display orientation relative to panel rotation 0."""

    calibration = _load_json(
        registry.runtime_file(rig_profile, "hik_camera_calibration")
    )
    phone = dict(calibration.get("phone") or {})
    viewer = dict(phone.get("viewer") or {})
    return int(
        phone.get(
            "orientation_quarter_turns",
            viewer.get("canonical_orientation_quarter_turns", 0),
        )
    ) % 4


def recompose_active_rig_game_profiles(
    rig_profile: Mapping[str, Any],
    *,
    registry: ProfileRegistry,
    activate: bool = True,
) -> list[Dict[str, Any]]:
    """Compose active portable phone-game geometry with one exact rig revision."""

    rig_context = ProfileContext.from_dict(rig_profile.get("context") or {})
    recomposed = []
    composed_variants = set()
    for item in registry.list_revisions(kind="phone_game", active_only=True):
        phone_profile = registry.revision(str(item["revision_id"]))
        phone_context = ProfileContext.from_dict(
            phone_profile.get("context") or {}
        )
        if phone_context.platform != rig_context.platform:
            continue
        if not _portable_panel_geometry_matches(rig_context, phone_context):
            continue
        target_variant = (
            phone_context.platform,
            phone_context.game_id,
            phone_context.game_display_signature,
        )
        if target_variant in composed_variants:
            continue
        composed_variants.add(target_variant)
        context = _rig_phone_game_context(rig_context, phone_context)
        result = registry.publish(
            "rig_game",
            context,
            _rig_game_payload_from_phone_game(phone_profile),
            dependencies={
                "rig": str(rig_profile["revision_id"]),
                "phone_game": str(phone_profile["revision_id"]),
            },
            provenance={
                "composition": "active_phone_game_plus_new_rig",
                "source_phone_game_revision": phone_profile["revision_id"],
                "source_rig_revision": rig_profile["revision_id"],
                "panel_compatibility": (
                    "same_natural_panel_raster; refresh, density, and source "
                    "phone metadata do not change portable game geometry"
                ),
                "source_phone_game_provenance": phone_profile.get("provenance"),
            },
            review_state="accepted",
            activate=activate,
        )
        recomposed.append(result)
    return recomposed


def recompose_active_rig_game_orientation_profiles(
    rig_profile: Mapping[str, Any],
    *,
    registry: ProfileRegistry,
    activate: bool = True,
) -> list[Dict[str, Any]]:
    """Re-express portable game-up orientation against one new rig revision."""

    rig_context = ProfileContext.from_dict(rig_profile.get("context") or {})
    target_display_turns = _rig_calibration_display_turns(rig_profile, registry)
    recomposed = []
    composed_variants = set()
    for item in registry.list_revisions(
        kind="rig_game_orientation", active_only=True
    ):
        source = registry.revision(str(item["revision_id"]))
        source_context = ProfileContext.from_dict(source.get("context") or {})
        if source_context.platform != rig_context.platform:
            continue
        if not _portable_panel_geometry_matches(rig_context, source_context):
            continue
        target_variant = (
            source_context.platform,
            source_context.game_id,
            source_context.game_display_signature,
        )
        if target_variant in composed_variants:
            continue
        composed_variants.add(target_variant)
        context = _rig_phone_game_context(rig_context, source_context)
        source_payload = dict(source.get("payload") or {})
        source_image_turns = int(
            source_payload.get(
                "camera_adapter_image_quarter_turns_clockwise_from_calibration_display",
                0,
            )
        ) % 4
        source_rig_revision = str(
            (source.get("dependencies") or {}).get("rig") or ""
        )
        portable_surface_turns = source_payload.get(
            "game_surface_quarter_turns_clockwise_from_phone_natural"
        )
        if portable_surface_turns is None:
            if not source_rig_revision:
                raise ProfileResolutionError(
                    "Game-orientation profile {} has neither a portable game "
                    "surface orientation nor its source rig dependency".format(
                        source["revision_id"]
                    )
                )
            source_rig = registry.revision(source_rig_revision)
            source_display_turns = _rig_calibration_display_turns(
                source_rig, registry
            )
            portable_surface_turns = (
                source_image_turns + source_display_turns
            ) % 4
            portable_basis = "derived_from_legacy_source_rig_and_relative_turn"
        else:
            portable_surface_turns = int(portable_surface_turns) % 4
            source_display_turns = None
            portable_basis = "stored_portable_game_surface_orientation"
        target_image_turns = (
            int(portable_surface_turns) - target_display_turns
        ) % 4
        target_payload = dict(source_payload)
        target_payload.update(
            game_surface_quarter_turns_clockwise_from_phone_natural=int(
                portable_surface_turns
            ),
            camera_adapter_image_quarter_turns_clockwise_from_calibration_display=(
                target_image_turns
            ),
            orientation_space_contract={
                "portable_source_space": "phone_natural_rotation_0",
                "adapter_base_space": "rig_calibration_display",
                "composition": (
                    "adapter_turn = game_surface_turn - "
                    "rig_calibration_display_turn (mod 4)"
                ),
            },
        )
        result = registry.publish(
            "rig_game_orientation",
            context,
            target_payload,
            dependencies={"rig": str(rig_profile["revision_id"])},
            provenance={
                "composition": "portable_game_surface_orientation_plus_new_rig",
                "source_orientation_revision": source["revision_id"],
                "source_rig_revision": source_rig_revision or None,
                "target_rig_revision": rig_profile["revision_id"],
                "source_adapter_turns": source_image_turns,
                "source_calibration_display_turns": source_display_turns,
                "portable_game_surface_turns": int(portable_surface_turns),
                "portable_orientation_basis": portable_basis,
                "target_calibration_display_turns": target_display_turns,
                "target_adapter_turns": target_image_turns,
                "transfer_basis": (
                    "game surface is portable in phone-natural rotation-0 space; "
                    "the adapter-relative turn is recomputed for the target rig"
                ),
                "source_orientation_provenance": source.get("provenance"),
            },
            review_state="accepted",
            activate=activate,
        )
        recomposed.append(result)
    return recomposed


def reconcile_active_rig_dependents(
    rig_profile: Mapping[str, Any],
    *,
    registry: ProfileRegistry,
    activate: bool = True,
) -> Dict[str, Any]:
    """Reconcile every active game profile affected by one new rig revision."""

    rig_context = ProfileContext.from_dict(rig_profile.get("context") or {})
    stale_color = []
    for item in registry.list_revisions(kind="rig_game_color", active_only=True):
        source = registry.revision(str(item["revision_id"]))
        source_context = ProfileContext.from_dict(source.get("context") or {})
        if source_context.platform != rig_context.platform:
            continue
        if source_context.camera_id != rig_context.camera_id:
            continue
        if not _portable_panel_geometry_matches(rig_context, source_context):
            continue
        source_rig = str((source.get("dependencies") or {}).get("rig") or "")
        if source_rig == str(rig_profile["revision_id"]):
            continue
        stale_color.append(
            {
                "profile_revision": source["revision_id"],
                "game_id": source_context.game_id,
                "source_rig_revision": source_rig or None,
                "target_rig_revision": rig_profile["revision_id"],
                "action": (
                    "adapter_falls_back_to_rig_locked_until_fresh_synchronized_"
                    "game_calibration_publishes_a_local_hik_fit"
                ),
            }
        )
    return {
        "schema_version": "1.0",
        "rig_revision": rig_profile["revision_id"],
        "recomposed": {
            "rig_game": recompose_active_rig_game_profiles(
                rig_profile, registry=registry, activate=activate
            ),
            "rig_game_orientation": recompose_active_rig_game_orientation_profiles(
                rig_profile, registry=registry, activate=activate
            ),
        },
        "requires_fresh_evidence": {"rig_game_color": stale_color},
    }


def _session_context(summary: Mapping[str, Any]) -> tuple[Path, Dict[str, Any], Dict[str, Any]]:
    session_path = Path(summary["provenance"]["session_path"]).resolve()
    manifest_path = session_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("Localization session manifest is missing: {}".format(manifest_path))
    manifest = _load_json(manifest_path)
    context = dict(manifest.get("context") or {})
    return session_path, manifest, context


def _phone_id_from_manifest(
    manifest: Mapping[str, Any], fallback: Optional[str] = None
) -> str:
    for source in manifest.get("frame_sources") or []:
        shared = source.get("shared_capture") or {}
        if source.get("stream_id") == "android_phone" and shared.get("serial"):
            return str(shared["serial"])
        if source.get("stream_id") == "android_phone" and source.get("serial"):
            return str(source["serial"])
    if fallback:
        return str(fallback)
    raise ValueError("Session does not identify its Android phone")


def _logical_minimap_crop(summary: Mapping[str, Any]) -> list[int]:
    if summary.get("crop_xywh") is not None:
        return [int(value) for value in summary["crop_xywh"]]
    calibration_config = summary.get("config") or {}
    if calibration_config.get("crop_xywh") is not None:
        return [int(value) for value in calibration_config["crop_xywh"]]
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
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    rig_context: Optional[ProfileContext],
    *,
    game_id: Optional[str] = None,
    phone_id: Optional[str] = None,
) -> ProfileContext:
    capture_context = manifest.get("context") or {}
    surface = capture_context.get("phone_surface_orientation") or {}
    launch = capture_context.get("game_launch") or {}
    selected_phone_id = _phone_id_from_manifest(manifest, phone_id)
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
        game_id=str(
            game_id
            or capture_context.get("game_id")
            or launch.get("game_id")
            or ""
        ),
        platform="android",
        package=launch.get("package"),
        camera_adapter=(rig_context.camera_adapter if rig_context else "hik_mvs"),
        camera_id=(rig_context.camera_id if rig_context else None),
        phone_id=selected_phone_id,
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
    game_id: Optional[str] = None,
    phone_id: Optional[str] = None,
    camera_id: Optional[str] = None,
    activate: bool = False,
    compose_rig: bool = True,
) -> Dict[str, Any]:
    summary_path = Path(localization_summary)
    if summary_path.is_dir():
        localization_file = summary_path / "localization_summary.json"
        summary_path = (
            localization_file
            if localization_file.is_file()
            else summary_path / "calibration.json"
        )
    summary_path = summary_path.resolve()
    summary = _load_json(summary_path)
    if summary.get("status") not in ("review_required", "accepted", "complete"):
        raise ValueError("Localization result is not publishable: {}".format(summary.get("status")))
    session_path, manifest, capture_context = _session_context(summary)
    selected_rig = rig_calibration if compose_rig else None
    if compose_rig and selected_rig is None:
        selected_rig = ((capture_context.get("hik_capture") or {}).get("rig_calibration"))
    store = registry or ProfileRegistry(profile_root)
    source_phone_payload: Dict[str, Any] = {}
    source_phone_revision = str(
        (summary.get("provenance") or {}).get("source_phone_game_revision") or ""
    )
    if source_phone_revision:
        try:
            source_profile = store.revision(source_phone_revision)
        except (KeyError, FileNotFoundError, ProfileResolutionError):
            source_profile = None
        if (
            source_profile is not None
            and (source_profile.get("identity") or {}).get("kind") == "phone_game"
        ):
            source_phone_payload = dict(source_profile.get("payload") or {})
    rig_profile = rig_context = None
    if selected_rig:
        rig_file = _rig_calibration_file(Path(selected_rig))
        rig_document = _load_json(rig_file)
        rig_context = context_from_rig_calibration(rig_document)
        rig_profile = publish_rig_calibration(
            rig_file, registry=store, activate=True
        )
    if compose_rig and rig_profile is None:
        try:
            rig_profile = store.resolve(
                "rig",
                ProfileContext(camera_id=camera_id, phone_id=phone_id),
            )
        except ProfileResolutionError:
            pass
        else:
            rig_context = ProfileContext.from_dict(rig_profile.get("context") or {})
    context = _profile_context_from_localization(
        summary,
        manifest,
        rig_context,
        game_id=game_id,
        phone_id=phone_id,
    )
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
    if canonical_rotation_center is None and isinstance(
        source_phone_payload.get("rotation_center"), Mapping
    ):
        canonical_rotation_center = dict(source_phone_payload["rotation_center"])
    cursor_shape = summary.get("cursor_shape") or android_result.get("cursor_shape")
    cursor_geometry = (
        dict(source_phone_payload.get("cursor_geometry") or {}) or None
    )
    if canonical_rotation_center is not None:
        cursor_geometry = cursor_geometry or {"schema_version": "1.0"}
        cursor_geometry["rotation_center"] = canonical_rotation_center
        cursor_geometry["measurement_space"] = canonical_rotation_center["space"]
        if canonical_rotation_center.get("centroid_orbit_radius_px") is not None:
            cursor_geometry["centroid_orbit_radius_px"] = float(
                canonical_rotation_center["centroid_orbit_radius_px"]
            )
            cursor_geometry["centroid_orbit_diameter_px"] = float(
                2.0 * canonical_rotation_center["centroid_orbit_radius_px"]
            )
        cursor_geometry["rotation_center_quality"] = {
            key: canonical_rotation_center.get(key)
            for key in (
                "confidence",
                "confidence_level",
                "method",
                "analyzed_frames",
                "detected_frames",
                "total_frames",
                "detection_rate",
                "angular_coverage_10deg_bins",
                "symmetry_score",
                "symmetry_peak_margin",
                "localization_sigma_px",
                "subpixel_registration_response",
                "circle_fit_rmse_px",
                "bootstrap_center_sigma_px",
            )
            if canonical_rotation_center.get(key) is not None
        }
    if isinstance(cursor_shape, Mapping):
        cursor_geometry = cursor_geometry or {
            "schema_version": "1.0",
            "rotation_center": canonical_rotation_center,
        }
        cursor_geometry["rotation_center"] = canonical_rotation_center
        cursor_geometry["measurement_space"] = canonical_boundary["space"]
        cursor_geometry["cursor_polygon_max_span_px"] = float(
            cursor_shape.get("cursor_polygon_max_span_px", 0.0)
        )
        if cursor_shape.get("observed_static_cursor_max_span_px") is not None:
            cursor_geometry["observed_static_cursor_max_span_px"] = float(
                cursor_shape["observed_static_cursor_max_span_px"]
            )
        cursor_geometry["latest_shape_source"] = str(
            cursor_shape.get("source") or summary.get("calibration_kind") or "unknown"
        )
        envelope_diameter = cursor_shape.get(
            "rotating_cursor_envelope_diameter_px"
        )
        envelope_radius = cursor_shape.get("rotating_cursor_envelope_radius_px")
        if (
            canonical_rotation_center is not None
            and envelope_diameter is not None
            and envelope_radius is not None
        ):
            cursor_geometry.update(
                rotating_cursor_envelope_radius_px=float(envelope_radius),
                rotating_cursor_envelope_diameter_px=float(envelope_diameter),
                centroid_orbit_radius_px=float(
                    canonical_rotation_center.get("centroid_orbit_radius_px", 0.0)
                ),
                centroid_orbit_diameter_px=float(
                    2.0
                    * float(
                        canonical_rotation_center.get(
                            "centroid_orbit_radius_px", 0.0
                        )
                    )
                ),
                measurement_space=canonical_rotation_center["space"],
                size_definition=str(cursor_shape.get("size_definition") or ""),
            )
    evidence = summary.get("evidence") or android_result.get(
        "verified_backend_evidence"
    ) or []
    evidence_files = [
        str(value.get("name")) if isinstance(value, Mapping) else str(value)
        for value in evidence
        if not isinstance(value, Mapping) or value.get("name")
    ]
    phone_payload = {
        "profile_kind": "phone_game",
        "coordinate_space": "phone_natural_display_pixels",
        "canonical_phone_crop_xywh": canonical_crop,
        "android_logical_crop_xywh": logical_crop,
        "phone_surface_orientation": surface,
        "outer_boundary": canonical_boundary,
        "rotation_center": canonical_rotation_center,
        "cursor_geometry": cursor_geometry,
        "source_boundary": source_boundary,
        "shift_estimation_mask": android_result.get("shift_estimation_mask"),
        "capabilities": {
            "adb_minimap": True,
            "camera_independent": True,
            "cursor_rotation_center": canonical_rotation_center is not None,
            "cursor_shape": isinstance(cursor_shape, Mapping),
            "cursor_result_level": (
                "rotation_center_and_shape"
                if canonical_rotation_center is not None
                and isinstance(cursor_shape, Mapping)
                else "rotation_center_only"
                if canonical_rotation_center is not None
                else "shape_only"
                if isinstance(cursor_shape, Mapping)
                else "unavailable"
            ),
        },
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
    model_file = summary.get("model_file")
    if model_file:
        model_path = summary_path.parent / str(model_file)
        if model_path.is_file():
            portable_files["minimap_model"] = model_path
            phone_payload["minimap_model_runtime_file"] = "minimap_model"
    for index, evidence_value in enumerate(evidence_files):
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
            "evidence": evidence_files,
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
        rig_game_payload = _rig_game_payload_from_phone_game(phone_profile)
        rig_game_payload["session_hik_observation"] = summary.get(
            "rig_observation"
        ) or summary.get("hik_session_observation")
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
    configure_game = subcommands.add_parser(
        "configure-game",
        help="declare cursor and mini-map behavior for one game ID",
    )
    configure_game.add_argument("game_id")
    configure_game.add_argument(
        "--cursor-follows", choices=CURSOR_FOLLOWS_VALUES
    )
    configure_game.add_argument(
        "--minimap-orientation", choices=MINIMAP_ORIENTATION_VALUES
    )
    configure_game.add_argument("--platform", default="android")
    configure_game.add_argument("--package-id")
    configure_game.add_argument("--game-version")
    show_game = subcommands.add_parser(
        "show-game", help="show the effective behavior model for one game ID"
    )
    show_game.add_argument("game_id")
    show_game.add_argument("--platform", default="android")
    show_game.add_argument("--package-id")
    show_game.add_argument("--game-version")
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
    resolve.add_argument(
        "--mask-policy", choices=("none", "minimap_circle"), default="none"
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
    export.add_argument(
        "--mask-policy", choices=("none", "minimap_circle"), default="none"
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
    deployment_export = subcommands.add_parser(
        "export-deployment",
        help="export all active portable game calibrations as one package",
    )
    deployment_export.add_argument("output", type=Path)
    deployment_import = subcommands.add_parser(
        "import-deployment",
        help="import every game calibration in an IRIS deployment package",
    )
    deployment_import.add_argument("package", type=Path)
    deployment_import.add_argument("--camera-id")
    deployment_import.add_argument("--phone-id")
    deployment_import.add_argument("--panel-size", type=int, nargs=2, metavar=("W", "H"))
    deployment_import.add_argument("--activate", action="store_true")
    deployment_import.add_argument("--no-compose-rig", action="store_true")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    registry = ProfileRegistry(arguments.profile_root)
    if arguments.command == "publish-rig":
        result = publish_rig_calibration(
            arguments.calibration, registry=registry, activate=not arguments.candidate
        )
        print("Rig profile: {} ({})".format(result["revision_id"], result["publication"]))
        if not arguments.candidate:
            print(
                "Recomposed {} rig-game and {} game-orientation profile(s) "
                "for this panel.".format(
                    len(result.get("recomposed_rig_game_profiles") or []),
                    len(
                        result.get(
                            "recomposed_rig_game_orientation_profiles"
                        )
                        or []
                    ),
                )
            )
            stale_color = (
                ((result.get("rig_dependent_reconciliation") or {}).get(
                    "requires_fresh_evidence"
                ) or {}).get("rig_game_color")
                or []
            )
            if stale_color:
                print(
                    "Game color: {} previous rig-specific fit(s) now use safe "
                    "rig-locked fallback; run game-calibration with synchronized "
                    "HIK images to refresh them.".format(len(stale_color))
                )
        return 0
    if arguments.command == "publish-minimap":
        from rig_runtime.adapters.filesystem.system_configuration import (
            load_system_configuration,
        )

        settings = load_system_configuration(arguments.profile_root)
        result = publish_minimap_profiles(
            arguments.localization,
            registry=registry,
            rig_calibration=arguments.rig_calibration,
            game_id=settings["game"].get("game_id"),
            phone_id=settings["devices"].get("phone_id"),
            camera_id=settings["devices"].get("camera_id"),
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
    if arguments.command == "configure-game":
        result = publish_game_model(
            registry,
            arguments.game_id,
            cursor_follows=arguments.cursor_follows,
            minimap_orientation=arguments.minimap_orientation,
            platform=arguments.platform,
            package=arguments.package_id,
            game_version=arguments.game_version,
        )
        print("Game model: {}".format(result["revision_id"]))
        print(json.dumps(result["payload"], indent=2))
        return 0
    if arguments.command == "show-game":
        print(
            json.dumps(
                resolve_game_model(
                    registry,
                    arguments.game_id,
                    platform=arguments.platform,
                    package=arguments.package_id,
                    game_version=arguments.game_version,
                ),
                indent=2,
            )
        )
        return 0
    if arguments.command == "export-portable":
        from rig_runtime.workflows.portable_profiles import export_portable_profile

        result = export_portable_profile(
            arguments.revision_id, arguments.output, registry=registry
        )
        print("Portable calibration: {}".format(result["output"]))
        print("Source revision: {}".format(result["source_revision"]))
        return 0
    if arguments.command == "export-deployment":
        from rig_runtime.workflows.portable_profiles import export_deployment_package

        result = export_deployment_package(arguments.output, registry=registry)
        print("IRIS deployment: {}".format(result["output"]))
        print("Portable profiles: {}".format(result["profile_count"]))
        return 0
    from rig_runtime.adapters.filesystem.system_configuration import (
        load_system_configuration,
    )

    settings = load_system_configuration(arguments.profile_root)
    if arguments.command == "import-deployment":
        from rig_runtime.workflows.portable_profiles import import_deployment_package

        requested = ProfileContext(
            camera_id=arguments.camera_id or settings["devices"].get("camera_id"),
            phone_id=arguments.phone_id or settings["devices"].get("phone_id"),
            panel_display=(
                {"natural_panel_px": arguments.panel_size}
                if arguments.panel_size
                else {}
            ),
        )
        result = import_deployment_package(
            arguments.package,
            registry=registry,
            requested_context=requested,
            activate=arguments.activate,
            compose_local_rig=not arguments.no_compose_rig,
        )
        print("Imported portable profiles: {}".format(result["profile_count"]))
        for profile in result["profiles"]:
            print(
                "  {profile_kind} {game_id}: {revision_id}".format(**profile)
            )
        for message in result["warnings"]:
            print("Warning: {}".format(message))
        if not arguments.activate:
            print("Imported revisions are review-required candidates.")
        return 0
    if arguments.command == "import-portable":
        from rig_runtime.workflows.portable_profiles import import_portable_profile

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
        mask_policy=arguments.mask_policy,
    )
    if arguments.command == "export-adapter":
        from rig_runtime.workflows.adapter_export import export_resolved_adapter

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
