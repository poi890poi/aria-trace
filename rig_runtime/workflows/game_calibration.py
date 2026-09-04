"""Task-oriented calibration of every game capability supported by a session."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np

from rig_runtime.adapters.filesystem.profile_registry import ProfileContext, ProfileRegistry
from rig_runtime.adapters.filesystem.session import SessionReader
from rig_runtime.adapters.filesystem.system_configuration import (
    load_system_configuration,
)
from rig_runtime.domain.spatial import raster_space
from rig_runtime.domain.spaces import RigSpaceId
from rig_runtime.services.calibration.minimap.calibration import (
    calibrate_cursor_orbit_frames,
    calibrate_cursor_static_frames,
    calibrate_minimap_boundary_frames,
)
from rig_runtime.services.calibration.minimap.discovery import (
    discover_android_minimap_crop,
)
from rig_runtime.workflows.game_orientation_calibration import (
    calibrate_portable_game_orientation_session,
)
from rig_runtime.workflows.hik_game_color_calibration import (
    calibrate_game_color_session,
    decode_session_records,
)
from rig_runtime.workflows.profile_management import (
    DEFAULT_GAME_MODEL,
    cursor_behavior_by_acquisition,
    publish_minimap_profiles,
    resolve_game_model,
)
from rig_runtime.workflows.game_repeatability import (
    _logical_profile_crop,
    _profile_for_current_game,
)


def _safe_label(value: Optional[str]) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "game")).strip("-.")
    return cleaned or "game"


def _default_output(registry: ProfileRegistry, game_id: Optional[str]) -> Path:
    return (
        registry.root
        / "calibrations"
        / "game"
        / "{}-{}".format(
            _safe_label(game_id), datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    )


def _touch_intervals(
    reader: SessionReader, kind: str = "zigzag_touch"
) -> list[tuple[int, int]]:
    intervals = []
    active_start = None
    for event in reader.inputs:
        if event.get("kind") != kind:
            continue
        payload = dict(event.get("payload") or {})
        action = str(payload.get("action") or "").upper()
        time_ns = int(event["session_time_ns"])
        if action == "DOWN":
            active_start = time_ns
        elif action == "UP" and active_start is not None:
            intervals.append((active_start, time_ns))
            active_start = None
        elif action == "SWIPE":
            duration_ns = max(
                0,
                int(payload.get("command_end_host_time_ns") or 0)
                - int(payload.get("command_start_host_time_ns") or 0),
            )
            if duration_ns <= 0:
                duration_ns = int(
                    float(payload.get("duration_ms") or 0.0) * 1.0e6
                )
            intervals.append((max(0, time_ns - duration_ns), time_ns))
    return sorted(intervals)


def _select_android_records(
    reader: SessionReader,
    maximum_frames: int = 48,
    touch_kind: str = "zigzag_touch",
) -> tuple[list[Mapping[str, object]], Mapping[str, object]]:
    stream_id = "android_phone" if reader.frames_by_stream.get("android_phone") else "main"
    records = list(reader.frames_by_stream.get(stream_id) or [])
    if not records:
        raise ValueError("Session has no Android/main image frames")
    intervals = _touch_intervals(reader, touch_kind)
    candidates = [
        record for record in records
        if any(
            start <= int(record["session_time_ns"]) <= end
            for start, end in intervals
        )
    ] if intervals else []
    selection_basis = "{}_down_to_up_intervals".format(touch_kind)
    if len(candidates) < 4:
        candidates = records
        selection_basis = "all_available_frames_no_usable_touch_intervals"
    count = min(int(maximum_frames), len(candidates))
    positions = np.linspace(0, len(candidates) - 1, count).round().astype(int)
    selected = [candidates[int(index)] for index in positions]
    return selected, {
        "stream_id": stream_id,
        "selection_basis": selection_basis,
        "touch_interval_count": len(intervals),
        "candidate_count": len(candidates),
        "selected_frame_indices": [int(item["frame_index"]) for item in selected],
    }


def _calibrate_available_minimap_boundary(
    reader: SessionReader,
    output: Path,
    *,
    registry: ProfileRegistry,
    game_id: Optional[str],
    phone_id: Optional[str],
    camera_id: Optional[str],
    discovery_config: Optional[Mapping[str, object]],
    activate: bool,
) -> Mapping[str, object]:
    records, selection = _select_android_records(reader)
    frames = decode_session_records(reader, selection["stream_id"], records)
    discovery = discover_android_minimap_crop(frames, discovery_config)
    minimum_margin = float(
        discovery["search_bounds"].get("minimum_priority_margin_fraction", 0.04)
    )
    if (
        not discovery.get("operator_selected_candidate")
        and float(discovery["priority_margin_fraction"]) < minimum_margin
    ):
        raise ValueError(
            "Automatic mini-map discovery is ambiguous (priority margin {:.3f}, "
            "required {:.3f}); review candidates or pass discovery-config "
            "candidate_index instead of activating a guess".format(
                float(discovery["priority_margin_fraction"]), minimum_margin
            )
        )
    refined = []
    for hypothesis_index, hypothesis in enumerate(discovery["rough_hypotheses"][:8]):
        trial_crop = [int(value) for value in hypothesis["crop_xywh"]]
        tx, ty, tw, th = trial_crop
        trial_frames = frames[:, ty : ty + th, tx : tx + tw, :]
        try:
            trial = calibrate_minimap_boundary_frames(
                trial_frames,
                output / "_unused_trial_evidence",
                config=dict(hypothesis["boundary_config"]),
                write_evidence=False,
            )
        except (ValueError, RuntimeError):
            continue
        refined.append(
            {
                "hypothesis_index": hypothesis_index,
                "crop_xywh": trial_crop,
                "boundary_config": dict(hypothesis["boundary_config"]),
                "confidence": float(trial["outer_boundary"]["confidence"]),
                "radial_rmse_px": float(trial["outer_boundary"]["radial_rmse_px"]),
            }
        )
    if not refined:
        raise ValueError("No coarse mini-map hypothesis passed the verified boundary fitter")
    # Discovery priority decides among plausible HUD circles; the verified
    # fitter is a validity/refinement gate, not a second global detector whose
    # confidence can prefer a different circular widget.
    plausible = [
        item for item in refined
        if item["confidence"] >= 0.35 and item["radial_rmse_px"] <= 2.0
    ]
    selected_refinement = min(
        plausible or refined,
        key=lambda item: (
            item["hypothesis_index"]
            if plausible else -item["confidence"]
        ),
    )
    crop = [int(value) for value in selected_refinement["crop_xywh"]]
    x, y, width, height = crop
    cropped = frames[:, y : y + height, x : x + width, :]
    source_height, source_width = frames.shape[1:3]
    crop_space = raster_space(
        "game_calibration_android_minimap_discovery_crop_pixels",
        [width, height],
        parent_space_id=RigSpaceId.ANDROID_LOGICAL_DISPLAY,
        local_to_parent_3x3=[
            [1.0, 0.0, float(x)],
            [0.0, 1.0, float(y)],
            [0.0, 0.0, 1.0],
        ],
    )
    evidence = output / "evidence"
    result = calibrate_minimap_boundary_frames(
        cropped,
        evidence,
        config=dict(selected_refinement["boundary_config"]),
        write_evidence=True,
        frame_space=crop_space,
    )
    diagnostics = discovery.pop("diagnostics")
    discovery["verified_backend_hypotheses"] = refined
    discovery["selected_verified_hypothesis"] = selected_refinement
    discovery["crop_xywh"] = crop
    discovery["boundary_config"] = selected_refinement["boundary_config"]
    cv2.imwrite(
        str(evidence / "discovery_temporal_heatmap.png"),
        cv2.applyColorMap(
            cv2.normalize(
                diagnostics["temporal_heatmap"], None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8),
            cv2.COLORMAP_TURBO,
        ),
    )
    cv2.imwrite(str(evidence / "discovery_edges.png"), diagnostics["edges"])
    overlay = frames[len(frames) // 2].copy()
    boundary = result["outer_boundary"]
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (255, 0, 255), 2)
    cv2.circle(
        overlay,
        (int(round(x + boundary["center_x"])), int(round(y + boundary["center_y"]))),
        int(round(boundary["radius"])),
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(evidence / "discovery_source_overlay.png"), overlay)
    summary = {
        "schema_version": "1.0",
        "status": "accepted" if activate else "review_required",
        "calibration_kind": "minimap_boundary",
        "outer_boundary": boundary,
        "geometry_space": result["geometry_space"],
        "crop_xywh": crop,
        "config": {"crop_xywh": crop, "boundary": result["config"]},
        "android": {
            "frame_size_px": [source_width, source_height],
            "outer_boundary": boundary,
        },
        "automatic_discovery": discovery,
        "frame_selection": selection,
        "provenance": {
            "session_path": str(reader.path.resolve()),
            "session_id": reader.manifest.get("session_id"),
            "stream_id": selection["stream_id"],
            "movement_required": False,
        },
        "evidence": list(result["evidence"]) + [
            {"name": "discovery_temporal_heatmap.png", "category": "discovery"},
            {"name": "discovery_edges.png", "category": "discovery"},
            {"name": "discovery_source_overlay.png", "category": "discovery"},
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "calibration.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    profiles = publish_minimap_profiles(
        summary_path,
        registry=registry,
        game_id=game_id,
        phone_id=phone_id,
        camera_id=camera_id,
        activate=activate,
        compose_rig=False,
    )
    summary["profiles"] = {
        name: value["revision_id"] if value is not None else None
        for name, value in profiles.items()
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _available_cursor_acquisition_series(
    reader: SessionReader,
) -> list[Mapping[str, str]]:
    """Describe recorded motion patterns without guessing cursor behavior."""

    context = dict(reader.manifest.get("context") or {})
    capture_kind = str(context.get("capture_kind") or "")
    result = []
    if _touch_intervals(reader, "zigzag_touch"):
        result.append(
            {"acquisition_pattern": "zigzag", "touch_kind": "zigzag_touch"}
        )
    if (
        _touch_intervals(reader, "micro_movement_touch")
        or _touch_intervals(reader, "cursor_orbit_touch")
        or capture_kind in (
            "micro_movement_game_calibration_source_data",
            "cursor_orbit_game_calibration_source_data",
        )
    ):
        touch_kind = (
            "micro_movement_touch"
            if _touch_intervals(reader, "micro_movement_touch")
            else "cursor_orbit_touch"
        )
        result.append(
            {
                "acquisition_pattern": "micro_movement",
                "touch_kind": touch_kind,
            }
        )
    if not result and reader.frames_by_stream:
        result.append(
            {"acquisition_pattern": "uncontrolled", "touch_kind": ""}
        )
    return result


def _session_calibration_descriptor(session: Path) -> Mapping[str, object]:
    """Inspect one immutable session and describe how it should be ordered."""

    path = Path(session).resolve()
    reader = SessionReader(path)
    if reader.manifest.get("status") != "complete":
        raise ValueError(
            "Game calibration requires a complete immutable session: {}".format(path)
        )
    patterns = [
        str(item["acquisition_pattern"])
        for item in _available_cursor_acquisition_series(reader)
    ]
    if not patterns:
        patterns = ["uncontrolled"]
    if "zigzag" in patterns:
        order = 0
    elif "micro_movement" in patterns:
        order = 1
    else:
        order = 2
    return {
        "path": path,
        "session_id": reader.manifest.get("session_id"),
        "game_id": (reader.manifest.get("context") or {}).get("game_id"),
        "acquisition_patterns": patterns,
        "order": order,
    }


def _latest_phone_game_revision(result: Mapping[str, object]) -> Optional[str]:
    """Return the newest portable profile emitted by one calibration pass."""

    capabilities = dict(result.get("capabilities") or {})
    orientation = dict(capabilities.get("screen_orientation") or {})
    if orientation.get("profile_revision"):
        return str(orientation["profile_revision"])
    cursor = dict(capabilities.get("cursor_pose") or {})
    if cursor.get("selected_profile_revision"):
        return str(cursor["selected_profile_revision"])
    minimap = dict(capabilities.get("minimap_boundary") or {})
    profiles = dict(minimap.get("profiles") or {})
    if profiles.get("phone_game"):
        return str(profiles["phone_game"])
    return None


def _calibrate_available_cursor_series(
    reader: SessionReader,
    output: Path,
    *,
    registry: ProfileRegistry,
    game_id: Optional[str],
    phone_id: Optional[str],
    camera_id: Optional[str],
    activate: bool,
    acquisition_pattern: str,
    cursor_behavior: str,
    touch_kind: str = "",
    phone_game_revision: Optional[str] = None,
) -> Mapping[str, object]:
    """Fit one modeled cursor response against the verified map boundary."""

    if cursor_behavior not in ("rotating", "static"):
        raise ValueError("cursor_behavior must be rotating or static")

    capture_context = dict(reader.manifest.get("context") or {})
    surface = dict(capture_context.get("phone_surface_orientation") or {})
    if not surface.get("natural_size_px") or not surface.get("logical_size_px"):
        raise ValueError("Cursor-orbit session has no Android surface geometry")
    context = ProfileContext(
        game_id=game_id,
        camera_id=camera_id,
        phone_id=phone_id,
        panel_display={"natural_panel_px": surface["natural_size_px"]},
    )
    profile = _profile_for_current_game(
        registry, context, revision_id=phone_game_revision
    )
    crop, boundary = _logical_profile_crop(profile, surface)
    records, selection = _select_android_records(
        reader,
        maximum_frames=48,
        touch_kind=touch_kind or "cursor_series_without_touch_labels",
    )
    frames = decode_session_records(reader, selection["stream_id"], records)
    x, y, width, height = crop
    if x < 0 or y < 0 or x + width > frames.shape[2] or y + height > frames.shape[1]:
        raise ValueError(
            "Active mini-map crop {} exceeds captured Android raster {}x{}".format(
                crop, frames.shape[2], frames.shape[1]
            )
        )
    cropped = frames[:, y : y + height, x : x + width, :]
    provenance = {
        "session_path": str(reader.path.resolve()),
        "session_id": reader.manifest.get("session_id"),
        "stream_id": selection["stream_id"],
        "frame_selection": selection,
        "source_phone_game_revision": profile.get("revision_id"),
        "acquisition_pattern": acquisition_pattern,
        "expected_cursor_behavior": cursor_behavior,
        "touch_kind": touch_kind or None,
    }
    calibrator = (
        calibrate_cursor_orbit_frames
        if cursor_behavior == "rotating"
        else calibrate_cursor_static_frames
    )
    result = calibrator(
        cropped,
        output,
        outer_boundary=boundary,
        provenance=provenance,
        frame_space=boundary["space"],
    )
    backend_status = str(result.get("status") or "review_required")
    result.update(
        {
            "status": (
                backend_status
                if backend_status == "partial"
                else "accepted" if activate else "review_required"
            ),
            "profile_activation_requested": bool(activate),
            "crop_xywh": crop,
            "canonical_phone_crop_xywh": (profile.get("payload") or profile).get(
                "canonical_phone_crop_xywh"
            ),
            "android": {
                "frame_size_px": [int(frames.shape[2]), int(frames.shape[1])],
                "outer_boundary": boundary,
                "rotation_center": result["rotation_center"],
            },
            "frame_selection": selection,
            "acquisition_pattern": acquisition_pattern,
            "expected_cursor_behavior": cursor_behavior,
        }
    )
    summary_path = output / "calibration.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    profiles = publish_minimap_profiles(
        summary_path,
        registry=registry,
        game_id=game_id,
        phone_id=phone_id,
        camera_id=camera_id,
        activate=activate,
        compose_rig=False,
    )
    result["profiles"] = {
        name: value["revision_id"] if value is not None else None
        for name, value in profiles.items()
    }
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _outcome(status: str, **values) -> Mapping[str, object]:
    return {"status": status, **values}


def calibrate_game_session(
    session: Path,
    output: Path,
    *,
    profile_root: Optional[Path] = None,
    game_id: Optional[str] = None,
    maximum_pairs: int = 12,
    discovery_config: Optional[Mapping[str, object]] = None,
    activate: bool = True,
    include_color: bool = False,
    activate_color: bool = False,
    phone_game_revision: Optional[str] = None,
) -> Mapping[str, object]:
    """Run independent game capabilities, skipping only unavailable inputs."""

    session = Path(session).resolve()
    output = Path(output).resolve()
    reader = SessionReader(session)
    if reader.manifest.get("status") != "complete":
        raise ValueError("Game calibration requires a complete immutable session")
    registry = ProfileRegistry(profile_root)
    settings = load_system_configuration(profile_root)
    selected_game = (
        game_id
        or (reader.manifest.get("context") or {}).get("game_id")
        or settings["game"].get("game_id")
    )
    output.mkdir(parents=True, exist_ok=False)
    capabilities = {}
    current_phone_game_revision = phone_game_revision
    game_model = (
        resolve_game_model(registry, selected_game)
        if selected_game
        else dict(DEFAULT_GAME_MODEL)
    )
    cursor_series = _available_cursor_acquisition_series(reader)
    behavior_map = dict(
        game_model.get("cursor_behavior_by_acquisition")
        or cursor_behavior_by_acquisition(
            str(game_model.get("cursor_follows") or "character")
        )
    )
    if set(behavior_map) != {"zigzag", "micro_movement"} or any(
        value not in ("static", "rotating") for value in behavior_map.values()
    ):
        raise ValueError(
            "Game model cursor_behavior_by_acquisition must map zigzag and "
            "micro_movement to static or rotating"
        )

    has_zigzag_series = any(
        item["acquisition_pattern"] == "zigzag" for item in cursor_series
    )
    has_micro_movement_series = any(
        item["acquisition_pattern"] == "micro_movement" for item in cursor_series
    )
    if has_micro_movement_series and not has_zigzag_series:
        capabilities["minimap_boundary"] = _outcome(
            "skipped_existing_profile_reused",
            reason=(
                "This session has no zigzag series; cursor calibration preserves "
                "the active verified mini-map boundary"
            ),
        )
    else:
        try:
            value = _calibrate_available_minimap_boundary(
                reader,
                output / "minimap",
                registry=registry,
                game_id=selected_game,
                phone_id=settings["devices"].get("phone_id"),
                camera_id=settings["devices"].get("camera_id"),
                discovery_config=discovery_config,
                activate=activate,
            )
        except (ValueError, FileNotFoundError) as exc:
            capabilities["minimap_boundary"] = _outcome(
                "skipped_missing_or_ineligible_data", reason=str(exc)
            )
        except Exception as exc:
            capabilities["minimap_boundary"] = _outcome(
                "failed", error="{}: {}".format(type(exc).__name__, exc)
            )
        else:
            current_phone_game_revision = value["profiles"].get("phone_game")
            capabilities["minimap_boundary"] = _outcome(
                value["status"],
                calibration=str(output / "minimap" / "calibration.json"),
                profiles=value["profiles"],
            )

    if cursor_series:
        series_outcomes = {}
        # A rotating series is authoritative for the pivot. Run it before a
        # static series so profile composition can preserve that geometry.
        ordered = sorted(
            cursor_series,
            key=lambda item: 0
            if behavior_map.get(item["acquisition_pattern"], "static") == "rotating"
            else 1,
        )
        for series in ordered:
            pattern = series["acquisition_pattern"]
            behavior = behavior_map.get(pattern, "static")
            series_output = output / "cursor" / pattern
            try:
                value = _calibrate_available_cursor_series(
                    reader,
                    series_output,
                    registry=registry,
                    game_id=selected_game,
                    phone_id=settings["devices"].get("phone_id"),
                    camera_id=settings["devices"].get("camera_id"),
                    activate=activate,
                    acquisition_pattern=pattern,
                    cursor_behavior=behavior,
                    touch_kind=series["touch_kind"],
                    phone_game_revision=current_phone_game_revision,
                )
            except (ValueError, FileNotFoundError, RuntimeError) as exc:
                series_outcomes[pattern] = _outcome(
                    "skipped_missing_or_ineligible_data",
                    cursor_behavior=behavior,
                    reason=str(exc),
                )
            except Exception as exc:
                series_outcomes[pattern] = _outcome(
                    "failed",
                    cursor_behavior=behavior,
                    error="{}: {}".format(type(exc).__name__, exc),
                )
            else:
                current_phone_game_revision = value["profiles"].get("phone_game")
                partial_reason = "; ".join(
                    "{}: {}".format(
                        item.get("stage", "cursor calibration"),
                        item.get("reason", "unavailable"),
                    )
                    for item in (value.get("failure_reasons") or [])
                )
                series_outcomes[pattern] = _outcome(
                    value["status"],
                    cursor_behavior=behavior,
                    calibration=str(series_output / "calibration.json"),
                    profiles=value["profiles"],
                    result_level=value.get("result_level"),
                    capabilities=value.get("capabilities"),
                    reason=partial_reason or None,
                )
        accepted_series = [
            name for name, value in series_outcomes.items()
            if value["status"] in ("accepted", "review_required", "partial")
        ]
        failed_series = [
            name for name, value in series_outcomes.items()
            if value["status"] == "failed"
        ]
        partial_series = [
            name for name, value in series_outcomes.items()
            if value["status"] == "partial"
        ]
        skipped_reasons = [
            "{} ({}): {}".format(
                name,
                value.get("cursor_behavior", "unknown behavior"),
                value.get("reason") or value.get("error") or value["status"],
            )
            for name, value in series_outcomes.items()
            if value["status"] not in ("accepted", "review_required", "partial")
        ]
        capabilities["cursor_pose"] = _outcome(
            "complete" if accepted_series and not failed_series and not partial_series else (
                "partial" if accepted_series else "skipped_missing_or_ineligible_data"
            ),
            series=series_outcomes,
            selected_profile_revision=current_phone_game_revision,
            reason="; ".join(skipped_reasons) if skipped_reasons else None,
        )
    else:
        capabilities["cursor_pose"] = _outcome(
            "skipped_missing_or_ineligible_data",
            reason=(
                "No Android image series is available for cursor calibration"
            ),
        )

    try:
        value = calibrate_portable_game_orientation_session(
            session,
            output / "orientation",
            profile_root=registry.root,
            game_id=selected_game,
            activate=activate,
            phone_game_revision=current_phone_game_revision,
        )
    except (ValueError, FileNotFoundError) as exc:
        capabilities["screen_orientation"] = _outcome(
            "skipped_missing_or_ineligible_data", reason=str(exc)
        )
    except Exception as exc:
        capabilities["screen_orientation"] = _outcome(
            "failed", error="{}: {}".format(type(exc).__name__, exc)
        )
    else:
        current_phone_game_revision = value["profile_revision"]
        capabilities["screen_orientation"] = _outcome(
            value["status"],
            calibration=str(
                output / "orientation" / "game_orientation_calibration.json"
            ),
            profile_revision=value["profile_revision"],
            profile_activated=value["profile_activated"],
            rig_dependency=None,
        )

    if not include_color:
        capabilities["game_color"] = _outcome(
            "optional_not_requested",
            reason=(
                "Locked rig imaging and HIK auto white balance remain active; "
                "pass --include-color to produce an optional reviewed fit"
            ),
        )
    elif not (session / "coordinate_spaces.yaml").is_file() or not reader.frames_by_stream.get("hik_phone"):
        capabilities["game_color"] = _outcome(
            "optional_skipped_missing_data",
            reason="Synchronized ADB/HIK space conversion is unavailable",
        )
    else:
        try:
            value = calibrate_game_color_session(
                session,
                output / "color",
                profile_root=registry.root,
                game_id=selected_game,
                maximum_pairs=maximum_pairs,
                activate=bool(activate and activate_color),
                phone_game_revision=current_phone_game_revision,
            )
        except (ValueError, FileNotFoundError) as exc:
            capabilities["game_color"] = _outcome(
                "optional_skipped_ineligible_data", reason=str(exc)
            )
        except Exception as exc:
            capabilities["game_color"] = _outcome(
                "optional_failed_non_gating",
                error="{}: {}".format(type(exc).__name__, exc),
            )
        else:
            capabilities["game_color"] = _outcome(
                value["status"], calibration=str(output / "color" / "game_color_calibration.json"),
                profile_revision=value["profile_revision"],
            )

    successful = [
        name for name, value in capabilities.items()
        if value["status"] in ("accepted", "review_required", "complete", "partial")
    ]
    failed = [name for name, value in capabilities.items() if value["status"] == "failed"]
    summary = {
        "schema_version": "1.0",
        "calibration_kind": "game",
        "status": "complete" if successful and not failed else (
            "partial" if successful else "no_capabilities_calibrated"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "session": str(session),
        "game_id": selected_game,
        "game_model": game_model,
        "profile_root": str(registry.root),
        "activation_policy": "accepted_capabilities_only" if activate else "candidate_only",
        "capabilities": capabilities,
        "successful_capabilities": successful,
    }
    (output / "game_calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def calibrate_game_sessions(
    sessions: Sequence[Path],
    output: Path,
    *,
    profile_root: Optional[Path] = None,
    game_id: Optional[str] = None,
    maximum_pairs: int = 12,
    discovery_config: Optional[Mapping[str, object]] = None,
    activate: bool = True,
    include_color: bool = False,
    activate_color: bool = False,
) -> Mapping[str, object]:
    """Calibrate one game from independently captured acquisition sessions.

    Session contents, rather than caller ordering, determine the processing order.
    A zigzag session establishes the portable mini-map geometry before a separate
    micro-movement session adds its modeled cursor evidence.
    """

    unique_sessions = []
    seen = set()
    for session in sessions:
        path = Path(session).resolve()
        key = str(path).casefold()
        if key not in seen:
            seen.add(key)
            unique_sessions.append(path)
    if not unique_sessions:
        raise ValueError("At least one game-acquisition session is required")
    if len(unique_sessions) == 1:
        return calibrate_game_session(
            unique_sessions[0],
            output,
            profile_root=profile_root,
            game_id=game_id,
            maximum_pairs=maximum_pairs,
            discovery_config=discovery_config,
            activate=activate,
            include_color=include_color,
            activate_color=activate_color,
        )

    descriptors = [_session_calibration_descriptor(path) for path in unique_sessions]
    descriptors.sort(key=lambda item: (int(item["order"]), str(item["path"])))
    recorded_games = {
        str(item["game_id"])
        for item in descriptors
        if item.get("game_id")
    }
    if game_id is None and len(recorded_games) > 1:
        raise ValueError(
            "Sessions identify different games ({}); pass --game-id only if this "
            "combination is intentional".format(", ".join(sorted(recorded_games)))
        )
    selected_game = game_id or (next(iter(recorded_games)) if recorded_games else None)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)

    session_results = []
    current_phone_game_revision = None
    for index, descriptor in enumerate(descriptors, 1):
        patterns = list(descriptor["acquisition_patterns"])
        label = "-".join(pattern.replace("_", "-") for pattern in patterns)
        session_output = output / "sessions" / "{:02d}-{}".format(index, label)
        value = calibrate_game_session(
            descriptor["path"],
            session_output,
            profile_root=profile_root,
            game_id=selected_game,
            maximum_pairs=maximum_pairs,
            discovery_config=discovery_config,
            activate=activate,
            include_color=include_color,
            activate_color=activate_color,
            phone_game_revision=current_phone_game_revision,
        )
        current_phone_game_revision = (
            _latest_phone_game_revision(value) or current_phone_game_revision
        )
        session_results.append(
            {
                "session": str(descriptor["path"]),
                "session_id": descriptor.get("session_id"),
                "acquisition_patterns": patterns,
                "output": str(session_output),
                "result": value,
            }
        )

    successful = sorted(
        {
            name
            for item in session_results
            for name in item["result"].get("successful_capabilities", [])
        }
    )
    child_statuses = [str(item["result"].get("status")) for item in session_results]
    summary = {
        "schema_version": "1.0",
        "calibration_kind": "game_multi_session",
        "status": (
            "complete"
            if successful and all(status == "complete" for status in child_statuses)
            else "partial" if successful else "no_capabilities_calibrated"
        ),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sessions": [str(item["path"]) for item in descriptors],
        "game_id": selected_game,
        "profile_root": str(ProfileRegistry(profile_root).root),
        "activation_policy": "accepted_capabilities_only" if activate else "candidate_only",
        "session_results": session_results,
        "successful_capabilities": successful,
        "selected_phone_game_revision": current_phone_game_revision,
    }
    (output / "game_calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Calibrate every supported game capability from one or more immutable "
            "session folders. Zigzag and micro-movement sessions are identified and "
            "ordered automatically; missing inputs are reported and skipped."
        )
    )
    value.add_argument(
        "sessions",
        type=Path,
        nargs="+",
        metavar="SESSION",
        help="one or more captured session folders",
    )
    value.add_argument(
        "--output",
        type=Path,
        help="evidence output folder; defaults under the configured profile root",
    )
    value.add_argument("--profile-root", type=Path)
    value.add_argument("--game-id")
    value.add_argument("--maximum-pairs", type=int, default=12)
    value.add_argument("--discovery-config", type=Path)
    value.add_argument("--candidate", action="store_true")
    value.add_argument(
        "--include-color",
        action="store_true",
        help="fit optional HIK/ADB game color; locked rig auto-WB is the default",
    )
    value.add_argument(
        "--activate-color",
        action="store_true",
        help="activate an accepted optional color fit (requires --include-color)",
    )
    return value


def format_game_calibration_report(
    result: Mapping[str, object], output: Path
) -> list[str]:
    """Render one stable, final operator report; no live values are rewritten."""

    labels = {
        "accepted": "OK",
        "complete": "OK",
        "review_required": "REVIEW",
        "partial": "REVIEW",
        "optional_not_requested": "OPTIONAL",
        "optional_skipped_missing_data": "OPTIONAL",
        "optional_skipped_ineligible_data": "OPTIONAL",
        "optional_failed_non_gating": "OPTIONAL-WARN",
        "skipped_existing_profile_reused": "REUSED",
        "skipped_missing_or_ineligible_data": "SKIPPED",
        "failed": "ERROR",
    }
    lines = [
        "",
        "IRIS game calibration summary",
        "  Overall: {}".format(str(result.get("status") or "unknown").upper()),
        "  Output:  {}".format(Path(output).resolve()),
        "  Game:    {}".format(result.get("game_id") or "unidentified"),
        "",
    ]
    if result.get("session_results"):
        lines.extend(["", "Input sessions (automatically classified)"])
        for item in result["session_results"]:
            patterns = ", ".join(item.get("acquisition_patterns") or ["uncontrolled"])
            child = dict(item.get("result") or {})
            lines.append("  [{}] {}".format(patterns, item.get("session")))
            lines.append(
                "                  Result: {}  Evidence: {}".format(
                    str(child.get("status") or "unknown").upper(),
                    item.get("output"),
                )
            )
        if result.get("selected_phone_game_revision"):
            lines.append(
                "  Composed profile: {}".format(result["selected_phone_game_revision"])
            )
        lines.extend(
            ["", "Review each session evidence folder above before accepting REVIEW results."]
        )
        return lines

    lines.append("Capabilities")
    for name, outcome_value in (result.get("capabilities") or {}).items():
        outcome = dict(outcome_value or {})
        status = str(outcome.get("status") or "unknown")
        label = labels.get(status, status.upper())
        lines.append("  [{:13s}] {:18s} {}".format(label, name, status))
        detail = outcome.get("reason") or outcome.get("error")
        if detail:
            lines.append("                  Reason: {}".format(detail))
        calibration = outcome.get("calibration")
        if calibration:
            lines.append("                  Evidence: {}".format(calibration))
        if name == "cursor_pose":
            for series_name, series_value in (outcome.get("series") or {}).items():
                series = dict(series_value or {})
                series_status = str(series.get("status") or "unknown")
                lines.append(
                    "                  - {} / {}: {}".format(
                        series_name,
                        series.get("cursor_behavior") or "unknown behavior",
                        series_status,
                    )
                )
                series_detail = series.get("reason") or series.get("error")
                if series_detail:
                    lines.append("                    Reason: {}".format(series_detail))
    lines.extend(["", "Review the evidence paths above before accepting REVIEW results."])
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.activate_color and not arguments.include_color:
        raise ValueError("--activate-color requires --include-color")
    registry = ProfileRegistry(arguments.profile_root)
    settings = load_system_configuration(arguments.profile_root)
    game_id = arguments.game_id or settings["game"].get("game_id")
    output = arguments.output or _default_output(registry, game_id)
    discovery = (
        json.loads(arguments.discovery_config.read_text(encoding="utf-8"))
        if arguments.discovery_config else None
    )
    result = calibrate_game_sessions(
        arguments.sessions,
        output,
        profile_root=registry.root,
        game_id=game_id,
        maximum_pairs=arguments.maximum_pairs,
        discovery_config=discovery,
        activate=not arguments.candidate,
        include_color=arguments.include_color,
        activate_color=arguments.activate_color,
    )
    for line in format_game_calibration_report(result, output):
        print(line)
    return 0 if result["successful_capabilities"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "calibrate_game_session",
    "calibrate_game_sessions",
    "format_game_calibration_report",
    "main",
]
