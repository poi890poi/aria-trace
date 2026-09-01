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

from aria_trace.adapters.filesystem.profile_registry import ProfileRegistry
from aria_trace.adapters.filesystem.session import SessionReader
from aria_trace.adapters.filesystem.system_configuration import (
    load_system_configuration,
)
from aria_trace.domain.spatial import raster_space
from aria_trace.services.calibration.minimap.calibration import (
    calibrate_minimap_boundary_frames,
)
from aria_trace.services.calibration.minimap.discovery import (
    discover_android_minimap_crop,
)
from aria_trace.workflows.game_orientation_calibration import (
    calibrate_game_orientation_session,
)
from aria_trace.workflows.hik_game_color_calibration import (
    calibrate_game_color_session,
    decode_session_records,
)
from aria_trace.workflows.profile_management import publish_minimap_profiles


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


def _touch_intervals(reader: SessionReader) -> list[tuple[int, int]]:
    intervals = []
    active_start = None
    for event in reader.inputs:
        if event.get("kind") != "zigzag_touch":
            continue
        payload = dict(event.get("payload") or {})
        action = str(payload.get("action") or "").upper()
        time_ns = int(event["session_time_ns"])
        if action == "DOWN":
            active_start = time_ns
        elif action == "UP" and active_start is not None:
            intervals.append((active_start, time_ns))
            active_start = None
    return sorted(intervals)


def _select_android_records(
    reader: SessionReader, maximum_frames: int = 48
) -> tuple[list[Mapping[str, object]], Mapping[str, object]]:
    stream_id = "android_phone" if reader.frames_by_stream.get("android_phone") else "main"
    records = list(reader.frames_by_stream.get(stream_id) or [])
    if not records:
        raise ValueError("Session has no Android/main image frames")
    intervals = _touch_intervals(reader)
    candidates = [
        record for record in records
        if any(
            start <= int(record["session_time_ns"]) <= end
            for start, end in intervals
        )
    ] if intervals else []
    selection_basis = "zigzag_touch_down_to_up_intervals"
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
        parent_space_id="android_logical_display_pixels",
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
    )
    summary["profiles"] = {
        name: value["revision_id"] if value is not None else None
        for name, value in profiles.items()
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


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

    try:
        value = calibrate_game_orientation_session(
            session,
            output / "orientation",
            profile_root=registry.root,
            game_id=selected_game,
            maximum_pairs=maximum_pairs,
            activate=activate,
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
        capabilities["screen_orientation"] = _outcome(
            value["status"],
            calibration=str(output / "orientation" / "game_orientation_calibration.json"),
            profile_revision=value["profile_revision"],
            profile_activated=value["profile_activated"],
        )

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
        capabilities["minimap_boundary"] = _outcome(
            value["status"], calibration=str(output / "minimap" / "calibration.json"),
            profiles=value["profiles"],
        )

    if not (session / "coordinate_spaces.yaml").is_file() or not reader.frames_by_stream.get("hik_phone"):
        capabilities["game_color"] = _outcome(
            "skipped_missing_or_ineligible_data",
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
                activate=activate,
            )
        except (ValueError, FileNotFoundError) as exc:
            capabilities["game_color"] = _outcome(
                "skipped_missing_or_ineligible_data", reason=str(exc)
            )
        except Exception as exc:
            capabilities["game_color"] = _outcome(
                "failed", error="{}: {}".format(type(exc).__name__, exc)
            )
        else:
            capabilities["game_color"] = _outcome(
                value["status"], calibration=str(output / "color" / "game_color_calibration.json"),
                profile_revision=value["profile_revision"],
            )

    capabilities["cursor_pose"] = _outcome(
        "skipped_missing_or_ineligible_data",
        reason="No explicitly labeled rotation-only and movement-only segments; movement is optional",
    )
    successful = [
        name for name, value in capabilities.items()
        if value["status"] in ("accepted", "review_required")
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
        "profile_root": str(registry.root),
        "activation_policy": "accepted_capabilities_only" if activate else "candidate_only",
        "capabilities": capabilities,
        "successful_capabilities": successful,
    }
    (output / "game_calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Calibrate every game capability supported by one immutable session; "
            "missing movement or image sources are reported and skipped"
        )
    )
    value.add_argument("session", type=Path)
    value.add_argument("output", type=Path, nargs="?")
    value.add_argument("--profile-root", type=Path)
    value.add_argument("--game-id")
    value.add_argument("--maximum-pairs", type=int, default=12)
    value.add_argument("--discovery-config", type=Path)
    value.add_argument("--candidate", action="store_true")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    registry = ProfileRegistry(arguments.profile_root)
    settings = load_system_configuration(arguments.profile_root)
    game_id = arguments.game_id or settings["game"].get("game_id")
    output = arguments.output or _default_output(registry, game_id)
    discovery = (
        json.loads(arguments.discovery_config.read_text(encoding="utf-8"))
        if arguments.discovery_config else None
    )
    result = calibrate_game_session(
        arguments.session,
        output,
        profile_root=registry.root,
        game_id=game_id,
        maximum_pairs=arguments.maximum_pairs,
        discovery_config=discovery,
        activate=not arguments.candidate,
    )
    print("Game calibration: {}".format(Path(output).resolve()))
    for name, outcome in result["capabilities"].items():
        detail = outcome.get("reason") or outcome.get("error") or ""
        print("  {:18s} {}{}".format(name, outcome["status"], ": " + detail if detail else ""))
    return 0 if result["successful_capabilities"] else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["calibrate_game_session", "main"]
