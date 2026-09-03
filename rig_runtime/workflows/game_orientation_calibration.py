"""Calibrate a game's screen-up orientation from synchronized ADB/HIK frames."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np

from rig_runtime.adapters.filesystem.profile_registry import (
    ProfileContext,
    ProfileRegistry,
)
from rig_runtime.adapters.filesystem.session import SessionReader
from rig_runtime.adapters.hik.capture import rotate_quarter_turns_clockwise
from rig_runtime.services.calibration.rig.cross_source import (
    match_game_camera_orientation,
)
from rig_runtime.services.calibration.rig.hik.color_match import (
    synchronized_frame_pairs,
)
from rig_runtime.workflows.hik_game_color_calibration import (
    decode_session_records,
    session_game_context,
    sha256_file,
)
from rig_runtime.workflows.profile_management import (
    game_orientation_from_frame_size,
)


def _portable_game_context(
    reader: SessionReader,
    game_id: Optional[str],
    natural_size_px: Sequence[int],
    logical_size_px: Sequence[int],
    surface_turns: int,
) -> ProfileContext:
    capture = dict(reader.manifest.get("context") or {})
    launch = dict(capture.get("game_launch") or {})
    devices = dict(reader.manifest.get("devices") or {})
    phone = dict(devices.get("phone") or {})
    selected_game = game_id or capture.get("game_id") or launch.get("game_id")
    if not selected_game:
        raise ValueError("Portable game orientation requires a game_id")
    return ProfileContext(
        game_id=str(selected_game),
        platform="android",
        package=launch.get("package") or launch.get("foreground_package_at_capture"),
        phone_id=phone.get("serial") or capture.get("phone_serial"),
        phone_model=phone.get("model"),
        panel_display={
            "natural_panel_px": list(map(int, natural_size_px)),
            "logical_frame_px": list(map(int, natural_size_px)),
        },
        game_display={
            "natural_panel_px": list(map(int, natural_size_px)),
            "logical_frame_px": list(map(int, logical_size_px)),
            "game_viewport_xywh": [
                0, 0, int(logical_size_px[0]), int(logical_size_px[1])
            ],
            "rotation_quarter_turns": int(surface_turns) % 4,
            "ui_layout_id": "default",
        },
    )


def calibrate_portable_game_orientation_session(
    session: Path,
    output: Path,
    *,
    profile_root: Optional[Path] = None,
    game_id: Optional[str] = None,
    activate: bool = True,
    phone_game_revision: Optional[str] = None,
    minimum_consensus: float = 0.80,
) -> Mapping[str, object]:
    """Publish game-up in canonical phone space without any rig dependency."""

    session = Path(session).resolve()
    output = Path(output).resolve()
    reader = SessionReader(session)
    if reader.manifest.get("status") != "complete":
        raise ValueError("Game orientation requires a complete capture session")
    stream_id = (
        "android_phone" if reader.frames_by_stream.get("android_phone") else "main"
    )
    records = list(reader.frames_by_stream.get(stream_id) or [])
    if not records:
        raise ValueError("Session has no Android/main image frames")

    observations = []
    for record in records:
        image_space = dict((record.get("metadata") or {}).get("image_space") or {})
        turns = image_space.get(
            "surface_quarter_turns_clockwise_from_canonical"
        )
        natural = image_space.get("canonical_size_px")
        logical = image_space.get("source_logical_size_px")
        if turns is None or not natural or not logical:
            continue
        observations.append(
            {
                "frame_index": int(record["frame_index"]),
                "turns": int(turns) % 4,
                "natural_size_px": list(map(int, natural)),
                "logical_size_px": list(map(int, logical)),
                "orientation_source": image_space.get("orientation_source"),
            }
        )
    if not observations:
        surface = dict(
            (reader.manifest.get("context") or {}).get(
                "phone_surface_orientation"
            ) or {}
        )
        natural = surface.get("natural_size_px")
        logical = surface.get("logical_size_px")
        turns = surface.get("quarter_turns_clockwise_from_natural")
        if not natural or not logical or turns is None:
            raise ValueError(
                "Android frames and session context have no canonical surface orientation"
            )
        observations.append(
            {
                "frame_index": None,
                "turns": int(turns) % 4,
                "natural_size_px": list(map(int, natural)),
                "logical_size_px": list(map(int, logical)),
                "orientation_source": surface.get("source") or "session_context",
            }
        )

    counts = Counter(item["turns"] for item in observations)
    selected_turns, selected_count = counts.most_common(1)[0]
    consensus = float(selected_count) / float(len(observations))
    selected_observations = [
        item for item in observations if item["turns"] == selected_turns
    ]
    sizes = Counter(
        (
            tuple(item["natural_size_px"]),
            tuple(item["logical_size_px"]),
        )
        for item in selected_observations
    )
    (natural_size, logical_size), size_count = sizes.most_common(1)[0]
    size_consensus = float(size_count) / float(len(selected_observations))
    accepted = consensus >= float(minimum_consensus) and size_consensus >= 0.80
    context = _portable_game_context(
        reader, game_id, natural_size, logical_size, selected_turns
    )
    game_orientation = game_orientation_from_frame_size(logical_size)
    registry = ProfileRegistry(profile_root)
    source_profile = None
    if phone_game_revision:
        source_profile = registry.resolve_revision(
            str(phone_game_revision), context, expected_kind="phone_game"
        )
    else:
        try:
            candidate = registry.resolve("phone_game", context)
        except Exception:
            candidate = None
        if candidate is not None:
            identity = dict(candidate.get("identity") or {})
            if (
                identity.get("panel_signature") == context.panel_signature
                and identity.get("game_display_signature")
                == context.game_display_signature
            ):
                source_profile = candidate
    payload = dict((source_profile or {}).get("payload") or {})
    capabilities = dict(payload.get("capabilities") or {})
    capabilities.update(
        camera_independent=True,
        game_screen_orientation=True,
    )
    payload.update(
        profile_kind="phone_game",
        coordinate_space="phone_natural_display_pixels",
        game_orientation=game_orientation,
        game_surface_quarter_turns_clockwise_from_phone_natural=int(
            selected_turns
        ),
        orientation_source="android_per_frame_space_metadata_consensus",
        orientation_quality={
            "observation_count": len(observations),
            "quarter_turn_counts": {
                str(turns): int(count) for turns, count in sorted(counts.items())
            },
            "consensus": consensus,
            "size_consensus": size_consensus,
            "minimum_consensus": float(minimum_consensus),
        },
        capabilities=capabilities,
    )
    output.mkdir(parents=True, exist_ok=False)
    sample_record = next(
        (
            record for record in records
            if int(record["frame_index"]) == selected_observations[0]["frame_index"]
        ),
        records[0],
    )
    sample = decode_session_records(reader, stream_id, [sample_record])[0]
    _write_image(output / "android_game_orientation_sample.png", sample)
    summary = {
        "schema_version": "1.0",
        "calibration_kind": "portable_game_screen_orientation",
        "status": (
            "accepted" if accepted and activate else "review_required"
        ),
        "session": str(session),
        "profile_context": context.as_dict(),
        "game_orientation": game_orientation,
        "game_surface_quarter_turns_clockwise_from_phone_natural": int(
            selected_turns
        ),
        "rig_dependency": None,
        "method": "android_per_frame_space_metadata_consensus",
        "quality": payload["orientation_quality"],
        "selected_observations": selected_observations,
        "evidence": ["android_game_orientation_sample.png"],
    }
    summary_path = output / "game_orientation_calibration.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    profile = registry.publish(
        "phone_game",
        context,
        payload,
        dependencies={},
        provenance={
            "session": str(session),
            "calibration": str(summary_path),
            "source_phone_game_revision": (
                source_profile.get("revision_id") if source_profile else None
            ),
        },
        review_state="accepted" if accepted and activate else "review_required",
        activate=bool(accepted and activate),
    )
    summary["profile_revision"] = profile["revision_id"]
    summary["profile_activated"] = bool(accepted and activate)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError("Cannot write game-orientation evidence {}".format(path))


def _stored_hik_to_calibration_display(
    image: np.ndarray, record: Mapping[str, object]
) -> np.ndarray:
    metadata = dict(record.get("metadata") or {})
    turns = int(
        metadata.get(
            "output_quarter_turns_clockwise_from_calibration_display", 0
        )
    ) % 4
    padding = metadata.get("video_encoding_padding_right_bottom_px") or [0, 0]
    right, bottom = [int(value) for value in padding]
    content = image
    if right:
        content = content[:, :-right]
    if bottom:
        content = content[:-bottom, :]
    return rotate_quarter_turns_clockwise(content, -turns)


def calibrate_game_orientation_session(
    session: Path,
    output: Path,
    *,
    profile_root: Optional[Path] = None,
    game_id: Optional[str] = None,
    maximum_pairs: int = 12,
    activate: bool = True,
    preferred_confidence: float = 0.50,
    preferred_margin: float = 0.06,
) -> Mapping[str, object]:
    """Fit one discrete game-up rotation and publish an independent profile."""

    session = Path(session).resolve()
    output = Path(output).resolve()
    reader = SessionReader(session)
    if reader.manifest.get("status") != "complete":
        raise ValueError("Game orientation requires a complete capture session")
    for stream_id in ("android_phone", "hik_phone"):
        if not reader.frames_by_stream.get(stream_id):
            raise ValueError("Session has no {} frames".format(stream_id))

    calibration_value = (
        ((reader.manifest.get("context") or {}).get("hik_capture") or {}).get(
            "rig_calibration"
        )
    )
    if not calibration_value:
        raise ValueError("Session does not identify its HIK rig calibration")
    rig_calibration = Path(str(calibration_value)).resolve()
    if not rig_calibration.is_file():
        raise FileNotFoundError("Rig calibration is unavailable: {}".format(rig_calibration))
    rig_document = json.loads(rig_calibration.read_text(encoding="utf-8"))
    context = session_game_context(reader, rig_document, game_id=game_id)
    registry = ProfileRegistry(profile_root)
    active_rig = registry.resolve(
        "rig",
        ProfileContext(
            camera_id=context.camera_id,
            phone_id=context.phone_id,
            phone_model=context.phone_model,
            panel_display=context.panel_display,
        ),
    )
    active_rig_file = registry.runtime_file(
        active_rig, "hik_camera_calibration"
    ).resolve()
    if sha256_file(active_rig_file) != sha256_file(rig_calibration):
        raise ValueError(
            "Session HIK frames were not captured with the active rig revision"
        )

    android_records = list(reader.frames_by_stream["android_phone"])
    hik_records = list(reader.frames_by_stream["hik_phone"])
    pairs = synchronized_frame_pairs(
        np.asarray(
            [int(item["host_capture_time_ns"]) for item in android_records],
            dtype=np.int64,
        ),
        np.asarray(
            [int(item["host_capture_time_ns"]) for item in hik_records],
            dtype=np.int64,
        ),
        maximum_pairs=maximum_pairs,
    )
    if not pairs:
        raise ValueError("No synchronized ADB/HIK frame pairs are available")
    selected_android = [android_records[a] for a, _h, _d in pairs]
    selected_hik = [hik_records[h] for _a, h, _d in pairs]
    android_frames = decode_session_records(reader, "android_phone", selected_android)
    hik_frames = decode_session_records(reader, "hik_phone", selected_hik)

    pair_results = []
    pair_images = []
    candidate_scores = {turns: [] for turns in range(4)}
    for pair_index, (adb, stored_hik, pair) in enumerate(
        zip(android_frames, hik_frames, pairs)
    ):
        hik = _stored_hik_to_calibration_display(
            stored_hik, selected_hik[pair_index]
        )
        result, images = match_game_camera_orientation(
            adb,
            hik,
            rig_calibration,
            preferred_confidence=preferred_confidence,
            preferred_margin=preferred_margin,
        )
        scored = [
            item for item in result["candidates"]
            if item.get("status") == "scored"
        ]
        best_information = max(
            float((item.get("metrics") or {}).get("information_quality", 0.0))
            for item in scored
        )
        eligible = best_information > 0.0
        if eligible:
            for item in scored:
                turns = int(
                    item[
                        "camera_adapter_image_quarter_turns_clockwise_from_calibration_display"
                    ]
                )
                candidate_scores[turns].append(
                    float(item["metrics"]["confidence"])
                )
        pair_results.append(
            {
                "pair_index": pair_index,
                "android_frame_index": int(selected_android[pair_index]["frame_index"]),
                "hik_frame_index": int(selected_hik[pair_index]["frame_index"]),
                "host_delta_ms": float(pair[2]),
                "information_quality": best_information,
                "eligible": eligible,
                "selected_quarter_turns_clockwise": int(
                    result[
                        "selected_camera_adapter_image_quarter_turns_clockwise_from_calibration_display"
                    ]
                ),
                "selected_confidence": float(result["selected_confidence"]),
                "confidence_margin": result["confidence_margin"],
                "candidates": scored,
            }
        )
        pair_images.append((adb, hik, images))

    aggregated = []
    for turns, scores in candidate_scores.items():
        if scores:
            aggregated.append(
                {
                    "quarter_turns_clockwise_from_calibration_display": turns,
                    "median_confidence": float(np.median(scores)),
                    "minimum_confidence": float(np.min(scores)),
                    "pair_count": len(scores),
                }
            )
    if not aggregated:
        raise ValueError(
            "All synchronized pairs have sparse/full masks or insufficient edges; "
            "orientation was not guessed"
        )
    aggregated.sort(key=lambda item: item["median_confidence"], reverse=True)
    selected = aggregated[0]
    runner = aggregated[1] if len(aggregated) > 1 else None
    margin = (
        float(selected["median_confidence"] - runner["median_confidence"])
        if runner is not None else None
    )
    selected_turns = int(selected["quarter_turns_clockwise_from_calibration_display"])
    phone = dict(rig_document.get("phone") or {})
    viewer = dict(phone.get("viewer") or {})
    rig_display_turns = int(
        phone.get(
            "orientation_quarter_turns",
            viewer.get("canonical_orientation_quarter_turns", 0),
        )
    ) % 4
    game_surface_turns = (selected_turns + rig_display_turns) % 4
    game_orientation = game_orientation_from_frame_size(
        context.game_display.get("logical_frame_px") or []
    )
    eligible_pairs = [item for item in pair_results if item["eligible"]]
    consensus = float(
        np.mean(
            [item["selected_quarter_turns_clockwise"] == selected_turns for item in eligible_pairs]
        )
    )
    accepted = (
        float(selected["median_confidence"]) >= float(preferred_confidence)
        and (margin is None or margin >= float(preferred_margin))
        and consensus >= 0.60
    )

    output.mkdir(parents=True, exist_ok=False)
    for index, (adb, hik, _images) in enumerate(pair_images[:4]):
        _write_image(output / "pairs" / "pair-{:02d}-adb.png".format(index), adb)
        _write_image(
            output / "pairs" / "pair-{:02d}-hik-calibration-display.png".format(index),
            hik,
        )
    best_pair_index = max(
        range(len(pair_results)),
        key=lambda index: (
            pair_results[index]["eligible"],
            pair_results[index]["selected_confidence"],
        ),
    )
    for name, image in pair_images[best_pair_index][2].items():
        _write_image(output / "review" / name, image)
    selected_preview = rotate_quarter_turns_clockwise(
        pair_images[best_pair_index][1], selected_turns
    )
    _write_image(output / "review" / "selected_hik_game_upright.png", selected_preview)

    summary = {
        "schema_version": "1.0",
        "calibration_kind": "game_screen_orientation",
        "status": "accepted" if accepted and activate else "review_required",
        "session": str(session),
        "rig_calibration": str(rig_calibration),
        "rig_revision": active_rig["revision_id"],
        "profile_context": context.as_dict(),
        "game_orientation": game_orientation,
        "game_surface_quarter_turns_clockwise_from_phone_natural": game_surface_turns,
        "camera_adapter_image_quarter_turns_clockwise_from_calibration_display": selected_turns,
        "orientation_space_contract": {
            "portable_source_space": "phone_natural_rotation_0",
            "adapter_base_space": "rig_calibration_display",
            "rig_calibration_display_quarter_turns_clockwise_from_phone_natural": (
                rig_display_turns
            ),
            "composition": (
                "adapter_turn = game_surface_turn - "
                "rig_calibration_display_turn (mod 4)"
            ),
        },
        "runtime_operation": (
            "precompose_into_rectification_lookup; "
            "discrete_quarter_turn_only_when_rectification_is_disabled"
        ),
        "aggregated_candidates": aggregated,
        "median_confidence": float(selected["median_confidence"]),
        "confidence_margin": margin,
        "pair_consensus": consensus,
        "eligible_pair_count": len(eligible_pairs),
        "selected_pairs": pair_results,
        "evidence": {
            "lossless_source_stills": "pairs/*.png",
            "review": "review/*.png",
            "best_pair_index": best_pair_index,
        },
    }
    summary_path = output / "game_orientation_calibration.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    profile = registry.publish(
        "rig_game_orientation",
        context,
        {
            "profile_kind": "rig_game_orientation",
            "game_orientation": game_orientation,
            "game_surface_quarter_turns_clockwise_from_phone_natural": (
                game_surface_turns
            ),
            "camera_adapter_image_quarter_turns_clockwise_from_calibration_display": selected_turns,
            "orientation_space_contract": summary["orientation_space_contract"],
            "runtime_operation": (
                "precompose_into_rectification_lookup; "
                "discrete_quarter_turn_only_when_rectification_is_disabled"
            ),
            "capabilities": {
                "game_screen_upright": True,
                "game_world_north": False,
                "runtime_interpolation": False,
            },
            "quality": {
                "median_confidence": float(selected["median_confidence"]),
                "confidence_margin": margin,
                "pair_consensus": consensus,
                "eligible_pair_count": len(eligible_pairs),
            },
        },
        dependencies={"rig": active_rig["revision_id"]},
        provenance={
            "session": str(session),
            "calibration": str(summary_path),
            "selected_android_frame_indices": [
                item["android_frame_index"] for item in pair_results
            ],
            "selected_hik_frame_indices": [
                item["hik_frame_index"] for item in pair_results
            ],
        },
        review_state="accepted" if accepted and activate else "review_required",
        activate=bool(accepted and activate),
    )
    summary["profile_revision"] = profile["revision_id"]
    summary["profile_activated"] = bool(accepted and activate)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


__all__ = [
    "calibrate_game_orientation_session",
    "calibrate_portable_game_orientation_session",
]
