"""Calibrate a game's screen-up orientation from synchronized ADB/HIK frames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np

from aria_trace.adapters.filesystem.profile_registry import (
    ProfileContext,
    ProfileRegistry,
)
from aria_trace.adapters.filesystem.session import SessionReader
from aria_trace.adapters.hik.capture import rotate_quarter_turns_clockwise
from aria_trace.services.calibration.rig.cross_source import (
    match_game_camera_orientation,
)
from aria_trace.services.calibration.rig.hik.color_match import (
    synchronized_frame_pairs,
)
from aria_trace.workflows.hik_game_color_calibration import (
    decode_session_records,
    session_game_context,
    sha256_file,
)


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


__all__ = ["calibrate_game_orientation_session"]
