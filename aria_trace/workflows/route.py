"""Compile recorded gameplay routes against a canonical map atlas."""

import json
import math
from pathlib import Path
from typing import Optional

from replay.route_tracking import compile_route_tracking_package, describe_minimap
from replay.session_tools import decode_frames, sample_frames

from aria_trace.services.tracking.runtime import MinimapExtractor
from aria_trace.services.mapping.layers import LayeredGlobalLocalizer
from aria_trace.adapters.filesystem.session import SessionReader


def localize_route_frames(
    frame_records,
    images,
    extractor,
    localizer,
    *,
    max_step_px: float = 80.0,
    search_radius_px: float = 160.0,
    progress=None,
):
    """Convert sampled source frames into continuous canonical observations."""

    accepted = []
    rejected = []
    previous_xy = None
    total = len(frame_records)
    for index, (frame_record, image) in enumerate(zip(frame_records, images)):
        observation, mask = extractor.extract(image)
        fix = localizer.localize(
            observation,
            mask,
            search_center_xy=previous_xy,
            search_radius_px=search_radius_px if previous_xy is not None else None,
        )
        reason = None
        if not fix.valid:
            reason = "localization:" + ",".join(fix.rejection_reasons)
        elif previous_xy is not None:
            distance = math.hypot(fix.x - previous_xy[0], fix.y - previous_xy[1])
            if distance > float(max_step_px):
                reason = "discontinuous:{:.2f}px".format(distance)
        if reason:
            rejected.append(
                {
                    "source_frame_index": int(frame_record["frame_index"]),
                    "session_time_ns": int(frame_record["session_time_ns"]),
                    "reason": reason,
                    "score": float(fix.score),
                    "margin": float(fix.margin),
                }
            )
        else:
            map_layer = (fix.diagnostics or {}).get("map_layer") or {}
            mode_id = str(map_layer.get("selected_mode_id") or "default")
            mode_likelihoods = dict(map_layer.get("mode_likelihoods") or {})
            accepted.append(
                {
                    "source_frame_index": int(frame_record["frame_index"]),
                    "session_time_ns": int(frame_record["session_time_ns"]),
                    "x": float(fix.x),
                    "y": float(fix.y),
                    "map_alignment_deg": float(fix.yaw_deg),
                    "map_scale": float(fix.scale),
                    "mode_id": mode_id,
                    "mode_likelihoods": mode_likelihoods,
                    "localization_score": float(fix.score),
                    "localization_margin": float(fix.margin),
                    "descriptor": describe_minimap(observation, mask),
                }
            )
            previous_xy = (float(fix.x), float(fix.y))
        if progress and (index % max(1, total // 20) == 0 or index + 1 == total):
            progress(
                "Localizing route sample {} / {} ({} accepted)".format(
                    index + 1, total, len(accepted)
                )
            )
    if len(accepted) < 3:
        raise RuntimeError(
            "Route compilation produced only {} continuous localized samples"
            .format(len(accepted))
        )
    return accepted, rejected


def compile_route_session(
    session_path: Path,
    output_path: Path,
    *,
    stream_id: str,
    route_id: str,
    atlas_path: Path,
    minimap_config: dict,
    minimap_calibration: dict,
    reference_rate_hz: float = 5.0,
    max_step_px: float = 80.0,
    corridor_radius_px: float = 35.0,
    progress=None,
) -> dict:
    """Compile a labeled route session into a ready route-tracking artifact."""

    reader = SessionReader(Path(session_path))
    if stream_id not in reader.frames_by_stream:
        raise ValueError("Route session has no stream: {}".format(stream_id))
    source_frames = reader.frames_by_stream[stream_id]
    if not source_frames:
        raise RuntimeError("Route session contains no frames")
    selected = sample_frames(
        source_frames,
        int(source_frames[0]["session_time_ns"]),
        int(source_frames[-1]["session_time_ns"]),
        float(reference_rate_hz),
    )
    if progress:
        progress("Decoding {} sampled route frames".format(len(selected)))
    images = decode_frames(reader, stream_id, selected)
    extractor = MinimapExtractor(
        minimap_config["crop_xywh"], minimap_calibration
    )
    localizer = LayeredGlobalLocalizer(Path(atlas_path))
    try:
        observations, rejected = localize_route_frames(
            selected,
            images,
            extractor,
            localizer,
            max_step_px=max_step_px,
            progress=progress,
        )
        manifest = compile_route_tracking_package(
            observations,
            Path(output_path),
            route_id=route_id,
            atlas_id=localizer.manifest["atlas_id"],
            coordinate_space_id=localizer.manifest["coordinate_space_id"],
            corridor_radius_px=corridor_radius_px,
            max_step_px=max_step_px,
        )
    finally:
        localizer.close()
    manifest["source_session"] = {
        "session_id": reader.manifest.get("session_id"),
        "session_path": str(Path(session_path)),
        "stream_id": stream_id,
        "source_frame_count": len(source_frames),
        "sampled_frame_count": len(selected),
        "accepted_observation_count": len(observations),
        "rejected_observation_count": len(rejected),
    }
    manifest["localization_rejections"] = rejected
    manifest["reference_rate_hz"] = float(reference_rate_hz)
    (Path(output_path) / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
