"""Monotonic visual alignment between a live/replayed session and a demonstration."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from aria_trace.adapters.filesystem.session import SessionReader

from .descriptors import extract_many
from .package import ReplayPackage
from .session_tools import (
    decode_frames,
    make_stages,
    route_annotations,
    route_bounds,
    sample_frames,
    stage_for_time,
)


def cosine_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if query.ndim != 2 or reference.ndim != 2 or query.shape[1] != reference.shape[1]:
        raise ValueError("Descriptor matrices must be two-dimensional with equal width")
    query_norm = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1.0e-8)
    reference_norm = reference / np.maximum(
        np.linalg.norm(reference, axis=1, keepdims=True), 1.0e-8
    )
    return np.clip(1.0 - np.matmul(query_norm, reference_norm.T), 0.0, 2.0)


def align_descriptors(
    query: np.ndarray,
    reference: np.ndarray,
    max_advance: int = 4,
    stay_penalty: float = 0.02,
    skip_penalty: float = 0.05,
    force_end: bool = False,
) -> dict:
    """Viterbi/DTW alignment constrained to non-decreasing route progress."""
    if len(query) == 0 or len(reference) == 0:
        raise ValueError("Alignment requires query and reference descriptors")
    if max_advance < 1:
        raise ValueError("max_advance must be at least one")
    costs = cosine_distances(query, reference)
    query_count, reference_count = costs.shape
    accumulated = np.full((query_count, reference_count), np.inf, dtype=np.float64)
    previous = np.full((query_count, reference_count), -1, dtype=np.int32)
    accumulated[0, 0] = float(costs[0, 0])
    for query_index in range(1, query_count):
        for reference_index in range(reference_count):
            lower = max(0, reference_index - max_advance)
            best_cost = np.inf
            best_previous = -1
            for prior_index in range(lower, reference_index + 1):
                advance = reference_index - prior_index
                transition = stay_penalty if advance == 0 else skip_penalty * max(0, advance - 1)
                candidate = accumulated[query_index - 1, prior_index] + transition
                if candidate < best_cost:
                    best_cost = candidate
                    best_previous = prior_index
            if best_previous >= 0:
                accumulated[query_index, reference_index] = best_cost + float(
                    costs[query_index, reference_index]
                )
                previous[query_index, reference_index] = best_previous
    end_index = reference_count - 1 if force_end else int(np.argmin(accumulated[-1]))
    if not np.isfinite(accumulated[-1, end_index]):
        raise RuntimeError("No monotonic alignment path satisfies the transition limits")
    path = [end_index]
    for query_index in range(query_count - 1, 0, -1):
        end_index = int(previous[query_index, end_index])
        if end_index < 0:
            raise RuntimeError("Alignment backtracking encountered an incomplete path")
        path.append(end_index)
    path.reverse()
    return {
        "path": path,
        "visual_costs": costs,
        "total_cost": float(accumulated[-1, path[-1]]),
    }


def _write_jsonl(path: Path, records) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


def align_session(
    package_path: Path,
    query_session_path: Path,
    output_path: Path,
    stream_id: Optional[str] = None,
    route_id: Optional[str] = None,
    query_rate_hz: float = 5.0,
    max_advance: int = 4,
    distance_threshold: float = 0.45,
) -> dict:
    package = ReplayPackage(package_path)
    reader = SessionReader(query_session_path)
    stream_id = stream_id or package.manifest["stream_id"]
    route_id = route_id or package.manifest["route_id"]
    if stream_id not in reader.frames_by_stream:
        raise KeyError("Unknown query stream: {}".format(stream_id))
    annotations = route_annotations(query_session_path, stream_id, route_id)
    start, complete = route_bounds(annotations, require_complete=False)
    stream_frames = reader.frames_by_stream[stream_id]
    start_ns = start["session_time_ns"] if start else stream_frames[0]["session_time_ns"]
    end_ns = complete["session_time_ns"] if complete else stream_frames[-1]["session_time_ns"]
    frames = sample_frames(stream_frames, start_ns, end_ns, query_rate_hz)
    images = decode_frames(reader, stream_id, frames)
    query_descriptors = extract_many(images, package.manifest["visual_descriptor"])
    aligned = align_descriptors(
        query_descriptors,
        package.descriptors,
        max_advance=max_advance,
        force_end=complete is not None,
    )

    expected_stages = None
    if start and complete:
        expected_stages = make_stages(start, complete, annotations)
    stage_by_id = {stage["stage_id"]: stage for stage in package.stages}
    records = []
    expected_count = 0
    expected_correct = 0
    for query_index, (frame, reference_index) in enumerate(zip(frames, aligned["path"])):
        row = aligned["visual_costs"][query_index]
        distance = float(row[reference_index])
        outside = np.concatenate(
            (row[: max(0, reference_index - 2)], row[min(len(row), reference_index + 3) :])
        )
        alternative = float(np.min(outside)) if len(outside) else distance
        margin = alternative - distance
        confidence = float(
            np.clip(1.0 - distance / max(distance_threshold, 1.0e-8), 0.0, 1.0) * 0.75
            + np.clip(margin / 0.15, 0.0, 1.0) * 0.25
        )
        reference_record = package.references[reference_index]
        stage = stage_by_id[reference_record["stage_id"]]
        expected_label = None
        if expected_stages:
            expected_label = stage_for_time(expected_stages, frame["session_time_ns"])["label"]
            expected_count += 1
            if expected_label == stage["label"]:
                expected_correct += 1
        records.append(
            {
                "query_index": query_index,
                "query_frame_index": frame["frame_index"],
                "query_session_time_ns": frame["session_time_ns"],
                "reference_index": reference_index,
                "reference_id": reference_record["reference_id"],
                "stage_id": stage["stage_id"],
                "stage_label": stage["label"],
                "expected_stage_label": expected_label,
                "progress": reference_record["progress"],
                "visual_distance": distance,
                "alternative_margin": margin,
                "confidence": confidence,
                "accepted": distance <= distance_threshold,
            }
        )

    output_path = Path(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise RuntimeError("Alignment output directory is not empty: {}".format(output_path))
    output_path.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path / "alignment.jsonl", records)
    accepted = [record for record in records if record["accepted"]]
    summary = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "package_id": package.manifest["package_id"],
        "query_session_id": reader.manifest.get("session_id"),
        "route_id": route_id,
        "stream_id": stream_id,
        "query_bounds_source": "route_annotations" if start else "full_session",
        "completion_observed": complete is not None,
        "query_reference_count": len(records),
        "accepted_count": len(accepted),
        "accepted_fraction": len(accepted) / float(len(records)),
        "mean_visual_distance": float(np.mean([record["visual_distance"] for record in records])),
        "mean_confidence": float(np.mean([record["confidence"] for record in records])),
        "final_progress": records[-1]["progress"],
        "monotonic": all(
            records[index]["reference_index"] <= records[index + 1]["reference_index"]
            for index in range(len(records) - 1)
        ),
        "total_alignment_cost": aligned["total_cost"],
        "stage_label_accuracy": (
            expected_correct / float(expected_count) if expected_count else None
        ),
        "visual_source_quality": "decoded_primary_video",
        "files": {"alignment": "alignment.jsonl"},
    }
    _write_json_atomic(output_path / "summary.json", summary)
    return summary
