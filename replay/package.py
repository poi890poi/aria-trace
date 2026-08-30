"""Compile a recorded demonstration into a portable ReplayPackage."""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from aria_trace.adapters.filesystem.session import SessionReader

from . import SCHEMA_VERSION
from .descriptors import describe, extract_many
from .session_tools import (
    decode_frames,
    make_stages,
    route_annotations,
    route_bounds,
    sample_frames,
    sha256_file,
    stage_for_time,
)


def _write_jsonl(path: Path, records) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


def compile_replay_package(
    session_path: Path,
    output_path: Path,
    stream_id: str,
    route_id: str,
    reference_rate_hz: float = 5.0,
    descriptor_config: Optional[dict] = None,
) -> dict:
    session_path = Path(session_path)
    output_path = Path(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise RuntimeError("ReplayPackage directory is not empty: {}".format(output_path))
    output_path.mkdir(parents=True, exist_ok=True)
    reader = SessionReader(session_path)
    if stream_id not in reader.frames_by_stream:
        raise KeyError("Unknown stream: {}".format(stream_id))
    annotations = route_annotations(session_path, stream_id, route_id)
    start, complete = route_bounds(annotations, require_complete=True)
    frames = sample_frames(
        reader.frames_by_stream[stream_id],
        start["session_time_ns"],
        complete["session_time_ns"],
        reference_rate_hz,
    )
    images = decode_frames(reader, stream_id, frames)
    config = describe(descriptor_config)
    descriptors = extract_many(images, config)
    np.savez_compressed(output_path / "descriptors.npz", descriptors=descriptors)

    stages = make_stages(start, complete, annotations)
    references = []
    for index, frame in enumerate(frames):
        references.append(
            {
                "reference_id": "ref_{:06d}".format(index),
                "descriptor_index": index,
                "stream_id": stream_id,
                "source_frame_index": frame["frame_index"],
                "source_session_time_ns": frame["session_time_ns"],
                "route_time_ns": frame["session_time_ns"] - start["session_time_ns"],
                "progress": index / float(max(1, len(frames) - 1)),
                "stage_id": stage_for_time(stages, frame["session_time_ns"])["stage_id"],
                "online_source_evidence": reader.online_features_for_frame(
                    stream_id, frame["frame_index"]
                ),
            }
        )

    action_priors = []
    for index in range(len(references) - 1):
        lower = references[index]["source_session_time_ns"]
        upper = references[index + 1]["source_session_time_ns"]
        events = []
        for event in reader.inputs:
            if lower <= event["session_time_ns"] < upper:
                copied = dict(event)
                copied["offset_from_interval_start_ns"] = event["session_time_ns"] - lower
                copied["route_time_ns"] = event["session_time_ns"] - start["session_time_ns"]
                events.append(copied)
        action_priors.append(
            {
                "prior_id": "prior_{:06d}".format(index),
                "from_reference_id": references[index]["reference_id"],
                "to_reference_id": references[index + 1]["reference_id"],
                "stage_id": references[index]["stage_id"],
                "duration_ns": upper - lower,
                "events": events,
            }
        )

    _write_jsonl(output_path / "stages.jsonl", stages)
    _write_jsonl(output_path / "references.jsonl", references)
    _write_jsonl(output_path / "action_priors.jsonl", action_priors)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "package_id": str(uuid.uuid4()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "route_id": route_id,
        "portal_id": start.get("portal_id"),
        "stream_id": stream_id,
        "source_session": {
            "session_id": reader.manifest.get("session_id"),
            "schema_version": reader.manifest.get("schema_version"),
            "manifest_sha256": sha256_file(session_path / "manifest.json"),
            "annotations_sha256": sha256_file(session_path / "annotations.jsonl"),
            "context": reader.manifest.get("context", {}),
        },
        "route_interval": {
            "start_session_time_ns": start["session_time_ns"],
            "end_session_time_ns": complete["session_time_ns"],
            "start_frame_index": start["frame_index"],
            "end_frame_index": complete["frame_index"],
            "start_annotation_id": start["annotation_id"],
            "complete_annotation_id": complete["annotation_id"],
        },
        "visual_descriptor": config,
        "visual_source_quality": "decoded_primary_video",
        "reference_rate_hz": float(reference_rate_hz),
        "counts": {
            "stages": len(stages),
            "references": len(references),
            "action_priors": len(action_priors),
            "input_events": sum(len(item["events"]) for item in action_priors),
        },
        "files": {
            "descriptors": "descriptors.npz",
            "stages": "stages.jsonl",
            "references": "references.jsonl",
            "action_priors": "action_priors.jsonl",
        },
        "action_semantics": "recorded_input_prior_not_timed_macro",
    }
    _write_json_atomic(output_path / "manifest.json", manifest)
    return manifest


class ReplayPackage:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.manifest = json.loads((self.path / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(
                "Unsupported ReplayPackage schema: {}".format(self.manifest.get("schema_version"))
            )
        self.stages = self._read_jsonl(self.manifest["files"]["stages"])
        self.references = self._read_jsonl(self.manifest["files"]["references"])
        self.action_priors = self._read_jsonl(self.manifest["files"]["action_priors"])
        with np.load(str(self.path / self.manifest["files"]["descriptors"])) as archive:
            self.descriptors = archive["descriptors"].astype(np.float32)
        if len(self.references) != len(self.descriptors):
            raise RuntimeError("ReplayPackage reference/descriptor counts disagree")

    def _read_jsonl(self, filename: str):
        records = []
        with (self.path / filename).open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    records.append(json.loads(line))
        return records
