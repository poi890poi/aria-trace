"""Shared selection and decoding helpers for replay compilation and alignment."""

import hashlib
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

from rig_runtime.adapters.filesystem.annotations import AnnotationStore
from rig_runtime.adapters.filesystem.session import SessionReader


def sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def route_annotations(
    session_path: Path,
    stream_id: str,
    route_id: Optional[str],
) -> List[dict]:
    values = []
    for annotation in AnnotationStore(session_path).list():
        if annotation.get("stream_id") != stream_id:
            continue
        if route_id is not None and annotation.get("route_id") != route_id:
            continue
        values.append(annotation)
    return values


def route_bounds(
    annotations: List[dict],
    require_complete: bool,
) -> Tuple[Optional[dict], Optional[dict]]:
    starts = [item for item in annotations if item["kind"] == "route_start"]
    if not starts:
        if require_complete:
            raise RuntimeError("No route_start annotation matches this route and stream")
        return None, None
    start = starts[0]
    completions = [
        item
        for item in annotations
        if item["kind"] == "route_complete"
        and item["session_time_ns"] >= start["session_time_ns"]
    ]
    if not completions:
        if require_complete:
            raise RuntimeError("No route_complete annotation follows route_start")
        return start, None
    return start, completions[0]


def sample_frames(frames: List[dict], start_ns: int, end_ns: int, rate_hz: float) -> List[dict]:
    if rate_hz <= 0:
        raise ValueError("Sample rate must be positive")
    inside = [
        frame for frame in frames
        if start_ns <= frame["session_time_ns"] <= end_ns
    ]
    if not inside:
        raise RuntimeError("No frames exist inside the selected route interval")
    interval_ns = int(round(1.0e9 / rate_hz))
    selected = [inside[0]]
    next_time_ns = inside[0]["session_time_ns"] + interval_ns
    for frame in inside[1:-1]:
        if frame["session_time_ns"] >= next_time_ns:
            selected.append(frame)
            next_time_ns = frame["session_time_ns"] + interval_ns
    if inside[-1]["frame_index"] != selected[-1]["frame_index"]:
        selected.append(inside[-1])
    return selected


def decode_frames(reader: SessionReader, stream_id: str, frames: List[dict]):
    requested = {int(frame["frame_index"]): position for position, frame in enumerate(frames)}
    decoded = [None] * len(frames)
    capture = cv2.VideoCapture(str(reader.video_path(stream_id)))
    if not capture.isOpened():
        raise RuntimeError("Could not open video: {}".format(reader.video_path(stream_id)))
    try:
        index = 0
        maximum = max(requested) if requested else -1
        while index <= maximum:
            ok, image = capture.read()
            if not ok:
                break
            if index in requested:
                decoded[requested[index]] = image
            index += 1
    finally:
        capture.release()
    missing = [frames[index]["frame_index"] for index, image in enumerate(decoded) if image is None]
    if missing:
        raise RuntimeError("Could not decode selected video frames: {}".format(missing[:10]))
    return decoded


def make_stages(start: dict, complete: dict, annotations: List[dict]) -> List[dict]:
    boundaries = [start]
    boundaries.extend(
        item
        for item in annotations
        if item["kind"] == "route_stage"
        and start["session_time_ns"] < item["session_time_ns"] < complete["session_time_ns"]
    )
    boundaries.sort(key=lambda item: item["session_time_ns"])
    stages = []
    for index, boundary in enumerate(boundaries):
        end = boundaries[index + 1] if index + 1 < len(boundaries) else complete
        stages.append(
            {
                "stage_id": "stage_{:03d}".format(index),
                "label": boundary.get("note") or ("route_start" if index == 0 else "stage_{}".format(index)),
                "start_session_time_ns": boundary["session_time_ns"],
                "end_session_time_ns": end["session_time_ns"],
                "start_frame_index": boundary["frame_index"],
                "end_frame_index": end["frame_index"],
                "source_annotation_id": boundary["annotation_id"],
            }
        )
    return stages


def stage_for_time(stages: List[dict], session_time_ns: int) -> dict:
    for stage in reversed(stages):
        if session_time_ns >= stage["start_session_time_ns"]:
            return stage
    return stages[0]
