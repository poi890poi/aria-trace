"""Select crisp stable endpoint mini-maps from a transition recording."""

from pathlib import Path

import cv2
import numpy as np

from replay.session_tools import decode_frames

from rig_runtime.adapters.filesystem.session import SessionReader


def _endpoint_candidates(frames, count: int, from_end: bool):
    window = frames[-max(count * 3, count) :] if from_end else frames[: max(count * 3, count)]
    if len(window) <= count:
        return list(window)
    indexes = np.linspace(0, len(window) - 1, count).round().astype(int)
    return [window[int(index)] for index in indexes]


def transition_endpoint_references(
    session_path: Path,
    extractor,
    *,
    stream_id: str = "main",
    samples_per_endpoint: int = 10,
):
    """Return the sharpest mini-map near each stable end of one crossing."""

    reader = SessionReader(Path(session_path))
    frames = reader.frames_by_stream.get(stream_id) or []
    if len(frames) < 2 * max(3, int(samples_per_endpoint)):
        raise RuntimeError("Mini-map transition recording is too short")
    result = {}
    for name, from_end in (("source", False), ("target", True)):
        selected = _endpoint_candidates(
            frames, max(3, int(samples_per_endpoint)), from_end
        )
        images = decode_frames(reader, stream_id, selected)
        choices = []
        for record, image in zip(selected, images):
            observation, mask = extractor.extract(image)
            gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_32F)
            valid = mask > 0
            sharpness = float(np.var(laplacian[valid])) if np.any(valid) else 0.0
            choices.append((sharpness, record, observation, mask))
        sharpness, record, observation, mask = max(choices, key=lambda item: item[0])
        result[name] = {
            "image": observation,
            "mask": mask,
            "source_frame_index": int(record["frame_index"]),
            "session_time_ns": int(record["session_time_ns"]),
            "laplacian_variance": sharpness,
        }
    return result
