"""Compile and query route-constrained localization packages."""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional

import cv2
import numpy as np


SCHEMA_VERSION = "1.0"


def describe_minimap(image: np.ndarray, mask: Optional[np.ndarray] = None, size: int = 32):
    """Return a compact normalized gradient template for route matching."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    if mask is not None:
        gradient = gradient * (mask.astype(np.float32) / 255.0)
    resized = cv2.resize(gradient, (size, size), interpolation=cv2.INTER_AREA)
    vector = resized.reshape(-1).astype(np.float32)
    vector -= float(np.mean(vector))
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1.0e-6 else vector


def _angle_difference(first: float, second: float) -> float:
    return ((float(first) - float(second) + 180.0) % 360.0) - 180.0


def _percentiles(values) -> dict:
    rows = np.asarray(tuple(values), dtype=np.float64)
    if not len(rows):
        return {"median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "median": float(np.median(rows)),
        "p95": float(np.percentile(rows, 95)),
        "p99": float(np.percentile(rows, 99)),
        "max": float(np.max(rows)),
    }


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def compile_route_tracking_package(
    observations: Iterable[Mapping],
    output_path: Path,
    *,
    route_id: str,
    atlas_id: str,
    coordinate_space_id: str,
    corridor_radius_px: float = 35.0,
    max_step_px: Optional[float] = None,
) -> dict:
    """Precompute route geometry, descriptors, adjacency, and transition bands."""

    rows = [dict(item) for item in observations]
    if len(rows) < 3:
        raise ValueError("Route tracking needs at least three localized observations")
    if any(
        int(later["session_time_ns"]) <= int(earlier["session_time_ns"])
        for earlier, later in zip(rows, rows[1:])
    ):
        raise ValueError("Route observations must have increasing timestamps")
    descriptors = np.asarray([item["descriptor"] for item in rows], dtype=np.float32)
    if descriptors.ndim != 2 or descriptors.shape[0] != len(rows):
        raise ValueError("Route descriptors must have one fixed-length vector per state")
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptors = descriptors / np.maximum(norms, 1.0e-6)

    positions = np.asarray(
        [[float(item["x"]), float(item["y"])] for item in rows], dtype=np.float64
    )
    step_vectors = np.diff(positions, axis=0)
    step_distances = np.linalg.norm(step_vectors, axis=1)
    if max_step_px is not None and np.any(step_distances > float(max_step_px)):
        index = int(np.argmax(step_distances))
        raise ValueError(
            "Route localization is discontinuous between states {} and {}: {:.2f}px"
            .format(index, index + 1, step_distances[index])
        )
    cumulative = np.concatenate(([0.0], np.cumsum(step_distances)))
    elapsed_s = np.diff(
        np.asarray([int(item["session_time_ns"]) for item in rows], dtype=np.float64)
    ) / 1.0e9
    speeds = step_distances / np.maximum(elapsed_s, 1.0e-6)

    headings = []
    for index in range(len(rows)):
        if index == 0:
            vector = positions[1] - positions[0]
        elif index == len(rows) - 1:
            vector = positions[-1] - positions[-2]
        else:
            vector = positions[index + 1] - positions[index - 1]
        headings.append(math.degrees(math.atan2(float(vector[1]), float(vector[0]))) % 360.0)
    turn_rates = [
        abs(_angle_difference(headings[index + 1], headings[index]))
        / max(elapsed_s[index], 1.0e-6)
        for index in range(len(rows) - 1)
    ]

    states = []
    mode_indexes = {}
    for index, item in enumerate(rows):
        mode_id = str(item["mode_id"])
        state = {
            "state_index": index,
            "source_frame_index": int(item["source_frame_index"]),
            "session_time_ns": int(item["session_time_ns"]),
            "route_progress": index / float(len(rows) - 1),
            "route_distance_px": float(cumulative[index]),
            "canonical_xy": positions[index].tolist(),
            "route_heading_deg": float(headings[index]),
            "map_alignment_deg": (
                float(item["map_alignment_deg"])
                if item.get("map_alignment_deg") is not None
                else None
            ),
            "map_scale": float(item.get("map_scale") or 1.0),
            "mode_id": mode_id,
            "mode_likelihoods": dict(item.get("mode_likelihoods") or {}),
            "localization_score": float(item.get("localization_score") or 0.0),
            "localization_margin": float(item.get("localization_margin") or 0.0),
            "descriptor_index": index,
            "previous_state_index": index - 1 if index else None,
            "next_state_index": index + 1 if index + 1 < len(rows) else None,
        }
        states.append(state)
        mode_indexes.setdefault(mode_id, []).append(index)

    transitions = []
    for index in range(1, len(states)):
        source_mode = states[index - 1]["mode_id"]
        target_mode = states[index]["mode_id"]
        if source_mode == target_mode:
            continue
        left = max(0, index - 1)
        right = min(len(states) - 1, index + 1)
        transitions.append(
            {
                "transition_index": len(transitions),
                "source_mode_id": source_mode,
                "target_mode_id": target_mode,
                "first_state_index": left,
                "center_state_index": index,
                "last_state_index": right,
                "start_route_distance_px": states[left]["route_distance_px"],
                "end_route_distance_px": states[right]["route_distance_px"],
                "position_semantics": "continuous_no_displacement",
                "runtime_policy": "hold_pose_then_switch_layer_and_reset_reference",
            }
        )

    output_path = Path(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise RuntimeError("Route tracking package directory is not empty")
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path / "route_descriptors.npz", descriptors=descriptors)
    _write_jsonl(output_path / "route_states.jsonl", states)
    _write_jsonl(output_path / "map_transitions.jsonl", transitions)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "package_type": "route_tracking",
        "route_id": str(route_id),
        "atlas_id": str(atlas_id),
        "coordinate_space_id": str(coordinate_space_id),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "state_model": "directed_route_progress_with_lateral_and_heading_error",
        "position_semantics": "continuous_except_explicit_behavior_events",
        "corridor_radius_px": float(corridor_radius_px),
        "route_length_px": float(cumulative[-1]),
        "state_count": len(states),
        "mode_state_indexes": mode_indexes,
        "motion_envelope": {
            "step_distance_px": _percentiles(step_distances),
            "speed_px_s": _percentiles(speeds),
            "turn_rate_deg_s": _percentiles(turn_rates),
        },
        "transition_count": len(transitions),
        "files": {
            "states": "route_states.jsonl",
            "descriptors": "route_descriptors.npz",
            "transitions": "map_transitions.jsonl",
        },
    }
    temporary = output_path / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(output_path / "manifest.json"))
    return manifest


class RouteTrackingPackage:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.manifest = json.loads(
            (self.path / "manifest.json").read_text(encoding="utf-8")
        )
        self.states = self._read_jsonl(self.manifest["files"]["states"])
        self.transitions = self._read_jsonl(
            self.manifest["files"]["transitions"]
        )
        with np.load(str(self.path / self.manifest["files"]["descriptors"])) as archive:
            self.descriptors = archive["descriptors"].astype(np.float32)
        if len(self.states) != len(self.descriptors):
            raise RuntimeError("Route state and descriptor counts disagree")

    def _read_jsonl(self, filename: str):
        values = []
        with (self.path / filename).open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    values.append(json.loads(line))
        return values

    def candidates(
        self,
        descriptor,
        *,
        previous_state_index: Optional[int] = None,
        backward_states: int = 2,
        forward_states: int = 12,
        mode_id: Optional[str] = None,
        top_k: int = 5,
    ):
        query = np.asarray(descriptor, dtype=np.float32).reshape(-1)
        query /= max(float(np.linalg.norm(query)), 1.0e-6)
        if query.shape[0] != self.descriptors.shape[1]:
            raise ValueError("Route query descriptor has the wrong length")
        if previous_state_index is None:
            indexes = np.arange(len(self.states), dtype=np.int32)
        else:
            left = max(0, int(previous_state_index) - int(backward_states))
            right = min(
                len(self.states), int(previous_state_index) + int(forward_states) + 1
            )
            indexes = np.arange(left, right, dtype=np.int32)
        if mode_id is not None:
            indexes = np.asarray(
                [index for index in indexes if self.states[index]["mode_id"] == mode_id],
                dtype=np.int32,
            )
        if not len(indexes):
            return []
        scores = self.descriptors[indexes].dot(query)
        order = np.argsort(scores)[::-1][: max(1, int(top_k))]
        return [
            {
                "state_index": int(indexes[offset]),
                "score": float(scores[offset]),
                "state": dict(self.states[int(indexes[offset])]),
            }
            for offset in order
        ]
