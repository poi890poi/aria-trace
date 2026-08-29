"""Learn and apply continuous mini-map representation transitions.

This module deliberately consumes mode likelihoods rather than game-specific
pixels.  Map-layer localizers own those likelihoods; transition learning owns
their temporal interpretation and the continuity invariant.
"""

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class ModeObservation:
    frame_index: int
    session_time_ns: int
    likelihoods: Mapping[str, float]
    canonical_xy: Optional[Tuple[float, float]] = None

    def score(self, mode_id: str) -> float:
        return float(self.likelihoods.get(mode_id, 0.0))


def _stable_run(
    observations: Sequence[ModeObservation],
    mode_id: str,
    competing_mode_id: str,
    count: int,
    margin: float,
    reverse: bool = False,
):
    indexes = range(len(observations) - count, -1, -1) if reverse else range(
        0, len(observations) - count + 1
    )
    for start in indexes:
        rows = observations[start : start + count]
        if all(
            row.score(mode_id) - row.score(competing_mode_id) >= margin
            for row in rows
        ):
            return start, start + count - 1
    return None


def learn_transition_model(
    observations: Iterable[ModeObservation],
    source_mode_id: str,
    target_mode_id: str,
    *,
    stable_count: int = 3,
    stable_margin: float = 0.08,
) -> dict:
    """Learn one directed transition from temporally ordered mode evidence."""

    rows = tuple(observations)
    if source_mode_id == target_mode_id:
        raise ValueError("Transition source and target modes must differ")
    if stable_count < 2:
        raise ValueError("A stable transition endpoint needs at least 2 samples")
    if len(rows) < stable_count * 2:
        raise ValueError("Transition evidence is too short for stable endpoints")
    if any(
        later.session_time_ns <= earlier.session_time_ns
        for earlier, later in zip(rows, rows[1:])
    ):
        raise ValueError("Transition observations must have increasing timestamps")

    source_run = _stable_run(
        rows, source_mode_id, target_mode_id, stable_count, stable_margin
    )
    target_run = _stable_run(
        rows,
        target_mode_id,
        source_mode_id,
        stable_count,
        stable_margin,
        reverse=True,
    )
    if source_run is None:
        raise ValueError("No stable source-mode evidence was found")
    if target_run is None:
        raise ValueError("No stable target-mode evidence was found")
    if source_run[1] >= target_run[0]:
        raise ValueError("Stable source evidence does not precede stable target evidence")

    source_end = source_run[1]
    target_start = target_run[0]
    transition_rows = rows[source_end : target_start + 1]
    signed = np.asarray(
        [row.score(target_mode_id) - row.score(source_mode_id) for row in rows],
        dtype=np.float64,
    )
    crossings = [
        index
        for index in range(source_end + 1, target_start + 1)
        if signed[index - 1] <= 0.0 < signed[index]
    ]
    center_index = (
        crossings[0]
        if crossings
        else min(
            range(source_end, target_start + 1),
            key=lambda index: abs(float(signed[index])),
        )
    )
    source_margins = [
        row.score(source_mode_id) - row.score(target_mode_id)
        for row in rows[source_run[0] : source_run[1] + 1]
    ]
    target_margins = [
        row.score(target_mode_id) - row.score(source_mode_id)
        for row in rows[target_run[0] : target_run[1] + 1]
    ]
    endpoint_margin = min(float(np.median(source_margins)), float(np.median(target_margins)))
    monotonic_fraction = float(
        np.mean(np.diff(signed[source_end : target_start + 1]) >= -0.05)
    ) if target_start > source_end else 1.0
    confidence = float(
        np.clip(0.65 * endpoint_margin + 0.35 * monotonic_fraction, 0.0, 1.0)
    )
    positions = [row.canonical_xy for row in transition_rows if row.canonical_xy]
    boundary = None
    if positions:
        boundary = {
            "center_xy": np.mean(np.asarray(positions, dtype=np.float64), axis=0).tolist(),
            "radius_px": float(
                max(
                    np.linalg.norm(
                        np.asarray(point, dtype=np.float64)
                        - np.mean(np.asarray(positions, dtype=np.float64), axis=0)
                    )
                    for point in positions
                )
            ),
            "sample_count": len(positions),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "measurement": "minimap_representation_transition",
        "source_mode_id": source_mode_id,
        "target_mode_id": target_mode_id,
        "position_semantics": "continuous_no_displacement",
        "reference_policy": "hold_pose_then_reset_local_reference_on_target_lock",
        "source_stable": {
            "first_frame_index": rows[source_run[0]].frame_index,
            "last_frame_index": rows[source_run[1]].frame_index,
        },
        "transition": {
            "first_frame_index": rows[source_end].frame_index,
            "center_frame_index": rows[center_index].frame_index,
            "last_frame_index": rows[target_start].frame_index,
            "start_session_time_ns": rows[source_end].session_time_ns,
            "center_session_time_ns": rows[center_index].session_time_ns,
            "end_session_time_ns": rows[target_start].session_time_ns,
            "duration_ms": (
                rows[target_start].session_time_ns - rows[source_end].session_time_ns
            )
            / 1.0e6,
        },
        "target_stable": {
            "first_frame_index": rows[target_run[0]].frame_index,
            "last_frame_index": rows[target_run[1]].frame_index,
        },
        "canonical_boundary": boundary,
        "quality": {
            "confidence": confidence,
            "endpoint_margin": endpoint_margin,
            "monotonic_fraction": monotonic_fraction,
            "observation_count": len(rows),
        },
    }


class TransitionController:
    """Debounce a learned directed mode switch without changing position."""

    def __init__(self, model: Mapping, confirmation_count: int = 3) -> None:
        self.model = dict(model)
        self.source_mode_id = str(model["source_mode_id"])
        self.target_mode_id = str(model["target_mode_id"])
        self.confirmation_count = max(2, int(confirmation_count))
        self.active_mode_id = self.source_mode_id
        self._target_wins = 0

    def update(self, likelihoods: Mapping[str, float]) -> dict:
        source = float(likelihoods.get(self.source_mode_id, 0.0))
        target = float(likelihoods.get(self.target_mode_id, 0.0))
        if self.active_mode_id == self.source_mode_id and target > source:
            self._target_wins += 1
        else:
            self._target_wins = 0
        switched = self._target_wins >= self.confirmation_count
        if switched:
            self.active_mode_id = self.target_mode_id
            self._target_wins = 0
        return {
            "active_mode_id": self.active_mode_id,
            "state": "target_locked"
            if self.active_mode_id == self.target_mode_id
            else ("transitioning" if self._target_wins else "source_locked"),
            "switched": switched,
            "reset_local_reference": switched,
            "position_delta_xy": [0.0, 0.0],
            "target_confirmation_count": self._target_wins,
        }
