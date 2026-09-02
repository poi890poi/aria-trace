"""Causal fallback strategies and explicit end-to-end metric semantics.

The strategies in this module never inspect a future frame or reference label.
They receive a chronological stream of primary measurements.  A rejected
measurement may yield a held/predicted state, but that output is never relabeled
as a fresh accepted measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np


def circular_delta_deg(value: float, reference: float) -> float:
    return float((float(value) - float(reference) + 180.0) % 360.0 - 180.0)


@dataclass(frozen=True)
class Measurement:
    frame_index: int
    session_time_ns: int
    angle_deg: Optional[float]
    confidence: float
    accepted: bool
    rejection_reason: Optional[str]
    latency_ns: int


@dataclass(frozen=True)
class PublishedState:
    frame_index: int
    session_time_ns: int
    angle_deg: Optional[float]
    provenance: str
    primary_accepted: bool
    fallback_invoked: bool
    fallback_produced_output: bool
    state_age_ms: Optional[float]


class _FallbackStrategy:
    strategy_id = "base"

    def __init__(self) -> None:
        self.accepted_history: List[Measurement] = []
        self.previous_output: Optional[PublishedState] = None

    def _fallback_angle(self, measurement: Measurement) -> Optional[float]:
        raise NotImplementedError

    def _fallback_provenance(self) -> str:
        return "predicted"

    def update(self, measurement: Measurement) -> PublishedState:
        if measurement.accepted and measurement.angle_deg is not None:
            self.accepted_history.append(measurement)
            state = PublishedState(
                frame_index=measurement.frame_index,
                session_time_ns=measurement.session_time_ns,
                angle_deg=float(measurement.angle_deg) % 360.0,
                provenance="fresh_measurement",
                primary_accepted=True,
                fallback_invoked=False,
                fallback_produced_output=False,
                state_age_ms=0.0,
            )
            self.previous_output = state
            return state

        angle = self._fallback_angle(measurement)
        if angle is None:
            state = PublishedState(
                frame_index=measurement.frame_index,
                session_time_ns=measurement.session_time_ns,
                angle_deg=None,
                provenance="unavailable",
                primary_accepted=False,
                fallback_invoked=True,
                fallback_produced_output=False,
                state_age_ms=None,
            )
            self.previous_output = state
            return state

        last_fresh = self.accepted_history[-1]
        state = PublishedState(
            frame_index=measurement.frame_index,
            session_time_ns=measurement.session_time_ns,
            angle_deg=float(angle) % 360.0,
            provenance=self._fallback_provenance(),
            primary_accepted=False,
            fallback_invoked=True,
            fallback_produced_output=True,
            state_age_ms=max(
                0.0,
                (measurement.session_time_ns - last_fresh.session_time_ns) / 1.0e6,
            ),
        )
        self.previous_output = state
        return state


class NoFallback(_FallbackStrategy):
    strategy_id = "no_fallback"

    def _fallback_angle(self, measurement: Measurement) -> Optional[float]:
        return None


class ReusePreviousState(_FallbackStrategy):
    strategy_id = "reuse_previous_state"

    def _fallback_angle(self, measurement: Measurement) -> Optional[float]:
        if self.previous_output is None:
            return None
        return self.previous_output.angle_deg

    def _fallback_provenance(self) -> str:
        return "held"


class ConstantVelocity(_FallbackStrategy):
    strategy_id = "constant_velocity_last_2_accepted"

    def _fallback_angle(self, measurement: Measurement) -> Optional[float]:
        if len(self.accepted_history) < 2:
            return None
        first, last = self.accepted_history[-2:]
        elapsed_s = (last.session_time_ns - first.session_time_ns) / 1.0e9
        if elapsed_s <= 0.0:
            return float(last.angle_deg)
        velocity = circular_delta_deg(last.angle_deg, first.angle_deg) / elapsed_s
        horizon_s = (measurement.session_time_ns - last.session_time_ns) / 1.0e9
        return float(last.angle_deg + velocity * max(0.0, horizon_s))


FALLBACK_FACTORIES: Dict[str, Callable[[], _FallbackStrategy]] = {
    "no_fallback": NoFallback,
    "reuse_previous_state": ReusePreviousState,
    "constant_velocity_last_2_accepted": ConstantVelocity,
}


def apply_fallback_strategy(
    measurements: Iterable[Measurement], strategy_id: str
) -> List[PublishedState]:
    if strategy_id not in FALLBACK_FACTORIES:
        raise ValueError("unknown fallback strategy: {}".format(strategy_id))
    strategy = FALLBACK_FACTORIES[strategy_id]()
    return [strategy.update(measurement) for measurement in measurements]


def _distribution(values: Sequence[float], *, include_worst: bool) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        result = {"sample_count": 0, "mean": None, "median": None, "p95": None}
        if include_worst:
            result["worst"] = None
        return result
    result = {
        "sample_count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
    }
    if include_worst:
        result["worst"] = float(np.max(array))
    return result


def _loss_episodes(available: Sequence[bool]) -> dict:
    episodes = []
    active = 0
    for value in available:
        if value:
            if active:
                episodes.append(active)
                active = 0
        else:
            active += 1
    if active:
        episodes.append(active)
    return {
        "unavailable_episode_count": int(len(episodes)),
        "unavailable_frame_count": int(sum(episodes)),
        "longest_unavailable_episode_frames": int(max(episodes, default=0)),
    }


def summarize_e2e_rows(rows: Sequence[dict]) -> dict:
    """Summarize chronological rows with non-overloaded rate names.

    Rates use all primary attempts as their denominator.  Accuracy uses only
    rows with an independent reference and a published output.  The maximum
    error is deliberately named ``worst`` and appears only in the complete E2E
    accuracy distribution, never in fresh-only or fallback-only diagnostics.
    """
    attempted = len(rows)
    accepted = sum(bool(row["primary_accepted"]) for row in rows)
    rejected = attempted - accepted
    fallback_invoked = sum(bool(row["fallback_invoked"]) for row in rows)
    fallback_output = sum(bool(row["fallback_produced_output"]) for row in rows)
    provenance_counts = {
        name: sum(row["provenance"] == name for row in rows)
        for name in (
            "fresh_measurement",
            "held",
            "predicted",
            "unavailable",
        )
    }
    candidates = sum(bool(row.get("primary_candidate_produced")) for row in rows)
    confidence_passed = sum(bool(row.get("confidence_gate_passed")) for row in rows)
    confidence_rejected = sum(
        row.get("final_rejection_reason") == "confidence_below_threshold"
        for row in rows
    )
    temporal_rejected = sum(
        row.get("final_rejection_reason")
        == "large_innovation_without_high_confidence"
        for row in rows
    )
    evaluable = [
        row
        for row in rows
        if row.get("reference_angle_deg") is not None
        and row.get("output_angle_deg") is not None
    ]
    errors = [float(row["absolute_error_deg"]) for row in evaluable]
    fresh_errors = [
        float(row["absolute_error_deg"])
        for row in evaluable
        if row["provenance"] == "fresh_measurement"
    ]
    fallback_errors = [
        float(row["absolute_error_deg"])
        for row in evaluable
        if row["provenance"] in ("held", "filtered", "predicted")
    ]
    latency_ms = [float(row["e2e_latency_ns"]) / 1.0e6 for row in rows]
    fallback_overhead_ms = [
        float(row["fallback_strategy_latency_ns"]) / 1.0e6 for row in rows
    ]
    worst_case = None
    if evaluable:
        worst = max(evaluable, key=lambda row: float(row["absolute_error_deg"]))
        worst_case = {
            key: worst.get(key)
            for key in (
                "session",
                "frame_index",
                "session_time_ns",
                "provenance",
                "output_angle_deg",
                "reference_angle_deg",
                "absolute_error_deg",
            )
        }
    denominator = max(attempted, 1)
    return {
        "rate_denominator": "all chronological primary measurement attempts",
        "primary_measurement_attempt_count": int(attempted),
        "primary_candidate_produced_rate": float(candidates / denominator),
        "confidence_gate_pass_rate": float(confidence_passed / denominator),
        "confidence_gate_rejected_rate": float(confidence_rejected / denominator),
        "temporal_outlier_gate_rejected_rate": float(
            temporal_rejected / denominator
        ),
        "primary_measurement_accepted_rate": float(accepted / denominator),
        "primary_measurement_rejected_rate": float(rejected / denominator),
        "fallback_invocation_rate": float(fallback_invoked / denominator),
        "fallback_output_success_rate": (
            float(fallback_output / fallback_invoked) if fallback_invoked else None
        ),
        "final_output_available_rate": float(
            (attempted - provenance_counts["unavailable"]) / denominator
        ),
        "output_provenance_rate": {
            key: float(value / denominator) for key, value in provenance_counts.items()
        },
        "e2e_absolute_error_deg": _distribution(errors, include_worst=True),
        "fresh_measurement_error_deg": _distribution(
            fresh_errors, include_worst=False
        ),
        "fallback_output_error_deg": _distribution(
            fallback_errors, include_worst=False
        ),
        "e2e_latency_ms": _distribution(latency_ms, include_worst=True),
        "fallback_strategy_overhead_ms": _distribution(
            fallback_overhead_ms, include_worst=False
        ),
        "worst_e2e_case": worst_case,
        "continuity": _loss_episodes(
            [row["provenance"] != "unavailable" for row in rows]
        ),
    }
