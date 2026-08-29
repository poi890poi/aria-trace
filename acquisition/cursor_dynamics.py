"""Temporal cursor-motion measurements produced by standard calibration."""

import math
from typing import Mapping, Sequence

import numpy as np

from .cursor_pose import circular_difference_degrees


def _percentile(values: np.ndarray, percentile: float) -> float:
    return float(np.percentile(values, percentile)) if len(values) else 0.0


def _source_summary(poses: Sequence[dict], confidence_min: float) -> dict:
    transitions = []
    rates = []
    accelerations = []
    previous_rate = None
    previous_transition_end = None
    detected = sum(bool(item.get("detected")) for item in poses)
    accepted = sum(
        bool(item.get("detected"))
        and float(item.get("confidence") or 0.0) >= confidence_min
        for item in poses
    )
    for first, second in zip(poses, poses[1:]):
        first_ok = bool(first.get("detected")) and float(
            first.get("confidence") or 0.0
        ) >= confidence_min
        second_ok = bool(second.get("detected")) and float(
            second.get("confidence") or 0.0
        ) >= confidence_min
        if not (first_ok and second_ok):
            previous_rate = None
            previous_transition_end = None
            continue
        first_time = first.get("session_time_ns")
        second_time = second.get("session_time_ns")
        if first_time is None or second_time is None:
            continue
        elapsed_s = (int(second_time) - int(first_time)) / 1.0e9
        if not 0.005 <= elapsed_s <= 0.5:
            previous_rate = None
            previous_transition_end = None
            continue
        delta = float(
            circular_difference_degrees(
                float(second["angle_screen_deg"]),
                float(first["angle_screen_deg"]),
            )
        )
        rate = delta / elapsed_s
        transitions.append((elapsed_s, delta))
        rates.append(rate)
        if previous_rate is not None and previous_transition_end == first_time:
            accelerations.append((rate - previous_rate) / elapsed_s)
        previous_rate = rate
        previous_transition_end = second_time

    elapsed = np.asarray([item[0] for item in transitions], dtype=np.float64)
    jumps = np.abs(np.asarray([item[1] for item in transitions], dtype=np.float64))
    absolute_rates = np.abs(np.asarray(rates, dtype=np.float64))
    absolute_acceleration = np.abs(np.asarray(accelerations, dtype=np.float64))
    return {
        "frame_count": int(len(poses)),
        "detected_frame_count": int(detected),
        "accepted_frame_count": int(accepted),
        "accepted_frame_rate": float(accepted / max(1, len(poses))),
        "valid_transition_count": int(len(transitions)),
        "transition_coverage": float(len(transitions) / max(1, len(poses) - 1)),
        "median_sample_interval_ms": _percentile(elapsed * 1000.0, 50),
        "heading_jump_abs_deg": {
            "p50": _percentile(jumps, 50),
            "p95": _percentile(jumps, 95),
            "p99": _percentile(jumps, 99),
            "max": float(jumps.max()) if len(jumps) else 0.0,
        },
        "turn_rate_abs_deg_s": {
            "p50": _percentile(absolute_rates, 50),
            "p95": _percentile(absolute_rates, 95),
            "p99": _percentile(absolute_rates, 99),
            "max": float(absolute_rates.max()) if len(absolute_rates) else 0.0,
        },
        "angular_acceleration_abs_deg_s2": {
            "p50": _percentile(absolute_acceleration, 50),
            "p95": _percentile(absolute_acceleration, 95),
            "p99": _percentile(absolute_acceleration, 99),
        },
    }


def summarize_cursor_dynamics(
    pose_sequences: Mapping[str, Sequence[dict]],
    source_provenance: Mapping[str, dict] = None,
    confidence_min: float = 0.45,
) -> dict:
    """Summarize ordinary and stress motion without interpreting teleports."""
    if "ordinary_cruise" not in pose_sequences:
        raise ValueError("ordinary_cruise pose evidence is required")
    if "movement_only" not in pose_sequences:
        raise ValueError("movement_only pose evidence is required")
    sources = {
        role: _source_summary(list(poses), float(confidence_min))
        for role, poses in pose_sequences.items()
    }
    for role, provenance in (source_provenance or {}).items():
        if role in sources:
            sources[role]["provenance"] = dict(provenance)
    ordinary = sources["ordinary_cruise"]
    stress = sources["movement_only"]
    ordinary_rate = ordinary["turn_rate_abs_deg_s"]
    stress_rate = stress["turn_rate_abs_deg_s"]
    acceleration = stress["angular_acceleration_abs_deg_s2"]
    coverage = min(
        ordinary["transition_coverage"], stress["transition_coverage"]
    )
    confidence = float(
        np.clip(
            math.sqrt(
                ordinary["accepted_frame_rate"] * stress["accepted_frame_rate"]
            )
            * min(1.0, coverage / 0.80),
            0.0,
            1.0,
        )
    )
    return {
        "schema_version": "1.0",
        "measurement": "cursor_temporal_dynamics",
        "angle_convention": "screen degrees: 0=right, +clockwise",
        "confidence_threshold": float(confidence_min),
        "sources": sources,
        "recommended_runtime_envelope": {
            "normal_turn_rate_p95_deg_s": ordinary_rate["p95"],
            "normal_turn_rate_p99_deg_s": ordinary_rate["p99"],
            "calibrated_turn_rate_p99_deg_s": max(
                ordinary_rate["p99"], stress_rate["p99"]
            ),
            "calibrated_angular_acceleration_p99_deg_s2": acceleration["p99"],
            "ordinary_heading_jump_p99_deg": ordinary["heading_jump_abs_deg"][
                "p99"
            ],
            "stress_heading_jump_p99_deg": stress["heading_jump_abs_deg"]["p99"],
        },
        "confidence": confidence,
        "confidence_scale": "0..1 evidence coverage and accepted-pose rate",
        "runtime_scope": "temporal search bounds only; no teleport detection",
    }
