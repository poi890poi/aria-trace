"""Alternating-signal control-to-perception latency measurement."""

from bisect import bisect_right
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .contracts import ControlEvent, SignalObservation


def _clock_parameters(value: Optional[Mapping[str, Any]]) -> Tuple[float, float, float]:
    value = value or {}
    scale = float(value.get("scale", 1.0))
    offset_ns = float(value.get("offset_ns", 0.0))
    uncertainty_ns = float(value.get("uncertainty_ns", 0.0))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Clock-transform scale must be finite and positive")
    if not np.isfinite(offset_ns) or not np.isfinite(uncertainty_ns):
        raise ValueError("Clock-transform values must be finite")
    if uncertainty_ns < 0:
        raise ValueError("Clock-transform uncertainty must be non-negative")
    return scale, offset_ns, uncertainty_ns


def _mapped_time(time_ns: int, scale: float, offset_ns: float) -> int:
    return int(round(float(time_ns) * scale + offset_ns))


def _dominant_state(observation: SignalObservation) -> Tuple[str, float, float]:
    ordered = sorted(
        observation.probabilities.items(), key=lambda item: item[1], reverse=True
    )
    best_state, best_probability = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
    return str(best_state), float(best_probability), float(best_probability - runner_up)


def _global_correlation_lag(
    mapped_events: Sequence[Tuple[int, ControlEvent]],
    observations: Sequence[SignalObservation],
    states: Sequence[str],
    maximum_lag_ns: int,
) -> Dict[str, Any]:
    if len(states) != 2 or len(mapped_events) < 3 or len(observations) < 3:
        return {"lag_ns": None, "correlation": None}
    observation_times = np.asarray([item.time_ns for item in observations], dtype=np.int64)
    intervals = np.diff(observation_times)
    step_ns = int(max(1, round(float(np.median(intervals))))) if len(intervals) else 1
    event_times = [item[0] for item in mapped_events]
    state_sign = {states[0]: -1.0, states[1]: 1.0}
    observed_values = np.asarray(
        [
            float(item.probabilities.get(states[1], 0.0))
            - float(item.probabilities.get(states[0], 0.0))
            for item in observations
        ],
        dtype=np.float64,
    )
    best_lag = None
    best_correlation = -np.inf
    for lag_ns in range(0, int(maximum_lag_ns) + 1, step_ns):
        expected, measured = [], []
        for observation, observed_value in zip(observations, observed_values):
            event_index = bisect_right(event_times, observation.time_ns - lag_ns) - 1
            if event_index < 0:
                continue
            expected.append(state_sign[mapped_events[event_index][1].state])
            measured.append(observed_value)
        if len(expected) < 3 or np.std(expected) < 1.0e-9 or np.std(measured) < 1.0e-9:
            continue
        correlation = float(np.corrcoef(expected, measured)[0, 1])
        if correlation > best_correlation:
            best_correlation = correlation
            best_lag = lag_ns
    return {
        "lag_ns": int(best_lag) if best_lag is not None else None,
        "correlation": (
            float(best_correlation) if np.isfinite(best_correlation) else None
        ),
        "step_ns": step_ns,
    }


def estimate_latency(
    events: Iterable[ControlEvent],
    observations: Iterable[SignalObservation],
    clock_transform: Optional[Mapping[str, Any]] = None,
    probability_threshold: float = 0.80,
    ambiguity_margin: float = 0.15,
    stable_observations: int = 2,
    maximum_latency_ns: int = 500_000_000,
) -> Dict[str, Any]:
    """Measure first-stable observation latency for timestamped signal changes."""

    if not 0.0 < probability_threshold <= 1.0:
        raise ValueError("Probability threshold must be within (0, 1]")
    if not 0.0 <= ambiguity_margin <= 1.0:
        raise ValueError("Ambiguity margin must be within [0, 1]")
    if stable_observations < 1:
        raise ValueError("stable_observations must be positive")
    if maximum_latency_ns <= 0:
        raise ValueError("maximum_latency_ns must be positive")

    event_values = sorted(list(events), key=lambda item: item.time_ns)
    observation_values = sorted(list(observations), key=lambda item: item.time_ns)
    if len(event_values) < 2:
        raise ValueError("At least two alternating control events are required")
    states = {item.state for item in event_values}
    if len(states) != 2:
        raise ValueError("Alternating latency measurement requires exactly two states")
    if any(
        first.state == second.state
        for first, second in zip(event_values, event_values[1:])
    ):
        raise ValueError("Latency control events must alternate state")
    if not observation_values:
        raise ValueError("At least one signal observation is required")
    event_clocks = {item.clock_id for item in event_values}
    observation_clocks = {item.clock_id for item in observation_values}
    if len(event_clocks) != 1 or len(observation_clocks) != 1:
        raise ValueError("Each latency endpoint must use one declared clock")
    scale, offset_ns, clock_uncertainty_ns = _clock_parameters(clock_transform)
    if event_clocks != observation_clocks and clock_transform is None:
        raise ValueError("Different endpoint clocks require an explicit clock transform")

    mapped_events = [
        (_mapped_time(item.time_ns, scale, offset_ns), item) for item in event_values
    ]
    accepted: List[Dict[str, Any]] = []
    missed: List[str] = []
    ambiguous: List[str] = []
    for event_index, (issue_time, event) in enumerate(mapped_events):
        next_issue = (
            mapped_events[event_index + 1][0]
            if event_index + 1 < len(mapped_events)
            else issue_time + maximum_latency_ns + 1
        )
        window_end = min(next_issue, issue_time + maximum_latency_ns + 1)
        candidates = [
            observation
            for observation in observation_values
            if issue_time <= observation.time_ns < window_end
        ]
        stable_start = None
        for index in range(0, len(candidates) - stable_observations + 1):
            group = candidates[index : index + stable_observations]
            passes = True
            for observation in group:
                dominant, probability, margin = _dominant_state(observation)
                if (
                    dominant != event.state
                    or probability < probability_threshold
                    or margin < ambiguity_margin
                ):
                    passes = False
                    break
            if passes:
                stable_start = group[0]
                break
        if stable_start is not None:
            accepted.append(
                {
                    "token": event.token,
                    "state": event.state,
                    "control_time_ns": int(issue_time),
                    "observation_time_ns": int(stable_start.time_ns),
                    "latency_ns": int(stable_start.time_ns - issue_time),
                    "source_id": stable_start.source_id,
                    "probability": float(
                        stable_start.probabilities.get(event.state, 0.0)
                    ),
                }
            )
            continue
        saw_candidate_state = any(
            _dominant_state(observation)[0] == event.state for observation in candidates
        )
        if saw_candidate_state:
            ambiguous.append(event.token)
        else:
            missed.append(event.token)

    if not accepted:
        raise RuntimeError("No stable alternating-signal transitions were observed")
    latencies = np.asarray([item["latency_ns"] for item in accepted], dtype=np.float64)
    state_medians = {}
    for state in sorted({item["state"] for item in accepted}):
        values = [item["latency_ns"] for item in accepted if item["state"] == state]
        state_medians[state] = float(np.median(values))
    state_values = list(state_medians.values())
    rising_falling_bias = (
        float(max(state_values) - min(state_values)) if len(state_values) == 2 else None
    )
    observation_intervals = np.diff(
        np.asarray([item.time_ns for item in observation_values], dtype=np.int64)
    )
    frame_interval_ns = (
        float(np.median(observation_intervals)) if len(observation_intervals) else None
    )
    states = sorted(states)
    correlation = _global_correlation_lag(
        mapped_events, observation_values, states, maximum_latency_ns
    )
    accepted_ratio = len(accepted) / float(len(event_values))
    sample_quality = min(1.0, len(accepted) / 64.0)
    separability_quality = max(0.0, float(correlation.get("correlation") or 0.0))
    confidence = float(
        np.clip(accepted_ratio * (0.65 * sample_quality + 0.35 * separability_quality), 0.0, 1.0)
    )
    return {
        "endpoint": "first_stable",
        "event_clock": next(iter(event_clocks)),
        "observation_clock": next(iter(observation_clocks)),
        "clock_transform": {
            "scale": scale,
            "offset_ns": offset_ns,
            "uncertainty_ns": clock_uncertainty_ns,
        },
        "probability_threshold": probability_threshold,
        "ambiguity_margin": ambiguity_margin,
        "stable_observations": stable_observations,
        "issued_transitions": len(event_values),
        "accepted_transitions": len(accepted),
        "missed_transitions": len(missed),
        "ambiguous_transitions": len(ambiguous),
        "missed_tokens": missed,
        "ambiguous_tokens": ambiguous,
        "median_ns": float(np.median(latencies)),
        "p05_ns": float(np.percentile(latencies, 5)),
        "p95_ns": float(np.percentile(latencies, 95)),
        "maximum_ns": float(np.max(latencies)),
        "robust_jitter_ns": float(
            (np.percentile(latencies, 95) - np.percentile(latencies, 5)) / 2.0
        ),
        "state_median_ns": state_medians,
        "rising_falling_bias_ns": rising_falling_bias,
        "frame_interval_median_ns": frame_interval_ns,
        "timestamp_uncertainty_ns": float(
            clock_uncertainty_ns + (frame_interval_ns or 0.0) / 2.0
        ),
        "cross_correlation": correlation,
        "confidence": confidence,
        "transitions": accepted,
    }


def estimate_paired_delay(
    earlier_endpoint: Mapping[str, Any], later_endpoint: Mapping[str, Any]
) -> Dict[str, Any]:
    """Measure endpoint delay from transitions sharing the same control token."""

    earlier = {
        str(item["token"]): item for item in earlier_endpoint.get("transitions", [])
    }
    later = {str(item["token"]): item for item in later_endpoint.get("transitions", [])}
    tokens = sorted(set(earlier).intersection(later))
    if not tokens:
        raise ValueError("Latency endpoints have no paired transition tokens")
    delays = np.asarray(
        [
            int(later[token]["observation_time_ns"])
            - int(earlier[token]["observation_time_ns"])
            for token in tokens
        ],
        dtype=np.float64,
    )
    return {
        "paired_transition_count": len(tokens),
        "tokens": tokens,
        "median_ns": float(np.median(delays)),
        "p05_ns": float(np.percentile(delays, 5)),
        "p95_ns": float(np.percentile(delays, 95)),
        "minimum_ns": float(np.min(delays)),
        "maximum_ns": float(np.max(delays)),
        "source": "derived_from_paired_transitions",
    }
