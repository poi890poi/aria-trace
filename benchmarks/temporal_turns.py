"""Causal temporal-response diagnostics from independent turn signals.

The helpers deliberately keep control input, observed scene motion, and an
algorithm output as separate channels.  They estimate sign and time alignment
before judging response direction; equal frame indexes are never assumed.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np


def _summary(values) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"count": 0, "median": None, "p95": None, "maximum": None}
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def _signal(rows: Iterable[Mapping]) -> tuple[np.ndarray, np.ndarray]:
    retained = []
    for row in rows:
        if row.get("valid") is False:
            continue
        timestamp = row.get("session_time_ns")
        value = row.get("value")
        if timestamp is None or value is None:
            continue
        if not math.isfinite(float(value)):
            continue
        retained.append((int(timestamp) / 1.0e9, float(value)))
    retained.sort()
    if len(retained) < 3:
        raise ValueError("Turn evidence needs at least three finite samples")
    # Duplicate timestamps can occur when an input burst is aggregated.  Their
    # median is deterministic and prevents undefined interpolation behavior.
    grouped = {}
    for timestamp, value in retained:
        grouped.setdefault(timestamp, []).append(value)
    times = np.asarray(sorted(grouped), dtype=np.float64)
    values = np.asarray([np.median(grouped[item]) for item in times], np.float64)
    return times, values


def _standardize(values: np.ndarray) -> np.ndarray:
    centered = values - np.median(values)
    scale = float(np.percentile(np.abs(centered), 90))
    if scale <= 1.0e-9:
        raise ValueError("Turn evidence has no useful variation")
    return centered / scale


def align_turn_signals(
    evidence_rows: Iterable[Mapping],
    observed_rows: Iterable[Mapping],
    *,
    maximum_lag_ms: float = 750.0,
    sample_hz: float = 30.0,
) -> dict:
    """Fit observed sign and lag against independent turn evidence.

    Positive lag means the observed channel responds after the evidence.  Both
    signs are tested explicitly because screen, camera, and input conventions
    may be opposite.
    """

    evidence_t, evidence_v = _signal(evidence_rows)
    observed_t, observed_v = _signal(observed_rows)
    start = max(float(evidence_t[0]), float(observed_t[0]))
    end = min(float(evidence_t[-1]), float(observed_t[-1]))
    interval = 1.0 / float(sample_hz)
    grid = np.arange(start, end + interval * 0.25, interval)
    if len(grid) < 10:
        raise ValueError("Turn signals have insufficient overlapping duration")
    evidence = _standardize(np.interp(grid, evidence_t, evidence_v))
    observed = _standardize(np.interp(grid, observed_t, observed_v))
    maximum_steps = max(0, int(round(maximum_lag_ms / 1000.0 * sample_hz)))
    candidates = []
    for lag_steps in range(-maximum_steps, maximum_steps + 1):
        if lag_steps < 0:
            left, right = evidence[-lag_steps:], observed[:lag_steps]
        elif lag_steps > 0:
            left, right = evidence[:-lag_steps], observed[lag_steps:]
        else:
            left, right = evidence, observed
        if len(left) < 10:
            continue
        correlation = float(np.corrcoef(left, right)[0, 1])
        if math.isfinite(correlation):
            candidates.append((abs(correlation), correlation, lag_steps))
    if not candidates:
        raise ValueError("Turn signals cannot be correlated")
    _, correlation, lag_steps = max(candidates)
    lag_ms = lag_steps / float(sample_hz) * 1000.0
    sign = 1 if correlation >= 0 else -1
    return {
        "sign": sign,
        "sign_relation": "same" if sign == 1 else "opposite",
        "lag_ms": float(lag_ms),
        "correlation": float(abs(correlation)),
        "signed_correlation": float(correlation),
        "sample_hz": float(sample_hz),
        "maximum_lag_ms": float(maximum_lag_ms),
        "overlap_sample_count": int(len(grid) - abs(lag_steps)),
    }


def find_sharp_reversals(
    evidence_rows: Iterable[Mapping],
    *,
    minimum_magnitude: float = 0.35,
    quiet_s: float = 0.10,
    window_s: float = 0.25,
) -> list[dict]:
    """Find sign reversals using robust local medians, without lap boundaries."""

    times, raw = _signal(evidence_rows)
    values = _standardize(raw)
    output = []
    for index in range(1, len(times) - 1):
        before = (times >= times[index] - window_s) & (times < times[index] - quiet_s)
        after = (times > times[index] + quiet_s) & (times <= times[index] + window_s)
        if not np.any(before) or not np.any(after):
            continue
        old = float(np.median(values[before]))
        new = float(np.median(values[after]))
        if abs(old) < minimum_magnitude or abs(new) < minimum_magnitude:
            continue
        if np.sign(old) == np.sign(new):
            continue
        if output and times[index] - output[-1]["session_time_ns"] / 1.0e9 < window_s:
            continue
        output.append(
            {
                "session_time_ns": int(round(times[index] * 1.0e9)),
                "before_sign": int(np.sign(old)),
                "after_sign": int(np.sign(new)),
                "sharpness": float(min(abs(old), abs(new))),
            }
        )
    return output


def evaluate_reversal_response(
    evidence_rows: Iterable[Mapping],
    observed_rows: Iterable[Mapping],
    *,
    maximum_lag_ms: float = 750.0,
    response_threshold: float = 0.20,
    search_after_ms: float = 1000.0,
    stable_window_ms: float = 200.0,
    alignment_sample_hz: float = 30.0,
    reversal_minimum_magnitude: float = 0.35,
    reversal_quiet_s: float = 0.10,
    reversal_window_s: float = 0.25,
) -> dict:
    """Measure response lag and wrong-direction persistence at sharp turns."""

    evidence_rows = list(evidence_rows)
    observed_rows = list(observed_rows)
    alignment = align_turn_signals(
        evidence_rows,
        observed_rows,
        maximum_lag_ms=maximum_lag_ms,
        sample_hz=alignment_sample_hz,
    )
    events = find_sharp_reversals(
        evidence_rows,
        minimum_magnitude=reversal_minimum_magnitude,
        quiet_s=reversal_quiet_s,
        window_s=reversal_window_s,
    )
    observed_t, observed_raw = _signal(observed_rows)
    observed = _standardize(observed_raw) * alignment["sign"]
    onset_lags = []
    settling_times = []
    wrong_durations = []
    peak_ratios = []
    normalized_overshoots = []
    event_rows = []
    for event in events:
        event_s = event["session_time_ns"] / 1.0e9
        end_s = event_s + search_after_ms / 1000.0
        indexes = np.flatnonzero((observed_t >= event_s) & (observed_t <= end_s))
        expected_sign = int(event["after_sign"])
        onset = None
        for index in indexes:
            if np.sign(observed[index]) == expected_sign and abs(observed[index]) >= response_threshold:
                onset = float(observed_t[index])
                break
        if onset is None:
            event_rows.append({**event, "status": "no_response_in_window"})
            continue
        onset_ms = max(0.0, (onset - event_s) * 1000.0)
        onset_lags.append(onset_ms)
        settled = None
        stable_s = stable_window_ms / 1000.0
        for index in indexes:
            start_time = float(observed_t[index])
            if start_time < onset:
                continue
            stable = (observed_t >= start_time) & (observed_t <= start_time + stable_s)
            selected = observed[stable]
            if len(selected) < 2:
                continue
            correct = (np.sign(selected) == expected_sign) & (
                np.abs(selected) >= response_threshold
            )
            if float(np.mean(correct)) >= 0.8:
                settled = start_time
                break
        if settled is not None:
            settling_times.append(max(0.0, (settled - event_s) * 1000.0))
        wrong_duration_ms = 0.0
        prior_time = event_s
        for index in indexes:
            current = min(float(observed_t[index]), onset)
            if np.sign(observed[index]) == -expected_sign and abs(observed[index]) >= response_threshold:
                wrong_duration_ms += max(0.0, current - prior_time) * 1000.0
            prior_time = current
            if current >= onset:
                break
        wrong_durations.append(wrong_duration_ms)
        evidence_t, evidence_raw = _signal(evidence_rows)
        evidence_normalized = _standardize(evidence_raw)
        evidence_window = (
            (evidence_t > event_s) & (evidence_t <= end_s)
        )
        observed_window = (
            (observed_t > event_s) & (observed_t <= end_s)
        )
        reference_peak = float(
            np.max(np.abs(evidence_normalized[evidence_window]))
        ) if np.any(evidence_window) else 0.0
        response_peak = float(
            np.max(np.abs(observed[observed_window]))
        ) if np.any(observed_window) else 0.0
        peak_ratio = response_peak / reference_peak if reference_peak > 1.0e-9 else None
        if peak_ratio is not None:
            peak_ratios.append(peak_ratio)
            normalized_overshoots.append(max(0.0, peak_ratio - 1.0))
        event_rows.append(
            {
                **event,
                "status": "responded",
                "onset_lag_ms": onset_ms,
                "wrong_direction_duration_ms": wrong_duration_ms,
                "settling_time_ms": (
                    max(0.0, (settled - event_s) * 1000.0)
                    if settled is not None
                    else None
                ),
                "normalized_peak_ratio": peak_ratio,
            }
        )
    return {
        "parameters": {
            "maximum_lag_ms": float(maximum_lag_ms),
            "response_threshold": float(response_threshold),
            "search_after_ms": float(search_after_ms),
            "stable_window_ms": float(stable_window_ms),
            "alignment_sample_hz": float(alignment_sample_hz),
            "reversal_minimum_magnitude": float(reversal_minimum_magnitude),
            "reversal_quiet_s": float(reversal_quiet_s),
            "reversal_window_s": float(reversal_window_s),
        },
        "alignment": alignment,
        "sharp_reversal_count": len(events),
        "responded_reversal_count": len(onset_lags),
        "onset_lag_ms": _summary(onset_lags),
        "settling_time_ms": _summary(settling_times),
        "wrong_direction_duration_ms": _summary(wrong_durations),
        "normalized_peak_ratio": _summary(peak_ratios),
        "normalized_overshoot": _summary(normalized_overshoots),
        "events": event_rows,
        "interpretation": (
            "Input is intent and KLT is observed scene motion; neither is player-heading "
            "truth. A wrong-direction defect requires corroborating evidence after lag fit."
        ),
    }
