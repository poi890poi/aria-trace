"""Run a causal, traceable cursor-pose publication benchmark.

This benchmark measures an entire cursor-pose publication chain:

    decoded frame -> primary pose measurement -> confidence decision
    -> optional causal fallback -> published pose

The independent motion reference is evaluator-only and is never visible to the
estimator or fallback.  Natural confidence rejections and deterministic outage
stress tests are reported separately so a dataset with no natural rejection
cannot make every fallback look equally good.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from rig_runtime.adapters.filesystem.session import SessionReader
from rig_runtime.services.calibration.cursor.pose import CursorPoseEstimator
from rig_runtime.services.calibration.minimap.verification import estimate_masked_shift
from aria_trace.services.tracking.profiles import resolve_tracking_profile
from benchmarks.cursor_pose.stateful import (
    FALLBACK_FACTORIES,
    Measurement,
    circular_delta_deg,
    summarize_e2e_rows,
)


SCHEMA_VERSION = "1.0"
DEFAULT_SESSIONS = ("run_03", "run_04", "run_12", "run_13")
DEFAULT_DEVELOPMENT_SESSIONS = frozenset(("run_03", "run_04"))
OUTAGE_SCENARIOS = {
    "natural": {"period_frames": None, "burst_frames": 0},
    "three_frame_burst_every_90": {"period_frames": 90, "burst_frames": 3},
}
SOLUTION_CONFIGS = {
    "realtime_confidence_hold": {
        "profile": "real-time",
        "fallback_strategy": "reuse_previous_state",
        "temporal_outlier_gate": False,
        "report_role": "current_solution",
    },
    "realtime_strict_hold": {
        "profile": "real-time",
        "fallback_strategy": "reuse_previous_state",
        "temporal_outlier_gate": True,
        "report_role": "experiment",
    },
    "realtime_confidence_predict": {
        "profile": "real-time",
        "fallback_strategy": "constant_velocity_last_2_accepted",
        "temporal_outlier_gate": False,
        "report_role": "experiment",
    },
    "realtime_confidence_reject": {
        "profile": "real-time",
        "fallback_strategy": "no_fallback",
        "temporal_outlier_gate": False,
        "report_role": "experiment",
    },
    "fast_confidence_hold": {
        "profile": "fast",
        "fallback_strategy": "reuse_previous_state",
        "temporal_outlier_gate": False,
        "report_role": "experiment",
    },
    "accurate_confidence_hold": {
        "profile": "accurate",
        "fallback_strategy": "reuse_previous_state",
        "temporal_outlier_gate": False,
        "report_role": "experiment",
    },
}
DEFAULT_SOLUTIONS = (
    "realtime_confidence_hold",
    "realtime_strict_hold",
    "realtime_confidence_predict",
    "realtime_confidence_reject",
    "fast_confidence_hold",
    "accurate_confidence_hold",
)


@dataclass
class CausalOutlierGate:
    """One final gate; rejected candidates never update temporal state."""

    confidence_min: float
    innovation_limit_deg: float
    large_innovation_confidence_min: float
    temporal_check_enabled: bool = True

    def __post_init__(self) -> None:
        self.accepted_history: List[Measurement] = []

    def decide(self, candidate: Measurement) -> Measurement:
        if candidate.angle_deg is None:
            return Measurement(
                candidate.frame_index,
                candidate.session_time_ns,
                candidate.angle_deg,
                candidate.confidence,
                False,
                "cursor_not_detected",
                candidate.latency_ns,
            )
        if candidate.confidence < self.confidence_min:
            return Measurement(
                candidate.frame_index,
                candidate.session_time_ns,
                candidate.angle_deg,
                candidate.confidence,
                False,
                "confidence_below_threshold",
                candidate.latency_ns,
            )
        if self.temporal_check_enabled and self.accepted_history:
            last = self.accepted_history[-1]
            innovation = abs(
                circular_delta_deg(candidate.angle_deg, last.angle_deg)
            )
            if (
                innovation > self.innovation_limit_deg
                and candidate.confidence < self.large_innovation_confidence_min
            ):
                return Measurement(
                    candidate.frame_index,
                    candidate.session_time_ns,
                    candidate.angle_deg,
                    candidate.confidence,
                    False,
                    "large_innovation_without_high_confidence",
                    candidate.latency_ns,
                )
        accepted = Measurement(
            candidate.frame_index,
            candidate.session_time_ns,
            candidate.angle_deg,
            candidate.confidence,
            True,
            None,
            candidate.latency_ns,
        )
        self.accepted_history.append(accepted)
        return accepted


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(value) -> dict:
    source_file = Path(inspect.getsourcefile(value)).resolve()
    source = inspect.getsource(value).encode("utf-8")
    return {
        "qualified_name": "{}.{}".format(value.__module__, value.__qualname__),
        "source_file": str(source_file),
        "source_file_sha256": _sha256(source_file),
        "implementation_sha256": hashlib.sha256(source).hexdigest(),
        "source_start_line": int(inspect.getsourcelines(value)[1]),
    }


def _git_output(root: Path, *arguments: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _repository_identity(root: Path, tested_files: Sequence[Path]) -> dict:
    relative = []
    for path in tested_files:
        try:
            relative.append(str(Path(path).resolve().relative_to(root.resolve())))
        except ValueError:
            relative.append(str(Path(path).resolve()))
    tested_status = _git_output(root, "status", "--porcelain", "--", *relative)
    whole_status = _git_output(root, "status", "--porcelain")
    return {
        "revision": _git_output(root, "rev-parse", "HEAD"),
        "branch": _git_output(root, "branch", "--show-current"),
        "repository_dirty": bool(whole_status),
        "tested_source_dirty": bool(tested_status),
        "tested_source_status": tested_status.splitlines() if tested_status else [],
    }


def _file_identity(path: Path, *, hash_content: bool = True) -> dict:
    path = Path(path).resolve()
    stat = path.stat()
    result = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "modified_time_ns": int(stat.st_mtime_ns),
    }
    if hash_content:
        result["sha256"] = _sha256(path)
    return result


def _decode_session(path: Path) -> Tuple[List[np.ndarray], List[dict], SessionReader]:
    reader = SessionReader(path)
    records = list(reader.frames_by_stream.get("main") or ())
    capture = cv2.VideoCapture(str(reader.video_path("main")))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    count = min(len(frames), len(records))
    if count < 100:
        raise RuntimeError("{} has too few synchronized frames".format(path))
    return frames[:count], records[:count], reader


def _reference_mask(estimator: CursorPoseEstimator) -> np.ndarray:
    _x, _y, width, height = estimator.crop_xywh
    boundary = estimator.calibration["outer_boundary"]
    yy, xx = np.ogrid[:height, :width]
    mask = (
        (xx - float(boundary["center_x"])) ** 2
        + (yy - float(boundary["center_y"])) ** 2
        <= max(float(boundary["radius"]) - 5.0, 1.0) ** 2
    ).astype(np.uint8) * 255
    cursor_hole = (
        (xx - float(estimator.pivot[0])) ** 2
        + (yy - float(estimator.pivot[1])) ** 2
        <= 15.0 ** 2
    )
    mask[cursor_hole] = 0
    return mask


def _session_e2e_reference(
    estimator: CursorPoseEstimator,
    mask: np.ndarray,
    frames: Sequence[np.ndarray],
) -> dict:
    margin = max(1, min(8, len(frames) // 8))
    first_index = margin
    last_index = len(frames) - margin - 1
    x, y, width, height = estimator.crop_xywh
    first = frames[first_index][y : y + height, x : x + width]
    last = frames[last_index][y : y + height, x : x + width]
    shift, response = estimate_masked_shift(first, last, mask)
    magnitude = float(np.hypot(float(shift[0]), float(shift[1])))
    shift_angle = float(np.degrees(np.arctan2(shift[1], shift[0])) % 360.0)
    return {
        "valid": bool(response >= 0.05 and magnitude >= 1.5),
        "reason": None if response >= 0.05 and magnitude >= 1.5 else "weak_motion_reference",
        "first_frame_index": int(first_index),
        "last_frame_index": int(last_index),
        "phase_correlation_response": float(response),
        "map_content_shift_magnitude_px": magnitude,
        "travel_angle_screen_deg": float((shift_angle + 180.0) % 360.0),
    }


def _forward_only_input_audit(reader: SessionReader) -> dict:
    keys = set()
    mouse_motion_event_count = 0
    for event in reader.inputs:
        payload = event.get("payload") or {}
        if event.get("kind") == "pc_raw_keyboard":
            name = str(payload.get("key_name") or "").upper()
            if name:
                keys.add(name)
        elif event.get("kind") == "pc_raw_mouse" and (
            int(payload.get("delta_x") or 0) or int(payload.get("delta_y") or 0)
        ):
            mouse_motion_event_count += 1
    direction_change_keys = sorted(keys.intersection(("A", "S", "D")))
    valid = bool(
        "W" in keys
        and not direction_change_keys
        and mouse_motion_event_count == 0
    )
    return {
        "valid_forward_only_control": valid,
        "observed_keys": sorted(keys),
        "direction_change_keys": direction_change_keys,
        "mouse_motion_event_count": int(mouse_motion_event_count),
        "requirement": "W present; A/S/D absent; no relative mouse motion",
    }


def _outage_for_frame(scenario: str, ordinal: int) -> bool:
    config = OUTAGE_SCENARIOS[scenario]
    period = config["period_frames"]
    if period is None:
        return False
    return bool(
        int(ordinal) >= int(period)
        and int(ordinal) % int(period) < int(config["burst_frames"])
    )


def _gate_config(calibration: dict) -> dict:
    dynamics = calibration.get("cursor_temporal_dynamics") or {}
    envelope = dynamics.get("recommended_runtime_envelope") or {}
    innovation_limit = envelope.get("ordinary_heading_jump_p99_deg")
    high_confidence = (
        calibration.get("cursor_pose_validation") or {}
    ).get("median_confidence")
    if innovation_limit is None or high_confidence is None:
        raise ValueError(
            "Standard E2E gate needs ordinary-cruise cursor dynamics calibration"
        )
    return {
        "method": "last_accepted_state_innovation_gate",
        "innovation_limit_deg": float(innovation_limit),
        "large_innovation_confidence_min": float(high_confidence),
        "limit_provenance": (
            "calibration.cursor_temporal_dynamics."
            "recommended_runtime_envelope.ordinary_heading_jump_p99_deg"
        ),
        "large_innovation_confidence_provenance": (
            "calibration.cursor_pose_validation.median_confidence"
        ),
    }


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _profile_measurements(
    calibration_path: Path,
    profile_name: str,
    session: str,
    frames: Sequence[np.ndarray],
    records: Sequence[dict],
    references: Sequence[dict],
) -> List[dict]:
    profile = resolve_tracking_profile(profile_name)
    estimator = CursorPoseEstimator(
        calibration_path,
        gaussian_fit_method=profile["cursor_pose_method"],
        validation_policy=profile["cursor_validation_policy"],
    )
    estimator.estimate(frames[0])
    rows = []
    for ordinal, (frame, record, reference) in enumerate(
        zip(frames, records, references)
    ):
        started = time.perf_counter_ns()
        result = estimator.estimate(
            frame,
            frame_index=int(record["frame_index"]),
            session_time_ns=int(record["session_time_ns"]),
        )
        latency_ns = time.perf_counter_ns() - started
        detected = bool(result.get("detected"))
        confidence = float(result.get("confidence") or 0.0)
        candidate_produced = bool(
            detected
            and result.get("angle_screen_deg") is not None
        )
        confidence_passed = bool(
            candidate_produced
            and confidence >= float(profile["pose_confidence_min"])
        )
        if not candidate_produced:
            reason = "cursor_not_detected"
        elif not confidence_passed:
            reason = "confidence_below_threshold"
        else:
            reason = None
        rows.append(
            {
                "profile": profile_name,
                "session": session,
                "ordinal": int(ordinal),
                "frame_index": int(record["frame_index"]),
                "session_time_ns": int(record["session_time_ns"]),
                "angle_deg": (
                    float(result["angle_screen_deg"])
                    if result.get("angle_screen_deg") is not None
                    else None
                ),
                "confidence": confidence,
                "primary_candidate_produced": candidate_produced,
                "confidence_gate_passed": confidence_passed,
                "natural_primary_accepted": confidence_passed,
                "natural_rejection_reason": reason,
                "confidence_threshold": float(profile["pose_confidence_min"]),
                "primary_latency_ns": int(latency_ns),
                "pixel_validation_performed": bool(
                    result.get("pixel_validation_performed")
                ),
                "gaussian_fitter_fallback_used": bool(
                    result.get("gaussian_fallback_used")
                ),
                "reference_valid": bool(reference.get("valid")),
                "reference_angle_deg": reference.get("travel_angle_screen_deg"),
                "reference_response": reference.get("phase_correlation_response"),
                "reference_displacement_px": reference.get(
                    "map_content_shift_magnitude_px"
                ),
            }
        )
    return rows


def _replay_strategy(
    primary_rows: Sequence[dict],
    scenario: str,
    solution_id: str,
    strategy_id: str,
    gate_config: dict,
    temporal_outlier_gate: bool,
    report_role: str,
) -> List[dict]:
    gate = CausalOutlierGate(
        confidence_min=float(primary_rows[0]["confidence_threshold"]),
        innovation_limit_deg=float(gate_config["innovation_limit_deg"]),
        large_innovation_confidence_min=float(
            gate_config["large_innovation_confidence_min"]
        ),
        temporal_check_enabled=bool(temporal_outlier_gate),
    )
    strategy = FALLBACK_FACTORIES[strategy_id]()
    rows = []
    for row in primary_rows:
        injected = bool(
            row["confidence_gate_passed"]
            and _outage_for_frame(scenario, row["ordinal"])
        )
        candidate = Measurement(
                frame_index=int(row["frame_index"]),
                session_time_ns=int(row["session_time_ns"]),
                angle_deg=row["angle_deg"],
                confidence=float(row["confidence"]),
                accepted=bool(row["confidence_gate_passed"]),
                rejection_reason=row["natural_rejection_reason"],
                latency_ns=int(row["primary_latency_ns"]),
            )
        if injected:
            measurement = Measurement(
                candidate.frame_index,
                candidate.session_time_ns,
                candidate.angle_deg,
                candidate.confidence,
                False,
                "injected_outage:{}".format(scenario),
                candidate.latency_ns,
            )
        else:
            measurement = gate.decide(candidate)
        started = time.perf_counter_ns()
        state = strategy.update(measurement)
        strategy_latency_ns = time.perf_counter_ns() - started
        reference = (
            float(row["reference_angle_deg"])
            if row["reference_valid"]
            else None
        )
        error = (
            abs(circular_delta_deg(state.angle_deg, reference))
            if state.angle_deg is not None and reference is not None
            else None
        )
        rows.append(
            {
                "solution": solution_id,
                "report_role": report_role,
                "profile": row["profile"],
                "session": row["session"],
                "outage_scenario": scenario,
                "fallback_strategy": strategy_id,
                "frame_index": int(row["frame_index"]),
                "session_time_ns": int(row["session_time_ns"]),
                "confidence": float(row["confidence"]),
                "primary_candidate_produced": bool(
                    row["primary_candidate_produced"]
                ),
                "confidence_gate_passed": bool(row["confidence_gate_passed"]),
                "natural_primary_accepted": bool(
                    row["natural_primary_accepted"]
                ),
                "injected_outage": bool(injected),
                "final_rejection_reason": measurement.rejection_reason,
                "primary_accepted": bool(state.primary_accepted),
                "fallback_invoked": bool(state.fallback_invoked),
                "fallback_produced_output": bool(
                    state.fallback_produced_output
                ),
                "provenance": state.provenance,
                "state_age_ms": state.state_age_ms,
                "output_angle_deg": state.angle_deg,
                "reference_angle_deg": reference,
                "absolute_error_deg": error,
                "primary_latency_ns": int(row["primary_latency_ns"]),
                "fallback_strategy_latency_ns": int(strategy_latency_ns),
                "e2e_latency_ns": int(
                    row["primary_latency_ns"] + strategy_latency_ns
                ),
            }
        )
    return rows


def _aggregate(rows: Sequence[dict], development_sessions: Sequence[str]) -> List[dict]:
    development_sessions = set(development_sessions)
    groups: Dict[Tuple[str, str, str], List[dict]] = {}
    for row in rows:
        split = "development" if row["session"] in development_sessions else "holdout"
        for aggregate_split in (split, "all"):
            key = (
                row["solution"],
                row["outage_scenario"],
                aggregate_split,
            )
            groups.setdefault(key, []).append(row)
    output = []
    for key, selected in sorted(groups.items()):
        solution, scenario, split = key
        summary = summarize_e2e_rows(selected)
        summary.update(
            {
                "solution": solution,
                "report_role": selected[0]["report_role"],
                "profile": selected[0]["profile"],
                "outage_scenario": scenario,
                "fallback_strategy": selected[0]["fallback_strategy"],
                "split": split,
                "reference_evaluable_frame_count": int(
                    sum(row["reference_angle_deg"] is not None for row in selected)
                ),
                "injected_outage_rate": float(
                    sum(bool(row["injected_outage"]) for row in selected)
                    / max(len(selected), 1)
                ),
            }
        )
        output.append(summary)
    return output


def _format_percent(value: Optional[float]) -> str:
    return "n/a" if value is None else "{:.2%}".format(float(value))


def _format_number(value: Optional[float], suffix: str = "") -> str:
    return "n/a" if value is None else "{:.2f}{}".format(float(value), suffix)


def _write_report(path: Path, result: dict) -> None:
    selected = [row for row in result["aggregate"] if row["split"] == "holdout"]
    lookup = {
        (row["solution"], row["outage_scenario"]): row for row in selected
    }

    def add_result(lines, label, row):
        if row is None:
            lines.extend(["### {}".format(label), "", "Not run.", ""])
            return
        error = row["e2e_absolute_error_deg"]
        lines.extend(
            [
                "### {}".format(label),
                "",
                "- Candidate: {}".format(
                    _format_percent(row["primary_candidate_produced_rate"])
                ),
                "- Accepted fresh: {}".format(
                    _format_percent(row["primary_measurement_accepted_rate"])
                ),
                "- Held: {}".format(
                    _format_percent(row["output_provenance_rate"]["held"])
                ),
                "- Available: {}".format(
                    _format_percent(row["final_output_available_rate"])
                ),
                "- Error mean / median: {} / {}".format(
                    _format_number(error["mean"], " deg"),
                    _format_number(error["median"], " deg"),
                ),
                "- Error P95 / worst: {} / {}".format(
                    _format_number(error["p95"], " deg"),
                    _format_number(error["worst"], " deg"),
                ),
                "- Median latency: {}".format(
                    _format_number(row["e2e_latency_ms"]["median"], " ms")
                ),
                "",
            ]
        )

    lines = [
        "# Cursor pose layered comparison",
        "",
        "Holdout frames are chronological. Each section changes one layer only.",
        "",
        "## Layer 1 — estimator profile",
        "",
        "Gate is confidence-only. Publication is previous-state hold.",
        "",
    ]
    add_result(
        lines,
        "Accurate: vectorized Gaussian + full validation",
        lookup.get(("accurate_confidence_hold", "natural")),
    )
    add_result(
        lines,
        "Real-time: Gaussian cascade + ambiguous validation",
        lookup.get(("realtime_confidence_hold", "natural")),
    )
    add_result(
        lines,
        "Fast: Gaussian cascade + minimal validation",
        lookup.get(("fast_confidence_hold", "natural")),
    )
    lines.extend(
        [
            "## Layer 2 — final gate",
            "",
            "Estimator is real-time. Publication is previous-state hold.",
            "",
        ]
    )
    add_result(
        lines,
        "Confidence-only gate (current)",
        lookup.get(("realtime_confidence_hold", "natural")),
    )
    add_result(
        lines,
        "Strict temporal gate (negative control)",
        lookup.get(("realtime_strict_hold", "natural")),
    )
    lines.extend(
        [
            "## Layer 3 — rejected-state publication",
            "",
            "Estimator is real-time. Gate is confidence-only.",
            "A deterministic three-frame outage is injected every 90 frames.",
            "",
        ]
    )
    add_result(
        lines,
        "Hold previous accepted state",
        lookup.get(("realtime_confidence_hold", "three_frame_burst_every_90")),
    )
    add_result(
        lines,
        "Constant-velocity prediction",
        lookup.get(("realtime_confidence_predict", "three_frame_burst_every_90")),
    )
    add_result(
        lines,
        "Publish unavailable",
        lookup.get(("realtime_confidence_reject", "three_frame_burst_every_90")),
    )
    lines.extend(
        [
            "## Complete stack decision",
            "",
            "Keep: real-time estimator + confidence gate + previous-state hold.",
            "",
            "Do not land the strict temporal gate from this evidence. Its rejected "
            "frames must be checked against pose labels, because travel direction is "
            "not pixel-level cursor truth.",
            "",
            "Prediction and unavailable output remain negative controls unless they "
            "beat hold on both availability and worst E2E error.",
            "",
            "## Rate meanings",
            "",
            "- Candidate: first-layer pose produced / attempted frames.",
            "- Accepted fresh: final accepted measurements / attempted frames.",
            "- Held: reused states / attempted frames.",
            "- Available: fresh + held + predicted / attempted frames.",
            "- Internal validation invocation is not acceptance or rejection.",
            "",
            "## Accuracy limits",
            "",
            "Mean, median, P95, and worst are absolute circular E2E travel-heading "
            "errors. Worst is reported only for a complete chronological output.",
            "",
            "Whole-session travel is functional evidence, not per-frame cursor truth. "
            "It must not train a pose-outlier gate by itself.",
            "",
            "## Traceability",
            "",
            "- Git: `{}`".format(result["provenance"]["repository"]["revision"]),
            "- Tested source dirty: `{}`".format(
                result["provenance"]["repository"]["tested_source_dirty"]
            ),
            "- Config: `benchmark_config.json`",
            "- Methods: `method_manifest.json`",
            "- Raw measurements: `primary_measurements.csv`",
            "- Stateful rows: `e2e_rows.csv`",
            "- Aggregates: `results.json`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(
    calibration_path: Path,
    session_root: Path,
    output_path: Path,
    *,
    sessions: Sequence[str] = DEFAULT_SESSIONS,
    development_sessions: Sequence[str] = tuple(DEFAULT_DEVELOPMENT_SESSIONS),
    solutions: Sequence[str] = DEFAULT_SOLUTIONS,
    hash_videos: bool = True,
) -> dict:
    root = Path(__file__).resolve().parents[2]
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    calibration_path = Path(calibration_path).resolve()
    session_root = Path(session_root).resolve()
    calibration_document = json.loads(calibration_path.read_text(encoding="utf-8"))
    gate_config = _gate_config(calibration_document)
    unknown_solutions = sorted(set(solutions) - set(SOLUTION_CONFIGS))
    if unknown_solutions:
        raise ValueError("Unknown solutions: {}".format(", ".join(unknown_solutions)))
    selected_solution_configs = {
        name: dict(SOLUTION_CONFIGS[name]) for name in solutions
    }
    profiles = sorted(
        {value["profile"] for value in selected_solution_configs.values()}
    )
    config = {
        "schema_version": SCHEMA_VERSION,
        "sessions": list(sessions),
        "development_sessions": list(development_sessions),
        "holdout_sessions": sorted(set(sessions) - set(development_sessions)),
        "solutions": selected_solution_configs,
        "profiles": profiles,
        "decision_layers": [
            "one_frame_primary_candidate",
            "final_confidence_and_temporal_innovation_gate",
        ],
        "gate": gate_config,
        "outage_scenarios": OUTAGE_SCENARIOS,
        "reference": {
            "method": "whole_session_masked_minimap_phase_correlation",
            "scope": "controlled forward-only sessions",
            "minimum_response": 0.05,
            "minimum_displacement_px": 1.5,
            "evaluator_only": True,
        },
        "accuracy": {
            "quantity": "absolute circular screen-heading error",
            "unit": "degree",
            "statistics": ["mean", "median", "p95", "worst_e2e_only"],
        },
    }
    (output_path / "benchmark_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    fallback_identities = {
        name: _source_identity(factory().__class__)
        for name, factory in FALLBACK_FACTORIES.items()
    }
    tested_files = {
        Path(__file__).resolve(),
        Path(inspect.getsourcefile(CursorPoseEstimator)).resolve(),
        Path(inspect.getsourcefile(summarize_e2e_rows)).resolve(),
    }
    method_manifest = {
        "schema_version": SCHEMA_VERSION,
        "primary_estimator": _source_identity(CursorPoseEstimator.estimate),
        "fallback_strategies": fallback_identities,
        "final_outlier_gate": _source_identity(CausalOutlierGate.decide),
        "profile_resolver": _source_identity(resolve_tracking_profile),
    }
    (output_path / "method_manifest.json").write_text(
        json.dumps(method_manifest, indent=2), encoding="utf-8"
    )

    identity_estimator = CursorPoseEstimator(calibration_path)
    reference_mask = _reference_mask(identity_estimator)
    all_primary = []
    input_manifest = {
        "calibration": _file_identity(calibration_path),
        "model": _file_identity(calibration_path.parent / "model.npz"),
        "sessions": {},
    }
    for session in sessions:
        session_path = session_root / session
        frames, records, reader = _decode_session(session_path)
        input_audit = _forward_only_input_audit(reader)
        if not input_audit["valid_forward_only_control"]:
            raise RuntimeError(
                "{} is not a verified forward-only E2E reference session: {}".format(
                    session, input_audit
                )
            )
        session_reference = _session_e2e_reference(
            identity_estimator, reference_mask, frames
        )
        if not session_reference["valid"]:
            raise RuntimeError(
                "{} has no reliable whole-session E2E reference: {}".format(
                    session, session_reference
                )
            )
        references = [session_reference] * len(frames)
        input_manifest["sessions"][session] = {
            "manifest": _file_identity(session_path / "manifest.json"),
            "frames": _file_identity(session_path / "frames.jsonl"),
            "inputs": _file_identity(session_path / "inputs.jsonl"),
            "video": _file_identity(
                reader.video_path("main"), hash_content=bool(hash_videos)
            ),
            "decoded_frame_count": int(len(frames)),
            "reference_valid_frame_count": int(len(frames)),
            "forward_only_input_audit": input_audit,
            "whole_session_e2e_reference": session_reference,
        }
        for profile in profiles:
            all_primary.extend(
                _profile_measurements(
                    calibration_path,
                    profile,
                    session,
                    frames,
                    records,
                    references,
                )
            )

    (output_path / "input_manifest.json").write_text(
        json.dumps(input_manifest, indent=2), encoding="utf-8"
    )
    _write_csv(output_path / "primary_measurements.csv", all_primary)

    e2e_rows = []
    for solution_id, solution in selected_solution_configs.items():
        profile = solution["profile"]
        strategy_id = solution["fallback_strategy"]
        for session in sessions:
            selected = [
                row
                for row in all_primary
                if row["profile"] == profile and row["session"] == session
            ]
            for scenario in OUTAGE_SCENARIOS:
                e2e_rows.extend(
                    _replay_strategy(
                        selected,
                        scenario,
                        solution_id,
                        strategy_id,
                        gate_config,
                        bool(solution["temporal_outlier_gate"]),
                        str(solution["report_role"]),
                    )
                )
    _write_csv(output_path / "e2e_rows.csv", e2e_rows)
    aggregate = _aggregate(e2e_rows, development_sessions)
    provenance = {
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "repository": _repository_identity(root, sorted(tested_files)),
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "method_manifest": "method_manifest.json",
        "input_manifest": "input_manifest.json",
        "configuration": "benchmark_config.json",
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": "cursor_pose_causal_e2e",
        "provenance": provenance,
        "config": config,
        "aggregate": aggregate,
    }
    (output_path / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _write_report(output_path / "REPORT.md", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("session_root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--session", action="append", dest="sessions")
    parser.add_argument("--development-session", action="append")
    parser.add_argument(
        "--solution",
        action="append",
        dest="solutions",
        choices=tuple(SOLUTION_CONFIGS),
    )
    parser.add_argument(
        "--skip-video-hash",
        action="store_true",
        help="Record video size/mtime without SHA-256; report is less portable.",
    )
    arguments = parser.parse_args()
    result = run(
        arguments.calibration,
        arguments.session_root,
        arguments.output,
        sessions=arguments.sessions or DEFAULT_SESSIONS,
        development_sessions=(
            arguments.development_session or tuple(DEFAULT_DEVELOPMENT_SESSIONS)
        ),
        solutions=arguments.solutions or DEFAULT_SOLUTIONS,
        hash_videos=not arguments.skip_video_hash,
    )
    print(
        json.dumps(
            [
                row
                for row in result["aggregate"]
                if row["split"] == "holdout"
                and row["outage_scenario"] == "natural"
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
