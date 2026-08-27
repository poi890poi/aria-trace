"""Private legacy MR95 implementation retained for old internal artifacts.

New code must use :mod:`feature_matching`; this module is intentionally absent
from the package's public exports and from the rig-calibration application.
"""

from collections import defaultdict
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np

from .contracts import MatchResult, MatchTrial


def _gray_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("Matcher expects a grayscale or BGR image")
    return np.asarray(gray, dtype=np.float32)


class PhaseCorrelationMatcher:
    """Small translation matcher useful for focus sweeps and plumbing tests."""

    def __init__(self, minimum_response: float = 0.10) -> None:
        self.minimum_response = float(minimum_response)

    def match(self, reference: np.ndarray, observed: np.ndarray) -> MatchResult:
        if reference.shape[:2] != observed.shape[:2]:
            raise ValueError("Phase-correlation images must have the same size")
        first = _gray_float(reference)
        second = _gray_float(observed)
        height, width = first.shape
        window = cv2.createHanningWindow((width, height), cv2.CV_32F)
        shift, response = cv2.phaseCorrelate(first, second, window)
        return MatchResult(
            translation_xy=(float(shift[0]), float(shift[1])),
            rotation_deg=0.0,
            confidence=float(np.clip(response, 0.0, 1.0)),
            ambiguous=bool(response < self.minimum_response),
            diagnostics={"phase_correlation_response": float(response)},
        )

    def __call__(self, reference: np.ndarray, observed: np.ndarray) -> MatchResult:
        return self.match(reference, observed)


def generate_band_limited_target(
    size_px: Sequence[int],
    detail_cells_across: int,
    seed: int,
    pattern_family: str = "luminance",
) -> np.ndarray:
    """Generate a repeatable feature target at one dominant detail scale."""

    width, height = map(int, size_px)
    if min(width, height) <= 0 or detail_cells_across <= 1:
        raise ValueError("Target size and detail scale must be positive")
    grid_width = max(2, int(detail_cells_across))
    grid_height = max(2, int(round(grid_width * height / float(width))))
    random = np.random.RandomState(int(seed))
    coarse = random.normal(0.0, 1.0, (grid_height, grid_width)).astype(np.float32)
    field = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    cell_px = width / float(grid_width)
    low_frequency = cv2.GaussianBlur(
        field, (0, 0), sigmaX=max(0.5, cell_px * 1.5), sigmaY=max(0.5, cell_px * 1.5)
    )
    field -= low_frequency
    field /= float(np.std(field) + 1.0e-6)
    intensity = np.clip(127.5 + field * 42.0, 16.0, 239.0).astype(np.uint8)
    if pattern_family == "luminance":
        return cv2.cvtColor(intensity, cv2.COLOR_GRAY2BGR)
    inverse = 255 - intensity
    neutral = np.full_like(intensity, 127)
    if pattern_family == "red_green":
        return np.dstack([neutral, inverse, intensity])
    if pattern_family == "blue_yellow":
        return np.dstack([intensity, inverse, inverse])
    raise ValueError("Unknown target pattern family {}".format(pattern_family))


def warp_target(
    image: np.ndarray,
    translation_xy: Sequence[float] = (0.0, 0.0),
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """Apply a known in-plane transform for a controlled matchability trial."""

    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(
        ((width - 1) / 2.0, (height - 1) / 2.0), float(rotation_deg), 1.0
    )
    matrix[:, 2] += np.asarray(translation_xy, dtype=np.float64)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _matcher_call(matcher: Any, reference: np.ndarray, observed: np.ndarray) -> MatchResult:
    value = matcher.match(reference, observed) if hasattr(matcher, "match") else matcher(reference, observed)
    if not isinstance(value, MatchResult):
        if not isinstance(value, Mapping):
            raise TypeError("Matcher must return MatchResult or a compatible mapping")
        value = MatchResult(
            translation_xy=tuple(value["translation_xy"]),
            rotation_deg=float(value.get("rotation_deg", 0.0)),
            confidence=float(value.get("confidence", 0.0)),
            ambiguous=bool(value.get("ambiguous", False)),
            diagnostics=value.get("diagnostics", {}),
        )
    return value


def _angle_error_degrees(measured: float, expected: float) -> float:
    return float(abs((measured - expected + 180.0) % 360.0 - 180.0))


def _passing_cells(
    rows: Sequence[Mapping[str, Any]], reliability_threshold: float
) -> int:
    by_scale_and_condition: Dict[Tuple[int, str, str, bool], List[bool]] = defaultdict(list)
    for row in rows:
        key = (
            int(row["detail_cells_across"]),
            str(row["reference_mode"]),
            str(row["pattern_family"]),
            bool(row["moving"]),
        )
        by_scale_and_condition[key].append(bool(row["success"]))
    passing = []
    scales = sorted({key[0] for key in by_scale_and_condition})
    for scale in scales:
        condition_rates = [
            float(np.mean(values))
            for key, values in by_scale_and_condition.items()
            if key[0] == scale
        ]
        if condition_rates and min(condition_rates) >= reliability_threshold:
            passing.append(scale)
    return max(passing) if passing else 0


def evaluate_matchability(
    trials: Iterable[MatchTrial],
    matcher: Any,
    patch_size_mm: float = 20.0,
    reliability_threshold: float = 0.95,
    translation_tolerance_fraction: float = 0.01,
    rotation_tolerance_deg: float = 1.0,
    bootstrap_samples: int = 200,
    bootstrap_seed: int = 1977,
) -> Dict[str, Any]:
    """Evaluate held-out trials and return a conservative MR95-20 score."""

    values = list(trials)
    if not values:
        raise ValueError("At least one matchability trial is required")
    if patch_size_mm <= 0:
        raise ValueError("patch_size_mm must be positive")
    if not 0.0 < reliability_threshold <= 1.0:
        raise ValueError("Reliability threshold must be within (0, 1]")
    if translation_tolerance_fraction <= 0 or rotation_tolerance_deg < 0:
        raise ValueError("Match tolerances are invalid")

    rows: List[Dict[str, Any]] = []
    for index, trial in enumerate(values):
        result = _matcher_call(matcher, trial.reference, trial.observed)
        measured_translation = np.asarray(result.translation_xy, dtype=np.float64)
        expected_translation = np.asarray(
            trial.expected_translation_xy, dtype=np.float64
        )
        translation_error_px = float(
            np.linalg.norm(measured_translation - expected_translation)
        )
        patch_width_px = float(trial.reference.shape[1])
        translation_error_fraction = translation_error_px / max(patch_width_px, 1.0)
        rotation_error_deg = _angle_error_degrees(
            result.rotation_deg, trial.expected_rotation_deg
        )
        success = bool(
            not result.ambiguous
            and translation_error_fraction <= translation_tolerance_fraction
            and rotation_error_deg <= rotation_tolerance_deg
        )
        catastrophic = bool(
            result.ambiguous
            or translation_error_fraction > 5.0 * translation_tolerance_fraction
            or rotation_error_deg > 5.0 * max(rotation_tolerance_deg, 1.0e-9)
        )
        rows.append(
            {
                "trial_id": trial.trial_id or "trial-{:04d}".format(index),
                "detail_cells_across": int(trial.detail_cells_across),
                "reference_mode": trial.reference_mode,
                "pattern_family": trial.pattern_family,
                "moving": bool(trial.moving),
                "success": success,
                "catastrophic": catastrophic,
                "translation_error_px": translation_error_px,
                "translation_error_fraction": translation_error_fraction,
                "rotation_error_deg": rotation_error_deg,
                "confidence": float(result.confidence),
                "ambiguous": bool(result.ambiguous),
                "measured_translation_xy": list(map(float, result.translation_xy)),
                "measured_rotation_deg": float(result.rotation_deg),
                "diagnostics": dict(result.diagnostics),
            }
        )

    conditions_by_scale = defaultdict(set)
    for row in rows:
        conditions_by_scale[int(row["detail_cells_across"])].add(
            (row["reference_mode"], row["pattern_family"], row["moving"])
        )
    all_conditions = set().union(*conditions_by_scale.values())
    incomplete_scales = [
        scale
        for scale, conditions in conditions_by_scale.items()
        if conditions != all_conditions
    ]
    if incomplete_scales:
        raise ValueError(
            "Every detail scale must include the same reference/pattern/motion conditions; "
            "incomplete scales: {}".format(sorted(incomplete_scales))
        )

    primary = _passing_cells(rows, reliability_threshold)
    scale_summary = []
    for scale in sorted({int(row["detail_cells_across"]) for row in rows}):
        scale_rows = [row for row in rows if row["detail_cells_across"] == scale]
        scale_summary.append(
            {
                "detail_cells_across": scale,
                "trial_count": len(scale_rows),
                "success_rate": float(np.mean([row["success"] for row in scale_rows])),
                "translation_error_p95_fraction": float(
                    np.percentile(
                        [row["translation_error_fraction"] for row in scale_rows], 95
                    )
                ),
                "rotation_error_p95_deg": float(
                    np.percentile([row["rotation_error_deg"] for row in scale_rows], 95)
                ),
                "ambiguous_rate": float(
                    np.mean([row["ambiguous"] for row in scale_rows])
                ),
            }
        )

    bootstrap_scores = []
    if bootstrap_samples > 0:
        random = np.random.RandomState(int(bootstrap_seed))
        grouped: Dict[Tuple[int, str, str, bool], List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[
                (
                    row["detail_cells_across"],
                    row["reference_mode"],
                    row["pattern_family"],
                    row["moving"],
                )
            ].append(row)
        for _ in range(int(bootstrap_samples)):
            sampled = []
            for group in grouped.values():
                indices = random.randint(0, len(group), size=len(group))
                sampled.extend(group[index] for index in indices)
            bootstrap_scores.append(_passing_cells(sampled, reliability_threshold))

    smallest_detail = patch_size_mm / primary if primary > 0 else None
    if bootstrap_scores:
        bootstrap_width = float(
            np.percentile(bootstrap_scores, 97.5)
            - np.percentile(bootstrap_scores, 2.5)
        )
        maximum_scale = max(row["detail_cells_across"] for row in rows)
        stability_quality = max(0.0, 1.0 - bootstrap_width / max(maximum_scale, 1))
    else:
        stability_quality = 0.5
    sample_quality = min(1.0, len(rows) / 64.0)
    confidence = float(np.clip(0.6 * sample_quality + 0.4 * stability_quality, 0.0, 1.0))
    return {
        "metric": "MR{:02d}-{:g}".format(
            int(round(reliability_threshold * 100)), patch_size_mm
        ),
        "patch_size_mm": float(patch_size_mm),
        "reliability_threshold": float(reliability_threshold),
        "translation_tolerance_fraction": float(translation_tolerance_fraction),
        "rotation_tolerance_deg": float(rotation_tolerance_deg),
        "primary_cells_across_patch": int(primary),
        "smallest_matchable_detail_mm": (
            float(smallest_detail) if smallest_detail is not None else None
        ),
        "bootstrap_95_ci_cells": (
            [
                float(np.percentile(bootstrap_scores, 2.5)),
                float(np.percentile(bootstrap_scores, 97.5)),
            ]
            if bootstrap_scores
            else None
        ),
        "failure_rate": float(np.mean([not row["success"] for row in rows])),
        "catastrophic_mismatch_rate": float(np.mean([row["catastrophic"] for row in rows])),
        "confidence": confidence,
        "scale_summary": scale_summary,
        "trials": rows,
    }
