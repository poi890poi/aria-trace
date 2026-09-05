"""Causal recorded-video benchmark for route-assisted map tracking.

This module is deliberately outside the production live tracker.  It compares
small route-tracing mechanisms without changing workbench behavior or artifact
schemas.  A demonstrated route may propose a bounded map search, but it never
supplies the reported pose or the post-run compliance score.
"""

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from aria_trace.services.tracking.runtime import MinimapExtractor, _gradient
from aria_trace.services.mapping.layers import LayeredGlobalLocalizer
from rig_runtime.adapters.filesystem.session import SessionReader
from replay.route_similarity import route_similarity_report
from replay.route_tracking import RouteTrackingPackage, describe_minimap


VARIANTS = (
    "route_descriptor",
    "route_refine_top1",
    "route_refine_top3",
    "continuous_local",
    "continuous_gated",
    "local_primary_gated",
)
CORRELATION_FEATURES = ("gradient", "intensity", "canny", "laplacian")
LOCAL_MATCHERS = ("ccorr_normed", "phase_correlation")
MODE_POLICIES = ("all", "sticky", "transition_zone")
INITIALIZATION_POLICIES = ("route", "global", "known_start")
CONTINUITY_CLOCKS = ("frame", "accepted")
RECOVERY_POLICIES = ("route", "global_consensus")


def _correlation_feature(image: np.ndarray, name: str) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if name == "gradient":
        return _gradient(image)
    if name == "intensity":
        return gray.astype(np.float32)
    if name == "canny":
        return cv2.Canny(np.uint8(gray), 60, 160).astype(np.float32)
    if name == "laplacian":
        return np.abs(
            cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=3)
        )
    raise ValueError("Unknown correlation feature: {}".format(name))


@dataclass
class TraceResult:
    valid: bool
    x: Optional[float]
    y: Optional[float]
    score: float
    margin: float
    source: str
    route_state_index: Optional[int] = None
    mode_id: Optional[str] = None
    measurement_accepted: bool = False
    primary_candidate_produced: bool = False
    primary_measurement_accepted: bool = False
    final_gate_rejected: bool = False


def _percentiles(values) -> dict:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not len(array):
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _loss_metrics(valid_flags) -> dict:
    episodes = []
    current = 0
    for valid in valid_flags:
        if valid:
            if current:
                episodes.append(current)
                current = 0
        else:
            current += 1
    if current:
        episodes.append(current)
    return {
        "tracked_fraction": float(np.mean(valid_flags)) if valid_flags else 0.0,
        "loss_episode_count": len(episodes),
        "longest_loss_frames": max(episodes, default=0),
        "lost_frame_count": int(sum(episodes)),
    }


class CausalRouteTracer:
    """One-frame-in, one-pose-out experimental route tracer.

    The class has no clock from the demonstration and does not consume route
    heading, motion vectors, or progress.  Route descriptors are only visual
    search proposals.  Every refined pose comes from the current mini-map's
    correlation against the map atlas.
    """

    def __init__(
        self,
        package: RouteTrackingPackage,
        atlas: LayeredGlobalLocalizer,
        variant: str,
        *,
        score_min: float = 0.55,
        recovery_radius_px: float = 55.0,
        local_radius_px: float = 18.0,
        correlation_feature: str = "gradient",
        mode_policy: str = "all",
        continuity_clock: str = "frame",
        continuity_speed_multiplier: float = 4.0,
        recovery_policy: str = "route",
        local_matcher: str = "ccorr_normed",
        transition_zone_radius_floor_px: float = 12.0,
        transition_exit_radius_px: float = 40.0,
        transition_confirmation_count: int = 2,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError("Unknown route tracer variant: {}".format(variant))
        self.package = package
        self.atlas = atlas
        self.variant = variant
        self.score_min = float(score_min)
        self.recovery_radius_px = float(recovery_radius_px)
        self.local_radius_px = float(local_radius_px)
        if correlation_feature not in CORRELATION_FEATURES:
            raise ValueError(
                "Unknown correlation feature: {}".format(correlation_feature)
            )
        if mode_policy not in MODE_POLICIES:
            raise ValueError("Unknown mode policy: {}".format(mode_policy))
        self.correlation_feature = correlation_feature
        if local_matcher not in LOCAL_MATCHERS:
            raise ValueError("Unknown local matcher: {}".format(local_matcher))
        self.local_matcher = local_matcher
        self.mode_policy = mode_policy
        self.transition_zone_radius_floor_px = max(
            0.0, float(transition_zone_radius_floor_px)
        )
        self.transition_exit_radius_px = max(
            self.transition_zone_radius_floor_px,
            float(transition_exit_radius_px),
        )
        self.transition_confirmation_count = max(
            1, int(transition_confirmation_count)
        )
        transition_model = getattr(atlas, "transition_model", None)
        if not isinstance(transition_model, dict):
            transition_model = {}
        zones = tuple(transition_model.get("transition_zones") or ())
        legacy_zone = transition_model.get("canonical_boundary")
        if not zones and legacy_zone:
            zones = (legacy_zone,)
        self.transition_zones = tuple(
            {
                "zone_id": str(
                    zone.get("zone_id") or "legacy-transition-zone"
                ),
                "center_xy": tuple(float(value) for value in zone["center_xy"]),
                "radius_px": max(0.0, float(zone.get("radius_px") or 0.0)),
            }
            for zone in zones
        )
        self._armed_transition_zone_id = None
        self._completed_transition_zone_id = None
        self._pending_mode_id = None
        self._pending_mode_wins = 0
        self.transition_switches = []
        if continuity_clock not in CONTINUITY_CLOCKS:
            raise ValueError(
                "Unknown continuity clock: {}".format(continuity_clock)
            )
        self.continuity_clock = continuity_clock
        self._correlation_feature_configured = correlation_feature == "gradient"
        self.previous_xy = None
        self.previous_time_ns = None
        self.previous_mode_id = None
        motion = package.manifest.get("motion_envelope") or {}
        speed = (motion.get("speed_px_s") or {}).get("p99")
        self.continuity_speed_multiplier = max(
            1.0, float(continuity_speed_multiplier)
        )
        self.continuity_speed_limit_px_s = max(
            30.0,
            self.continuity_speed_multiplier * float(speed or 0.0),
        )
        if recovery_policy not in RECOVERY_POLICIES:
            raise ValueError("Unknown recovery policy: {}".format(recovery_policy))
        self.recovery_policy = recovery_policy
        self.rejection_streak = 0
        self.global_recovery_hypotheses = []

    def _configure_correlation_feature(self) -> None:
        if self._correlation_feature_configured:
            return
        for localizer in self.atlas.localizers.values():
            localizer.map_gradient = _correlation_feature(
                localizer.mosaic, self.correlation_feature
            )
        self._correlation_feature_configured = True

    def _route_candidates(self, descriptor, top_k: int):
        # No previous_state_index: the replay may pause, reverse, deviate, or
        # travel at a completely different rate from the demonstration.
        return self.package.candidates(descriptor, top_k=top_k)

    def _transition_zone_mode_ids(self):
        """Choose active-only or all-layer search from causal spatial evidence."""

        if (
            self.previous_xy is None
            or self.previous_mode_id not in self.atlas.localizers
            or not self.transition_zones
        ):
            return None
        point = np.asarray(self.previous_xy, dtype=np.float64)
        distance, zone = min(
            (
                float(
                    np.linalg.norm(
                        point - np.asarray(item["center_xy"], dtype=np.float64)
                    )
                ),
                item,
            )
            for item in self.transition_zones
        )
        entry_radius = max(
            float(zone["radius_px"]), self.transition_zone_radius_floor_px
        )
        exit_radius = max(entry_radius, self.transition_exit_radius_px)
        if self._completed_transition_zone_id == zone["zone_id"]:
            if distance <= exit_radius:
                return [self.previous_mode_id]
            self._completed_transition_zone_id = None
        if self._armed_transition_zone_id == zone["zone_id"]:
            if distance <= exit_radius:
                return None
            self._armed_transition_zone_id = None
            self._pending_mode_id = None
            self._pending_mode_wins = 0
        if distance <= entry_radius:
            self._armed_transition_zone_id = zone["zone_id"]
            return None
        return [self.previous_mode_id]

    def _confirm_transition_mode(self, result: TraceResult) -> TraceResult:
        if (
            self.mode_policy != "transition_zone"
            or not result.valid
            or not result.measurement_accepted
            or self.previous_xy is None
            or self.previous_mode_id is None
            or result.mode_id is None
        ):
            return result
        if result.mode_id == self.previous_mode_id:
            self._pending_mode_id = None
            self._pending_mode_wins = 0
            return result
        if self._armed_transition_zone_id is None:
            return TraceResult(
                True,
                self.previous_xy[0],
                self.previous_xy[1],
                result.score,
                result.margin,
                "unarmed_mode_change_hold",
                result.route_state_index,
                self.previous_mode_id,
                measurement_accepted=False,
                primary_candidate_produced=True,
                primary_measurement_accepted=True,
                final_gate_rejected=True,
            )
        if self._pending_mode_id == result.mode_id:
            self._pending_mode_wins += 1
        else:
            self._pending_mode_id = result.mode_id
            self._pending_mode_wins = 1
        if self._pending_mode_wins < self.transition_confirmation_count:
            return TraceResult(
                True,
                self.previous_xy[0],
                self.previous_xy[1],
                result.score,
                result.margin,
                "transition_confirmation_hold",
                result.route_state_index,
                self.previous_mode_id,
                measurement_accepted=False,
                primary_candidate_produced=True,
                primary_measurement_accepted=True,
                final_gate_rejected=True,
            )
        self.transition_switches.append(
            {
                "from_mode_id": self.previous_mode_id,
                "to_mode_id": result.mode_id,
                "zone_id": self._armed_transition_zone_id,
            }
        )
        self._completed_transition_zone_id = self._armed_transition_zone_id
        self._armed_transition_zone_id = None
        self._pending_mode_id = None
        self._pending_mode_wins = 0
        return result

    def _refine_centers(
        self,
        feature,
        mask,
        centers,
        radius_px,
        source,
        mode_ids=None,
        apply_score=True,
    ):
        hypotheses = []
        for center, route_state_index in centers:
            selected_modes = mode_ids or self.atlas.localizers.keys()
            for mode_id in selected_modes:
                localizer = self.atlas.localizers[mode_id]
                match = self._observe_one_mode(
                    localizer, feature, mask, center, radius_px
                )
                if not match.get("valid"):
                    continue
                offset = match["best_offset_canonical_xy"]
                hypotheses.append(
                    {
                        "x": float(center[0]) + float(offset[0]),
                        "y": float(center[1]) + float(offset[1]),
                        "score": float(match["score"]),
                        "mode_id": str(mode_id),
                        "route_state_index": route_state_index,
                    }
                )
        if not hypotheses:
            return TraceResult(False, None, None, 0.0, 0.0, source)
        hypotheses.sort(key=lambda item: item["score"], reverse=True)
        best = hypotheses[0]
        distinct_scores = [
            item["score"]
            for item in hypotheses[1:]
            if math.hypot(item["x"] - best["x"], item["y"] - best["y"])
            >= 8.0
        ]
        second = max(distinct_scores, default=0.0)
        valid = bool(not apply_score or best["score"] >= self.score_min)
        return TraceResult(
            valid,
            best["x"],
            best["y"],
            best["score"],
            best["score"] - second,
            source,
            best["route_state_index"],
            best["mode_id"],
            measurement_accepted=valid,
            primary_candidate_produced=True,
            primary_measurement_accepted=valid,
        )

    def _observe_one_mode(self, localizer, feature, mask, center, radius_px):
        if self.local_matcher == "ccorr_normed":
            return self.atlas._observe_one_mode(
                localizer, feature, mask, center, radius_px
            )
        height, width = feature.shape[:2]
        center_x, center_y = localizer._localization_xy(center)
        left = int(round(center_x - width / 2.0))
        top = int(round(center_y - height / 2.0))
        right = left + width
        bottom = top + height
        if (
            left < 0
            or top < 0
            or right > localizer.map_gradient.shape[1]
            or bottom > localizer.map_gradient.shape[0]
        ):
            return {
                "valid": False,
                "score": 0.0,
                "coverage_fraction": 0.0,
                "reason": "insufficient-local-map-area",
            }
        map_patch = localizer.map_gradient[top:bottom, left:right]
        weight = mask.astype(np.float32) / 255.0
        window = cv2.createHanningWindow((width, height), cv2.CV_32F) * weight
        shift, response = cv2.phaseCorrelate(
            map_patch.astype(np.float32) * weight,
            feature.astype(np.float32) * weight,
            window,
        )
        # phaseCorrelate(map patch, new observation) reports map content moving
        # opposite to the player-centered observation.  Negating it gives the
        # new map center relative to the prior center.
        match_center_local = (center_x - shift[0], center_y - shift[1])
        match_center_canonical = localizer._original_xy(match_center_local)
        offset = (
            float(match_center_canonical[0] - center[0]),
            float(match_center_canonical[1] - center[1]),
        )
        scale_x = math.hypot(
            localizer.original_to_localization[0, 0],
            localizer.original_to_localization[1, 0],
        )
        scale_y = math.hypot(
            localizer.original_to_localization[0, 1],
            localizer.original_to_localization[1, 1],
        )
        local_radius = max(4.0, float(radius_px) * (scale_x + scale_y) / 2.0)
        match_left = int(round(match_center_local[0] - width / 2.0))
        match_top = int(round(match_center_local[1] - height / 2.0))
        if (
            match_left < 0
            or match_top < 0
            or match_left + width > localizer.coverage.shape[1]
            or match_top + height > localizer.coverage.shape[0]
        ):
            coverage_fraction = 0.0
        else:
            coverage_patch = localizer.coverage[
                match_top : match_top + height,
                match_left : match_left + width,
            ]
            selected = mask > 0
            coverage_fraction = float(
                np.mean(coverage_patch[selected] > 0)
                if np.any(selected)
                else 0.0
            )
        shift_distance = math.hypot(float(shift[0]), float(shift[1]))
        valid = bool(
            math.isfinite(response)
            and response >= 0.0
            and shift_distance <= local_radius
            and coverage_fraction >= 0.75
        )
        return {
            "valid": valid,
            "score": max(0.0, float(response)) if valid else 0.0,
            "coverage_fraction": coverage_fraction,
            "best_offset_canonical_xy": list(offset),
            "reason": None if valid else "phase-correlation-outside-local-support",
        }

    def _route_refine(
        self, observation, feature, mask, top_k: int, *, apply_score=True
    ):
        descriptor = describe_minimap(observation, mask)
        candidates = self._route_candidates(descriptor, top_k)
        centers = [
            (item["state"]["canonical_xy"], int(item["state_index"]))
            for item in candidates
        ]
        return self._refine_centers(
            feature,
            mask,
            centers,
            self.recovery_radius_px,
            "route_recovery",
            apply_score=apply_score,
        )

    def track(self, observation, mask, session_time_ns=None) -> TraceResult:
        # Keep the atlas's canonical gradient representation intact for any
        # independent global initialization performed before the first frame.
        self._configure_correlation_feature()
        if self.variant == "route_descriptor":
            candidates = self._route_candidates(
                describe_minimap(observation, mask), top_k=2
            )
            if not candidates:
                return TraceResult(False, None, None, 0.0, 0.0, "route_descriptor")
            best = candidates[0]
            second = candidates[1]["score"] if len(candidates) > 1 else -1.0
            margin = float(best["score"] - second)
            valid = bool(best["score"] >= 0.25 and margin >= 0.015)
            state = best["state"]
            result = TraceResult(
                valid,
                float(state["canonical_xy"][0]) if valid else None,
                float(state["canonical_xy"][1]) if valid else None,
                float(best["score"]),
                margin,
                "route_descriptor",
                int(best["state_index"]),
                str(state["mode_id"]),
                measurement_accepted=valid,
                primary_candidate_produced=True,
                primary_measurement_accepted=valid,
            )
        else:
            feature = _correlation_feature(
                observation, self.correlation_feature
            )
            if self.variant == "route_refine_top1":
                result = self._route_refine(observation, feature, mask, 1)
            elif self.variant == "route_refine_top3":
                result = self._route_refine(observation, feature, mask, 3)
            else:
                result = TraceResult(False, None, None, 0.0, 0.0, "local")
                if self.previous_xy is not None:
                    if (
                        self.mode_policy == "sticky"
                        and self.previous_mode_id in self.atlas.localizers
                    ):
                        mode_ids = [self.previous_mode_id]
                    elif self.mode_policy == "transition_zone":
                        mode_ids = self._transition_zone_mode_ids()
                    else:
                        mode_ids = None
                    result = self._refine_centers(
                        feature,
                        mask,
                        [(self.previous_xy, None)],
                        self.local_radius_px,
                        "local",
                        mode_ids=mode_ids,
                        apply_score=self.variant != "local_primary_gated",
                    )
                if not result.valid and not (
                    self.variant == "local_primary_gated"
                    and self.previous_xy is not None
                ):
                    result = self._route_refine(
                        observation,
                        feature,
                        mask,
                        3,
                        apply_score=self.variant != "local_primary_gated",
                    )
                elif not result.valid and self.previous_xy is not None:
                    result = TraceResult(
                        True,
                        self.previous_xy[0],
                        self.previous_xy[1],
                        result.score,
                        result.margin,
                        "primary_rejection_hold",
                        measurement_accepted=False,
                        primary_candidate_produced=False,
                        primary_measurement_accepted=False,
                    )
        result = self._confirm_transition_mode(result)
        if (
            self.variant == "local_primary_gated"
            and result.valid
            and result.measurement_accepted
            and result.score < self.score_min
        ):
            if self.previous_xy is None:
                result = TraceResult(
                    False,
                    None,
                    None,
                    result.score,
                    result.margin,
                    "initialization_rejected_by_final_gate",
                    result.route_state_index,
                    result.mode_id,
                    measurement_accepted=False,
                    primary_candidate_produced=True,
                    primary_measurement_accepted=False,
                    final_gate_rejected=True,
                )
            else:
                result = TraceResult(
                    True,
                    self.previous_xy[0],
                    self.previous_xy[1],
                    result.score,
                    result.margin,
                    "confidence_hold",
                    result.route_state_index,
                    result.mode_id,
                    measurement_accepted=False,
                    primary_candidate_produced=True,
                    primary_measurement_accepted=False,
                    final_gate_rejected=True,
                )
        if (
            self.variant in ("continuous_gated", "local_primary_gated")
            and result.valid
            and result.measurement_accepted
            and self.previous_xy is not None
            and self.previous_time_ns is not None
            and session_time_ns is not None
        ):
            elapsed_s = max(
                0.0, (int(session_time_ns) - int(self.previous_time_ns)) / 1.0e9
            )
            maximum_step = max(6.0, self.continuity_speed_limit_px_s * elapsed_s)
            step = math.hypot(
                float(result.x) - self.previous_xy[0],
                float(result.y) - self.previous_xy[1],
            )
            if step > maximum_step:
                result = TraceResult(
                    True,
                    self.previous_xy[0],
                    self.previous_xy[1],
                    result.score,
                    result.margin,
                    "continuity_hold",
                    result.route_state_index,
                    result.mode_id,
                    measurement_accepted=False,
                    primary_candidate_produced=True,
                    primary_measurement_accepted=True,
                    final_gate_rejected=True,
                )
        if result.valid and result.measurement_accepted:
            self.rejection_streak = 0
            self.global_recovery_hypotheses = []
        else:
            self.rejection_streak += 1
        if (
            self.recovery_policy == "global_consensus"
            and self.rejection_streak >= 6
        ):
            fix = self.atlas.localize(observation, mask)
            if fix.valid:
                mode_id = (
                    ((fix.diagnostics or {}).get("map_layer") or {}).get(
                        "selected_mode_id"
                    )
                )
                hypothesis = {
                    "x": float(fix.x),
                    "y": float(fix.y),
                    "score": float(fix.score),
                    "margin": float(fix.margin),
                    "mode_id": mode_id,
                }
                if self.global_recovery_hypotheses:
                    previous = self.global_recovery_hypotheses[-1]
                    agrees = (
                        math.hypot(
                            hypothesis["x"] - previous["x"],
                            hypothesis["y"] - previous["y"],
                        )
                        <= 30.0
                        and (
                            not hypothesis["mode_id"]
                            or not previous["mode_id"]
                            or hypothesis["mode_id"] == previous["mode_id"]
                        )
                    )
                    if not agrees:
                        self.global_recovery_hypotheses = []
                self.global_recovery_hypotheses.append(hypothesis)
                if len(self.global_recovery_hypotheses) >= 2:
                    rows = self.global_recovery_hypotheses[-2:]
                    result = TraceResult(
                        True,
                        float(np.mean([item["x"] for item in rows])),
                        float(np.mean([item["y"] for item in rows])),
                        float(np.mean([item["score"] for item in rows])),
                        float(np.mean([item["margin"] for item in rows])),
                        "global_recovery",
                        mode_id=rows[-1]["mode_id"],
                        measurement_accepted=True,
                    )
                    self.rejection_streak = 0
                    self.global_recovery_hypotheses = []
        if result.valid:
            if result.measurement_accepted:
                self.previous_xy = (float(result.x), float(result.y))
                self.previous_mode_id = result.mode_id
            if session_time_ns is not None and (
                result.measurement_accepted
                or self.continuity_clock == "frame"
            ):
                self.previous_time_ns = int(session_time_ns)
        return result


def _attach_reference_errors(rows, reference_package, session_id) -> None:
    if reference_package is None:
        return
    source = reference_package.manifest.get("source_session") or {}
    if str(source.get("session_id")) != str(session_id):
        return
    states = reference_package.states
    times = np.asarray([int(item["session_time_ns"]) for item in states], np.float64)
    xs = np.asarray([float(item["canonical_xy"][0]) for item in states])
    ys = np.asarray([float(item["canonical_xy"][1]) for item in states])
    modes = [str(item.get("mode_id") or "unknown") for item in states]
    reference_rate_hz = float(
        reference_package.manifest.get("reference_rate_hz") or 5.0
    )
    nominal_interval_ns = 1.0e9 / max(reference_rate_hz, 1.0e-6)
    maximum_bracket_gap_ns = 1.5 * nominal_interval_ns
    for row in rows:
        row["reference_error_px"] = None
        row["reference_mode_id"] = None
        if not row["valid"]:
            continue
        timestamp = float(row["session_time_ns"])
        insertion = int(np.searchsorted(times, timestamp))
        if insertion < len(times) and times[insertion] == timestamp:
            reference = np.asarray([xs[insertion], ys[insertion]])
            reference_mode_id = modes[insertion]
        elif insertion == 0 or insertion >= len(times):
            continue
        else:
            left = insertion - 1
            right = insertion
            gap_ns = float(times[right] - times[left])
            if (
                gap_ns > maximum_bracket_gap_ns
                or modes[left] != modes[right]
            ):
                continue
            fraction = (timestamp - times[left]) / max(gap_ns, 1.0)
            reference = np.asarray(
                [
                    xs[left] + fraction * (xs[right] - xs[left]),
                    ys[left] + fraction * (ys[right] - ys[left]),
                ]
            )
            reference_mode_id = modes[left]
        row["reference_error_px"] = float(
            np.linalg.norm(np.asarray([row["x"], row["y"]]) - reference)
        )
        row["reference_mode_id"] = reference_mode_id


def _reference_mode_metrics(rows):
    supported = [
        row
        for row in rows
        if row.get("reference_mode_id") is not None and row.get("valid")
    ]
    if not supported:
        return None
    mismatches = [
        row
        for row in supported
        if str(row.get("mode_id")) != str(row["reference_mode_id"])
    ]
    mismatch_flags = [
        str(row.get("mode_id")) != str(row["reference_mode_id"])
        for row in supported
    ]
    return {
        "role": "offline-sparse-reference-mode-agreement",
        "sample_count": len(supported),
        "mismatch_count": len(mismatches),
        "mismatch_rate": len(mismatches) / float(len(supported)),
        "agreement_rate": 1.0 - len(mismatches) / float(len(supported)),
        "mismatch_episodes": _loss_metrics(
            [not value for value in mismatch_flags]
        ),
    }


def _reference_errors(rows, reference_package, session_id, *, fresh_only=False):
    if reference_package is None:
        return None
    source = reference_package.manifest.get("source_session") or {}
    if str(source.get("session_id")) != str(session_id):
        return None
    errors = []
    for row in rows:
        if not row["valid"] or (fresh_only and not row["measurement_accepted"]):
            continue
        error = row.get("reference_error_px")
        if error is not None:
            errors.append(float(error))
    if not errors:
        return None
    values = np.asarray(errors, dtype=np.float64)
    return {
        "role": "offline-sparse-map-localization-reference",
        "sample_count": len(errors),
        "rmse_px": float(math.sqrt(float(np.mean(values * values)))),
        "mean_px": float(np.mean(values)),
        "median_px": float(np.median(values)),
        "p95_px": float(np.percentile(values, 95)),
        "max_px": float(np.max(values)),
    }


def _supported_interval_metrics(rows, reference_package, session_id, demonstrated):
    if reference_package is None:
        return None
    source = reference_package.manifest.get("source_session") or {}
    if str(source.get("session_id")) != str(session_id):
        return None
    states = reference_package.states
    if not states:
        return None
    start_ns = int(states[0]["session_time_ns"])
    end_ns = int(states[-1]["session_time_ns"])
    selected = [
        row for row in rows if start_ns <= int(row["session_time_ns"]) <= end_ns
    ]
    valid = [row for row in selected if row["valid"]]
    points = [[row["x"], row["y"]] for row in valid]
    return {
        "role": "post-run-map-supported-interval-review",
        "feeds_tracker": False,
        "start_session_time_ns": start_ns,
        "end_session_time_ns": end_ns,
        "frame_count": len(selected),
        "continuity": _loss_metrics([row["valid"] for row in selected]),
        "visual_measurement_continuity": _loss_metrics(
            [row["measurement_accepted"] for row in selected]
        ),
        "algorithm_latency_ms": _percentiles(
            [row["algorithm_elapsed_ms"] for row in selected]
        ),
        "route_compliance": route_similarity_report(
            points,
            demonstrated,
            float(reference_package.manifest.get("corridor_radius_px") or 35.0),
        ),
    }


def benchmark_session(
    session_path: Path,
    package_path: Path,
    atlas_path: Path,
    minimap_config: dict,
    minimap_calibration: dict,
    variant: str,
    *,
    score_min: float = 0.55,
    local_radius_px: float = 18.0,
    correlation_feature: str = "gradient",
    mode_policy: str = "all",
    continuity_clock: str = "frame",
    continuity_speed_multiplier: float = 4.0,
    recovery_policy: str = "route",
    initialization: str = "route",
    local_matcher: str = "ccorr_normed",
    transition_zone_radius_floor_px: float = 12.0,
    transition_exit_radius_px: float = 40.0,
    transition_confirmation_count: int = 2,
    reference_package_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> dict:
    reader = SessionReader(Path(session_path))
    records = reader.frames_by_stream["main"]
    package = RouteTrackingPackage(Path(package_path))
    reference_package = (
        RouteTrackingPackage(Path(reference_package_path))
        if reference_package_path
        else None
    )
    extractor = MinimapExtractor(minimap_config["crop_xywh"], minimap_calibration)
    atlas = LayeredGlobalLocalizer(Path(atlas_path))
    tracer = CausalRouteTracer(
        package,
        atlas,
        variant,
        score_min=score_min,
        local_radius_px=local_radius_px,
        correlation_feature=correlation_feature,
        mode_policy=mode_policy,
        continuity_clock=continuity_clock,
        continuity_speed_multiplier=continuity_speed_multiplier,
        recovery_policy=recovery_policy,
        local_matcher=local_matcher,
        transition_zone_radius_floor_px=transition_zone_radius_floor_px,
        transition_exit_radius_px=transition_exit_radius_px,
        transition_confirmation_count=transition_confirmation_count,
    )
    if initialization not in INITIALIZATION_POLICIES:
        raise ValueError("Unknown initialization policy: {}".format(initialization))
    capture = cv2.VideoCapture(str(reader.video_path("main")))
    if not capture.isOpened():
        atlas.close()
        raise RuntimeError("Could not open video: {}".format(reader.video_path("main")))
    rows = []
    decode_times = []
    extraction_times = []
    initialization_result = None
    known_start = None
    if initialization == "known_start":
        if reference_package is None or not reference_package.states:
            raise ValueError("known_start initialization needs a reference package")
        state = reference_package.states[0]
        known_start = {
            "session_time_ns": int(state["session_time_ns"]),
            "x": float(state["canonical_xy"][0]),
            "y": float(state["canonical_xy"][1]),
            "mode_id": str(state["mode_id"]),
        }
    try:
        for record in records:
            decode_started = time.perf_counter_ns()
            ok, frame = capture.read()
            decode_elapsed_ms = (
                time.perf_counter_ns() - decode_started
            ) / 1.0e6
            decode_times.append(decode_elapsed_ms)
            if not ok:
                raise RuntimeError(
                    "Video ended before frame {}".format(record["frame_index"])
                )
            extraction_started = time.perf_counter_ns()
            observation, mask = extractor.extract(frame)
            extraction_elapsed_ms = (
                time.perf_counter_ns() - extraction_started
            ) / 1.0e6
            extraction_times.append(extraction_elapsed_ms)
            if (
                known_start is not None
                and initialization_result is None
                and int(record["session_time_ns"]) < known_start["session_time_ns"]
            ):
                continue
            known_start_initializing = bool(
                known_start is not None and initialization_result is None
            )
            if known_start_initializing:
                tracer.previous_xy = (known_start["x"], known_start["y"])
                tracer.previous_time_ns = known_start["session_time_ns"]
                tracer.previous_mode_id = known_start["mode_id"]
                initialization_result = {
                    "valid": True,
                    "elapsed_ms": 0.0,
                    "x": known_start["x"],
                    "y": known_start["y"],
                    "mode_id": known_start["mode_id"],
                    "source": "evaluator_declared_known_start",
                    "feeds_tracker_once": True,
                    "reference_future_positions_used": False,
                }
            if initialization == "global" and initialization_result is None:
                initialization_started = time.perf_counter_ns()
                fix = atlas.localize(observation, mask)
                initialization_result = {
                    "valid": bool(fix.valid),
                    "elapsed_ms": (
                        time.perf_counter_ns() - initialization_started
                    )
                    / 1.0e6,
                    "score": float(fix.score),
                    "margin": float(fix.margin),
                    "x": float(fix.x),
                    "y": float(fix.y),
                    "mode_id": (
                        ((fix.diagnostics or {}).get("map_layer") or {}).get(
                            "selected_mode_id"
                        )
                    ),
                    "rejection_reasons": list(fix.rejection_reasons),
                    "feeds_tracker_once": bool(fix.valid),
                }
                if fix.valid:
                    tracer.previous_xy = (float(fix.x), float(fix.y))
                    tracer.previous_mode_id = initialization_result["mode_id"]
                else:
                    raise RuntimeError(
                        "Independent global initialization failed: {}".format(
                            ",".join(fix.rejection_reasons)
                        )
                    )
            started = time.perf_counter_ns()
            route_initializing = bool(
                initialization == "route" and initialization_result is None
            )
            result = tracer.track(
                observation, mask, session_time_ns=record["session_time_ns"]
            )
            elapsed_ms = (time.perf_counter_ns() - started) / 1.0e6
            if route_initializing:
                initialization_result = {
                    "valid": bool(result.valid and result.measurement_accepted),
                    "elapsed_ms": elapsed_ms,
                    "score": float(result.score),
                    "margin": float(result.margin),
                    "x": result.x,
                    "y": result.y,
                    "mode_id": result.mode_id,
                    "source": result.source,
                    "feeds_tracker_once": bool(
                        result.valid and result.measurement_accepted
                    ),
                }
            core_elapsed_ms = extraction_elapsed_ms + elapsed_ms
            end_to_end_serial_elapsed_ms = decode_elapsed_ms + core_elapsed_ms
            rows.append(
                {
                    "frame_index": int(record["frame_index"]),
                    "session_time_ns": int(record["session_time_ns"]),
                    "valid": bool(result.valid),
                    "measurement_accepted": bool(
                        result.valid and result.measurement_accepted
                    ),
                    "primary_candidate_produced": bool(
                        result.primary_candidate_produced
                    ),
                    "primary_measurement_accepted": bool(
                        result.primary_measurement_accepted
                    ),
                    "final_gate_rejected": bool(result.final_gate_rejected),
                    "x": result.x,
                    "y": result.y,
                    "score": float(result.score),
                    "margin": float(result.margin),
                    "source": result.source,
                    "route_state_index": result.route_state_index,
                    "mode_id": result.mode_id,
                    "algorithm_elapsed_ms": elapsed_ms,
                    "decode_elapsed_ms": decode_elapsed_ms,
                    "extraction_elapsed_ms": extraction_elapsed_ms,
                    "localization_core_elapsed_ms": core_elapsed_ms,
                    "end_to_end_serial_elapsed_ms": end_to_end_serial_elapsed_ms,
                    "initialization_frame": bool(
                        route_initializing or known_start_initializing
                    ),
                }
            )
    finally:
        capture.release()
        atlas.close()

    _attach_reference_errors(
        rows, reference_package, reader.manifest.get("session_id")
    )

    valid_rows = [row for row in rows if row["valid"]]
    points = [[row["x"], row["y"]] for row in valid_rows]
    demonstrated = [state["canonical_xy"] for state in package.states]
    adjacent_jumps = []
    for first, second in zip(rows, rows[1:]):
        if first["valid"] and second["valid"]:
            adjacent_jumps.append(
                math.hypot(second["x"] - first["x"], second["y"] - first["y"])
            )
    algorithm_times = [row["algorithm_elapsed_ms"] for row in rows]
    core_times = [row["localization_core_elapsed_ms"] for row in rows]
    end_to_end_times = [row["end_to_end_serial_elapsed_ms"] for row in rows]
    fresh_flags = np.asarray(
        [row["measurement_accepted"] for row in rows], dtype=bool
    )
    core_time_array = np.asarray(core_times, dtype=np.float64)
    end_to_end_time_array = np.asarray(end_to_end_times, dtype=np.float64)
    source_counts = {}
    for row in rows:
        source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1
    report = {
        "schema_version": "1.0",
        "experiment": "causal-route-tracer-poc",
        "variant": variant,
        "causal_constraints": {
            "demo_timing_used": False,
            "demo_motion_vector_used": False,
            "demo_heading_used": False,
            "route_pose_used": variant == "route_descriptor",
            "known_start_reference_feeds_tracker_once": bool(
                initialization == "known_start"
            ),
            "reference_future_positions_feed_tracker": False,
            "cold_start_localization_measured": bool(
                initialization != "known_start"
            ),
            "route_role": (
                "pose baseline only"
                if variant == "route_descriptor"
                else "visual bounded-search proposal only"
            ),
        },
        "session": {
            "path": str(Path(session_path)),
            "session_id": reader.manifest.get("session_id"),
            "frame_count": len(rows),
        },
        "route_package": str(Path(package_path)),
        "reference_package": (
            str(Path(reference_package_path))
            if reference_package_path is not None
            else None
        ),
        "atlas": str(Path(atlas_path)),
        "parameters": {
            "score_min": float(score_min),
            "recovery_radius_px": tracer.recovery_radius_px,
            "local_radius_px": tracer.local_radius_px,
            "continuity_speed_limit_px_s": tracer.continuity_speed_limit_px_s,
            "continuity_speed_multiplier": tracer.continuity_speed_multiplier,
            "recovery_policy": tracer.recovery_policy,
            "correlation_feature": tracer.correlation_feature,
            "mode_policy": tracer.mode_policy,
            "continuity_clock": tracer.continuity_clock,
            "initialization": initialization,
            "local_matcher": tracer.local_matcher,
            "transition_zone_radius_floor_px": (
                tracer.transition_zone_radius_floor_px
            ),
            "transition_exit_radius_px": tracer.transition_exit_radius_px,
            "transition_confirmation_count": tracer.transition_confirmation_count,
        },
        "initialization": initialization_result,
        "algorithm_latency_ms": _percentiles(algorithm_times),
        "algorithm_throughput_fps": (
            1000.0 / float(np.mean(algorithm_times)) if algorithm_times else 0.0
        ),
        "decode_latency_ms": _percentiles(decode_times),
        "extraction_latency_ms": _percentiles(extraction_times),
        "localization_core_latency_ms": _percentiles(core_times),
        "end_to_end_serial_latency_ms": _percentiles(end_to_end_times),
        "continuity": _loss_metrics([row["valid"] for row in rows]),
        "visual_measurement_continuity": _loss_metrics(
            [row["measurement_accepted"] for row in rows]
        ),
        "two_layer_contract": {
            "layer_1": "one current-frame local map measurement",
            "layer_2": "one final physical-continuity rejection gate",
            "route_or_global_initialization_is_not_a_per-frame_fallback": True,
            "primary_candidate_produced_rate": float(
                np.mean([row["primary_candidate_produced"] for row in rows])
            ),
            "primary_measurement_accepted_rate": float(
                np.mean([row["primary_measurement_accepted"] for row in rows])
            ),
            "final_gate_rejection_rate": float(
                np.mean([row["final_gate_rejected"] for row in rows])
            ),
            "fresh_measurement_accepted_rate": float(np.mean(fresh_flags)),
            "fresh_within_33_3ms_rate": float(
                np.mean(fresh_flags & (core_time_array <= (1000.0 / 30.0)))
            ),
            "fresh_within_66_7ms_rate": float(
                np.mean(fresh_flags & (core_time_array <= (1000.0 / 15.0)))
            ),
            "fresh_e2e_within_33_3ms_rate": float(
                np.mean(
                    fresh_flags
                    & (end_to_end_time_array <= (1000.0 / 30.0))
                )
            ),
            "fresh_e2e_within_66_7ms_rate": float(
                np.mean(
                    fresh_flags
                    & (end_to_end_time_array <= (1000.0 / 15.0))
                )
            ),
            "held_states_count_as_fresh": False,
        },
        "reference_mode_agreement": _reference_mode_metrics(rows),
        "transition_switches": list(tracer.transition_switches),
        "source_frame_counts": source_counts,
        "adjacent_pose_jump_px": _percentiles(adjacent_jumps),
        "route_compliance": route_similarity_report(
            points,
            demonstrated,
            float(package.manifest.get("corridor_radius_px") or 35.0),
        ),
        "reference_pose_error": _reference_errors(
            rows, reference_package, reader.manifest.get("session_id")
        ),
        "fresh_reference_pose_error": _reference_errors(
            rows,
            reference_package,
            reader.manifest.get("session_id"),
            fresh_only=True,
        ),
        "map_supported_interval": _supported_interval_metrics(
            rows,
            reference_package,
            reader.manifest.get("session_id"),
            demonstrated,
        ),
        "acceptance": {
            "target_algorithm_fps": 30.0,
            "target_frame_budget_ms": 1000.0 / 30.0,
            "meets_mean_30fps_budget": bool(
                core_times and float(np.mean(core_times)) <= 1000.0 / 30.0
            ),
            "meets_p95_30fps_budget": bool(
                core_times
                and float(np.percentile(core_times, 95)) <= 1000.0 / 30.0
            ),
        },
        "method_traceability": {
            "implementation": "poc.benchmark_route_tracer.CausalRouteTracer",
            "source_file": str(Path(__file__).resolve()),
            "source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "variant": variant,
            "local_matcher": tracer.local_matcher,
            "correlation_feature": tracer.correlation_feature,
        },
        "rows_file": "telemetry.jsonl" if output_path else None,
    }
    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / "telemetry.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        (output_path / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return report


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--route-package", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--minimap-calibration", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--score-min", type=float, default=0.55)
    parser.add_argument("--local-radius-px", type=float, default=18.0)
    parser.add_argument(
        "--correlation-feature", choices=CORRELATION_FEATURES, default="gradient"
    )
    parser.add_argument("--mode-policy", choices=MODE_POLICIES, default="all")
    parser.add_argument(
        "--initialization", choices=INITIALIZATION_POLICIES, default="route"
    )
    parser.add_argument(
        "--continuity-clock", choices=CONTINUITY_CLOCKS, default="frame"
    )
    parser.add_argument("--continuity-speed-multiplier", type=float, default=4.0)
    parser.add_argument(
        "--recovery-policy", choices=RECOVERY_POLICIES, default="route"
    )
    parser.add_argument(
        "--local-matcher", choices=LOCAL_MATCHERS, default="ccorr_normed"
    )
    parser.add_argument("--transition-zone-radius-floor-px", type=float, default=12.0)
    parser.add_argument("--transition-exit-radius-px", type=float, default=40.0)
    parser.add_argument("--transition-confirmation-count", type=int, default=2)
    parser.add_argument("--reference-package", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    calibration = json.loads(args.minimap_calibration.read_text(encoding="utf-8"))
    report = benchmark_session(
        args.session,
        args.route_package,
        args.atlas,
        profile["minimap_calibration"],
        calibration,
        args.variant,
        score_min=args.score_min,
        local_radius_px=args.local_radius_px,
        correlation_feature=args.correlation_feature,
        mode_policy=args.mode_policy,
        initialization=args.initialization,
        continuity_clock=args.continuity_clock,
        continuity_speed_multiplier=args.continuity_speed_multiplier,
        recovery_policy=args.recovery_policy,
        local_matcher=args.local_matcher,
        transition_zone_radius_floor_px=args.transition_zone_radius_floor_px,
        transition_exit_radius_px=args.transition_exit_radius_px,
        transition_confirmation_count=args.transition_confirmation_count,
        reference_package_path=args.reference_package,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
