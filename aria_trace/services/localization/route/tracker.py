"""Route-only localization and continuous progress estimation."""

import math
from typing import Optional

import numpy as np

from replay.route_tracking import RouteTrackingPackage, describe_minimap

from aria_trace.services.tracking.runtime import GlobalFix


def _angle_difference(first: float, second: float) -> float:
    return ((float(first) - float(second) + 180.0) % 360.0) - 180.0


class RouteFinishGate:
    """Confirm arrival from consecutive accepted visual map measurements.

    The demonstrated endpoint defines only the finish region. Route timing,
    progress, input, and state-index estimates never count as arrival evidence.
    The gate first observes the player outside that region, preventing an
    immediate stop when a lap starts and finishes at the same location.
    """

    def __init__(
        self,
        package: RouteTrackingPackage,
        *,
        radius_px: Optional[float] = None,
        consecutive_measurements: int = 3,
    ) -> None:
        final_state = package.states[-1]
        self.endpoint_xy = tuple(
            float(value) for value in final_state["canonical_xy"]
        )
        corridor = float(package.manifest.get("corridor_radius_px") or 35.0)
        self.radius_px = float(
            radius_px
            if radius_px is not None
            else max(6.0, min(18.0, corridor * 0.4))
        )
        self.departure_radius_px = max(12.0, self.radius_px * 2.0)
        self.final_mode_id = str(final_state.get("mode_id") or "")
        self.required_measurements = max(1, int(consecutive_measurements))
        self.armed = False
        self.confirmations = 0
        self.reached = False

    def update(self, state: dict) -> dict:
        pose = state.get("pose") or {}
        route = state.get("route_tracking") or {}
        accepted = bool(
            state.get("route_tracking_fresh")
            and route.get("measurement_accepted")
            and pose.get("x") is not None
            and pose.get("y") is not None
        )
        distance = None
        mode_matches = False
        if pose.get("x") is not None and pose.get("y") is not None:
            distance = math.hypot(
                float(pose["x"]) - self.endpoint_xy[0],
                float(pose["y"]) - self.endpoint_xy[1],
            )
            active_mode = str(
                state.get("active_map_mode_id")
                or route.get("active_mode_id")
                or ""
            )
            mode_matches = not self.final_mode_id or active_mode == self.final_mode_id
        if accepted and distance is not None:
            if not self.armed and distance > self.departure_radius_px:
                self.armed = True
            inside = bool(
                self.armed and mode_matches and distance <= self.radius_px
            )
            self.confirmations = self.confirmations + 1 if inside else 0
            self.reached = self.confirmations >= self.required_measurements
        elif state.get("route_tracking_fresh"):
            self.confirmations = 0
        return {
            "endpoint_xy": list(self.endpoint_xy),
            "endpoint_mode_id": self.final_mode_id or None,
            "distance_px": distance,
            "radius_px": self.radius_px,
            "departure_radius_px": self.departure_radius_px,
            "armed": self.armed,
            "measurement_accepted": accepted,
            "mode_matches": mode_matches,
            "confirmations": self.confirmations,
            "required_confirmations": self.required_measurements,
            "reached": self.reached,
            "evidence_policy": "consecutive-current-frame-map-measurements",
        }


class RouteGlobalLocalizer:
    """Fast initial/recovery localization restricted to demonstrated states."""

    supports_bounded_search = True

    def __init__(
        self,
        package: RouteTrackingPackage,
        *,
        score_min: float = 0.25,
        margin_min: float = 0.015,
    ) -> None:
        self.package = package
        self.score_min = float(score_min)
        self.margin_min = float(margin_min)
        self.previous_state_index = None
        self.active_mode_id = None
        self.last_selected_state_index = None

    def close(self) -> None:
        return None

    def set_active_mode(self, mode_id: Optional[str]) -> None:
        self.active_mode_id = mode_id

    def localize(
        self,
        observation,
        mask,
        yaw_prior_deg=None,
        search_center_xy=None,
        search_radius_px=None,
    ) -> GlobalFix:
        descriptor = describe_minimap(observation, mask)
        candidates = self.package.candidates(
            descriptor,
            previous_state_index=self.previous_state_index,
            backward_states=3,
            forward_states=18,
            mode_id=self.active_mode_id,
            top_k=3,
        )
        if not candidates:
            return GlobalFix(
                0.0,
                0.0,
                float(yaw_prior_deg or 0.0),
                1.0,
                0.0,
                0.0,
                0.0,
                valid=False,
                rejection_reasons=("no-route-candidates",),
            )
        best = candidates[0]
        second_score = candidates[1]["score"] if len(candidates) > 1 else -1.0
        margin = float(best["score"] - second_score)
        state = best["state"]
        valid = bool(best["score"] >= self.score_min and margin >= self.margin_min)
        if valid:
            self.previous_state_index = int(best["state_index"])
            self.last_selected_state_index = self.previous_state_index
        map_alignment = state.get("map_alignment_deg")
        if map_alignment is None:
            map_alignment = yaw_prior_deg or 0.0
        return GlobalFix(
            float(state["canonical_xy"][0]),
            float(state["canonical_xy"][1]),
            float(map_alignment),
            float(state.get("map_scale") or 1.0),
            float(best["score"]),
            margin,
            0.0,
            valid=valid,
            rejection_reasons=()
            if valid
            else ("ambiguous-route-template",),
            alternatives=tuple(
                {
                    "state_index": int(item["state_index"]),
                    "score": float(item["score"]),
                    "x": float(item["state"]["canonical_xy"][0]),
                    "y": float(item["state"]["canonical_xy"][1]),
                }
                for item in candidates[1:]
            ),
            search_area_fraction=len(candidates) / float(len(self.package.states)),
            diagnostics={
                "route": {
                    "route_id": self.package.manifest["route_id"],
                    "selected_state_index": int(best["state_index"]),
                    "mode_id": state["mode_id"],
                    "candidate_count": len(candidates),
                }
            },
        )


class RouteCandidateAdvisor:
    """Suggest bounded atlas searches without producing or accepting a pose.

    Adjacent demonstrated states often describe the same physical neighborhood.
    Grouping them before selecting a search window avoids treating neighboring
    frames as competing pose hypotheses.  The returned window is only a cache/
    search-order hint; the ordinary atlas localizer still measures and validates
    the actual fix.
    """

    def __init__(
        self,
        package: RouteTrackingPackage,
        *,
        top_k: int = 32,
        adjacent_state_gap: int = 4,
        spatial_cluster_radius_px: float = 45.0,
        search_padding_px: float = 90.0,
    ) -> None:
        self.package = package
        self.top_k = max(4, int(top_k))
        self.adjacent_state_gap = max(1, int(adjacent_state_gap))
        self.spatial_cluster_radius_px = max(
            1.0, float(spatial_cluster_radius_px)
        )
        self.search_padding_px = max(20.0, float(search_padding_px))

    def propose(self, observation, mask) -> Optional[dict]:
        descriptor = describe_minimap(observation, mask)
        candidates = self.package.candidates(
            descriptor,
            top_k=min(self.top_k, len(self.package.states)),
        )
        if not candidates:
            return None
        clusters = []
        for candidate in candidates:
            index = int(candidate["state_index"])
            point = np.asarray(candidate["state"]["canonical_xy"], dtype=np.float64)
            selected = None
            for cluster in clusters:
                index_gap = min(abs(index - item) for item in cluster["indexes"])
                spatial_gap = float(np.linalg.norm(point - cluster["center"]))
                if (
                    index_gap <= self.adjacent_state_gap
                    or spatial_gap <= self.spatial_cluster_radius_px
                ):
                    selected = cluster
                    break
            if selected is None:
                selected = {
                    "indexes": [],
                    "points": [],
                    "scores": [],
                    "center": point,
                }
                clusters.append(selected)
            selected["indexes"].append(index)
            selected["points"].append(point)
            selected["scores"].append(float(candidate["score"]))
            weights = np.maximum(np.asarray(selected["scores"]), 0.0) + 1.0e-3
            selected["center"] = np.average(
                np.asarray(selected["points"]), axis=0, weights=weights
            )
        for cluster in clusters:
            ordered = sorted(cluster["scores"], reverse=True)
            # Several consistent neighboring frames outrank one isolated match,
            # but never become a localization acceptance score.
            cluster["rank_score"] = float(
                ordered[0] + 0.10 * sum(ordered[1:4])
            )
        best = max(clusters, key=lambda item: item["rank_score"])
        distances = [
            float(np.linalg.norm(point - best["center"]))
            for point in best["points"]
        ]
        radius = self.search_padding_px + (max(distances) if distances else 0.0)
        return {
            "center_xy": [float(best["center"][0]), float(best["center"][1])],
            "radius_px": float(radius),
            "cluster_state_indexes": sorted(best["indexes"]),
            "cluster_candidate_count": len(best["indexes"]),
            "candidate_count": len(candidates),
            "cluster_count": len(clusters),
            "policy": "candidate-window-only",
        }


class RouteVisualTracker:
    """Track current atlas pixels; an optional route supplies search proposals."""

    def __init__(
        self,
        package: Optional[RouteTrackingPackage],
        map_localizer,
        *,
        score_min: float = 0.55,
        local_radius_px: float = 18.0,
        recovery_radius_px: float = 55.0,
        recovery_top_k: int = 3,
    ) -> None:
        if not callable(getattr(map_localizer, "refine_near", None)):
            raise ValueError("Route visual tracking needs local atlas refinement")
        self.package = package
        self.map_localizer = map_localizer
        self.score_min = float(score_min)
        self.local_radius_px = max(4.0, float(local_radius_px))
        self.recovery_radius_px = max(
            self.local_radius_px, float(recovery_radius_px)
        )
        self.recovery_top_k = max(1, int(recovery_top_k))
        self.previous_xy = None
        self.previous_time_ns = None
        self._trained_transition = None
        motion = (package.manifest.get("motion_envelope") or {}) if package is not None else {}
        speed_p99 = (motion.get("speed_px_s") or {}).get("p99")
        self.continuity_speed_limit_px_s = max(
            120.0, 4.0 * float(speed_p99 or 0.0)
        )

    def seed(self, x: float, y: float, timestamp_ns=None) -> None:
        self.previous_xy = (float(x), float(y))
        self.previous_time_ns = (
            int(timestamp_ns) if timestamp_ns is not None else None
        )
        self._trained_transition = None

    def arm_trained_transition(
        self, source_mode_id: str, target_mode_id: str
    ) -> Optional[dict]:
        """Hold XY and select the nearest trained post-transition search anchor."""

        if self.previous_xy is None or self.package is None:
            return None
        source_mode_id = str(source_mode_id)
        target_mode_id = str(target_mode_id)
        if self._trained_transition is not None:
            pending = self._trained_transition
            if (
                pending["source_mode_id"] == source_mode_id
                and pending["target_mode_id"] == target_mode_id
            ):
                return dict(pending)
        matches = [
            item
            for item in self.package.transitions
            if str(item.get("source_mode_id")) == source_mode_id
            and str(item.get("target_mode_id")) == target_mode_id
        ]
        if not matches:
            return None

        def source_position(item):
            explicit = item.get("last_source_canonical_xy")
            if explicit is not None:
                return tuple(float(value) for value in explicit)
            index = int(
                item.get(
                    "last_source_state_index",
                    item.get("first_state_index", 0),
                )
            )
            return tuple(
                float(value) for value in self.package.states[index]["canonical_xy"]
            )

        selected = min(
            matches,
            key=lambda item: math.hypot(
                source_position(item)[0] - self.previous_xy[0],
                source_position(item)[1] - self.previous_xy[1],
            ),
        )
        target_index = int(
            selected.get(
                "first_target_state_index",
                selected.get("center_state_index"),
            )
        )
        target_xy = selected.get("first_target_canonical_xy")
        if target_xy is None:
            target_xy = self.package.states[target_index]["canonical_xy"]
        self._trained_transition = {
            "transition_index": int(selected.get("transition_index", 0)),
            "source_mode_id": source_mode_id,
            "target_mode_id": target_mode_id,
            "target_state_index": target_index,
            "target_canonical_xy": [float(value) for value in target_xy],
            "target_layer_confirmed": False,
        }
        return dict(self._trained_transition)

    def confirm_trained_transition_layer(self, target_mode_id: str) -> bool:
        pending = self._trained_transition
        if pending is None or pending["target_mode_id"] != str(target_mode_id):
            return False
        pending["target_layer_confirmed"] = True
        return True

    def cancel_trained_transition(self) -> None:
        self._trained_transition = None

    def _held_transition_result(self) -> dict:
        pending = dict(self._trained_transition or {})
        return {
            "valid": False,
            "measurement_accepted": False,
            "pose_available": self.previous_xy is not None,
            "held": self.previous_xy is not None,
            "x": self.previous_xy[0] if self.previous_xy is not None else None,
            "y": self.previous_xy[1] if self.previous_xy is not None else None,
            "score": 0.0,
            "decision": "held:trained-map-transition",
            "route_role": "trained-transition-search-proposal-only",
            "continuity_rejected": False,
            "transition_waiting": True,
            "continuity_speed_limit_px_s": self.continuity_speed_limit_px_s,
            "trained_transition": pending,
        }

    def _refine(self, observation, mask, center, radius, source, state_index=None):
        refiner = getattr(
            self.map_localizer,
            "refine_active_near",
            self.map_localizer.refine_near,
        )
        result = dict(
            refiner(
                observation,
                mask,
                center,
                search_radius_px=radius,
                score_min=self.score_min,
            )
        )
        result.update(
            {
                "source": source,
                "route_state_index": state_index,
            }
        )
        return result

    def _recover(self, observation, mask):
        if self.package is None:
            return None
        descriptor = describe_minimap(observation, mask)
        candidates = self.package.candidates(
            descriptor,
            top_k=self.recovery_top_k,
        )
        hypotheses = []
        for candidate in candidates:
            result = self._refine(
                observation,
                mask,
                candidate["state"]["canonical_xy"],
                self.recovery_radius_px,
                "route-recovery",
                int(candidate["state_index"]),
            )
            if result.get("valid"):
                hypotheses.append(result)
        if not hypotheses:
            return None
        return max(hypotheses, key=lambda item: float(item["score"]))

    def track(self, observation, mask, timestamp_ns=None) -> dict:
        result = None
        pending_transition = self._trained_transition
        if pending_transition is not None and not pending_transition[
            "target_layer_confirmed"
        ]:
            return self._held_transition_result()
        if pending_transition is not None:
            result = self._refine(
                observation,
                mask,
                pending_transition["target_canonical_xy"],
                max(
                    self.local_radius_px,
                    float(self.package.manifest.get("corridor_radius_px") or 0.0),
                ),
                "route-transition-anchor",
                int(pending_transition["target_state_index"]),
            )
        elif self.previous_xy is not None:
            result = self._refine(
                observation,
                mask,
                self.previous_xy,
                self.recovery_radius_px if self.previous_time_ns is None else self.local_radius_px,
                "continuous-local",
            )
        if pending_transition is None and (
            result is None or not result.get("valid")
        ):
            recovered = self._recover(observation, mask)
            if recovered is not None:
                result = recovered
        measurement_accepted = bool(result and result.get("valid"))
        continuity_rejected = False
        if (
            measurement_accepted
            and self.previous_xy is not None
            and self.previous_time_ns is not None
            and timestamp_ns is not None
        ):
            elapsed_s = max(
                0.0,
                (int(timestamp_ns) - self.previous_time_ns) / 1.0e9,
            )
            step_px = math.hypot(
                float(result["x"]) - self.previous_xy[0],
                float(result["y"]) - self.previous_xy[1],
            )
            step_limit_px = max(
                6.0, self.continuity_speed_limit_px_s * elapsed_s
            )
            result["measured_x"] = float(result["x"])
            result["measured_y"] = float(result["y"])
            result["continuity_step_px"] = float(step_px)
            result["continuity_step_limit_px"] = float(step_limit_px)
            if step_px > step_limit_px:
                measurement_accepted = False
                continuity_rejected = True
        if measurement_accepted:
            self.previous_xy = (float(result["x"]), float(result["y"]))
            if pending_transition is not None:
                self._trained_transition = None
        if measurement_accepted and timestamp_ns is not None:
            self.previous_time_ns = int(timestamp_ns)
        pose_available = self.previous_xy is not None
        public = dict(result or {})
        public.update(
            {
                "valid": measurement_accepted,
                "measurement_accepted": measurement_accepted,
                "pose_available": pose_available,
                "held": bool(pose_available and not measurement_accepted),
                "x": self.previous_xy[0] if pose_available else None,
                "y": self.previous_xy[1] if pose_available else None,
                "decision": (
                    "accepted-current-frame-map-pose"
                    if measurement_accepted
                    else "held:trained-transition-awaiting-visual-confirmation"
                    if pending_transition is not None
                    else "held:continuity-jump"
                    if continuity_rejected
                    else "held-no-map-measurement"
                    if pose_available
                    else "unlocalized"
                ),
                "route_role": "bounded-search-proposal-only" if self.package is not None else "none",
                "continuity_rejected": continuity_rejected,
                "transition_waiting": bool(
                    pending_transition is not None and not measurement_accepted
                ),
                "continuity_speed_limit_px_s": self.continuity_speed_limit_px_s,
                "trained_transition": (
                    dict(self._trained_transition)
                    if self._trained_transition is not None
                    else None
                ),
            }
        )
        return public


class RouteLockedStateEstimator:
    """Collapse live position onto a directed route with bounded progress."""

    def __init__(
        self,
        package: RouteTrackingPackage,
        *,
        backward_states: int = 2,
        forward_states: int = 12,
        observation_weight: float = 0.65,
    ) -> None:
        self.package = package
        self.backward_states = max(0, int(backward_states))
        self.forward_states = max(1, int(forward_states))
        self.observation_weight = float(observation_weight)
        self.corridor_radius_px = float(package.manifest["corridor_radius_px"])
        self.state_index = None
        self.last_time_ns = None
        self.active_mode_id = None

    def initialize_near(self, x: float, y: float, state_index: Optional[int] = None):
        if state_index is None:
            distances = [
                math.hypot(
                    float(state["canonical_xy"][0]) - float(x),
                    float(state["canonical_xy"][1]) - float(y),
                )
                for state in self.package.states
            ]
            state_index = int(np.argmin(distances))
        self.state_index = int(state_index)
        self.active_mode_id = self.package.states[self.state_index]["mode_id"]
        return self.package.states[self.state_index]

    @staticmethod
    def _closest_point(point, start, end):
        point = np.asarray(point, dtype=np.float64)
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        segment = end - start
        denominator = float(segment.dot(segment))
        fraction = 0.0 if denominator <= 1.0e-9 else float(
            np.clip((point - start).dot(segment) / denominator, 0.0, 1.0)
        )
        projected = start + fraction * segment
        return projected, fraction, float(np.linalg.norm(point - projected))

    def _project(self, point, center_index: int):
        left = max(0, center_index - 1)
        right = min(len(self.package.states) - 1, center_index + 1)
        choices = []
        for start_index in range(left, right):
            start = self.package.states[start_index]["canonical_xy"]
            end = self.package.states[start_index + 1]["canonical_xy"]
            projected, fraction, distance = self._closest_point(point, start, end)
            choices.append((distance, start_index, fraction, projected))
        if not choices:
            state = self.package.states[center_index]
            projected = np.asarray(state["canonical_xy"], dtype=np.float64)
            return projected, 0.0, center_index, 0.0
        distance, start_index, fraction, projected = min(choices)
        return projected, distance, start_index, fraction

    def update(
        self,
        observation,
        mask,
        predicted_xy,
        *,
        player_heading_deg: Optional[float] = None,
        timestamp_ns: Optional[int] = None,
    ) -> dict:
        if self.state_index is None:
            self.initialize_near(float(predicted_xy[0]), float(predicted_xy[1]))
        descriptor = describe_minimap(observation, mask)
        candidates = self.package.candidates(
            descriptor,
            previous_state_index=self.state_index,
            backward_states=self.backward_states,
            forward_states=self.forward_states,
            top_k=self.backward_states + self.forward_states + 1,
        )
        scored = []
        for candidate in candidates:
            state = candidate["state"]
            distance = math.hypot(
                float(state["canonical_xy"][0]) - float(predicted_xy[0]),
                float(state["canonical_xy"][1]) - float(predicted_xy[1]),
            )
            spatial = math.exp(-0.5 * (distance / max(self.corridor_radius_px, 1.0)) ** 2)
            heading = 1.0
            if player_heading_deg is not None:
                error = abs(
                    _angle_difference(player_heading_deg, state["route_heading_deg"])
                )
                heading = math.exp(-0.5 * (error / 50.0) ** 2)
            posterior = (
                self.observation_weight * max(-1.0, candidate["score"])
                + (1.0 - self.observation_weight) * spatial * heading
            )
            scored.append((posterior, candidate, distance))
        scored.sort(key=lambda item: item[0], reverse=True)
        best_posterior, best, _ = scored[0]
        second_posterior = scored[1][0] if len(scored) > 1 else -1.0
        previous_index = self.state_index
        selected_index = int(best["state_index"])
        accepted = bool(best_posterior >= 0.10)
        if accepted:
            self.state_index = selected_index
        projected, cross_track, segment_index, segment_fraction = self._project(
            predicted_xy, self.state_index
        )
        previous_mode = self.active_mode_id
        self.active_mode_id = self.package.states[self.state_index]["mode_id"]
        mode_switched = previous_mode is not None and previous_mode != self.active_mode_id
        self.last_time_ns = int(timestamp_ns) if timestamp_ns is not None else None
        start_distance = float(
            self.package.states[segment_index]["route_distance_px"]
        )
        end_distance = float(
            self.package.states[min(segment_index + 1, len(self.package.states) - 1)][
                "route_distance_px"
            ]
        )
        route_distance = start_distance + segment_fraction * (end_distance - start_distance)
        return {
            "accepted": accepted,
            "state_index": self.state_index,
            "previous_state_index": previous_index,
            "canonical_xy": projected.tolist(),
            "route_distance_px": route_distance,
            "route_progress": route_distance
            / max(float(self.package.manifest["route_length_px"]), 1.0e-6),
            "cross_track_error_px": cross_track,
            "corridor_radius_px": self.corridor_radius_px,
            "observation_score": float(best["score"]),
            "posterior_score": float(best_posterior),
            "posterior_margin": float(best_posterior - second_posterior),
            "active_mode_id": self.active_mode_id,
            "mode_switched": mode_switched,
            "reset_local_reference": mode_switched,
            "state": "route_track" if accepted else "lost_near_route",
        }
