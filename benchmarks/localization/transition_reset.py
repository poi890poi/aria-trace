"""One-variable experiment: reset template search after a confirmed layer switch."""

from contextlib import contextmanager
from aria_trace.services.tracking.runtime import TwoRateRealtimeTracker as Runtime
from aria_trace.services.localization.route.tracker import RouteVisualTracker


@contextmanager
def installed(reset=True, hold=False):
    original = Runtime._consume_representation_observation
    original_track = RouteVisualTracker.track
    def consume(self, timestamp_ns):
        previous = self._last_map_transition
        result = original(self, timestamp_ns)
        if reset and self._last_map_transition is not previous and self.route_visual_tracker is not None:
            self.route_visual_tracker.previous_time_ns = None
        if hold and self.route_visual_tracker is not None:
            observation = self._last_representation_observation or {}
            controller = observation.get("controller") or {}
            scores = sorted((observation.get("likelihoods") or {}).values(), reverse=True)
            self.route_visual_tracker._hold_ambiguous_scale = bool(
                controller.get("transition_armed") and len(scores) >= 2
                and scores[0]-scores[1] < self.transition_controller.minimum_margin)
        return result
    def track(self, *args, **kwargs):
        if getattr(self, "_hold_ambiguous_scale", False) and self.previous_xy is not None:
            return {"valid": False, "measurement_accepted": False, "pose_available": True,
                    "held": True, "x": self.previous_xy[0], "y": self.previous_xy[1],
                    "decision": "held:ambiguous-map-scale", "transition_waiting": True,
                    "route_role": "none" if self.package is None else "bounded-search-proposal-only"}
        return original_track(self, *args, **kwargs)
    try:
        Runtime._consume_representation_observation = consume
        RouteVisualTracker.track = track
        yield
    finally:
        Runtime._consume_representation_observation = original
        RouteVisualTracker.track = original_track
