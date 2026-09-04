"""Named, reproducible execution profiles for localization and pose tracking."""

from copy import deepcopy
from typing import Mapping, Optional


TRACKING_PROFILES = {
    "real-time": {
        "cursor_pose_core": "angular_projection_ncc_parabolic",
        "cursor_pose_method": "cascade",
        "cursor_validation_policy": "minimal",
        "cursor_interval_s": 0.0,
        "global_interval_s": 2.0,
        "representation_interval_s": 0.25,
        "temporal_pose_search": False,
        "pose_confidence_min": 0.0,
        "cursor_worker_process": True,
        "cursor_opencv_threads": 1,
        "route_map_score_min": 0.50,
    },
    "real-time-legacy": {
        "cursor_pose_core": "polygon_gaussian",
        "cursor_pose_method": "cascade",
        "cursor_validation_policy": "ambiguous",
        "cursor_interval_s": 0.05,
        "global_interval_s": 2.0,
        "representation_interval_s": 0.25,
        "temporal_pose_search": True,
        "pose_confidence_min": 0.45,
        "cursor_worker_process": True,
        "cursor_opencv_threads": 1,
        "route_map_score_min": 0.50,
    },
    "fast": {
        "cursor_pose_core": "polygon_gaussian",
        "cursor_pose_method": "cascade",
        "cursor_validation_policy": "minimal",
        "cursor_interval_s": 0.10,
        "global_interval_s": 3.0,
        "representation_interval_s": 0.35,
        "temporal_pose_search": True,
        "pose_confidence_min": 0.50,
        "cursor_worker_process": True,
        "cursor_opencv_threads": 1,
        "route_map_score_min": 0.50,
    },
    "accurate": {
        "cursor_pose_core": "polygon_gaussian",
        "cursor_pose_method": "vectorized_grid",
        "cursor_validation_policy": "full",
        "cursor_interval_s": 0.15,
        "global_interval_s": 1.5,
        "representation_interval_s": 0.15,
        "temporal_pose_search": True,
        "pose_confidence_min": 0.40,
        "cursor_worker_process": True,
        "cursor_opencv_threads": 1,
        "route_map_score_min": 0.50,
    },
    "offline": {
        "cursor_pose_core": "polygon_gaussian",
        "cursor_pose_method": "vectorized_grid",
        "cursor_validation_policy": "full",
        "cursor_interval_s": 0.0,
        "global_interval_s": 0.5,
        "representation_interval_s": 0.0,
        "temporal_pose_search": False,
        "pose_confidence_min": 0.0,
        "cursor_worker_process": True,
        "cursor_opencv_threads": 2,
        "route_map_score_min": 0.50,
    },
}


def resolve_tracking_profile(
    name: str = "real-time", overrides: Optional[Mapping] = None
) -> dict:
    """Resolve a named profile plus explicit developer overrides."""
    if name not in TRACKING_PROFILES:
        raise ValueError(
            "Unknown tracking profile {!r}; expected one of {}".format(
                name, ", ".join(TRACKING_PROFILES)
            )
        )
    value = deepcopy(TRACKING_PROFILES[name])
    value["profile"] = name
    allowed = set(value) - {"profile"}
    for key, override in dict(overrides or {}).items():
        if key not in allowed:
            raise ValueError("Unknown tracking-profile override: {}".format(key))
        value[key] = override
    if float(value["cursor_interval_s"]) < 0.0:
        raise ValueError("cursor_interval_s cannot be negative")
    if not 0.5 <= float(value["global_interval_s"]) <= 30.0:
        raise ValueError("global_interval_s must be 0.5–30 seconds")
    if float(value["representation_interval_s"]) < 0.0:
        raise ValueError("representation_interval_s cannot be negative")
    if not 0.0 <= float(value["pose_confidence_min"]) <= 1.0:
        raise ValueError("pose_confidence_min must be within 0..1")
    if int(value["cursor_opencv_threads"]) < 1:
        raise ValueError("cursor_opencv_threads must be at least one")
    if not 0.0 <= float(value["route_map_score_min"]) <= 1.0:
        raise ValueError("route_map_score_min must be within 0..1")
    return value
