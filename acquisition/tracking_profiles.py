"""Named, reproducible execution profiles for localization and pose tracking."""

from copy import deepcopy
from typing import Mapping, Optional


TRACKING_PROFILES = {
    "real-time": {
        "cursor_pose_method": "fast_grid",
        "cursor_interval_s": 0.05,
        "global_interval_s": 2.0,
        "temporal_pose_search": True,
        "pose_confidence_min": 0.45,
        "local_xy_stride": 1,
        "diagnostics_stride": 10,
        "quality_policy": "adaptive_fallback",
    },
    "fast": {
        "cursor_pose_method": "fast_grid",
        "cursor_interval_s": 0.10,
        "global_interval_s": 3.0,
        "temporal_pose_search": True,
        "pose_confidence_min": 0.50,
        "local_xy_stride": 1,
        "diagnostics_stride": 20,
        "quality_policy": "early_exit",
    },
    "accurate": {
        "cursor_pose_method": "vectorized_grid",
        "cursor_interval_s": 0.15,
        "global_interval_s": 1.5,
        "temporal_pose_search": True,
        "pose_confidence_min": 0.40,
        "local_xy_stride": 1,
        "diagnostics_stride": 5,
        "quality_policy": "validate_ambiguous",
    },
    "offline": {
        "cursor_pose_method": "vectorized_grid",
        "cursor_interval_s": 0.0,
        "global_interval_s": 0.5,
        "temporal_pose_search": False,
        "pose_confidence_min": 0.0,
        "local_xy_stride": 1,
        "diagnostics_stride": 1,
        "quality_policy": "full_evidence",
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
    if not 0.0 <= float(value["pose_confidence_min"]) <= 1.0:
        raise ValueError("pose_confidence_min must be within 0..1")
    return value
