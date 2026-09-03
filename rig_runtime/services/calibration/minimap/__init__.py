"""Mini-map boundary, cursor, verification, and transition services."""

from .calibration import (
    ORDINARY_MOTION_SEGMENT_LABELS,
    calibrate_minimap_boundary_frames,
    calibrate_minimap_frames,
)
from .transition import TransitionController
from .spatial import (
    MINIMAP_CROP_SPACE_ID,
    bind_minimap_boundary,
    minimap_crop_space,
    normalize_minimap_geometry,
)

__all__ = [
    "ORDINARY_MOTION_SEGMENT_LABELS",
    "TransitionController",
    "MINIMAP_CROP_SPACE_ID",
    "bind_minimap_boundary",
    "calibrate_minimap_boundary_frames",
    "calibrate_minimap_frames",
    "minimap_crop_space",
    "normalize_minimap_geometry",
]
