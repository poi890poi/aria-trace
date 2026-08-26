"""Independent camera-to-phone rig calibration primitives.

The package owns calibration math and artifacts. Camera drivers, ADB, game
profiles, workbench UI, maps, and dataset registries integrate through the
small adapter interfaces exported here.
"""

from .adapters import AlternatingStimulus, FrameStream, SignalObserver, TargetPresenter
from .artifact import (
    CALIBRATION_SCHEMA_VERSION,
    load_calibration_yaml,
    validate_calibration,
    write_calibration_yaml,
)
from .bundle import build_calibration, valid_output_mask, write_calibration_bundle
from .contracts import (
    ControlEvent,
    FrameSample,
    GeometryEstimate,
    MatchResult,
    MatchTrial,
    SignalObservation,
)
from .geometry import (
    CharucoLayout,
    calibrate_intrinsics_from_views,
    detect_charuco_correspondences,
    estimate_screen_geometry,
    generate_charuco_target,
)
from .latency import estimate_latency, estimate_paired_delay
from .inspection import (
    extract_one_to_one_patch,
    nearest_neighbor_magnify,
    render_geometry_overlay,
    render_latency_timeline,
    render_matchability_curve,
)
from .matchability import (
    PhaseCorrelationMatcher,
    evaluate_matchability,
    generate_band_limited_target,
    warp_target,
)
from .normalizer import FrameNormalizer, build_rectification_maps
from .spatial_export import export_spatial_fragment, validate_spatial_fragment

__all__ = [
    "AlternatingStimulus",
    "CALIBRATION_SCHEMA_VERSION",
    "CharucoLayout",
    "ControlEvent",
    "FrameNormalizer",
    "FrameSample",
    "FrameStream",
    "GeometryEstimate",
    "MatchResult",
    "MatchTrial",
    "PhaseCorrelationMatcher",
    "SignalObservation",
    "SignalObserver",
    "TargetPresenter",
    "build_calibration",
    "build_rectification_maps",
    "calibrate_intrinsics_from_views",
    "detect_charuco_correspondences",
    "estimate_latency",
    "estimate_paired_delay",
    "estimate_screen_geometry",
    "evaluate_matchability",
    "extract_one_to_one_patch",
    "export_spatial_fragment",
    "generate_charuco_target",
    "generate_band_limited_target",
    "load_calibration_yaml",
    "nearest_neighbor_magnify",
    "render_geometry_overlay",
    "render_latency_timeline",
    "render_matchability_curve",
    "validate_calibration",
    "validate_spatial_fragment",
    "valid_output_mask",
    "warp_target",
    "write_calibration_bundle",
    "write_calibration_yaml",
]
