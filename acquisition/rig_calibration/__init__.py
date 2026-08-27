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
    SignalObservation,
)
from .geometry import (
    CharucoLayout,
    calibrate_intrinsics_from_views,
    detect_charuco_correspondences,
    estimate_screen_geometry,
    generate_charuco_target,
    select_visible_quality_region,
    visible_screen_mask,
)
from .feature_matching import (
    aggregate_feature_matching,
    evaluate_feature_matching,
    generate_feature_target,
)
from .image_quality import (
    aggregate_esfr_measurements,
    generate_slanted_edge_target,
    measure_slanted_edge_esfr,
)
from .latency import estimate_latency, estimate_paired_delay
from .inspection import (
    extract_one_to_one_patch,
    nearest_neighbor_magnify,
    render_geometry_overlay,
    render_esfr_curve,
    render_feature_matching_curve,
    render_feature_matching_overlay,
    render_latency_timeline,
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
    "SignalObservation",
    "SignalObserver",
    "TargetPresenter",
    "aggregate_esfr_measurements",
    "aggregate_feature_matching",
    "build_calibration",
    "build_rectification_maps",
    "calibrate_intrinsics_from_views",
    "detect_charuco_correspondences",
    "estimate_latency",
    "estimate_paired_delay",
    "estimate_screen_geometry",
    "evaluate_feature_matching",
    "extract_one_to_one_patch",
    "export_spatial_fragment",
    "generate_charuco_target",
    "generate_feature_target",
    "generate_slanted_edge_target",
    "load_calibration_yaml",
    "measure_slanted_edge_esfr",
    "nearest_neighbor_magnify",
    "render_geometry_overlay",
    "render_esfr_curve",
    "render_feature_matching_curve",
    "render_feature_matching_overlay",
    "render_latency_timeline",
    "select_visible_quality_region",
    "validate_calibration",
    "validate_spatial_fragment",
    "valid_output_mask",
    "visible_screen_mask",
    "write_calibration_bundle",
    "write_calibration_yaml",
]
