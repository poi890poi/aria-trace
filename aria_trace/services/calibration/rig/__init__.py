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
from .data_matrix_readability import (
    DataMatrixObservation,
    DataMatrixTarget,
    alternating_data_matrix_targets,
    decode_data_matrix_payloads,
    encode_data_matrix_modules,
    grade_data_matrix_decode,
    render_data_matrix_target,
    summarize_data_matrix_decode_sweep,
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
from .presentation import (
    HOST_MONOTONIC_CLOCK,
    matching_paint_acknowledgement,
    presentation_freshness_boundary_ns,
    sample_host_time_ns,
)
from .spatial_export import export_spatial_fragment, validate_spatial_fragment

__all__ = [
    "AlternatingStimulus",
    "CALIBRATION_SCHEMA_VERSION",
    "CharucoLayout",
    "ControlEvent",
    "DataMatrixObservation",
    "DataMatrixTarget",
    "FrameNormalizer",
    "FrameSample",
    "FrameStream",
    "GeometryEstimate",
    "HOST_MONOTONIC_CLOCK",
    "SignalObservation",
    "SignalObserver",
    "TargetPresenter",
    "aggregate_esfr_measurements",
    "aggregate_feature_matching",
    "alternating_data_matrix_targets",
    "build_calibration",
    "build_rectification_maps",
    "calibrate_intrinsics_from_views",
    "detect_charuco_correspondences",
    "decode_data_matrix_payloads",
    "encode_data_matrix_modules",
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
    "grade_data_matrix_decode",
    "measure_slanted_edge_esfr",
    "matching_paint_acknowledgement",
    "nearest_neighbor_magnify",
    "render_geometry_overlay",
    "render_esfr_curve",
    "render_feature_matching_curve",
    "render_feature_matching_overlay",
    "render_latency_timeline",
    "presentation_freshness_boundary_ns",
    "render_data_matrix_target",
    "select_visible_quality_region",
    "sample_host_time_ns",
    "summarize_data_matrix_decode_sweep",
    "validate_calibration",
    "validate_spatial_fragment",
    "valid_output_mask",
    "visible_screen_mask",
    "write_calibration_bundle",
    "write_calibration_yaml",
]
