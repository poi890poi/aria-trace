"""GUI-independent workflow calculations and evidence persistence."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

from ..bundle import build_calibration, write_calibration_bundle
from ..geometry import (
    CharucoLayout,
    detect_charuco_correspondences,
    select_visible_quality_region,
    transform_points,
)
from ..inspection import (
    extract_one_to_one_patch,
    nearest_neighbor_magnify,
    render_esfr_curve,
    render_feature_matching_curve,
    render_geometry_overlay,
    render_latency_timeline,
)


@dataclass(frozen=True)
class CalibrationInputs:
    screen_size_px: tuple[int, int]
    required_roi_xywh: tuple[int, int, int, int]
    phone_diagonal_in: float = 6.5
    camera_horizontal_fov_deg: float = 70.0
    patch_size_mm: float = 20.0
    calibration_name: str = "phone-camera"

    def __post_init__(self) -> None:
        if min(self.screen_size_px) <= 0:
            raise ValueError("Phone screen dimensions must be positive")
        x, y, width, height = self.required_roi_xywh
        screen_width, screen_height = self.screen_size_px
        if min(width, height) <= 0:
            raise ValueError("Required ROI dimensions must be positive")
        if x < 0 or y < 0 or x + width > screen_width or y + height > screen_height:
            raise ValueError("Required ROI must lie within the phone screen")
        if self.phone_diagonal_in <= 0 or self.camera_horizontal_fov_deg <= 0:
            raise ValueError("Physical display size and camera field of view must be positive")
        if not 5.0 < self.camera_horizontal_fov_deg < 175.0:
            raise ValueError("Camera horizontal field of view is implausible")
        if self.patch_size_mm <= 0:
            raise ValueError("Matchability patch size must be positive")

    @property
    def required_polygon(self) -> list[list[float]]:
        x, y, width, height = self.required_roi_xywh
        return [
            [float(x), float(y)],
            [float(x + width), float(y)],
            [float(x + width), float(y + height)],
            [float(x), float(y + height)],
        ]


@dataclass
class CalibrationAnalysis:
    inputs: CalibrationInputs
    layout: CharucoLayout
    raw_frame: np.ndarray
    target_image: np.ndarray
    overlay: np.ndarray
    normalized: np.ndarray
    one_to_one_patch: np.ndarray
    magnified_patch: np.ndarray
    calibration: dict[str, Any]
    geometry: Any
    valid_mask: np.ndarray
    detection: Mapping[str, Any]
    guidance: Mapping[str, Any]
    camera_metadata: Mapping[str, Any]


def _physical_screen_mm(inputs: CalibrationInputs) -> tuple[float, float]:
    width_px, height_px = inputs.screen_size_px
    diagonal_mm = float(inputs.phone_diagonal_in) * 25.4
    ratio = math.sqrt(float(width_px * width_px + height_px * height_px))
    return diagonal_mm * width_px / ratio, diagonal_mm * height_px / ratio


def _estimate_pose(
    screen_polygon_camera_xy: np.ndarray,
    camera_size_px: Sequence[int],
    inputs: CalibrationInputs,
) -> dict[str, Any]:
    camera_width, camera_height = map(int, camera_size_px)
    physical_width, physical_height = _physical_screen_mm(inputs)
    focal = camera_width / (
        2.0 * math.tan(math.radians(inputs.camera_horizontal_fov_deg) / 2.0)
    )
    camera_matrix = np.asarray(
        [
            [focal, 0.0, (camera_width - 1.0) / 2.0],
            [0.0, focal, (camera_height - 1.0) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    objects = np.asarray(
        [
            [-physical_width / 2.0, -physical_height / 2.0, 0.0],
            [physical_width / 2.0, -physical_height / 2.0, 0.0],
            [physical_width / 2.0, physical_height / 2.0, 0.0],
            [-physical_width / 2.0, physical_height / 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    image_points = np.asarray(screen_polygon_camera_xy, dtype=np.float64).reshape((4, 2))
    ok, rotation_vector, translation = cv2.solvePnP(
        objects,
        image_points,
        camera_matrix,
        np.zeros(5, dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("Approximate camera pose could not be fitted")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    normal = rotation[:, 2]
    off_axis = math.degrees(math.acos(float(np.clip(abs(normal[2]), 0.0, 1.0))))
    top_edge = image_points[1] - image_points[0]
    roll = math.degrees(math.atan2(float(top_edge[1]), float(top_edge[0])))
    return {
        "distance_mm": float(abs(translation.reshape(-1)[2])),
        "off_axis_deg": float(off_axis),
        "roll_deg": float(roll),
        "assumed_phone_diagonal_in": float(inputs.phone_diagonal_in),
        "assumed_camera_horizontal_fov_deg": float(inputs.camera_horizontal_fov_deg),
        "camera_matrix_3x3": camera_matrix.tolist(),
        "source": "estimated_from_assumed_physical_size_and_fov",
    }


def _focus_metrics(image: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return {
        "laplacian_variance_relative": float(np.var(laplacian)),
        "tenengrad_relative": float(np.mean(gradient_x * gradient_x + gradient_y * gradient_y)),
    }


def positioning_guidance(calibration: Mapping[str, Any], pose: Mapping[str, Any]) -> list[str]:
    geometry = calibration["geometry"]
    messages = []
    if float(geometry["required_region_coverage"]) < 0.999:
        messages.append("Move or pull back until the complete required ROI is visible.")
    if float(geometry["required_region_detected_hull_coverage"]) < 0.999:
        messages.append("Show more ChArUco corners around the required ROI; extrapolation is unsafe.")
    coverage = float(geometry["screen_coverage"])
    if coverage < 0.85:
        messages.append("Increase camera distance if full-screen reuse is required.")
    utilization = float(geometry["camera_utilization"])
    if utilization < 0.35:
        messages.append("Move the camera closer; too many camera pixels are outside the phone.")
    if abs(float(pose["roll_deg"])) > 3.0:
        messages.append("Rotate the camera to reduce in-plane roll below 3 degrees.")
    if float(pose["off_axis_deg"]) > 12.0:
        messages.append("Align the camera more squarely with the phone to reduce perspective tilt.")
    if float(geometry["reprojection_p95_px"]) > 2.0:
        messages.append("Hold the rig steady and refit; corner reprojection error is high.")
    if not messages:
        messages.append("Geometry is well positioned; continue to focus and image-quality measurement.")
    return messages


def analyze_frame(
    frame: np.ndarray,
    inputs: CalibrationInputs,
    layout: CharucoLayout,
    target_image: np.ndarray,
    camera_metadata: Optional[Mapping[str, Any]] = None,
) -> CalibrationAnalysis:
    """Detect one reviewed geometry sample and produce all immediate evidence."""

    detection = detect_charuco_correspondences(frame, layout)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    calibration_id = "{}-{}".format(inputs.calibration_name, timestamp)
    camera_size = (int(frame.shape[1]), int(frame.shape[0]))
    rig = {
        "camera": {
            "adapter": dict(camera_metadata or {}),
            "mode": {"width_px": camera_size[0], "height_px": camera_size[1]},
        },
        "phone": {
            "screen_size_px": list(inputs.screen_size_px),
            "orientation": "portrait" if inputs.screen_size_px[1] >= inputs.screen_size_px[0] else "landscape",
            "diagonal_in": float(inputs.phone_diagonal_in),
        },
    }
    optics = {
        "lens_model": {"source": "unavailable"},
        "pose_assumptions": {
            "camera_horizontal_fov_deg": float(inputs.camera_horizontal_fov_deg),
            "phone_diagonal_in": float(inputs.phone_diagonal_in),
        },
    }
    calibration, geometry, valid_mask = build_calibration(
        calibration_id=calibration_id,
        camera_points_xy=detection["camera_points_xy"],
        screen_points_xy=detection["screen_points_xy"],
        camera_size_px=camera_size,
        screen_size_px=inputs.screen_size_px,
        input_frame_id="aria://rig/{}/camera/undistorted".format(calibration_id),
        canonical_screen_frame_id="aria://device/phone/screen/layout-{}x{}".format(*inputs.screen_size_px),
        required_region_screen_xy=inputs.required_polygon,
        rig=rig,
        optics=optics,
        required_roi={
            "kind": "screen-rectangle",
            "xywh": list(inputs.required_roi_xywh),
            "polygon_xy": inputs.required_polygon,
        },
        status="warning",
    )
    normalized = cv2.warpPerspective(
        frame,
        np.asarray(calibration["normalization"]["matrix_3x3"], dtype=np.float64),
        tuple(map(int, inputs.screen_size_px)),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    try:
        supported_hull = cv2.convexHull(
            np.asarray(detection["screen_points_xy"], dtype=np.float32)[
                geometry.inlier_mask
            ]
        ).reshape((-1, 2))
        quality_region = select_visible_quality_region(
            camera_size,
            inputs.screen_size_px,
            geometry.matrix_3x3,
            required_region_screen_xy=inputs.required_polygon,
            supported_region_screen_xy=supported_hull,
            supported_region_margin_display_px=int(round(layout.square_px)),
        )
    except (ValueError, RuntimeError) as exc:
        quality_region = {
            "status": "unavailable",
            "xywh": None,
            "space": "canonical_phone_screen_px",
            "selection": "camera_visible_required_intersection",
            "error": str(exc),
        }
    if quality_region["xywh"] is not None:
        qx, qy, qwidth, qheight = quality_region["xywh"]
        screen_center = np.asarray(
            [[qx + (qwidth - 1) / 2.0, qy + (qheight - 1) / 2.0]],
            dtype=np.float64,
        )
        camera_center = transform_points(screen_center, geometry.inverse_matrix_3x3)[0]
    else:
        camera_center = np.asarray(
            [(frame.shape[1] - 1) / 2.0, (frame.shape[0] - 1) / 2.0],
            dtype=np.float64,
        )
    patch_width = min(320, frame.shape[1])
    patch_height = min(240, frame.shape[0])
    patch, patch_metadata = extract_one_to_one_patch(
        frame, camera_center, (patch_width, patch_height)
    )
    pose = _estimate_pose(geometry.screen_polygon_input_xy, camera_size, inputs)
    focus = _focus_metrics(patch)
    messages = positioning_guidance(calibration, pose)
    if quality_region["status"] != "available":
        messages.append(
            "No safe quality patch is available: {}".format(quality_region["error"])
        )
    calibration["geometry"]["approximate_pose"] = pose
    calibration["geometry"]["charuco_atlas"] = {
        "fit_precedes_quality_measurement": True,
        "screen_size_px": list(inputs.screen_size_px),
        "detected_corner_ids": [int(value) for value in detection["corner_ids"]],
        "detected_corner_count": int(detection["corner_count"]),
        "detected_marker_count": int(detection["marker_count"]),
        "screen_view_iou": float(geometry.metrics["screen_view_iou"]),
        "screen_coverage": float(geometry.metrics["screen_coverage"]),
    }
    calibration["geometry"]["quality_region"] = dict(quality_region)
    calibration["confidence"]["assumptions"].extend(
        ["phone_diagonal_in", "camera_horizontal_fov_deg"]
    )
    guidance = {
        "messages": messages,
        "pose": pose,
        "focus": focus,
        "inspection_patch": patch_metadata,
        "quality_region": quality_region,
    }
    return CalibrationAnalysis(
        inputs=inputs,
        layout=layout,
        raw_frame=frame.copy(),
        target_image=target_image.copy(),
        overlay=render_geometry_overlay(
            frame,
            geometry,
            detection["camera_points_xy"],
            quality_region["xywh"],
            canonical_screen_size_px=inputs.screen_size_px,
        ),
        normalized=normalized,
        one_to_one_patch=patch,
        magnified_patch=nearest_neighbor_magnify(patch, 4),
        calibration=calibration,
        geometry=geometry,
        valid_mask=valid_mask,
        detection=detection,
        guidance=guidance,
        camera_metadata=dict(camera_metadata or {}),
    )


def save_analysis_bundle(
    output_directory: Path,
    analysis: CalibrationAnalysis,
    image_quality: Optional[Mapping[str, Any]] = None,
    feature_matching: Optional[Mapping[str, Any]] = None,
    timing: Optional[Mapping[str, Any]] = None,
    adb_reference: Optional[np.ndarray] = None,
    quality_evidence: Optional[Mapping[str, np.ndarray]] = None,
) -> Path:
    """Save review evidence and the consumer-facing commented YAML contract."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    calibration = copy.deepcopy(analysis.calibration)
    calibration["image_quality"] = dict(image_quality or {})
    calibration["feature_matching"] = dict(feature_matching or {})
    calibration["timing"] = dict(timing or {})
    measured = [float(calibration["confidence"]["geometry"])]
    if image_quality:
        calibration["confidence"]["image_quality"] = float(image_quality.get("confidence", 0.0))
        measured.append(calibration["confidence"]["image_quality"])
    if feature_matching:
        calibration["confidence"]["feature_matching"] = float(feature_matching.get("confidence", 0.0))
        measured.append(calibration["confidence"]["feature_matching"])
    if timing:
        camera_timing = timing.get("camera", timing)
        calibration["confidence"]["timing"] = float(camera_timing.get("confidence", 0.0))
        measured.append(calibration["confidence"]["timing"])
    calibration["confidence"]["overall"] = min(measured)
    geometry = calibration["geometry"]
    measurements_complete = bool(
        float(geometry["required_region_coverage"]) >= 0.999
        and float(geometry["required_region_detected_hull_coverage"]) >= 0.999
        and float(geometry["reprojection_p95_px"]) <= 2.0
        and image_quality
        and image_quality.get("display_referred", {}).get("mtf50_conservative") is not None
        and feature_matching
        and int(feature_matching.get("evaluated_match_count", 0)) > 0
    )
    # These measurements are evidence, not a universal pass/fail gate.  A task
    # integration must evaluate its own declared thresholds before changing the
    # status to accepted; merely finding a non-empty policy object is not proof
    # that its requirements were satisfied.
    calibration["status"] = "warning"
    if measurements_complete:
        calibration["confidence"]["warnings"].append(
            "task_acceptance_thresholds_not_configured"
        )
    files = {
        "raw_geometry_frame": "geometry-raw.png",
        "geometry_overlay": "geometry-overlay.png",
        "normalized_phone": "normalized-phone.png",
        "inspection_1_to_1": "focus-1to1.png",
        "inspection_4x_nearest": "focus-4x-nearest.png",
        "charuco_target": "charuco-target.png",
        "valid_screen_mask": "valid_screen_mask.png",
    }
    images = {
        files["raw_geometry_frame"]: analysis.raw_frame,
        files["geometry_overlay"]: analysis.overlay,
        files["normalized_phone"]: analysis.normalized,
        files["inspection_1_to_1"]: analysis.one_to_one_patch,
        files["inspection_4x_nearest"]: analysis.magnified_patch,
        files["charuco_target"]: analysis.target_image,
    }
    if image_quality:
        files["esfr_mtf_curve"] = "esfr-mtf-curve.png"
        images[files["esfr_mtf_curve"]] = render_esfr_curve(image_quality)
    if feature_matching:
        files["feature_matching_curve"] = "feature-matching-curve.png"
        images[files["feature_matching_curve"]] = render_feature_matching_curve(feature_matching)
    for evidence_name, evidence_image in dict(quality_evidence or {}).items():
        safe_name = "".join(
            character if character.isalnum() or character in ("-", "_") else "-"
            for character in str(evidence_name)
        ).strip("-")
        if not safe_name:
            continue
        filename = "{}.png".format(safe_name)
        files[safe_name] = filename
        images[filename] = evidence_image
    if timing:
        camera_timing = timing.get("camera", timing)
        if camera_timing.get("transitions"):
            files["latency_timeline"] = "latency-timeline.png"
            images[files["latency_timeline"]] = render_latency_timeline(camera_timing)
    if adb_reference is not None:
        files["adb_reference"] = "adb-reference.png"
        images[files["adb_reference"]] = adb_reference
    for name, image in images.items():
        if not cv2.imwrite(str(output / name), image):
            raise RuntimeError("Cannot write evidence image {}".format(name))
    calibration["evidence"] = files
    return write_calibration_bundle(output, calibration, analysis.valid_mask)
