"""Headless mini-map calibration from synchronized Android and HIK frames.

Reusable mini-map geometry is learned in phone/display coordinates. HIK
geometry is projected through a transform estimated from the current recording
and is retained only as session-local review evidence. Supplying a rig
calibration composes its existing camera-to-phone mapping with the phone-game
result; it never replaces the optical rig geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .calibration_profiles import (
    ScopedCalibrationProfileStore,
    ScopedProfileKey,
)
from .commented_yaml import PROFILE_COMMENTS, PROFILE_HEADER, write_commented_yaml
from .minimap_calibration import (
    _color_heatmap,
    _stacked_difference_heatmap,
    calibrate_minimap_boundary_frames,
)
from .session import SessionReader


SCHEMA_VERSION = "1.0"


DEFAULT_ANDROID_DISCOVERY = {
    "strategy": "relative_prior",
    "center_region_xyxy_fraction": [0.0, 0.0, 0.35, 0.35],
    "radius_fraction_range": [0.07, 0.22],
    "minimum_circle_visible_fraction": 0.85,
}


MINIMAP_HEADER = """# AriaTrace mini-map calibration result.
#
# This result contains only mini-map isolation geometry. XY/WH values are in
# the explicitly named source coordinate space with top-left origin, +X right,
# and +Y down. It does not contain cursor, pose, tracking, or optical-rig fits.
# The JSON companion contains identical machine-readable data."""

MINIMAP_COMMENTS = {
    "scope": "Exactly what this calibration does and does not own.",
    "coordinate_space": "Authority for every source-space crop, center, radius, and mask.",
    "selection": "How the candidate crop was selected before the verified boundary fit.",
    "outer_boundary": "Fitted circle in this source's pixels; source_center_xy includes crop offset.",
    "shift_estimation": "The exact circular mask intended for downstream shift estimation.",
    "evidence": "Images tied only to candidate selection, source activity, and the fitted boundary.",
    "provenance": "Session and stream that produced this immutable result.",
}


def _atomic_json(path: Path, value: Mapping[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))
    return path


def _parse_crop(value: Optional[str]) -> Optional[list[int]]:
    if value is None:
        return None
    crop = [int(item.strip()) for item in str(value).split(",")]
    if len(crop) != 4:
        raise argparse.ArgumentTypeError("crop must be x,y,width,height")
    return crop


def _fraction_list(
    value: object, count: int, label: str
) -> list[float]:
    if isinstance(value, str):
        values = [float(item.strip()) for item in value.split(",")]
    elif isinstance(value, Sequence):
        values = [float(item) for item in value]
    else:
        raise ValueError("{} must contain {} comma-separated fractions".format(label, count))
    if len(values) != count:
        raise ValueError("{} must contain {} fractions".format(label, count))
    if not all(math.isfinite(item) for item in values):
        raise ValueError("{} fractions must be finite".format(label))
    return values


def android_discovery_config(
    strategy: str = "relative_prior",
    center_region_xyxy_fraction: object = (0.0, 0.0, 0.35, 0.35),
    radius_fraction_range: object = (0.07, 0.22),
    minimum_circle_visible_fraction: float = 0.85,
) -> dict:
    """Validate configurable, resolution-relative Android discovery limits."""

    normalized_strategy = str(strategy).strip().lower().replace("-", "_")
    if normalized_strategy not in ("relative_prior", "unrestricted", "legacy_hint"):
        raise ValueError(
            "Android discovery strategy must be relative_prior, unrestricted, or legacy_hint"
        )
    center = _fraction_list(
        center_region_xyxy_fraction, 4, "Android center region"
    )
    radius = _fraction_list(radius_fraction_range, 2, "Android radius range")
    if not (
        0.0 <= center[0] < center[2] <= 1.0
        and 0.0 <= center[1] < center[3] <= 1.0
    ):
        raise ValueError("Android center region must be ordered within [0, 1]")
    if not (0.0 < radius[0] < radius[1] <= 0.5):
        raise ValueError("Android radius fractions must be ordered within (0, 0.5]")
    visible = float(minimum_circle_visible_fraction)
    if not 0.0 <= visible <= 1.0:
        raise ValueError("Minimum visible circle fraction must be within [0, 1]")
    return {
        "strategy": normalized_strategy,
        "center_region_xyxy_fraction": center,
        "radius_fraction_range": radius,
        "minimum_circle_visible_fraction": visible,
    }


def logical_crop_to_natural(
    crop_xywh: Sequence[int], orientation: Mapping[str, object]
) -> list[int]:
    """Convert an Android logical-display rectangle to the natural raster."""

    x, y, width, height = map(int, crop_xywh)
    natural = orientation.get("natural_size_px")
    if not isinstance(natural, Sequence) or len(natural) != 2:
        raise ValueError("phone_surface_orientation.natural_size_px is required")
    natural_width, natural_height = map(int, natural)
    quarter_turns = int(
        orientation.get("quarter_turns_clockwise_from_natural", 0)
    ) % 4
    if quarter_turns == 0:
        converted = [x, y, width, height]
    elif quarter_turns == 1:
        converted = [y, natural_height - x - width, height, width]
    elif quarter_turns == 2:
        converted = [
            natural_width - x - width,
            natural_height - y - height,
            width,
            height,
        ]
    else:
        converted = [natural_width - y - height, x, height, width]
    if min(converted) < 0:
        raise ValueError("Converted mini-map crop exceeds the natural phone raster")
    return converted


def logical_point_to_natural(
    point_xy: Sequence[float], orientation: Mapping[str, object]
) -> list[float]:
    """Convert an Android logical-display pixel center to natural coordinates."""

    x, y = map(float, point_xy)
    natural_width, natural_height = map(int, orientation["natural_size_px"])
    quarter_turns = int(
        orientation.get("quarter_turns_clockwise_from_natural", 0)
    ) % 4
    if quarter_turns == 0:
        return [x, y]
    if quarter_turns == 1:
        return [y, natural_height - x]
    if quarter_turns == 2:
        return [natural_width - x, natural_height - y]
    return [natural_width - y, x]


def read_session_stream_frames(
    session: SessionReader,
    stream_id: str,
    maximum_frames: int = 64,
    crop_xywh: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode representative video frames and their authoritative timestamps."""

    records = list(session.frames_by_stream.get(stream_id) or [])
    if len(records) < 12:
        raise ValueError("Stream {} contains fewer than 12 frames".format(stream_id))
    selected = set(
        np.linspace(0, len(records) - 1, min(maximum_frames, len(records)))
        .round()
        .astype(int)
        .tolist()
    )
    capture = cv2.VideoCapture(str(session.video_path(stream_id)))
    if not capture.isOpened():
        raise RuntimeError("Cannot open stream video: {}".format(session.video_path(stream_id)))
    frames, times = [], []
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index in selected and index < len(records):
                if crop_xywh is not None:
                    x, y, width, height = map(int, crop_xywh)
                    frame = frame[y : y + height, x : x + width]
                frames.append(frame)
                times.append(int(records[index]["host_capture_time_ns"]))
            index += 1
    finally:
        capture.release()
    if len(frames) < 12:
        raise ValueError("Stream {} yielded fewer than 12 decoded frames".format(stream_id))
    return np.stack(frames), np.asarray(times, dtype=np.int64)


def _known_android_hint(game_id: str, frame_size: Sequence[int]) -> Optional[list[float]]:
    normalized = str(game_id).strip().lower().replace("_", "-")
    if normalized not in ("genshin", "genshin-impact", "genshin-impact-pc"):
        return None
    width, height = map(float, frame_size)
    return [111.0 * width / 1280.0, 83.0 * height / 720.0, 68.5 * height / 720.0]


def _circle_candidates(
    frames: np.ndarray,
    expected: Optional[Sequence[float]] = None,
    discovery: Optional[Mapping[str, object]] = None,
) -> list[dict]:
    average = frames.mean(axis=0).astype(np.uint8)
    gray = cv2.GaussianBlur(cv2.cvtColor(average, cv2.COLOR_BGR2GRAY), (7, 7), 1.4)
    equalized = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    heat = cv2.GaussianBlur(_stacked_difference_heatmap(frames), (0, 0), 2.0)
    heat_image = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    height, width = gray.shape
    relative = None
    if discovery is not None:
        relative = android_discovery_config(
            discovery.get("strategy", "relative_prior"),
            discovery.get(
                "center_region_xyxy_fraction",
                DEFAULT_ANDROID_DISCOVERY["center_region_xyxy_fraction"],
            ),
            discovery.get(
                "radius_fraction_range",
                DEFAULT_ANDROID_DISCOVERY["radius_fraction_range"],
            ),
            discovery.get(
                "minimum_circle_visible_fraction",
                DEFAULT_ANDROID_DISCOVERY["minimum_circle_visible_fraction"],
            ),
        )
        if relative["strategy"] != "relative_prior":
            relative = None
    shorter = min(width, height)
    if relative is None:
        minimum_radius = max(18, int(round(shorter * 0.025)))
        maximum_radius = max(minimum_radius + 2, int(round(shorter * 0.22)))
        center_bounds = [0.0, 0.0, float(width), float(height)]
        minimum_visible = 0.0
    else:
        radius_fractions = relative["radius_fraction_range"]
        minimum_radius = max(18, int(round(shorter * radius_fractions[0])))
        maximum_radius = max(
            minimum_radius + 2, int(round(shorter * radius_fractions[1]))
        )
        center_fractions = relative["center_region_xyxy_fraction"]
        center_bounds = [
            center_fractions[0] * width,
            center_fractions[1] * height,
            center_fractions[2] * width,
            center_fractions[3] * height,
        ]
        minimum_visible = float(relative["minimum_circle_visible_fraction"])
    detected = []
    for source, edge_threshold, center_threshold in (
        (gray, 80.0, 20.0),
        (equalized, 70.0, 16.0),
        (heat_image, 55.0, 14.0),
    ):
        circles = cv2.HoughCircles(
            source,
            cv2.HOUGH_GRADIENT,
            dp=1.25,
            minDist=max(24, minimum_radius),
            param1=edge_threshold,
            param2=center_threshold,
            minRadius=minimum_radius,
            maxRadius=maximum_radius,
        )
        if circles is not None:
            detected.extend(circles[0])
    edge = cv2.magnitude(
        cv2.Sobel(equalized, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(equalized, cv2.CV_32F, 0, 1, ksize=3),
    )
    yy, xx = np.ogrid[:height, :width]
    candidates = []
    for center_x, center_y, radius in detected:
        if radius < minimum_radius or radius > maximum_radius:
            continue
        if not (
            center_bounds[0] <= center_x <= center_bounds[2]
            and center_bounds[1] <= center_y <= center_bounds[3]
        ):
            continue
        angles = np.linspace(0.0, 2.0 * np.pi, 180, endpoint=False)
        visible_fraction = float(
            np.mean(
                (center_x + np.cos(angles) * radius >= 0.0)
                & (center_x + np.cos(angles) * radius < width)
                & (center_y + np.sin(angles) * radius >= 0.0)
                & (center_y + np.sin(angles) * radius < height)
            )
        )
        if visible_fraction < minimum_visible:
            continue
        if any(
            math.hypot(center_x - item["center_x"], center_y - item["center_y"])
            < max(5.0, 0.12 * radius)
            and abs(radius - item["radius"]) < max(4.0, 0.10 * radius)
            for item in candidates
        ):
            continue
        distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        inside = distance <= radius * 0.82
        ring = np.abs(distance - radius) <= max(2.0, radius * 0.055)
        outside = (distance >= radius * 1.12) & (distance <= radius * 1.38)
        if not inside.any() or not ring.any() or not outside.any():
            continue
        motion_inside = float(np.mean(heat[inside]))
        motion_outside = float(np.mean(heat[outside]))
        ring_edge = float(np.mean(edge[ring]))
        if relative is None:
            score = ring_edge * math.sqrt(max(motion_inside, 0.01)) / (
                1.0 + 0.25 * max(motion_outside - motion_inside, 0.0)
            )
            score_kind = "legacy_motion_inside"
        else:
            stable_disc_contrast = max(0.0, motion_outside - motion_inside)
            score = ring_edge * math.sqrt(max(stable_disc_contrast, 0.01))
            score_kind = "stable_disc_boundary"
        if expected is not None:
            expected_x, expected_y, expected_radius = map(float, expected)
            distance_error = math.hypot(center_x - expected_x, center_y - expected_y) / max(expected_radius, 1.0)
            radius_error = abs(radius - expected_radius) / max(expected_radius, 1.0)
            score *= math.exp(-0.5 * (distance_error / 0.55) ** 2 - 0.5 * (radius_error / 0.35) ** 2)
        candidates.append(
            {
                "center_x": float(center_x),
                "center_y": float(center_y),
                "radius": float(radius),
                "score": float(score),
                "ring_edge": ring_edge,
                "motion_inside": motion_inside,
                "motion_outside": motion_outside,
                "stable_disc_contrast": max(0.0, motion_outside - motion_inside),
                "visible_circle_fraction": visible_fraction,
                "score_kind": score_kind,
            }
        )
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def _crop_from_circle(circle: Mapping[str, float], frame_size: Sequence[int]) -> list[int]:
    width, height = map(int, frame_size)
    radius = float(circle["radius"])
    margin = max(5, int(math.ceil(radius * 0.16)))
    left = max(0, int(math.floor(circle["center_x"] - radius)) - margin)
    top = max(0, int(math.floor(circle["center_y"] - radius)) - margin)
    right = min(width, int(math.ceil(circle["center_x"] + radius)) + margin + 1)
    bottom = min(height, int(math.ceil(circle["center_y"] + radius)) + margin + 1)
    return [left, top, right - left, bottom - top]


def calibrate_source_frames(
    frames: np.ndarray,
    output_path: Path,
    *,
    image_source: str,
    coordinate_space_id: str,
    selected_crop_xywh: Optional[Sequence[int]] = None,
    expected_circle_xy_radius: Optional[Sequence[float]] = None,
    discovery_config: Optional[Mapping[str, object]] = None,
    phone_surface_orientation: Optional[Mapping[str, object]] = None,
    provenance: Optional[Mapping[str, object]] = None,
) -> dict:
    """Discover a crop, then delegate the fit to the verified boundary routine."""

    if frames.ndim != 4 or len(frames) < 12:
        raise ValueError("Zigzag calibration needs at least 12 color frames")
    height, width = frames.shape[1:3]
    candidates = _circle_candidates(
        frames, expected_circle_xy_radius, discovery=discovery_config
    )
    if selected_crop_xywh is not None:
        crop = list(map(int, selected_crop_xywh))
        method = "user_selected_crop_then_verified_boundary_fit"
        seed = None
    elif expected_circle_xy_radius is not None:
        seed = {
            "center_x": float(expected_circle_xy_radius[0]),
            "center_y": float(expected_circle_xy_radius[1]),
            "radius": float(expected_circle_xy_radius[2]),
        }
        crop = _crop_from_circle(seed, (width, height))
        method = "checked_in_game_hint_then_verified_boundary_fit"
    elif candidates:
        seed = candidates[0]
        crop = _crop_from_circle(seed, (width, height))
        method = (
            "relative_prior_ranked_circle_search_then_verified_boundary_fit"
            if discovery_config is not None
            and str(discovery_config.get("strategy", "")).replace("-", "_")
            == "relative_prior"
            else "automatic_ranked_circle_search_then_verified_boundary_fit"
        )
    else:
        raise RuntimeError("No circular mini-map candidate was found; pass a selected crop")
    if len(crop) != 4:
        raise ValueError("Selected crop must be x,y,width,height")
    x, y, crop_width, crop_height = crop
    if min(x, y) < 0 or min(crop_width, crop_height) <= 0 or x + crop_width > width or y + crop_height > height:
        raise ValueError("Selected mini-map crop exceeds the source frame")
    cropped = frames[:, y : y + crop_height, x : x + crop_width]
    if seed is None:
        contained = [
            item for item in candidates
            if x <= item["center_x"] < x + crop_width
            and y <= item["center_y"] < y + crop_height
        ]
        seed = contained[0] if contained else None
    local_center = (
        [seed["center_x"] - x, seed["center_y"] - y]
        if seed is not None
        else [crop_width / 2.0, crop_height / 2.0]
    )
    seed_radius = (
        float(seed["radius"])
        if seed is not None
        else min(crop_width, crop_height) * 0.42
    )
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    boundary = calibrate_minimap_boundary_frames(
        cropped,
        output_path,
        config={
            "expected_center_xy": local_center,
            "center_search_radius_px": max(8.0, seed_radius * 0.20),
            "radius_range_px": [max(8.0, seed_radius * 0.84), seed_radius * 1.16],
        },
    )
    metrics = boundary["outer_boundary"]
    source_center = [float(metrics["center_x"] + x), float(metrics["center_y"] + y)]
    average = frames.mean(axis=0).astype(np.uint8)
    heat = _stacked_difference_heatmap(frames)
    overlay = average.copy()
    for index, candidate in enumerate(candidates[:8]):
        cv2.circle(
            overlay,
            (round(candidate["center_x"]), round(candidate["center_y"])),
            round(candidate["radius"]),
            (0, 255, 255) if index == 0 else (90, 90, 90),
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )
    cv2.rectangle(overlay, (x, y), (x + crop_width - 1, y + crop_height - 1), (255, 255, 0), 2)
    cv2.circle(overlay, tuple(map(round, source_center)), round(metrics["radius"]), (0, 255, 0), 2, cv2.LINE_AA)
    mask = np.zeros((height, width), np.uint8)
    cv2.circle(mask, tuple(map(round, source_center)), round(metrics["radius"]), 255, -1)
    crop_mask = mask[y : y + crop_height, x : x + crop_width]
    evidence = {
        "source_average": "source_average.png",
        "source_stacked_difference_heatmap": "source_stacked_difference_heatmap.png",
        "candidate_and_fit_overlay": "crop_selection_overlay.png",
        "actual_shift_estimation_mask": "actual_shift_estimation_mask.png",
        "cropped_minimap": "cropped_minimap.png",
        "cropped_minimap_mask": "cropped_minimap_mask.png",
    }
    images = {
        evidence["source_average"]: average,
        evidence["source_stacked_difference_heatmap"]: _color_heatmap(heat),
        evidence["candidate_and_fit_overlay"]: overlay,
        evidence["actual_shift_estimation_mask"]: mask,
        evidence["cropped_minimap"]: average[y : y + crop_height, x : x + crop_width],
        evidence["cropped_minimap_mask"]: crop_mask,
    }
    for filename, image in images.items():
        if not cv2.imwrite(str(output_path / filename), image):
            raise RuntimeError("Could not save mini-map evidence {}".format(filename))
    np.savez_compressed(
        str(output_path / "model.npz"),
        minimap_mask=mask,
        crop_mask=crop_mask,
        boundary=np.asarray([source_center[0], source_center[1], metrics["radius"]]),
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "review_required",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_kind": "zigzag_minimap_isolation",
        "scope": {
            "includes": ["mini-map crop", "circular boundary", "shift-estimation mask"],
            "excludes": ["camera/phone optical geometry", "cursor", "pose", "tracking", "game north"],
        },
        "image_source": image_source,
        "coordinate_space": {
            "id": coordinate_space_id,
            "frame_size_px": [width, height],
            "origin": "top_left_pixel_center",
            "axes": "+X right, +Y down",
        },
        "phone_surface_orientation": dict(phone_surface_orientation or {}),
        "selection": {
            "method": method,
            "expected_circle_xy_radius": list(map(float, expected_circle_xy_radius)) if expected_circle_xy_radius is not None else None,
            "discovery_config": (
                dict(discovery_config) if discovery_config is not None else None
            ),
            "candidate_count": len(candidates),
            "ranked_candidates": candidates[:12],
        },
        "crop_xywh": crop,
        "outer_boundary": {**metrics, "source_center_xy": source_center},
        "shift_estimation": {
            "mask_file": evidence["actual_shift_estimation_mask"],
            "crop_mask_file": evidence["cropped_minimap_mask"],
            "mask_semantics": "255 strictly inside the fitted circular mini-map boundary",
        },
        "evidence": evidence,
        "verified_boundary_evidence": boundary["evidence"],
        "model_file": "model.npz",
        "provenance": dict(provenance or {}),
    }
    if image_source == "android_scrcpy":
        orientation = dict(phone_surface_orientation or {})
        canonical_crop = logical_crop_to_natural(crop, orientation)
        canonical_center = logical_point_to_natural(source_center, orientation)
        result["canonical_phone_crop_xywh"] = canonical_crop
        result["outer_boundary"]["canonical_phone_center_xy"] = canonical_center
        result["canonical_coordinate_space"] = {
            "id": "phone_natural_display_pixels",
            "frame_size_px": list(map(int, orientation["natural_size_px"])),
        }
    _atomic_json(output_path / "minimap_calibration.json", result)
    write_commented_yaml(
        output_path / "minimap_calibration.yaml",
        result,
        header=MINIMAP_HEADER,
        section_comments=MINIMAP_COMMENTS,
    )
    return result


def estimate_current_cross_source_homography(
    android_frames: np.ndarray, hik_frames: np.ndarray
) -> dict:
    """Estimate this recording's Android-to-HIK mapping without saved geometry."""

    if android_frames.ndim != 4 or hik_frames.ndim != 4:
        raise ValueError("Cross-source alignment requires two color-frame sequences")
    android_average = android_frames.mean(axis=0).astype(np.uint8)
    hik_average = hik_frames.mean(axis=0).astype(np.uint8)
    android_gray = cv2.cvtColor(android_average, cv2.COLOR_BGR2GRAY)
    hik_gray = cv2.cvtColor(hik_average, cv2.COLOR_BGR2GRAY)
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("Current cross-source alignment requires OpenCV SIFT")
    detector = cv2.SIFT_create(
        nfeatures=8000, contrastThreshold=0.02, edgeThreshold=12
    )
    android_keypoints, android_descriptors = detector.detectAndCompute(
        android_gray, None
    )
    hik_keypoints, hik_descriptors = detector.detectAndCompute(hik_gray, None)
    if android_descriptors is None or hik_descriptors is None:
        raise RuntimeError("Cross-source alignment found no matchable descriptors")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        android_descriptors, hik_descriptors, k=2
    )
    matches = [
        first for first, second in pairs if first.distance < 0.72 * second.distance
    ]
    if len(matches) < 8:
        raise RuntimeError(
            "Cross-source alignment needs at least 8 ratio-test matches; got {}".format(
                len(matches)
            )
        )
    source = np.float32(
        [android_keypoints[item.queryIdx].pt for item in matches]
    ).reshape((-1, 1, 2))
    target = np.float32([hik_keypoints[item.trainIdx].pt for item in matches]).reshape(
        (-1, 1, 2)
    )
    robust_method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    matrix, mask = cv2.findHomography(
        source,
        target,
        robust_method,
        3.0,
        maxIters=10000,
        confidence=0.999,
    )
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        raise RuntimeError("Cross-source alignment could not fit a finite homography")
    inliers = mask.ravel().astype(bool)
    inlier_count = int(inliers.sum())
    inlier_rate = float(inliers.mean())
    if inlier_count < 8 or inlier_rate < 0.25:
        raise RuntimeError(
            "Cross-source alignment is ambiguous: {} inliers, {:.1%} rate".format(
                inlier_count, inlier_rate
            )
        )
    projected = cv2.perspectiveTransform(source, matrix)
    errors = np.linalg.norm(
        projected.reshape((-1, 2)) - target.reshape((-1, 2)), axis=1
    )
    median_error = float(np.median(errors[inliers]))
    p95_error = float(np.percentile(errors[inliers], 95))
    if median_error > 4.0 or p95_error > 10.0:
        raise RuntimeError(
            "Cross-source alignment reprojection is too large: median {:.2f}px, p95 {:.2f}px".format(
                median_error, p95_error
            )
        )
    confidence = float(
        min(1.0, inlier_count / 40.0)
        * min(1.0, inlier_rate / 0.70)
        * math.exp(-((median_error / 2.5) ** 2))
    )
    inlier_matches = [
        item for item, accepted in zip(matches, inliers.tolist()) if accepted
    ]
    match_visualization = cv2.drawMatches(
        android_average,
        android_keypoints,
        hik_average,
        hik_keypoints,
        inlier_matches[:160],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    return {
        "method": "sift_average_ratio_test_usac_homography",
        "android_to_hik_3x3": matrix,
        "android_keypoints": len(android_keypoints),
        "hik_keypoints": len(hik_keypoints),
        "ratio_match_count": len(matches),
        "inlier_count": inlier_count,
        "inlier_rate": inlier_rate,
        "median_inlier_reprojection_px": median_error,
        "p95_inlier_reprojection_px": p95_error,
        "confidence": confidence,
        "android_average": android_average,
        "hik_average": hik_average,
        "match_visualization": match_visualization,
    }


def create_current_hik_observation(
    phone_result: Mapping[str, object],
    hik_frames: np.ndarray,
    alignment: Mapping[str, object],
    output_path: Path,
    *,
    image_source: str,
    coordinate_space_id: str,
    provenance: Optional[Mapping[str, object]] = None,
) -> dict:
    """Project canonical session geometry into the current, possibly clipped HIK view."""

    height, width = hik_frames.shape[1:3]
    phone_boundary = phone_result["outer_boundary"]
    phone_center = np.asarray(phone_boundary["source_center_xy"], np.float64)
    phone_radius = float(phone_boundary["radius"])
    angles = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    phone_points = np.column_stack(
        (
            phone_center[0] + np.cos(angles) * phone_radius,
            phone_center[1] + np.sin(angles) * phone_radius,
        )
    )
    matrix = np.asarray(alignment["android_to_hik_3x3"], np.float64)
    projected = _transform_points(phone_points, matrix)
    projected_center = _transform_points([phone_center], matrix)[0]
    projected_radius = float(
        np.mean(np.linalg.norm(projected - projected_center[None, :], axis=1))
    )
    visible_boundary = (
        (projected[:, 0] >= 0.0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0.0)
        & (projected[:, 1] < height)
    )
    visible_boundary_fraction = float(visible_boundary.mean())
    polygon = np.round(projected).astype(np.int32)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [polygon], 255, cv2.LINE_AA)
    nonzero = cv2.findNonZero(mask)
    if nonzero is None:
        raise RuntimeError("The mapped mini-map does not intersect the current HIK view")
    full_area = max(abs(float(cv2.contourArea(projected.astype(np.float32)))), 1.0)
    visible_area_fraction = float(np.count_nonzero(mask) / full_area)
    bound_x, bound_y, bound_width, bound_height = cv2.boundingRect(nonzero)
    margin = max(4, int(math.ceil(projected_radius * 0.08)))
    left = max(0, bound_x - margin)
    top = max(0, bound_y - margin)
    right = min(width, bound_x + bound_width + margin)
    bottom = min(height, bound_y + bound_height + margin)
    crop = [left, top, right - left, bottom - top]
    crop_mask = mask[top:bottom, left:right]
    average = np.asarray(alignment["hik_average"], np.uint8)
    heat = _stacked_difference_heatmap(hik_frames)
    overlay = average.copy()
    cv2.polylines(overlay, [polygon], True, (255, 0, 255), 4, cv2.LINE_AA)
    cv2.drawMarker(
        overlay,
        tuple(np.round(projected_center).astype(int)),
        (255, 0, 255),
        cv2.MARKER_CROSS,
        34,
        4,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        overlay, (left, top), (right - 1, bottom - 1), (255, 255, 0), 2
    )
    warped_android = cv2.warpPerspective(
        np.asarray(alignment["android_average"], np.uint8), matrix, (width, height)
    )
    valid = cv2.warpPerspective(
        np.full(alignment["android_average"].shape[:2], 255, np.uint8),
        matrix,
        (width, height),
    )
    blend = average.copy()
    blend[valid > 0] = cv2.addWeighted(
        average[valid > 0], 0.5, warped_android[valid > 0], 0.5, 0
    )
    ellipse = cv2.fitEllipse(projected.astype(np.float32))
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    evidence = {
        "source_average": "source_average.png",
        "source_stacked_difference_heatmap": "source_stacked_difference_heatmap.png",
        "mapped_boundary_overlay": "mapped_boundary_overlay.png",
        "cross_source_matches": "cross_source_matches.png",
        "cross_source_warp_blend": "cross_source_warp_blend.png",
        "actual_shift_estimation_mask": "actual_shift_estimation_mask.png",
        "cropped_minimap": "cropped_minimap.png",
        "cropped_minimap_mask": "cropped_minimap_mask.png",
    }
    images = {
        evidence["source_average"]: average,
        evidence["source_stacked_difference_heatmap"]: _color_heatmap(heat),
        evidence["mapped_boundary_overlay"]: overlay,
        evidence["cross_source_matches"]: np.asarray(
            alignment["match_visualization"], np.uint8
        ),
        evidence["cross_source_warp_blend"]: blend,
        evidence["actual_shift_estimation_mask"]: mask,
        evidence["cropped_minimap"]: average[top:bottom, left:right],
        evidence["cropped_minimap_mask"]: crop_mask,
    }
    for filename, image in images.items():
        if not cv2.imwrite(str(output_path / filename), image):
            raise RuntimeError("Could not save HIK observation evidence {}".format(filename))
    np.savez_compressed(
        str(output_path / "model.npz"),
        minimap_mask=mask,
        crop_mask=crop_mask,
        boundary=np.asarray(
            [projected_center[0], projected_center[1], projected_radius]
        ),
        boundary_polygon=projected,
    )
    alignment_metrics = {
        key: alignment[key]
        for key in (
            "method",
            "android_keypoints",
            "hik_keypoints",
            "ratio_match_count",
            "inlier_count",
            "inlier_rate",
            "median_inlier_reprojection_px",
            "p95_inlier_reprojection_px",
            "confidence",
        )
    }
    alignment_metrics["android_to_hik_3x3"] = matrix.tolist()
    ellipse_center, ellipse_axes, ellipse_angle = ellipse
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "review_required",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "calibration_kind": "current_session_mapped_minimap_observation",
        "scope": {
            "includes": [
                "current HIK mini-map observation",
                "projected boundary polygon",
                "visible shift-estimation mask",
            ],
            "excludes": [
                "reusable HIK crop",
                "camera positioning prior",
                "cursor",
                "pose",
                "tracking",
                "game north",
            ],
        },
        "image_source": image_source,
        "coordinate_space": {
            "id": coordinate_space_id,
            "frame_size_px": [width, height],
            "origin": "top_left_pixel_center",
            "axes": "+X right, +Y down",
        },
        "selection": {
            "method": "current_cross_source_homography_from_android_boundary",
            "candidate_count": 0,
            "ranked_candidates": [],
            "cross_source_alignment": alignment_metrics,
        },
        "crop_xywh": crop,
        "outer_boundary": {
            "center_x": float(projected_center[0] - left),
            "center_y": float(projected_center[1] - top),
            "radius": projected_radius,
            "source_center_xy": projected_center.tolist(),
            "geometry": "projected_phone_circle",
            "source_boundary_polygon_xy": projected[::10].tolist(),
            "projected_ellipse": {
                "center_xy": [float(ellipse_center[0]), float(ellipse_center[1])],
                "diameter_xy": [float(ellipse_axes[0]), float(ellipse_axes[1])],
                "angle_deg": float(ellipse_angle),
            },
            "confidence": float(
                float(phone_boundary.get("confidence", 1.0))
                * float(alignment["confidence"])
                * math.sqrt(max(visible_boundary_fraction, 0.0))
            ),
        },
        "visibility": {
            "boundary_fraction": visible_boundary_fraction,
            "area_fraction": min(1.0, visible_area_fraction),
            "clipped_by_source_frame": visible_boundary_fraction < 0.999,
        },
        "shift_estimation": {
            "mask_file": evidence["actual_shift_estimation_mask"],
            "crop_mask_file": evidence["cropped_minimap_mask"],
            "mask_semantics": "255 inside the visible projection of the phone-space mini-map",
        },
        "reuse": {
            "persistent": False,
            "rule": "Recompute after camera movement; never reuse this HIK crop.",
        },
        "evidence": evidence,
        "model_file": "model.npz",
        "provenance": dict(provenance or {}),
    }
    _atomic_json(output_path / "minimap_calibration.json", result)
    write_commented_yaml(
        output_path / "minimap_calibration.yaml",
        result,
        header=MINIMAP_HEADER,
        section_comments=MINIMAP_COMMENTS,
    )
    return result


def _load_rig(path_value: Path) -> tuple[Path, dict]:
    path = Path(path_value)
    if path.is_dir():
        path = path / "hik_camera_calibration.json"
    if not path.is_file():
        raise FileNotFoundError("Rig calibration does not exist: {}".format(path))
    return path.resolve(), json.loads(path.read_text(encoding="utf-8"))


def _transform_points(points: Sequence[Sequence[float]], matrix: Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(points, np.float64).reshape((-1, 1, 2))
    return cv2.perspectiveTransform(values, np.asarray(matrix, np.float64)).reshape((-1, 2))


def compose_rig_game_profile(
    rig_path: Path,
    phone_result: Mapping[str, object],
    hik_result: Mapping[str, object],
) -> dict:
    """Reference a rig and phone-game profile without estimating a transform."""

    resolved_path, rig = _load_rig(rig_path)
    matrix = rig["geometry"]["full_sensor_camera_to_screen_3x3"]
    hik_center = hik_result["outer_boundary"]["source_center_xy"]
    radius = float(hik_result["outer_boundary"]["radius"])
    phone_center = np.asarray(
        phone_result["outer_boundary"]["canonical_phone_center_xy"], np.float64
    )
    phone_radius = float(phone_result["outer_boundary"]["radius"])
    hik_coordinate_space = (hik_result.get("coordinate_space") or {}).get(
        "id", "native_hik_sensor_bgr_pixels"
    )
    if hik_coordinate_space == "native_hik_sensor_bgr_pixels":
        mapped = _transform_points(
            [
                hik_center,
                [hik_center[0] + radius, hik_center[1]],
                [hik_center[0], hik_center[1] + radius],
            ],
            matrix,
        )
        mapped_radius = float(
            np.mean(
                [
                    np.linalg.norm(mapped[1] - mapped[0]),
                    np.linalg.norm(mapped[2] - mapped[0]),
                ]
            )
        )
        cross_source_check = {
            "method": "apply_saved_rig_homography_only_no_fitting",
            "mapped_hik_center_in_phone_xy": mapped[0].tolist(),
            "adb_phone_center_xy": phone_center.tolist(),
            "center_error_phone_px": float(np.linalg.norm(mapped[0] - phone_center)),
            "mapped_hik_radius_phone_px": mapped_radius,
            "adb_radius_phone_px": phone_radius,
            "radius_error_phone_px": abs(mapped_radius - phone_radius),
            "non_gating": True,
        }
    else:
        cross_source_check = {
            "method": "not_applicable_to_rig_normalized_hik_observation",
            "reason": (
                "The saved full-sensor camera matrix cannot be applied a second "
                "time to an already rig-normalized HIK stream."
            ),
            "non_gating": True,
        }
    camera_id = str(rig["camera"]["device_id"])
    phone_id = str(rig["phone"]["serial"])
    rig_id = "{}--{}".format(camera_id, phone_id)
    return {
        "profile_kind": "rig_game",
        "rig_id": rig_id,
        "base_rig_calibration": str(resolved_path),
        "composition_rule": (
            "Use the base rig normalization unchanged, then crop its normalized "
            "phone output with canonical_phone_crop_xywh. No optical transform is fitted here."
        ),
        "canonical_coordinate_space": "phone_natural_display_pixels",
        "canonical_phone_crop_xywh": list(phone_result["canonical_phone_crop_xywh"]),
        "native_hik_observation": {
            "coordinate_space": hik_coordinate_space,
            "crop_xywh": list(hik_result["crop_xywh"]),
            "center_xy": list(map(float, hik_center)),
            "radius_px": radius,
            "session_local": not bool(
                (hik_result.get("reuse") or {}).get("persistent", False)
            ),
            "reuse_rule": (hik_result.get("reuse") or {}).get("rule"),
            "visibility": dict(hik_result.get("visibility") or {}),
        },
        "cross_source_coordinate_check": cross_source_check,
    }


def _rig_id(rig_path: Path) -> str:
    _, rig = _load_rig(rig_path)
    return "{}--{}".format(rig["camera"]["device_id"], rig["phone"]["serial"])


def calibrate_zigzag_session(
    session_path: Path,
    output_path: Path,
    *,
    profiles_root: Path = Path("profiles"),
    rig_calibration: Optional[Path] = None,
    android_selected_crop_xywh: Optional[Sequence[int]] = None,
    hik_selected_crop_xywh: Optional[Sequence[int]] = None,
    android_discovery: Optional[Mapping[str, object]] = None,
) -> dict:
    session = SessionReader(session_path)
    context = session.manifest.get("context") or {}
    if context.get("capture_kind") not in (
        "zigzag_minimap_source_data",
        "zigzag_minimap_calibration",
    ):
        raise ValueError("Session is not a zigzag mini-map capture")
    game_id = str(context.get("game_id") or "unknown-game")
    orientation = dict(context.get("phone_surface_orientation") or {})
    android_frames, _ = read_session_stream_frames(session, "android_phone")
    if "hik_full" in session.frames_by_stream:
        hik_stream = "hik_full"
        hik_coordinate_space = "native_hik_sensor_bgr_pixels"
    elif "hik_phone" in session.frames_by_stream:
        hik_stream = "hik_phone"
        hik_coordinate_space = "rig_normalized_hik_phone_pixels"
    else:
        raise ValueError(
            "Session has neither a hik_full nor hik_phone frame stream"
        )
    hik_frames, _ = read_session_stream_frames(session, hik_stream)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=False)
    provenance = {"session_path": str(Path(session_path).resolve())}
    discovery = android_discovery_config(
        **(
            dict(DEFAULT_ANDROID_DISCOVERY)
            if android_discovery is None
            else dict(android_discovery)
        )
    )
    strategy = discovery["strategy"]
    expected_android_circle = (
        _known_android_hint(
            game_id, (android_frames.shape[2], android_frames.shape[1])
        )
        if strategy == "legacy_hint" and android_selected_crop_xywh is None
        else None
    )
    source_discovery = (
        discovery
        if strategy == "relative_prior" and android_selected_crop_xywh is None
        else None
    )
    android = calibrate_source_frames(
        android_frames,
        output_path / "android_phone",
        image_source="android_scrcpy",
        coordinate_space_id="android_logical_display_pixels",
        selected_crop_xywh=android_selected_crop_xywh,
        expected_circle_xy_radius=expected_android_circle,
        discovery_config=source_discovery,
        phone_surface_orientation=orientation,
        provenance={**provenance, "stream_id": "android_phone"},
    )
    hik_output = output_path / hik_stream
    if hik_selected_crop_xywh is not None:
        hik = calibrate_source_frames(
            hik_frames,
            hik_output,
            image_source="hik_mvs_manual_override",
            coordinate_space_id=hik_coordinate_space,
            selected_crop_xywh=hik_selected_crop_xywh,
            phone_surface_orientation=orientation,
            provenance={
                **provenance,
                "stream_id": hik_stream,
                "manual_override": True,
            },
        )
    else:
        alignment = estimate_current_cross_source_homography(
            android_frames, hik_frames
        )
        hik = create_current_hik_observation(
            android,
            hik_frames,
            alignment,
            hik_output,
            image_source=(
                "hik_mvs_native" if hik_stream == "hik_full" else "hik_rig_normalized"
            ),
            coordinate_space_id=hik_coordinate_space,
            provenance={
                **provenance,
                "stream_id": hik_stream,
                "geometry_source": "current_android_to_hik_homography",
            },
        )
    android_source = next(
        (
            item
            for item in (session.manifest.get("frame_sources") or [])
            if item.get("stream_id") == "android_phone"
        ),
        {},
    )
    phone_id = str(
        (android_source.get("shared_capture") or {}).get("serial")
        or context.get("phone_serial")
        or "unknown-phone"
    )
    store = ScopedCalibrationProfileStore(Path(profiles_root))
    revision_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    phone_key = ScopedProfileKey("phone_game", phone_id, game_id)
    phone_revision = store.create_revision_directory(phone_key, revision_id)
    phone_profile = store.publish(
        phone_key,
        phone_revision,
        {
            "session_path": str(Path(session_path).resolve()),
            "coordinate_space": "phone_natural_display_pixels",
            "calibration": {
                "canonical_phone_crop_xywh": android["canonical_phone_crop_xywh"],
                "outer_boundary": android["outer_boundary"],
                "phone_surface_orientation": orientation,
            },
            "artifacts": {"minimap_calibration": str((output_path / "android_phone" / "minimap_calibration.json").resolve())},
            "reuse": {
                "scope": "same physical phone and game UI layout",
                "camera_independent": True,
                "rule": "A camera or rig is neither required nor referenced by this profile.",
            },
            "notes": {"status": "Review the saved heatmap, fit overlay, and exact mask before acceptance."},
        },
    )
    rig_profile = None
    if rig_calibration is not None:
        composition = compose_rig_game_profile(rig_calibration, android, hik)
        rig_key = ScopedProfileKey("rig_game", _rig_id(rig_calibration), game_id)
        rig_revision = store.create_revision_directory(rig_key, revision_id)
        rig_profile = store.publish(
            rig_key,
            rig_revision,
            {
                **composition,
                "session_path": str(Path(session_path).resolve()),
                "phone_game_profile": str((phone_revision / "profile.json").resolve()),
                "artifacts": {
                    "native_hik_minimap_calibration": str((hik_output / "minimap_calibration.json").resolve()),
                    "phone_game_profile": str((phone_revision / "profile.json").resolve()),
                },
                "reuse": {
                    "scope": "same saved rig calibration revision and game UI layout",
                    "small_shift_rule": "Rerun headless rig calibration after a physical shift, then compose a new rig-game revision.",
                },
                "notes": {"optical_fit": "None. Runtime uses only the referenced base rig calibration."},
            },
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "review_required",
        "session_path": str(Path(session_path).resolve()),
        "game_id": game_id,
        "native_hik_calibration": str((hik_output / "minimap_calibration.json").resolve()),
        "hik_stream_id": hik_stream,
        "hik_geometry_reusable": False,
        "android_discovery": discovery,
        "phone_game_profile": str((phone_revision / "profile.json").resolve()),
        "rig_game_profile": (
            str((store.profile_directory(ScopedProfileKey("rig_game", _rig_id(rig_calibration), game_id)) / "current.json").resolve())
            if rig_calibration is not None else None
        ),
        "rig_composition_skipped": rig_calibration is None,
    }
    _atomic_json(output_path / "calibration_summary.json", summary)
    write_commented_yaml(
        output_path / "calibration_summary.yaml",
        summary,
        header=MINIMAP_HEADER,
        section_comments=MINIMAP_COMMENTS,
    )
    return {"summary": summary, "android": android, "hik": hik, "phone_profile": phone_profile, "rig_profile": rig_profile}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Calibrate mini-map isolation from a recorded native-HIK zigzag session")
    value.add_argument("session", type=Path)
    value.add_argument("--output", type=Path)
    value.add_argument("--profiles-root", type=Path, default=Path("profiles"))
    value.add_argument("--rig-calibration", type=Path)
    value.add_argument("--android-crop", type=_parse_crop)
    value.add_argument(
        "--hik-crop",
        type=_parse_crop,
        help=(
            "legacy diagnostic override; automatic calibration derives the current "
            "HIK observation from Android and never reuses a HIK crop"
        ),
    )
    value.add_argument(
        "--android-discovery",
        choices=("relative-prior", "unrestricted", "legacy-hint"),
        default="relative-prior",
    )
    value.add_argument(
        "--android-center-region",
        default="0,0,0.35,0.35",
        help="relative x0,y0,x1,y1 bounds for automatic Android circle centers",
    )
    value.add_argument(
        "--android-radius-fraction",
        default="0.07,0.22",
        help="minimum,maximum radius as fractions of the shorter Android dimension",
    )
    value.add_argument(
        "--android-min-visible",
        type=float,
        default=0.85,
        help="minimum fraction of the candidate circumference visible in Android",
    )
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    output = arguments.output or Path("artifacts") / "game-minimap-calibration-{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    result = calibrate_zigzag_session(
        arguments.session,
        output,
        profiles_root=arguments.profiles_root,
        rig_calibration=arguments.rig_calibration,
        android_selected_crop_xywh=arguments.android_crop,
        hik_selected_crop_xywh=arguments.hik_crop,
        android_discovery=android_discovery_config(
            arguments.android_discovery,
            arguments.android_center_region,
            arguments.android_radius_fraction,
            arguments.android_min_visible,
        ),
    )
    print("Mini-map calibration: {}".format(Path(output).resolve()))
    print("Phone-game profile: {}".format(result["summary"]["phone_game_profile"]))
    if result["summary"]["rig_game_profile"]:
        print("Rig-game profile: {}".format(result["summary"]["rig_game_profile"]))
    else:
        print("Rig-game profile: skipped (no optional rig calibration supplied)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
