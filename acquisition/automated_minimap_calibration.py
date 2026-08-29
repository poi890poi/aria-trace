"""Headless mini-map calibration from native HIK and synchronized ADB frames.

The only learned geometry in this module is the mini-map crop and circular
boundary in each source's own pixel coordinates.  Supplying a rig calibration
composes its existing camera-to-phone mapping with the phone-game result; it
never fits or replaces optical rig geometry.
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
    frames: np.ndarray, expected: Optional[Sequence[float]] = None
) -> list[dict]:
    average = frames.mean(axis=0).astype(np.uint8)
    gray = cv2.GaussianBlur(cv2.cvtColor(average, cv2.COLOR_BGR2GRAY), (7, 7), 1.4)
    equalized = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    heat = cv2.GaussianBlur(_stacked_difference_heatmap(frames), (0, 0), 2.0)
    heat_image = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    height, width = gray.shape
    minimum_radius = max(18, int(round(min(width, height) * 0.025)))
    maximum_radius = max(minimum_radius + 2, int(round(min(width, height) * 0.22)))
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
        score = ring_edge * math.sqrt(max(motion_inside, 0.01)) / (
            1.0 + 0.25 * max(motion_outside - motion_inside, 0.0)
        )
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
    phone_surface_orientation: Optional[Mapping[str, object]] = None,
    provenance: Optional[Mapping[str, object]] = None,
) -> dict:
    """Discover a crop, then delegate the fit to the verified boundary routine."""

    if frames.ndim != 4 or len(frames) < 12:
        raise ValueError("Zigzag calibration needs at least 12 color frames")
    height, width = frames.shape[1:3]
    candidates = _circle_candidates(frames, expected_circle_xy_radius)
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
        method = "automatic_ranked_circle_search_then_verified_boundary_fit"
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
    mapped = _transform_points(
        [hik_center, [hik_center[0] + radius, hik_center[1]], [hik_center[0], hik_center[1] + radius]],
        matrix,
    )
    mapped_radius = float(
        np.mean([np.linalg.norm(mapped[1] - mapped[0]), np.linalg.norm(mapped[2] - mapped[0])])
    )
    phone_center = np.asarray(
        phone_result["outer_boundary"]["canonical_phone_center_xy"], np.float64
    )
    center_error = float(np.linalg.norm(mapped[0] - phone_center))
    phone_radius = float(phone_result["outer_boundary"]["radius"])
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
            "coordinate_space": "native_hik_sensor_bgr_pixels",
            "crop_xywh": list(hik_result["crop_xywh"]),
            "center_xy": list(map(float, hik_center)),
            "radius_px": radius,
        },
        "cross_source_coordinate_check": {
            "method": "apply_saved_rig_homography_only_no_fitting",
            "mapped_hik_center_in_phone_xy": mapped[0].tolist(),
            "adb_phone_center_xy": phone_center.tolist(),
            "center_error_phone_px": center_error,
            "mapped_hik_radius_phone_px": mapped_radius,
            "adb_radius_phone_px": phone_radius,
            "radius_error_phone_px": abs(mapped_radius - phone_radius),
            "non_gating": True,
        },
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
    if "hik_full" not in session.frames_by_stream:
        raise ValueError(
            "Session has no native hik_full stream; rectified rig-dependent HIK "
            "sessions cannot be relabeled as native sensor calibration"
        )
    hik_stream = "hik_full"
    hik_frames, _ = read_session_stream_frames(session, hik_stream)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=False)
    provenance = {"session_path": str(Path(session_path).resolve())}
    android = calibrate_source_frames(
        android_frames,
        output_path / "android_phone",
        image_source="android_scrcpy",
        coordinate_space_id="android_logical_display_pixels",
        selected_crop_xywh=android_selected_crop_xywh,
        expected_circle_xy_radius=(
            None if android_selected_crop_xywh is not None
            else _known_android_hint(game_id, (android_frames.shape[2], android_frames.shape[1]))
        ),
        phone_surface_orientation=orientation,
        provenance={**provenance, "stream_id": "android_phone"},
    )
    hik = calibrate_source_frames(
        hik_frames,
        output_path / "hik_full",
        image_source="hik_mvs_native",
        coordinate_space_id="native_hik_sensor_bgr_pixels",
        selected_crop_xywh=hik_selected_crop_xywh,
        phone_surface_orientation=orientation,
        provenance={**provenance, "stream_id": hik_stream},
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
                    "native_hik_minimap_calibration": str((output_path / "hik_full" / "minimap_calibration.json").resolve()),
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
        "native_hik_calibration": str((output_path / "hik_full" / "minimap_calibration.json").resolve()),
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
    value.add_argument("--hik-crop", type=_parse_crop)
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
