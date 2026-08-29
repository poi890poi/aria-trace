"""POC: fit the mini-map in ADB only, then project it into HIK adapter space."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from acquisition.commented_yaml import write_commented_yaml
from acquisition.minimap_calibration import (
    _color_heatmap,
    _stacked_difference_heatmap,
    calibrate_minimap_boundary_frames,
)
from acquisition.session import SessionReader


HEADER = """# ADB-only mini-map boundary projection POC.
#
# The circular boundary is fitted only in ADB pixels by the verified mini-map
# backend. Camera-adapter geometry is obtained only by applying the saved rig
# coordinate transform. HIK images are visualization evidence, never fitting
# input. Pixel coordinates use top-left [0, 0], +X right, +Y down."""

COMMENTS = {
    "source": "Immutable session, streams, and rig calibration used by this POC.",
    "adb_boundary": "The sole fitted mini-map result, expressed in the full ADB raster.",
    "camera_adapter_projection": "ADB result transformed with the saved rig matrix; no HIK fitting.",
    "evidence": "Images for reviewing discovery, verified ADB fit, transformed mask, and HIK overlays.",
}


def _parse_crop(text: Optional[str]) -> Optional[list[int]]:
    if text is None:
        return None
    values = [int(item.strip()) for item in text.split(",")]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("crop must be x,y,width,height")
    return values


def _read_frames(
    reader: SessionReader, stream_id: str, maximum: int = 64
) -> tuple[np.ndarray, list[int]]:
    records = list(reader.frames_by_stream.get(stream_id) or [])
    if len(records) < 12:
        raise ValueError("{} contains fewer than 12 frames".format(stream_id))
    selected = np.unique(
        np.linspace(0, len(records) - 1, min(maximum, len(records))).round().astype(int)
    )
    capture = cv2.VideoCapture(str(reader.video_path(stream_id)))
    if not capture.isOpened():
        raise RuntimeError("Cannot open {} video".format(stream_id))
    frames = []
    indices = []
    try:
        for record_index in selected:
            video_index = int(records[int(record_index)]["frame_index"])
            capture.set(cv2.CAP_PROP_POS_FRAMES, video_index)
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(frame)
                indices.append(video_index)
    finally:
        capture.release()
    if len(frames) < 12:
        raise RuntimeError("Could decode fewer than 12 {} frames".format(stream_id))
    return np.stack(frames), indices


def _ring_score(
    center_x: float,
    center_y: float,
    radius: float,
    edges: np.ndarray,
    heat: np.ndarray,
) -> dict:
    height, width = heat.shape
    y0 = max(0, int(math.floor(center_y - radius - 6)))
    y1 = min(height, int(math.ceil(center_y + radius + 7)))
    x0 = max(0, int(math.floor(center_x - radius - 6)))
    x1 = min(width, int(math.ceil(center_x + radius + 7)))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    ring = np.abs(distance - radius) <= max(2.0, radius * 0.025)
    inside = distance <= radius * 0.82
    outside = (distance >= radius * 1.08) & (distance <= radius * 1.24)
    ring_edge = float(np.mean(edges[y0:y1, x0:x1][ring]) / 255.0)
    inside_motion = float(np.mean(heat[y0:y1, x0:x1][inside]))
    outside_motion = float(np.mean(heat[y0:y1, x0:x1][outside]))
    motion_ratio = outside_motion / max(inside_motion, 1.0e-6)
    stable_disc_contrast = max(0.0, outside_motion - inside_motion)
    return {
        "ring_edge_fraction": ring_edge,
        "inside_motion": inside_motion,
        "outside_motion": outside_motion,
        "outside_inside_motion_ratio": motion_ratio,
        "stable_disc_contrast": stable_disc_contrast,
        "coarse_score": ring_edge * math.sqrt(max(stable_disc_contrast, 0.01)) * min(3.0, motion_ratio),
    }


def _three_class_otsu(image: np.ndarray) -> tuple[int, int, np.ndarray]:
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    probability = histogram / max(float(histogram.sum()), 1.0)
    cumulative_weight = np.cumsum(probability)
    cumulative_mean = np.cumsum(probability * np.arange(256, dtype=np.float64))
    total_mean = cumulative_mean[-1]
    best_score = -1.0
    best = (64, 160)
    for first in range(1, 254):
        weight0 = cumulative_weight[first]
        if weight0 <= 1.0e-8:
            continue
        mean0 = cumulative_mean[first] / weight0
        for second in range(first + 1, 255):
            weight1 = cumulative_weight[second] - cumulative_weight[first]
            weight2 = 1.0 - cumulative_weight[second]
            if min(weight1, weight2) <= 1.0e-8:
                continue
            mean1 = (cumulative_mean[second] - cumulative_mean[first]) / weight1
            mean2 = (total_mean - cumulative_mean[second]) / weight2
            score = (
                weight0 * (mean0 - total_mean) ** 2
                + weight1 * (mean1 - total_mean) ** 2
                + weight2 * (mean2 - total_mean) ** 2
            )
            if score > best_score:
                best_score = score
                best = (first, second)
    classes = np.zeros_like(image, dtype=np.uint8)
    classes[image > best[0]] = 127
    classes[image > best[1]] = 255
    return best[0], best[1], classes


def _candidate_circles(
    frames: np.ndarray,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    average = frames.mean(axis=0).astype(np.uint8)
    heat = cv2.GaussianBlur(_stacked_difference_heatmap(frames), (0, 0), 2.0)
    gray = cv2.GaussianBlur(cv2.cvtColor(average, cv2.COLOR_BGR2GRAY), (7, 7), 1.4)
    heat_u8 = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    first_threshold, _, heat_classes = _three_class_otsu(heat_u8)
    low_motion = (heat_u8 <= first_threshold).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    low_motion = cv2.morphologyEx(low_motion, cv2.MORPH_CLOSE, kernel)
    low_motion = cv2.morphologyEx(low_motion, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    low_motion_edges = cv2.Canny(low_motion, 40, 120)
    edges = cv2.Canny(gray, 45, 135)
    height, width = gray.shape
    scale = 0.5
    shorter = min(width, height)
    minimum_radius = max(32, int(round(shorter * 0.045)))
    maximum_radius = max(minimum_radius + 4, int(round(shorter * 0.18)))
    detected = []
    for source, threshold in (
        (gray, 24.0),
        (heat_u8, 18.0),
        (cv2.GaussianBlur(low_motion, (9, 9), 1.8), 15.0),
    ):
        reduced = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        circles = cv2.HoughCircles(
            reduced,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(12.0, minimum_radius * scale * 0.45),
            param1=80.0,
            param2=threshold,
            minRadius=max(4, int(minimum_radius * scale)),
            maxRadius=max(6, int(maximum_radius * scale)),
        )
        if circles is not None:
            detected.extend((np.asarray(circles[0]) / scale).tolist())
    contours, _ = cv2.findContours(low_motion, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if area <= 0.0 or perimeter <= 0.0:
            continue
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        fill = area / max(math.pi * radius * radius, 1.0)
        if minimum_radius <= radius <= maximum_radius and circularity >= 0.35 and fill >= 0.45:
            detected.append([center_x, center_y, radius])
    candidates = []
    for center_x, center_y, radius in detected:
        if not (
            radius <= center_x < width - radius
            and radius <= center_y < height - radius
        ):
            continue
        if any(
            math.hypot(center_x - item["center_x"], center_y - item["center_y"])
            < max(8.0, 0.18 * radius)
            and abs(radius - item["radius"]) < max(8.0, 0.15 * radius)
            for item in candidates
        ):
            continue
        score = _ring_score(center_x, center_y, radius, edges, heat)
        candidates.append(
            {
                "center_x": float(center_x),
                "center_y": float(center_y),
                "radius": float(radius),
                **score,
            }
        )
    candidates.sort(key=lambda item: item["coarse_score"], reverse=True)
    return candidates[:24], average, heat, heat_classes, low_motion_edges


def _crop_for_circle(circle: dict, frame_size: Sequence[int]) -> list[int]:
    frame_width, frame_height = map(int, frame_size)
    radius = float(circle["radius"])
    half = max(48, int(math.ceil(radius * 1.30)))
    center_x = int(round(circle["center_x"]))
    center_y = int(round(circle["center_y"]))
    x0 = max(0, center_x - half)
    y0 = max(0, center_y - half)
    x1 = min(frame_width, center_x + half + 1)
    y1 = min(frame_height, center_y + half + 1)
    return [x0, y0, x1 - x0, y1 - y0]


def _refine_candidates(
    frames: np.ndarray,
    candidates: Sequence[dict],
    selected_crop: Optional[Sequence[int]],
) -> tuple[list[int], dict, list[dict]]:
    height, width = frames.shape[1:3]
    if selected_crop is not None:
        x, y, crop_width, crop_height = map(int, selected_crop)
        seed = {
            "center_x": x + crop_width / 2.0,
            "center_y": y + crop_height / 2.0,
            "radius": min(crop_width, crop_height) * 0.46,
            "coarse_score": 0.0,
        }
        trials = [seed]
    else:
        trials = list(candidates[:12])
    if not trials:
        raise RuntimeError("No full-ADB circular candidates were found")
    refined = []
    for candidate in trials:
        crop = list(map(int, selected_crop)) if selected_crop is not None else _crop_for_circle(candidate, [width, height])
        x, y, crop_width, crop_height = crop
        if min(crop_width, crop_height) < 80 or x + crop_width > width or y + crop_height > height:
            continue
        crop_frames = frames[:, y : y + crop_height, x : x + crop_width]
        expected = [candidate["center_x"] - x, candidate["center_y"] - y]
        radius = float(candidate["radius"])
        selected_by_user = selected_crop is not None
        try:
            result = calibrate_minimap_boundary_frames(
                crop_frames,
                Path("unused"),
                config={
                    "expected_center_xy": expected,
                    "center_search_radius_px": max(
                        16.0, radius * (0.25 if selected_by_user else 0.35)
                    ),
                    "radius_range_px": (
                        [radius * 0.86, radius * 1.15]
                        if selected_by_user
                        else [max(20.0, radius * 0.72), radius * 1.30]
                    ),
                },
                write_evidence=False,
            )
        except Exception as exc:
            refined.append({**candidate, "crop_xywh": crop, "error": "{}: {}".format(type(exc).__name__, exc)})
            continue
        boundary = result["outer_boundary"]
        refined.append(
            {
                **candidate,
                "crop_xywh": crop,
                "verified_center_adb_xy": [boundary["center_x"] + x, boundary["center_y"] + y],
                "verified_radius_px": boundary["radius"],
                "verified_confidence": boundary["confidence"],
                "verified_confidence_level": boundary["confidence_level"],
                "verified_boundary": boundary,
            }
        )
    successful = [item for item in refined if "verified_boundary" in item]
    if not successful:
        raise RuntimeError("Verified boundary backend rejected every full-ADB candidate")
    successful.sort(
        key=lambda item: (item["verified_confidence"], item.get("coarse_score", 0.0)),
        reverse=True,
    )
    selected = successful[0]
    return selected["crop_xywh"], selected, refined


def _rotate_clockwise(image: np.ndarray, turns: int) -> np.ndarray:
    turns %= 4
    if turns == 0:
        return image
    if turns == 1:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if turns == 2:
        return cv2.rotate(image, cv2.ROTATE_180)
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _rotate_point_clockwise(
    point_xy: Sequence[float], image_size_px: Sequence[int], turns: int
) -> np.ndarray:
    x, y = map(float, point_xy)
    width, height = map(int, image_size_px)
    turns %= 4
    if turns == 0:
        return np.asarray([x, y], dtype=np.float64)
    if turns == 1:
        return np.asarray([height - 1 - y, x], dtype=np.float64)
    if turns == 2:
        return np.asarray([width - 1 - x, height - 1 - y], dtype=np.float64)
    return np.asarray([y, width - 1 - x], dtype=np.float64)


def run(
    session_path: Path,
    rig_calibration: Path,
    output_path: Path,
    adb_crop: Optional[Sequence[int]] = None,
) -> dict:
    reader = SessionReader(session_path)
    orientation = dict((reader.manifest.get("context") or {}).get("phone_surface_orientation") or {})
    turns = int(orientation.get("quarter_turns_clockwise_from_natural", 0))
    calibration = json.loads(Path(rig_calibration).read_text(encoding="utf-8"))
    calibration_turns = int(calibration["phone"]["orientation_quarter_turns"]) % 4
    if turns % 4 != calibration_turns:
        raise RuntimeError(
            "This POC requires rig calibration and game capture in the same "
            "Android orientation (rig {}, session {})".format(calibration_turns, turns % 4)
        )
    normalization = calibration["normalization"]
    adapter_width, adapter_height = map(int, normalization["output_size_px"])
    origin_adb = np.asarray(normalization["origin_screen_xy"], dtype=np.float64)
    scale_adb_per_adapter = np.asarray(
        normalization.get("screen_units_per_output_pixel_xy", [1.0, 1.0]),
        dtype=np.float64,
    )
    adapter_to_adb_3x3 = np.asarray(
        [
            [scale_adb_per_adapter[0], 0.0, origin_adb[0]],
            [0.0, scale_adb_per_adapter[1], origin_adb[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    adb_to_adapter_3x3 = np.linalg.inv(adapter_to_adb_3x3)
    adb_frames, adb_indices = _read_frames(reader, "android_phone", 64)
    hik_frames, hik_indices = _read_frames(reader, "hik_phone", 12)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=False)

    candidates, adb_average, adb_heat, heat_classes, heat_edges = _candidate_circles(adb_frames)
    crop, selected, trials = _refine_candidates(adb_frames, candidates, adb_crop)
    x, y, width, height = crop
    verified = calibrate_minimap_boundary_frames(
        adb_frames[:, y : y + height, x : x + width],
        output_path / "verified_adb_boundary",
        config={
            "expected_center_xy": [
                selected["verified_center_adb_xy"][0] - x,
                selected["verified_center_adb_xy"][1] - y,
            ],
            "center_search_radius_px": 14.0,
            "radius_range_px": [
                selected["verified_radius_px"] - 10.0,
                selected["verified_radius_px"] + 10.0,
            ],
        },
        write_evidence=True,
    )
    boundary = verified["outer_boundary"]
    center_adb = np.asarray([[boundary["center_x"] + x, boundary["center_y"] + y]], dtype=np.float64)
    radius = float(boundary["radius"])
    center_adapter = cv2.perspectiveTransform(
        center_adb.reshape(1, -1, 2).astype(np.float64), adb_to_adapter_3x3
    ).reshape(-1, 2)[0]
    probes_adb = np.asarray(
        [center_adb[0] + [radius, 0], center_adb[0] + [0, radius]],
        dtype=np.float64,
    )
    probes_adapter = cv2.perspectiveTransform(
        probes_adb.reshape(1, -1, 2).astype(np.float64), adb_to_adapter_3x3
    ).reshape(-1, 2)
    radii_adapter = np.linalg.norm(probes_adapter - center_adapter[None, :], axis=1)
    angles = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
    perimeter_x = center_adapter[0] + radii_adapter[0] * np.cos(angles)
    perimeter_y = center_adapter[1] + radii_adapter[1] * np.sin(angles)
    visible_perimeter = (
        (perimeter_x >= 0.0)
        & (perimeter_x < adapter_width)
        & (perimeter_y >= 0.0)
        & (perimeter_y < adapter_height)
    )
    adapter_bounds = [
        float(center_adapter[0] - radii_adapter[0]),
        float(center_adapter[1] - radii_adapter[1]),
        float(center_adapter[0] + radii_adapter[0]),
        float(center_adapter[1] + radii_adapter[1]),
    ]
    clipped_by_adapter_frame = not bool(np.all(visible_perimeter))

    adb_overlay = adb_average.copy()
    cv2.circle(adb_overlay, tuple(np.round(center_adb[0]).astype(int)), round(radius), (0, 255, 255), 3, cv2.LINE_AA)
    cv2.drawMarker(adb_overlay, tuple(np.round(center_adb[0]).astype(int)), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
    cv2.imwrite(str(output_path / "adb_full_boundary_overlay.png"), adb_overlay)
    cv2.imwrite(str(output_path / "adb_stacked_difference_heatmap.png"), _color_heatmap(adb_heat))
    cv2.imwrite(str(output_path / "adb_heatmap_three_class_otsu.png"), heat_classes)
    cv2.imwrite(str(output_path / "adb_heatmap_low_motion_edges.png"), heat_edges)

    candidate_overlay = adb_average.copy()
    for index, candidate in enumerate(candidates[:12]):
        color = (0, 255, 0) if index == 0 else (0, 120, 255)
        cv2.circle(candidate_overlay, (round(candidate["center_x"]), round(candidate["center_y"])), round(candidate["radius"]), color, 2, cv2.LINE_AA)
        cv2.putText(candidate_overlay, str(index + 1), (round(candidate["center_x"]), round(candidate["center_y"])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path / "adb_discovery_candidates.png"), candidate_overlay)

    adb_mask = np.zeros(adb_frames.shape[1:3], np.uint8)
    cv2.circle(adb_mask, tuple(np.round(center_adb[0]).astype(int)), round(radius), 255, -1)
    adapter_mask = cv2.warpPerspective(
        adb_mask,
        adb_to_adapter_3x3,
        (adapter_width, adapter_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    cv2.imwrite(str(output_path / "camera_adapter_projected_mask.png"), adapter_mask)

    adb_visible_x, adb_visible_y = np.rint(origin_adb).astype(int)
    hik_capture = dict((reader.manifest.get("context") or {}).get("hik_capture") or {})
    output_turns = int(
        hik_capture.get("output_image_quarter_turns_clockwise_from_phone_natural", 0)
    ) % 4
    if output_turns % 2:
        aligned_width, aligned_height = adapter_height, adapter_width
    else:
        aligned_width, aligned_height = adapter_width, adapter_height
    center_hik_video = _rotate_point_clockwise(
        center_adapter, [adapter_width, adapter_height], output_turns
    )
    adapter_overlays = []
    hik_video_overlays = []
    inverse_turns = (-output_turns) % 4
    for index, frame in zip(hik_indices, hik_frames):
        aligned = frame[:aligned_height, :aligned_width]
        hik_overlay = aligned.copy()
        cv2.circle(
            hik_overlay,
            tuple(np.round(center_hik_video).astype(int)),
            round(radius),
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )
        hik_name = "hik_video_boundary_overlay_{:06d}.png".format(index)
        cv2.imwrite(str(output_path / hik_name), hik_overlay)
        hik_video_overlays.append((hik_name, hik_overlay))
        adapter_image = _rotate_clockwise(aligned, inverse_turns)
        overlay = adapter_image.copy()
        axes = tuple(max(1, round(value)) for value in radii_adapter)
        cv2.ellipse(
            overlay,
            tuple(np.round(center_adapter).astype(int)),
            axes,
            0.0,
            0.0,
            360.0,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.drawMarker(overlay, tuple(np.round(center_adapter).astype(int)), (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
        name = "camera_adapter_boundary_overlay_{:06d}.png".format(index)
        cv2.imwrite(str(output_path / name), overlay)
        adapter_overlays.append((name, overlay))
    if hik_video_overlays:
        cv2.imwrite(
            str(output_path / "hik_video_first_frame.png"),
            hik_frames[0][:aligned_height, :aligned_width],
        )
    mosaic_columns = 3
    thumb_width = 480
    thumbs = []
    for _, image in adapter_overlays:
        thumb_height = max(1, round(image.shape[0] * thumb_width / image.shape[1]))
        thumbs.append(cv2.resize(image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA))
    if thumbs:
        row_height = max(item.shape[0] for item in thumbs)
        blank = np.zeros((row_height, thumb_width, 3), np.uint8)
        rows = []
        for offset in range(0, len(thumbs), mosaic_columns):
            row = thumbs[offset : offset + mosaic_columns]
            row += [blank] * (mosaic_columns - len(row))
            row = [cv2.copyMakeBorder(item, 0, row_height - item.shape[0], 0, 0, cv2.BORDER_CONSTANT) for item in row]
            rows.append(np.hstack(row))
        cv2.imwrite(str(output_path / "camera_adapter_boundary_overlays_mosaic.png"), np.vstack(rows))

    result = {
        "schema_version": "1.0",
        "status": "review_required",
        "method": "fit_ADB_once_then_apply_saved_rig_coordinate_transform",
        "source": {
            "session": str(Path(session_path).resolve()),
            "rig_calibration": str(Path(rig_calibration).resolve()),
            "adb_stream": "android_phone",
            "hik_stream": "hik_phone",
            "adb_frame_indices": adb_indices,
            "hik_frame_indices": hik_indices,
        },
        "discovery": {
            "method": "full_ADB_Hough_candidates_then_verified_boundary_backend",
            "selected_crop_xywh": crop,
            "candidate_count": len(candidates),
            "selected_candidate": selected,
            "trials": trials,
        },
        "adb_boundary": {
            "coordinate_space": "android_logical_display_pixels",
            "center_xy": center_adb[0].tolist(),
            "radius_px": radius,
            "confidence": boundary["confidence"],
            "confidence_level": boundary["confidence_level"],
            "verified_backend_metrics": boundary,
        },
        "camera_adapter_projection": {
            "coordinate_space": "hik_rig_rectified_visible_phone_pixels",
            "center_xy": center_adapter.tolist(),
            "radius_x_px": float(radii_adapter[0]),
            "radius_y_px": float(radii_adapter[1]),
            "adapter_size_px": [adapter_width, adapter_height],
            "projected_bounds_xyxy": adapter_bounds,
            "clipped_by_adapter_frame": clipped_by_adapter_frame,
            "visible_perimeter_fraction": float(np.mean(visible_perimeter)),
            "rig_calibration_orientation_quarter_turns": calibration_turns,
            "session_orientation_quarter_turns": turns % 4,
            "same_orientation_required_by_poc": True,
            "rig_normalization_origin_adb_xy": origin_adb.tolist(),
            "screen_units_per_adapter_pixel_xy": scale_adb_per_adapter.tolist(),
            "adb_to_camera_adapter_3x3": adb_to_adapter_3x3.tolist(),
            "fit_performed_in_camera_space": False,
        },
        "evidence": {
            "adb_candidates": "adb_discovery_candidates.png",
            "adb_heatmap": "adb_stacked_difference_heatmap.png",
            "adb_heatmap_three_class_otsu": "adb_heatmap_three_class_otsu.png",
            "adb_heatmap_low_motion_edges": "adb_heatmap_low_motion_edges.png",
            "adb_full_boundary": "adb_full_boundary_overlay.png",
            "verified_adb_directory": "verified_adb_boundary",
            "camera_adapter_mask": "camera_adapter_projected_mask.png",
            "hik_video_first_frame": "hik_video_first_frame.png",
            "hik_video_overlays": [name for name, _ in hik_video_overlays],
            "camera_adapter_overlays": [name for name, _ in adapter_overlays],
            "camera_adapter_mosaic": "camera_adapter_boundary_overlays_mosaic.png",
        },
    }
    (output_path / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_commented_yaml(
        output_path / "result.yaml",
        result,
        header=HEADER,
        section_comments=COMMENTS,
    )
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("session", type=Path)
    value.add_argument("rig_calibration", type=Path)
    value.add_argument("output", type=Path)
    value.add_argument("--adb-crop", type=_parse_crop)
    return value


def main(argv=None) -> int:
    arguments = parser().parse_args(argv)
    result = run(
        arguments.session,
        arguments.rig_calibration,
        arguments.output,
        arguments.adb_crop,
    )
    print("ADB boundary: center={} radius={:.2f} confidence={:.3f}".format(
        [round(value, 2) for value in result["adb_boundary"]["center_xy"]],
        result["adb_boundary"]["radius_px"],
        result["adb_boundary"]["confidence"],
    ))
    print("Camera adapter projection: center={} radii=({:.2f}, {:.2f})".format(
        [round(value, 2) for value in result["camera_adapter_projection"]["center_xy"]],
        result["camera_adapter_projection"]["radius_x_px"],
        result["camera_adapter_projection"]["radius_y_px"],
    ))
    print("Evidence: {}".format(arguments.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
