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
from acquisition.rig_calibration.hik.spaces import RigCalibratedSpaceConverter
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
    if adb_crop is None:
        raise ValueError(
            "This POC requires a reviewed wide --adb-crop; automatic candidate "
            "selection was removed"
        )
    reader = SessionReader(session_path)
    orientation = dict((reader.manifest.get("context") or {}).get("phone_surface_orientation") or {})
    turns = int(orientation.get("quarter_turns_clockwise_from_natural", 0))
    calibration = json.loads(Path(rig_calibration).read_text(encoding="utf-8"))
    calibration_turns = int(calibration["phone"]["orientation_quarter_turns"]) % 4
    converter = RigCalibratedSpaceConverter(calibration, turns)
    adapter_width, adapter_height = converter.adapter_size_px
    adapter_to_adb_3x3 = np.asarray(converter.adapter_to_adb_3x3, np.float64)
    adb_to_adapter_3x3 = np.asarray(converter.adb_to_adapter_3x3, np.float64)
    adb_frames, adb_indices = _read_frames(reader, "android_phone", 64)
    hik_frames, hik_indices = _read_frames(reader, "hik_phone", 12)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=False)

    adb_average = adb_frames.mean(axis=0).astype(np.uint8)
    adb_heat = _stacked_difference_heatmap(adb_frames)
    crop = list(map(int, adb_crop))
    x, y, width, height = crop
    if (
        min(x, y) < 0
        or min(width, height) < 80
        or x + width > adb_frames.shape[2]
        or y + height > adb_frames.shape[1]
    ):
        raise ValueError("Reviewed ADB crop is outside the full ADB raster")
    seed_radius = min(width, height) * 0.46
    verified = calibrate_minimap_boundary_frames(
        adb_frames[:, y : y + height, x : x + width],
        output_path / "verified_adb_boundary",
        config={
            "expected_center_xy": [width / 2.0, height / 2.0],
            "center_search_radius_px": max(16.0, seed_radius * 0.25),
            "radius_range_px": [
                seed_radius * 0.86,
                seed_radius * 1.15,
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

    input_overlay = adb_average.copy()
    cv2.rectangle(
        input_overlay,
        (x, y),
        (x + width - 1, y + height - 1),
        (255, 255, 0),
        3,
    )
    cv2.imwrite(str(output_path / "adb_input_rectangle.png"), input_overlay)

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

    hik_capture = dict((reader.manifest.get("context") or {}).get("hik_capture") or {})
    recorded_output_turns = int(
        hik_capture.get("output_image_quarter_turns_clockwise_from_phone_natural", 0)
    ) % 4
    corrected_output_turns = int(
        converter.output_image_quarter_turns_clockwise_from_phone_natural
    ) % 4
    correction_turns = (corrected_output_turns - recorded_output_turns) % 4
    if recorded_output_turns % 2:
        aligned_width, aligned_height = adapter_height, adapter_width
    else:
        aligned_width, aligned_height = adapter_width, adapter_height
    center_hik_video = _rotate_point_clockwise(
        center_adapter, [adapter_width, adapter_height], recorded_output_turns
    )
    center_corrected_hik_video = _rotate_point_clockwise(
        center_adapter, [adapter_width, adapter_height], corrected_output_turns
    )
    adapter_overlays = []
    hik_video_overlays = []
    corrected_hik_video_overlays = []
    inverse_turns = (-recorded_output_turns) % 4
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
        corrected_image = _rotate_clockwise(aligned, correction_turns)
        corrected_overlay = corrected_image.copy()
        cv2.circle(
            corrected_overlay,
            tuple(np.round(center_corrected_hik_video).astype(int)),
            round(radius),
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )
        corrected_name = "corrected_hik_video_boundary_overlay_{:06d}.png".format(index)
        cv2.imwrite(str(output_path / corrected_name), corrected_overlay)
        corrected_hik_video_overlays.append((corrected_name, corrected_overlay))
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
        cv2.imwrite(
            str(output_path / "corrected_hik_video_first_frame.png"),
            _rotate_clockwise(
                hik_frames[0][:aligned_height, :aligned_width], correction_turns
            ),
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
            "method": "reviewed_wide_ADB_crop_then_verified_boundary_backend_only",
            "selected_crop_xywh": crop,
            "candidate_count": 0,
            "automatic_candidate_selection": False,
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
            "surface_orientation_is_handled_by_saved_rig_transform": True,
            "corrected_adapter_bounds_in_adb_xywh": (
                converter.camera_adapter_bounds_in_adb_xywh()
            ),
            "adb_to_camera_adapter_3x3": adb_to_adapter_3x3.tolist(),
            "fit_performed_in_camera_space": False,
            "recorded_hik_video_quarter_turns": recorded_output_turns,
            "corrected_hik_video_quarter_turns": corrected_output_turns,
            "recorded_video_correction_quarter_turns": correction_turns,
        },
        "evidence": {
            "adb_input_rectangle": "adb_input_rectangle.png",
            "adb_heatmap": "adb_stacked_difference_heatmap.png",
            "adb_full_boundary": "adb_full_boundary_overlay.png",
            "verified_adb_directory": "verified_adb_boundary",
            "camera_adapter_mask": "camera_adapter_projected_mask.png",
            "hik_video_first_frame": "hik_video_first_frame.png",
            "corrected_hik_video_first_frame": "corrected_hik_video_first_frame.png",
            "hik_video_overlays": [name for name, _ in hik_video_overlays],
            "corrected_hik_video_overlays": [
                name for name, _ in corrected_hik_video_overlays
            ],
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
    value.add_argument("--adb-crop", type=_parse_crop, required=True)
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
