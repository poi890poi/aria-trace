"""Forward-session shift verification layered on the calibrated cursor model."""

import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np

from .cursor_pose import (
    CursorPoseEstimator,
    circular_difference_degrees,
    timing_summary_ms,
)
from .session import SessionReader


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


def estimate_masked_shift(first: np.ndarray, last: np.ndarray, mask: np.ndarray):
    """Estimate image-content translation inside a supplied mini-map mask."""
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY).astype(np.float32)
    last_gray = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY).astype(np.float32)
    weight = mask.astype(np.float32) / 255.0
    first_gray = (first_gray - cv2.GaussianBlur(first_gray, (0, 0), 3.0)) * weight
    last_gray = (last_gray - cv2.GaussianBlur(last_gray, (0, 0), 3.0)) * weight
    window = cv2.createHanningWindow(
        (first_gray.shape[1], first_gray.shape[0]), cv2.CV_32F
    )
    shift, response = cv2.phaseCorrelate(first_gray, last_gray, window * weight)
    return (float(shift[0]), float(shift[1])), float(response)


def benchmark_masked_shift(
    first: np.ndarray,
    last: np.ndarray,
    mask: np.ndarray,
    repeat_count: int = 20,
):
    """Return the normal shift result plus warm-cache wall-time statistics."""
    if repeat_count < 1:
        raise ValueError("Shift benchmark repeat count must be positive")
    estimate_masked_shift(first, last, mask)
    durations_ns = []
    shift = response = None
    for _ in range(repeat_count):
        started_ns = time.perf_counter_ns()
        shift, response = estimate_masked_shift(first, last, mask)
        durations_ns.append(time.perf_counter_ns() - started_ns)
    benchmark = timing_summary_ms(
        durations_ns,
        "one warm-up then repeated complete masked-shift wall time",
    )
    benchmark.update(
        {
            "warmup_count": 1,
            "image_size_wh": [int(first.shape[1]), int(first.shape[0])],
        }
    )
    return shift, response, benchmark


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError("Could not write verification evidence: {}".format(path))


def _labeled_panel(image: np.ndarray, label: str) -> np.ndarray:
    panel = np.zeros((image.shape[0] + 24, image.shape[1], 3), dtype=np.uint8)
    panel[24:] = image
    cv2.putText(
        panel,
        label,
        (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def _verification_graphic(
    crop: np.ndarray,
    pivot_xy,
    content_shift_xy,
    cursor_angle_deg,
    travel_angle_deg,
    response,
    angular_error,
) -> np.ndarray:
    map_image = cv2.resize(
        crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
    )
    graphic = np.zeros(
        (map_image.shape[0] + 72, map_image.shape[1], 3), dtype=np.uint8
    )
    graphic[: map_image.shape[0]] = map_image
    pivot = tuple(int(round(value * 2.0)) for value in pivot_xy)
    scale = 3.0
    shift_vector = np.asarray(content_shift_xy, dtype=np.float64) * scale
    shift_length = float(np.linalg.norm(shift_vector))
    shift_start = np.asarray(pivot, dtype=np.float64)
    if shift_length > 1.0e-6:
        shift_start += shift_vector / shift_length * 26.0
    content_end = (
        int(round(shift_start[0] + shift_vector[0])),
        int(round(shift_start[1] + shift_vector[1])),
    )
    cv2.arrowedLine(
        graphic,
        tuple(np.round(shift_start).astype(int)),
        content_end,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    for angle, color in (
        (cursor_angle_deg, (255, 100, 40)),
        (travel_angle_deg, (70, 255, 100)),
    ):
        radians = math.radians(angle)
        direction = np.array([math.cos(radians), math.sin(radians)])
        start = np.asarray(pivot, dtype=np.float64) + direction * 26.0
        end = (
            int(round(pivot[0] + direction[0] * 80)),
            int(round(pivot[1] + direction[1] * 80)),
        )
        cv2.arrowedLine(
            graphic,
            tuple(np.round(start).astype(int)),
            end,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        graphic,
        "map shift ({:.2f}, {:.2f}) px  response {:.3f}".format(
            content_shift_xy[0], content_shift_xy[1], response
        ),
        (8, map_image.shape[0] + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        graphic,
        "blue cursor {:.2f} deg  green travel {:.2f} deg  error {:.2f} deg".format(
            cursor_angle_deg, travel_angle_deg, angular_error
        ),
        (8, map_image.shape[0] + 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return graphic


def verify_forward_session(
    session_path: Path,
    calibration_path: Path,
    output_path: Path,
    progress=None,
) -> dict:
    """Verify calibrated cursor pose against one straight-forward map shift."""
    if progress:
        progress("Loading the forward session and calibrated cursor model")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    reader = SessionReader(session_path)
    records = reader.frames_by_stream.get("main", [])
    if len(records) < 4:
        raise ValueError("Forward verification needs at least four frames")
    capture = cv2.VideoCapture(str(reader.video_path("main")))
    decoded = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded.append(frame)
            if progress and len(decoded) % 90 == 0:
                progress(
                    "Decoding forward video: {} / {} frames".format(
                        len(decoded), len(records)
                    )
                )
    finally:
        capture.release()
    count = min(len(decoded), len(records))
    if count < 4:
        raise ValueError("Forward verification video is incomplete")
    estimator = CursorPoseEstimator(calibration_path)
    x, y, width, height = estimator.crop_xywh
    margin = max(1, min(8, count // 8))
    first_index = margin
    last_index = count - margin - 1
    first_frame = decoded[first_index]
    last_frame = decoded[last_index]
    first_crop = first_frame[y : y + height, x : x + width]
    last_crop = last_frame[y : y + height, x : x + width]
    calibration = estimator.calibration
    boundary = calibration["outer_boundary"]
    yy, xx = np.ogrid[:height, :width]
    mask = (
        (xx - float(boundary["center_x"])) ** 2
        + (yy - float(boundary["center_y"])) ** 2
        <= max(float(boundary["radius"]) - 5.0, 1.0) ** 2
    ).astype(np.uint8) * 255
    cursor_hole = (
        (xx - float(estimator.pivot[0])) ** 2
        + (yy - float(estimator.pivot[1])) ** 2
        <= 15.0 ** 2
    )
    mask[cursor_hole] = 0
    if progress:
        progress("Benchmarking masked shift estimation (20 repeats)")
    shift, response, shift_benchmark = benchmark_masked_shift(
        first_crop,
        last_crop,
        mask,
    )
    if progress:
        progress("Estimating cursor pose at the start and end frames")
    poses = [
        estimator.public_result(
            estimator.estimate(
                frame,
                frame_index=index,
                session_time_ns=int(records[index]["session_time_ns"]),
            )
        )
        for index, frame in ((first_index, first_frame), (last_index, last_frame))
    ]
    detected_angles = [
        float(item["angle_screen_deg"]) for item in poses if item.get("detected")
    ]
    cursor_angle = (
        float(np.median(detected_angles)) if detected_angles else float("nan")
    )
    shift_angle = math.degrees(math.atan2(shift[1], shift[0])) % 360.0
    travel_angle = (shift_angle + 180.0) % 360.0
    angular_error = (
        float(circular_difference_degrees(cursor_angle, travel_angle))
        if detected_angles
        else None
    )
    aligned_last = cv2.warpAffine(
        last_crop,
        np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]]),
        (width, height),
    )
    aligned_mask = cv2.warpAffine(
        mask,
        np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]]),
        (width, height),
    )
    overlap_mask = cv2.bitwise_and(mask, aligned_mask)
    first_gray = cv2.cvtColor(first_crop, cv2.COLOR_BGR2GRAY)
    aligned_gray = cv2.cvtColor(aligned_last, cv2.COLOR_BGR2GRAY)
    overlay = np.zeros_like(first_crop)
    overlay[:, :, 2] = first_gray
    overlay[:, :, 1] = aligned_gray
    overlay = cv2.bitwise_and(overlay, overlay, mask=overlap_mask)
    residual = cv2.absdiff(first_gray, aligned_gray)
    residual = cv2.bitwise_and(residual, residual, mask=overlap_mask)
    residual = cv2.applyColorMap(residual, cv2.COLORMAP_TURBO)
    registration_review = np.hstack(
        [
            _labeled_panel(overlay, "aligned overlay: yellow = agreement"),
            _labeled_panel(residual, "residual error: blue low / red high"),
        ]
    )
    evidence = [
        {"name": "forward_start.png", "title": "Raw forward start observation", "category": "shift"},
        {"name": "forward_end.png", "title": "Raw forward end observation", "category": "shift"},
        {"name": "forward_shift_mask.png", "title": "Exact shift-estimation mask (white used, black excluded)", "category": "shift"},
        {"name": "forward_registration_overlay.png", "title": "Aligned overlay and registration residual", "category": "shift"},
        {"name": "forward_pose_shift.png", "title": "Cursor pose and map-shift relationship", "category": "pose_verification"},
    ]
    if progress:
        progress("Rendering shift, correlation, and pose evidence")
    _write_image(output_path / "forward_start.png", first_crop)
    _write_image(output_path / "forward_end.png", last_crop)
    _write_image(output_path / "forward_shift_mask.png", mask)
    _write_image(output_path / "forward_registration_overlay.png", registration_review)
    _write_image(
        output_path / "forward_pose_shift.png",
        _verification_graphic(
            last_crop,
            estimator.pivot,
            shift,
            cursor_angle if detected_angles else 0.0,
            travel_angle,
            response,
            angular_error if angular_error is not None else 0.0,
        ),
    )
    result = {
        "schema_version": "1.0",
        "status": "review_required" if response >= 0.05 else "low_confidence",
        "source_session_path": str(Path(session_path).resolve()),
        "source_session_id": reader.manifest.get("session_id"),
        "source_frames": {
            "start": records[first_index],
            "end": records[last_index],
        },
        "map_content_shift_xy_px": [shift[0], shift[1]],
        "map_content_shift_angle_screen_deg": shift_angle,
        "inferred_travel_angle_screen_deg": travel_angle,
        "phase_correlation_response": response,
        "shift_estimation_benchmark": shift_benchmark,
        "cursor_angle_screen_deg": cursor_angle if detected_angles else None,
        "cursor_pose_start": poses[0],
        "cursor_pose_end": poses[1],
        "cursor_travel_angular_error_deg": angular_error,
        "world_heading_status": "relative_offset_only",
        "evidence": evidence,
    }
    if progress:
        progress("Writing the forward pose-verification result")
    _atomic_json(output_path / "forward_verification.json", result)
    return result
