"""Translation-based full-map stitching with inspectable registration evidence."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .session import SessionReader


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError("Could not write map-stitch evidence: {}".format(path))


def _registration_image(first: np.ndarray, second: np.ndarray, shift_xy) -> np.ndarray:
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    aligned = cv2.warpAffine(
        second,
        np.float32([[1, 0, -shift_xy[0]], [0, 1, -shift_xy[1]]]),
        (second.shape[1], second.shape[0]),
    )
    second_gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    image = np.zeros_like(first)
    image[:, :, 2] = first_gray
    image[:, :, 1] = second_gray
    return image


def _estimate_translation(first: np.ndarray, second: np.ndarray):
    height, width = first.shape[:2]
    first_small = cv2.resize(first, (width // 2, height // 2))
    second_small = cv2.resize(second, (width // 2, height // 2))
    first_gray = cv2.cvtColor(first_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    second_gray = cv2.cvtColor(second_small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    first_gray -= cv2.GaussianBlur(first_gray, (0, 0), 3.0)
    second_gray -= cv2.GaussianBlur(second_gray, (0, 0), 3.0)
    window = cv2.createHanningWindow(
        (first_gray.shape[1], first_gray.shape[0]), cv2.CV_32F
    )
    shift, response = cv2.phaseCorrelate(first_gray, second_gray, window)
    return (float(shift[0] * 2.0), float(shift[1] * 2.0)), float(response)


def stitch_map_frames(frames, output_path: Path, provenance=None, progress=None) -> dict:
    """Register overlapping map-view frames and write a reviewable mosaic."""
    if len(frames) < 2:
        raise ValueError("Map stitching needs at least two frames")
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    margins = {
        "left": min(180, width // 5),
        "top": min(65, height // 10),
        "right": min(100, width // 8),
        "bottom": min(55, height // 10),
    }
    x0, y0 = margins["left"], margins["top"]
    x1, y1 = width - margins["right"], height - margins["bottom"]
    if progress:
        progress("Cropping map viewports from {} decoded frames".format(len(frames)))
    viewports = [frame[y0:y1, x0:x1] for frame in frames]
    positions = [np.array([0.0, 0.0])]
    selected = [0]
    registrations = []
    reference = viewports[0]
    reference_index = 0
    reference_position = np.array([0.0, 0.0])
    for index in range(1, len(viewports)):
        if progress and (
            index == 1 or index % 25 == 0 or index == len(viewports) - 1
        ):
            progress(
                "Registering map frames: {} / {}".format(
                    index, len(viewports) - 1
                )
            )
        shift, response = _estimate_translation(reference, viewports[index])
        magnitude = float(np.linalg.norm(shift))
        accepted = bool(
            response >= 0.06
            and abs(shift[0]) < (x1 - x0) * 0.45
            and abs(shift[1]) < (y1 - y0) * 0.45
        )
        registrations.append(
            {
                "from_frame": reference_index,
                "to_frame": index,
                "content_shift_xy_px": [shift[0], shift[1]],
                "magnitude_px": magnitude,
                "response": response,
                "accepted": accepted,
            }
        )
        if accepted:
            position = reference_position - np.asarray(shift)
            if magnitude >= 28.0:
                selected.append(index)
                reference = viewports[index]
                reference_index = index
                reference_position = position.copy()
        else:
            position = positions[-1].copy()
        positions.append(position)
    if (
        selected[-1] != len(positions) - 1
        and np.linalg.norm(positions[-1] - positions[selected[-1]]) >= 5.0
    ):
        selected.append(len(positions) - 1)
    selected_positions = np.asarray([positions[index] for index in selected])
    viewport_height, viewport_width = viewports[0].shape[:2]
    minimum = np.floor(selected_positions.min(axis=0)).astype(int)
    maximum = np.ceil(selected_positions.max(axis=0)).astype(int)
    canvas_width = int(maximum[0] - minimum[0] + viewport_width)
    canvas_height = int(maximum[1] - minimum[1] + viewport_height)
    if canvas_width * canvas_height > 120_000_000:
        raise RuntimeError("Estimated map mosaic is implausibly large")
    if progress:
        progress(
            "Composing the observed {} x {} map mosaic".format(
                canvas_width, canvas_height
            )
        )
    accumulator = np.zeros((canvas_height, canvas_width, 3), np.float32)
    weights = np.zeros((canvas_height, canvas_width), np.float32)
    feather = cv2.createHanningWindow(
        (viewport_width, viewport_height), cv2.CV_32F
    )
    feather = np.maximum(feather, 0.08)
    for index in selected:
        origin = np.rint(positions[index] - minimum).astype(int)
        ox, oy = int(origin[0]), int(origin[1])
        accumulator[oy : oy + viewport_height, ox : ox + viewport_width] += (
            viewports[index].astype(np.float32) * feather[:, :, None]
        )
        weights[oy : oy + viewport_height, ox : ox + viewport_width] += feather
    mosaic = np.divide(
        accumulator,
        np.maximum(weights[:, :, None], 1.0e-6),
    ).clip(0, 255).astype(np.uint8)
    coverage = (weights > 0).astype(np.uint8) * 255
    coverage_heatmap = cv2.applyColorMap(
        np.uint8(np.clip(weights / max(float(weights.max()), 1.0) * 255, 0, 255)),
        cv2.COLORMAP_TURBO,
    )
    if progress:
        progress("Building registration-quality and coverage evidence")
    quality = np.full((360, 900, 3), 18, np.uint8)
    responses = [float(item["response"]) for item in registrations]
    for index in range(1, len(responses)):
        x_prev = 25 + int((index - 1) / max(len(responses) - 1, 1) * 850)
        x_now = 25 + int(index / max(len(responses) - 1, 1) * 850)
        y_prev = 320 - int(np.clip(responses[index - 1], 0, 1) * 280)
        y_now = 320 - int(np.clip(responses[index], 0, 1) * 280)
        color = (
            (80, 230, 120)
            if registrations[index - 1]["accepted"]
            else (80, 80, 255)
        )
        cv2.line(quality, (x_prev, y_prev), (x_now, y_now), color, 1, cv2.LINE_AA)
    cv2.putText(quality, "Pairwise registration response", (22, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 1, cv2.LINE_AA)
    sample_rows = [item for item in registrations if item["accepted"]]
    sample_rows = sample_rows[:: max(1, len(sample_rows) // 3)][:3]
    samples = []
    for item in sample_rows:
        overlay = _registration_image(
            viewports[item["from_frame"]],
            viewports[item["to_frame"]],
            item["content_shift_xy_px"],
        )
        samples.append(cv2.resize(overlay, (320, 180)))
    alignment_samples = (
        np.hstack(samples) if samples else np.zeros((180, 320, 3), np.uint8)
    )
    evidence = [
        {"name": "mosaic.png", "title": "Observed full-map mosaic", "category": "map"},
        {"name": "coverage.png", "title": "Observed viewport coverage", "category": "coverage"},
        {"name": "coverage_heatmap.png", "title": "Coverage overlap heatmap", "category": "coverage"},
        {"name": "registration_quality.png", "title": "Pairwise registration quality", "category": "quality"},
        {"name": "alignment_samples.png", "title": "Representative frame alignments", "category": "quality"},
    ]
    if progress:
        progress("Writing the mosaic and five review images")
    _write_image(output_path / "mosaic.png", mosaic)
    _write_image(output_path / "coverage.png", coverage)
    _write_image(output_path / "coverage_heatmap.png", coverage_heatmap)
    _write_image(output_path / "registration_quality.png", quality)
    _write_image(output_path / "alignment_samples.png", alignment_samples)
    accepted = sum(1 for item in registrations if item["accepted"])
    observed_coverage = float(np.count_nonzero(coverage) / coverage.size)
    result = {
        "schema_version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "review_required" if accepted else "failed",
        "coverage_scope": "observed_viewports_only",
        "provenance": provenance or {},
        "source_frame_count": len(frames),
        "selected_frame_indices": selected,
        "selected_frame_count": len(selected),
        "accepted_registrations": accepted,
        "rejected_registrations": len(registrations) - accepted,
        "median_registration_response": float(np.median(responses)) if responses else 0.0,
        "observed_canvas_coverage": observed_coverage,
        "viewport_crop_xywh": [x0, y0, viewport_width, viewport_height],
        "mosaic_size_wh": [canvas_width, canvas_height],
        "registrations": registrations,
        "warnings": [
            "Observed coverage does not certify every game region or layer."
        ],
        "evidence": evidence,
    }
    _atomic_json(output_path / "map_stitch.json", result)
    return result


def stitch_map_session(session_path: Path, output_path: Path, progress=None) -> dict:
    reader = SessionReader(session_path)
    records = reader.frames_by_stream.get("main", [])
    if progress:
        progress("Decoding the full-map recording")
    capture = cv2.VideoCapture(str(reader.video_path("main")))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
            if progress and len(frames) % 90 == 0:
                progress(
                    "Decoding full-map video: {} / {} frames".format(
                        len(frames), len(records)
                    )
                )
    finally:
        capture.release()
    return stitch_map_frames(
        frames,
        output_path,
        provenance={
            "source_session_path": str(Path(session_path).resolve()),
            "source_session_id": reader.manifest.get("session_id"),
            "source_frame_records": records,
        },
        progress=progress,
    )
