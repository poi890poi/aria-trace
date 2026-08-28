"""Translation-based full-map stitching with inspectable registration evidence."""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .session import SessionReader


def _gradient(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _minimap_reference(image: np.ndarray, calibration: dict):
    """Extract the calibrated circular map content used for cross-space scale."""
    boundary = calibration.get("outer_boundary") or {}
    center_x = float(boundary["center_x"])
    center_y = float(boundary["center_y"])
    radius = float(boundary["radius"])
    left = max(0, int(round(center_x - radius)))
    top = max(0, int(round(center_y - radius)))
    right = min(image.shape[1], int(round(center_x + radius)))
    bottom = min(image.shape[0], int(round(center_y + radius)))
    patch = image[top:bottom, left:right].copy()
    if min(patch.shape[:2]) < 48:
        raise RuntimeError("Mini-map scale reference is too small")
    mask = np.zeros(patch.shape[:2], np.uint8)
    patch_center = (patch.shape[1] // 2, patch.shape[0] // 2)
    usable_radius = max(8, min(patch.shape[:2]) // 2 - 3)
    cv2.circle(mask, patch_center, usable_radius, 255, -1)
    cv2.circle(mask, patch_center, max(5, int(round(radius * 0.20))), 0, -1)
    return patch, mask


def _estimate_minimap_similarity(
    reference: np.ndarray,
    mask: np.ndarray,
    mosaic: np.ndarray,
    coverage: np.ndarray,
):
    """Estimate mini-map pixels to original-mosaic pixels with SIFT geometry."""
    sift = cv2.SIFT_create(nfeatures=8000, contrastThreshold=0.005, edgeThreshold=15)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    mosaic_gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    map_mask = cv2.erode(coverage, np.ones((5, 5), np.uint8))
    reference_points, reference_descriptors = sift.detectAndCompute(reference_gray, mask)
    map_points, map_descriptors = sift.detectAndCompute(mosaic_gray, map_mask)
    if reference_descriptors is None or map_descriptors is None:
        raise RuntimeError("Mini-map/full-map scale calibration found no SIFT descriptors")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        reference_descriptors, map_descriptors, k=2
    )
    matches = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.80 * second.distance
    ]
    if len(matches) < 6:
        raise RuntimeError(
            "Mini-map/full-map scale calibration needs at least 6 ratio-test matches; got {}"
            .format(len(matches))
        )
    source = np.float32([reference_points[item.queryIdx].pt for item in matches])
    target = np.float32([map_points[item.trainIdx].pt for item in matches])
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=6.0,
        maxIters=20000,
        confidence=0.999,
    )
    if matrix is None or inlier_mask is None:
        raise RuntimeError("Mini-map/full-map SIFT matches have no consistent similarity")
    inliers = inlier_mask.ravel().astype(bool)
    inlier_count = int(np.count_nonzero(inliers))
    if inlier_count < 2:
        raise RuntimeError(
            "Mini-map/full-map similarity has fewer than 2 geometric inliers"
        )
    predicted = cv2.transform(source.reshape((-1, 1, 2)), matrix).reshape((-1, 2))
    errors = np.linalg.norm(predicted - target, axis=1)
    scale = math.hypot(float(matrix[0, 0]), float(matrix[0, 1]))
    rotation = math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
    center = cv2.transform(
        np.float32([[[reference.shape[1] / 2.0, reference.shape[0] / 2.0]]]),
        matrix,
    ).reshape(2)
    return {
        "matrix": matrix,
        "source_points": source,
        "target_points": target,
        "inlier_mask": inliers,
        "ratio_match_count": len(matches),
        "inlier_count": inlier_count,
        "inlier_ratio": float(np.mean(inliers)),
        "reprojection_median_px": float(np.median(errors[inliers])),
        "reprojection_p95_px": float(np.percentile(errors[inliers], 95)),
        "map_pixels_per_minimap_pixel": float(scale),
        "minimap_to_map_rotation_deg": float(rotation),
        "reference_center_map_xy": center,
    }


def _build_localization_derivative(
    mosaic: np.ndarray,
    coverage: np.ndarray,
    output_path: Path,
    reference: dict,
) -> dict:
    """Create a reversible map raster normalized to calibrated mini-map scale."""
    patch, mask = _minimap_reference(reference["image"], reference["calibration"])
    estimate = _estimate_minimap_similarity(patch, mask, mosaic, coverage)
    scale = estimate["map_pixels_per_minimap_pixel"]
    if not 0.5 <= scale <= 16.0:
        raise RuntimeError("Mini-map/full-map scale {:.3f} is implausible".format(scale))
    requested_factor = 1.0 / scale
    localization_size = (
        max(64, int(round(mosaic.shape[1] * requested_factor))),
        max(64, int(round(mosaic.shape[0] * requested_factor))),
    )
    interpolation = cv2.INTER_AREA if requested_factor < 1.0 else cv2.INTER_CUBIC
    localization_mosaic = cv2.resize(mosaic, localization_size, interpolation=interpolation)
    localization_coverage = cv2.resize(
        coverage, localization_size, interpolation=cv2.INTER_NEAREST
    )
    factor_x = localization_size[0] / float(mosaic.shape[1])
    factor_y = localization_size[1] / float(mosaic.shape[0])
    original_to_localization = np.asarray(
        [[factor_x, 0.0, 0.0], [0.0, factor_y, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    localization_to_original = np.linalg.inv(original_to_localization)

    rotation = estimate["minimap_to_map_rotation_deg"]
    rotation_matrix = cv2.getRotationMatrix2D(
        ((patch.shape[1] - 1) / 2.0, (patch.shape[0] - 1) / 2.0),
        rotation,
        1.0,
    )
    rotated_patch = cv2.warpAffine(
        _gradient(patch), rotation_matrix, (patch.shape[1], patch.shape[0])
    )
    rotated_mask = cv2.warpAffine(
        mask,
        rotation_matrix,
        (patch.shape[1], patch.shape[0]),
        flags=cv2.INTER_NEAREST,
    )
    response = cv2.matchTemplate(
        _gradient(localization_mosaic),
        rotated_patch,
        cv2.TM_CCORR_NORMED,
        mask=rotated_mask,
    )
    response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _, score, _, location = cv2.minMaxLoc(response)
    suppressed = response.copy()
    cv2.circle(
        suppressed,
        location,
        max(8, min(patch.shape[:2]) // 3),
        -1.0,
        -1,
    )
    _, second_score, _, _ = cv2.minMaxLoc(suppressed)
    correlation_center = np.asarray(
        [location[0] + patch.shape[1] / 2.0, location[1] + patch.shape[0] / 2.0],
        dtype=np.float64,
    )
    sift_center = np.asarray(
        [
            estimate["reference_center_map_xy"][0] * factor_x,
            estimate["reference_center_map_xy"][1] * factor_y,
        ],
        dtype=np.float64,
    )
    center_agreement = float(np.linalg.norm(correlation_center - sift_center))
    ready = bool(
        estimate["inlier_count"] >= 6
        and estimate["inlier_ratio"] >= 0.60
        and estimate["reprojection_p95_px"] <= 4.0
        and score >= 0.55
        and score - second_score >= 0.06
        and center_agreement <= 8.0
    )

    evidence = localization_mosaic.copy()
    for point, accepted in zip(estimate["target_points"], estimate["inlier_mask"]):
        if accepted:
            location_xy = (
                int(round(float(point[0]) * factor_x)),
                int(round(float(point[1]) * factor_y)),
            )
            cv2.circle(evidence, location_xy, 3, (40, 220, 245), -1, cv2.LINE_AA)
    cv2.circle(
        evidence,
        tuple(np.round(sift_center).astype(int)),
        max(8, min(patch.shape[:2]) // 2),
        (70, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.circle(
        evidence,
        tuple(np.round(correlation_center).astype(int)),
        max(5, min(patch.shape[:2]) // 2 - 5),
        (70, 235, 100),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        evidence,
        "scale {:.3f} map px/mini px  SIFT {}/{}  corr {:.3f} margin {:.3f}".format(
            scale,
            estimate["inlier_count"],
            estimate["ratio_match_count"],
            score,
            score - second_score,
        ),
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (250, 250, 250),
        2,
        cv2.LINE_AA,
    )
    _write_image(output_path / "localization_mosaic.png", localization_mosaic)
    _write_image(output_path / "localization_coverage.png", localization_coverage)
    _write_image(output_path / "localization_scale_evidence.png", evidence)
    return {
        "schema_version": "1.0",
        "status": "ready" if ready else "review_required",
        "method": "sift_similarity_with_masked_gradient_verification",
        "source_minimap_calibration_id": reference.get("calibration_id"),
        "source_minimap_image": reference.get("source_image_name"),
        "coordinate_space": "derived_localization_map_px",
        "mosaic_file": "localization_mosaic.png",
        "coverage_file": "localization_coverage.png",
        "size_wh": list(localization_size),
        "map_pixels_per_minimap_pixel": scale,
        "minimap_to_map_rotation_deg": estimate["minimap_to_map_rotation_deg"],
        "original_map_to_localization_3x3": original_to_localization.tolist(),
        "localization_to_original_map_3x3": localization_to_original.tolist(),
        "reference_center_original_map_xy": estimate["reference_center_map_xy"].tolist(),
        "reference_center_localization_xy": sift_center.tolist(),
        "quality": {
            "ratio_match_count": estimate["ratio_match_count"],
            "inlier_count": estimate["inlier_count"],
            "inlier_ratio": estimate["inlier_ratio"],
            "reprojection_median_original_map_px": estimate["reprojection_median_px"],
            "reprojection_p95_original_map_px": estimate["reprojection_p95_px"],
            "gradient_correlation_score": float(score),
            "gradient_correlation_margin": float(score - second_score),
            "sift_correlation_center_agreement_localization_px": center_agreement,
        },
        "evidence_file": "localization_scale_evidence.png",
    }


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


def stitch_map_frames(
    frames,
    output_path: Path,
    provenance=None,
    progress=None,
    localization_reference=None,
) -> dict:
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
    localization = None
    if localization_reference is not None:
        if progress:
            progress("Calibrating full-map pixels to the reviewed mini-map scale")
        localization = _build_localization_derivative(
            mosaic,
            coverage,
            output_path,
            localization_reference,
        )
        evidence.extend(
            [
                {
                    "name": "localization_mosaic.png",
                    "title": "Derived mosaic normalized to mini-map scale",
                    "category": "localization",
                },
                {
                    "name": "localization_coverage.png",
                    "title": "Valid coverage in localization coordinates",
                    "category": "localization",
                },
                {
                    "name": "localization_scale_evidence.png",
                    "title": "Mini-map/full-map scale and position verification",
                    "category": "localization",
                },
            ]
        )
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
        "localization": localization,
        "registrations": registrations,
        "warnings": [
            "Observed coverage does not certify every game region or layer."
        ],
        "evidence": evidence,
    }
    _atomic_json(output_path / "map_stitch.json", result)
    return result


def stitch_map_session(
    session_path: Path,
    output_path: Path,
    progress=None,
    localization_reference=None,
) -> dict:
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
        localization_reference=localization_reference,
    )
