"""Display-referred slanted-edge e-SFR/MTF measurement.

Camera samples remain in their native, pre-homography raster. Their sample
locations are mapped into canonical display pixels with the ChArUco-derived
homography, so the primary frequency axis is cycles per display pixel without
resampling the measured image.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .contracts import matrix_3x3
from .geometry import transform_points


_CHANNELS = ("luminance", "red", "green", "blue")


def _screen_rect_polygon(rect_xywh: Sequence[float]) -> np.ndarray:
    x, y, width, height = map(float, rect_xywh)
    if min(width, height) <= 0:
        raise ValueError("Quality target rectangle must have positive size")
    return np.asarray(
        [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
        dtype=np.float64,
    )


def _edge_geometry(
    rect_xywh: Sequence[float], edge_angle_deg: float, phase_display_px: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, width, height = map(float, rect_xywh)
    center = np.asarray([x + width / 2.0, y + height / 2.0], dtype=np.float64)
    radians = math.radians(float(edge_angle_deg))
    direction = np.asarray([math.cos(radians), math.sin(radians)], dtype=np.float64)
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    center = center + normal * float(phase_display_px)
    return center, direction, normal


def generate_slanted_edge_target(
    screen_size_px: Sequence[int],
    rect_screen_xywh: Sequence[float],
    edge_angle_deg: float = 5.0,
    phase_display_px: float = 0.0,
    channel: str = "luminance",
    low_code: int = 48,
    high_code: int = 192,
) -> np.ndarray:
    """Render a pixel-exact slanted edge within a declared display rectangle."""

    width, height = map(int, screen_size_px)
    if min(width, height) <= 0:
        raise ValueError("Screen size must be positive")
    channel = str(channel).lower()
    if channel not in _CHANNELS:
        raise ValueError("Unsupported e-SFR channel {}".format(channel))
    if not 0 <= int(low_code) < int(high_code) <= 255:
        raise ValueError("Slanted-edge code values must satisfy 0 <= low < high <= 255")
    rect = _screen_rect_polygon(rect_screen_xywh)
    if (
        np.min(rect[:, 0]) < 0
        or np.min(rect[:, 1]) < 0
        or np.max(rect[:, 0]) > width
        or np.max(rect[:, 1]) > height
    ):
        raise ValueError("Slanted-edge rectangle lies outside the display raster")

    image = np.full((height, width, 3), 24, dtype=np.uint8)
    x, y, patch_width, patch_height = map(int, rect_screen_xywh)
    yy, xx = np.mgrid[y : y + patch_height, x : x + patch_width]
    center, _, normal = _edge_geometry(
        rect_screen_xywh, edge_angle_deg, phase_display_px
    )
    signed = (
        (xx.astype(np.float64) + 0.5 - center[0]) * normal[0]
        + (yy.astype(np.float64) + 0.5 - center[1]) * normal[1]
    )
    high = signed >= 0.0
    if channel == "luminance":
        patch = np.where(high[..., None], int(high_code), int(low_code)).astype(np.uint8)
        patch = np.repeat(patch, 3, axis=2)
    else:
        patch = np.full((patch_height, patch_width, 3), int(low_code), dtype=np.uint8)
        index = {"blue": 0, "green": 1, "red": 2}[channel]
        patch[..., index] = np.where(high, int(high_code), int(low_code)).astype(np.uint8)
    image[y : y + patch_height, x : x + patch_width] = patch
    return image


def _channel_samples(image: np.ndarray, channel: str) -> np.ndarray:
    if image.ndim == 2:
        return np.asarray(image, dtype=np.float64)
    bgr = np.asarray(image, dtype=np.float64)
    if channel == "blue":
        return bgr[..., 0]
    if channel == "green":
        return bgr[..., 1]
    if channel == "red":
        return bgr[..., 2]
    if channel == "luminance":
        return 0.0722 * bgr[..., 0] + 0.7152 * bgr[..., 1] + 0.2126 * bgr[..., 2]
    raise ValueError("Unsupported e-SFR channel {}".format(channel))


def _crossing_frequency(
    frequencies: np.ndarray, response: np.ndarray, level: float
) -> Optional[float]:
    monotonic = np.minimum.accumulate(np.asarray(response, dtype=np.float64))
    indices = np.flatnonzero(monotonic <= float(level))
    if not len(indices):
        return None
    index = int(indices[0])
    if index == 0:
        return float(frequencies[0])
    x0, x1 = float(frequencies[index - 1]), float(frequencies[index])
    y0, y1 = float(monotonic[index - 1]), float(monotonic[index])
    if abs(y1 - y0) < 1.0e-12:
        return x1
    return float(x0 + (float(level) - y0) * (x1 - x0) / (y1 - y0))


def _camera_bbox_for_screen_rect(
    camera_shape: Sequence[int],
    rect_screen_xywh: Sequence[float],
    camera_to_screen_3x3: np.ndarray,
    padding_camera_px: int = 3,
) -> Tuple[int, int, int, int]:
    inverse = np.linalg.inv(camera_to_screen_3x3)
    camera_polygon = transform_points(_screen_rect_polygon(rect_screen_xywh), inverse)
    left = max(0, int(math.floor(float(np.min(camera_polygon[:, 0])))) - padding_camera_px)
    top = max(0, int(math.floor(float(np.min(camera_polygon[:, 1])))) - padding_camera_px)
    right = min(
        int(camera_shape[1]),
        int(math.ceil(float(np.max(camera_polygon[:, 0])))) + padding_camera_px + 1,
    )
    bottom = min(
        int(camera_shape[0]),
        int(math.ceil(float(np.max(camera_polygon[:, 1])))) + padding_camera_px + 1,
    )
    if right - left < 8 or bottom - top < 8:
        raise ValueError("Projected slanted-edge patch is too small in the camera image")
    return left, top, right, bottom


def _local_sampling(
    camera_to_screen_3x3: np.ndarray,
    edge_center_screen_xy: np.ndarray,
    edge_normal_screen_xy: np.ndarray,
) -> Dict[str, Any]:
    inverse = np.linalg.inv(camera_to_screen_3x3)
    camera_center = transform_points([edge_center_screen_xy], inverse)[0]
    camera_probe = np.asarray(
        [camera_center, camera_center + [1.0, 0.0], camera_center + [0.0, 1.0]],
        dtype=np.float64,
    )
    screen_probe = transform_points(camera_probe, camera_to_screen_3x3)
    jacobian = np.column_stack(
        [screen_probe[1] - screen_probe[0], screen_probe[2] - screen_probe[0]]
    )
    display_px_per_camera_px_normal = float(
        np.linalg.norm(jacobian.T.dot(edge_normal_screen_xy))
    )
    if display_px_per_camera_px_normal <= 1.0e-9:
        raise ValueError("Local camera/display sampling transform is degenerate")
    return {
        "roi_position_camera_xy": camera_center.tolist(),
        "jacobian_display_px_per_camera_px": jacobian.tolist(),
        "display_px_per_camera_px_along_edge_normal": display_px_per_camera_px_normal,
        "camera_px_per_display_px_along_edge_normal": float(
            1.0 / display_px_per_camera_px_normal
        ),
    }


def measure_slanted_edge_esfr(
    camera_image: np.ndarray,
    camera_to_screen_3x3: Sequence[Sequence[float]],
    rect_screen_xywh: Sequence[float],
    edge_angle_deg: float,
    phase_display_px: float = 0.0,
    channel: str = "luminance",
    oversampling: int = 4,
    oecf_lut: Optional[Sequence[float]] = None,
    geometry_confidence: float = 1.0,
    display_pixel_pitch_mm_xy: Optional[Sequence[float]] = None,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Measure one edge directly from native camera samples in display units.

    This follows the ISO 12233 slanted-edge e-SFR sequence (ESF binning, LSF,
    windowing, derivative-response correction, normalized Fourier magnitude).
    It is deliberately described as a non-certified implementation and records
    whether a measured OECF was supplied.
    """

    if camera_image is None or camera_image.size == 0:
        raise ValueError("Camera image is empty")
    channel = str(channel).lower()
    if channel not in _CHANNELS:
        raise ValueError("Unsupported e-SFR channel {}".format(channel))
    oversampling = int(oversampling)
    if oversampling < 2 or oversampling > 16:
        raise ValueError("e-SFR oversampling must be between 2 and 16")
    transform = matrix_3x3(camera_to_screen_3x3)
    left, top, right, bottom = _camera_bbox_for_screen_rect(
        camera_image.shape, rect_screen_xywh, transform
    )
    crop = camera_image[top:bottom, left:right]
    yy, xx = np.mgrid[top:bottom, left:right]
    camera_points = np.column_stack(
        [xx.reshape(-1).astype(np.float64), yy.reshape(-1).astype(np.float64)]
    )
    screen_points = transform_points(camera_points, transform)
    x, y, width, height = map(float, rect_screen_xywh)
    inset = max(2.0, min(width, height) * 0.025)
    inside = (
        (screen_points[:, 0] >= x + inset)
        & (screen_points[:, 0] < x + width - inset)
        & (screen_points[:, 1] >= y + inset)
        & (screen_points[:, 1] < y + height - inset)
    )
    center, direction, normal = _edge_geometry(
        rect_screen_xywh, edge_angle_deg, phase_display_px
    )
    signed_distance = (screen_points - center).dot(normal)
    along_edge = (screen_points - center).dot(direction)
    half_profile = max(8.0, min(width, height) * 0.22)
    half_length = max(16.0, min(width, height) * 0.40)
    selected = inside & (np.abs(signed_distance) <= half_profile) & (
        np.abs(along_edge) <= half_length
    )
    if int(np.count_nonzero(selected)) < 256:
        raise ValueError("Too few camera samples cover the slanted-edge analysis ROI")

    samples = _channel_samples(crop, channel).reshape(-1)[selected]
    if oecf_lut is not None:
        lut = np.asarray(oecf_lut, dtype=np.float64).reshape(-1)
        if len(lut) != 256 or not np.all(np.isfinite(lut)):
            raise ValueError("OECF LUT must contain 256 finite linearized values")
        sample_codes = np.clip(np.rint(samples), 0, 255).astype(np.uint8)
        samples = lut[sample_codes]
        oecf_status = "measured_lut_applied"
    else:
        samples = samples / 255.0
        oecf_status = "not_measured_code_values_used"

    distances = signed_distance[selected]
    bin_width = 1.0 / float(oversampling)
    edge_min = -half_profile
    bin_count = int(math.ceil((2.0 * half_profile) / bin_width))
    indices = np.floor((distances - edge_min) / bin_width).astype(np.int32)
    valid = (indices >= 0) & (indices < bin_count)
    indices = indices[valid]
    samples = samples[valid]
    sums = np.bincount(indices, weights=samples, minlength=bin_count).astype(np.float64)
    counts = np.bincount(indices, minlength=bin_count).astype(np.int32)
    occupied = counts > 0
    if int(np.count_nonzero(occupied)) < max(32, int(bin_count * 0.65)):
        raise ValueError("Slanted-edge bins have insufficient camera-sample coverage")
    esf = np.zeros(bin_count, dtype=np.float64)
    esf[occupied] = sums[occupied] / counts[occupied]
    positions = edge_min + (np.arange(bin_count, dtype=np.float64) + 0.5) * bin_width
    if not np.all(occupied):
        esf[~occupied] = np.interp(
            positions[~occupied], positions[occupied], esf[occupied]
        )

    tail = max(4, bin_count // 8)
    low = float(np.median(esf[:tail]))
    high = float(np.median(esf[-tail:]))
    if high < low:
        esf = esf[::-1]
        counts = counts[::-1]
        low, high = high, low
    contrast = high - low
    if contrast <= 0.03:
        raise ValueError("Slanted-edge contrast is too low for e-SFR")
    normalized_esf = np.clip((esf - low) / contrast, -0.25, 1.25)

    lsf = np.gradient(normalized_esf, bin_width)
    lsf -= float(np.mean(np.concatenate([lsf[:tail], lsf[-tail:]])))
    windowed = lsf * np.hamming(len(lsf))
    spectrum = np.abs(np.fft.rfft(windowed))
    frequencies = np.fft.rfftfreq(len(windowed), d=bin_width)
    if spectrum[0] <= 1.0e-12:
        raise ValueError("Slanted-edge LSF has no usable DC response")
    mtf = spectrum / spectrum[0]
    argument = 2.0 * np.pi * frequencies * bin_width
    derivative_correction = np.ones_like(argument)
    sine = np.sin(argument)
    correctable = (np.abs(argument) > 1.0e-12) & (np.abs(sine) > 1.0e-9)
    derivative_correction[correctable] = np.abs(
        argument[correctable] / sine[correctable]
    )
    derivative_correction = np.minimum(derivative_correction, 10.0)
    mtf *= derivative_correction
    supported = frequencies <= 0.5 + 1.0e-12
    frequencies = frequencies[supported]
    mtf = mtf[supported]
    mtf[0] = 1.0

    mtf50 = _crossing_frequency(frequencies, mtf, 0.50)
    mtf10 = _crossing_frequency(frequencies, mtf, 0.10)
    sampling = _local_sampling(transform, center, normal)
    native_frequency = frequencies / float(
        sampling["camera_px_per_display_px_along_edge_normal"]
    )

    sample_support = min(1.0, float(np.count_nonzero(selected)) / 4096.0)
    bin_support = float(np.mean(occupied))
    contrast_quality = float(np.clip(contrast / 0.35, 0.0, 1.0))
    geometry_quality = float(np.clip(geometry_confidence, 0.0, 1.0))
    confidence = float(
        0.30 * sample_support
        + 0.25 * bin_support
        + 0.20 * contrast_quality
        + 0.25 * geometry_quality
    )

    evidence = crop.copy() if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    camera_edge = transform_points(
        [center - direction * half_length, center + direction * half_length],
        np.linalg.inv(transform),
    )
    camera_edge -= np.asarray([left, top], dtype=np.float64)
    cv2.line(
        evidence,
        tuple(np.round(camera_edge[0]).astype(int)),
        tuple(np.round(camera_edge[1]).astype(int)),
        (40, 80, 255),
        2,
        cv2.LINE_AA,
    )
    result: Dict[str, Any] = {
        "standard": "ISO 12233:2024",
        "method": "slanted_edge_e_sfr",
        "implementation_conformance": "non_certified",
        "measurement_input_space": "camera_pre_homography_px",
        "primary_spatial_frequency_unit": "cycles_per_display_pixel",
        "native_analysis_frequency_unit": "cycles_per_camera_pixel",
        "rect_screen_xywh": list(map(float, rect_screen_xywh)),
        "channel": channel,
        "edge_angle_display_deg": float(edge_angle_deg),
        "phase_display_px": float(phase_display_px),
        "oecf_linearization": oecf_status,
        "oversampling": oversampling,
        "sample_count": int(np.count_nonzero(selected)),
        "occupied_bin_fraction": bin_support,
        "edge_contrast_normalized": float(contrast),
        "sampling": sampling,
        "display_referred": {
            "spatial_frequency_unit": "cycles_per_display_pixel",
            "nyquist": 0.5,
            "mtf50": mtf50,
            "mtf10": mtf10,
            "frequency": frequencies.tolist(),
            "mtf": mtf.tolist(),
        },
        "camera_analysis": {
            "spatial_frequency_unit": "cycles_per_camera_pixel",
            "frequency": native_frequency.tolist(),
            "mtf": mtf.tolist(),
            "mtf50": (
                float(mtf50 / sampling["camera_px_per_display_px_along_edge_normal"])
                if mtf50 is not None
                else None
            ),
            "mtf10": (
                float(mtf10 / sampling["camera_px_per_display_px_along_edge_normal"])
                if mtf10 is not None
                else None
            ),
        },
        "confidence": confidence,
        "confidence_components": {
            "sample_support": sample_support,
            "bin_occupancy": bin_support,
            "edge_contrast": contrast_quality,
            "geometry": geometry_quality,
        },
        "warnings": (
            ["measured_oecf_missing"] if oecf_lut is None else []
        ),
    }
    if display_pixel_pitch_mm_xy is not None:
        pitch = np.asarray(display_pixel_pitch_mm_xy, dtype=np.float64).reshape(-1)
        if len(pitch) != 2 or not np.all(np.isfinite(pitch)) or np.any(pitch <= 0):
            raise ValueError("Display pixel pitch must contain two positive mm values")
        normal_pitch_mm = float(np.linalg.norm(normal * pitch))
        result["display_physical"] = {
            "spatial_frequency_unit": "line_pairs_per_mm",
            "equivalent_unit": "cycles_per_mm",
            "pixel_pitch_mm_xy": pitch.tolist(),
            "pixel_pitch_mm_along_edge_normal": normal_pitch_mm,
            "frequency": (frequencies / normal_pitch_mm).tolist(),
            "mtf50": (
                float(mtf50 / normal_pitch_mm) if mtf50 is not None else None
            ),
            "mtf10": (
                float(mtf10 / normal_pitch_mm) if mtf10 is not None else None
            ),
        }
    return result, evidence


def aggregate_esfr_measurements(
    measurements: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Aggregate declared e-SFR conditions without hiding the per-trial curves."""

    rows = [dict(item) for item in measurements]
    if not rows:
        raise ValueError("At least one e-SFR measurement is required")
    grid = np.linspace(0.0, 0.5, 251)
    curves = []
    mtf50_values = []
    mtf10_values = []
    for row in rows:
        display = row["display_referred"]
        frequency = np.asarray(display["frequency"], dtype=np.float64)
        mtf = np.asarray(display["mtf"], dtype=np.float64)
        curves.append(np.interp(grid, frequency, mtf))
        if display.get("mtf50") is not None:
            mtf50_values.append(float(display["mtf50"]))
        if display.get("mtf10") is not None:
            mtf10_values.append(float(display["mtf10"]))
    curve_array = np.asarray(curves, dtype=np.float64)
    conservative = np.min(curve_array, axis=0)
    median = np.median(curve_array, axis=0)
    warnings = sorted(
        {warning for row in rows for warning in row.get("warnings", [])}
    )
    condition_summaries = []
    for row in rows:
        display = row["display_referred"]
        condition_summaries.append(
            {
                "channel": row["channel"],
                "edge_angle_display_deg": row["edge_angle_display_deg"],
                "phase_display_px": row["phase_display_px"],
                "mtf50": display.get("mtf50"),
                "mtf10": display.get("mtf10"),
                "confidence": row.get("confidence", 0.0),
                "warnings": list(row.get("warnings", [])),
            }
        )
    return {
        "standard": "ISO 12233:2024",
        "method": "slanted_edge_e_sfr",
        "implementation_conformance": "non_certified",
        "measurement_input_space": "camera_pre_homography_px",
        "primary_spatial_frequency_unit": "cycles_per_display_pixel",
        "native_analysis_frequency_unit": "cycles_per_camera_pixel",
        "display_nyquist_cycles_per_display_pixel": 0.5,
        "condition_count": len(rows),
        "display_referred": {
            "aggregation": "minimum_response_curve_across_declared_conditions",
            "spatial_frequency_unit": "cycles_per_display_pixel",
            "mtf50_conservative": (
                float(min(mtf50_values)) if mtf50_values else None
            ),
            "mtf10_conservative": (
                float(min(mtf10_values)) if mtf10_values else None
            ),
            "frequency": grid.tolist(),
            "mtf_conservative": conservative.tolist(),
            "mtf_median": median.tolist(),
        },
        "conditions": condition_summaries,
        "measurements": rows,
        "confidence": float(min(float(row.get("confidence", 0.0)) for row in rows)),
        "warnings": warnings,
    }
