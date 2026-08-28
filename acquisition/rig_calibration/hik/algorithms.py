"""Pure calculations used by the HIK calibration and streaming CLIs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import cv2
import numpy as np

from ..geometry import transform_points


@dataclass(frozen=True)
class ExposureObservation:
    shutter_refresh_multiplier: float
    exposure_us: float
    gain: float
    mean_bgr: tuple[float, float, float]
    clipped_fraction_bgr: tuple[float, float, float]
    temporal_noise_bgr: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def brightness(self) -> float:
        return float(np.mean(self.mean_bgr))

    @property
    def white_balance_reference_brightness(self) -> float:
        """Predict balanced-white level for the residual-WB method.

        ``manual_white_balance_ratios`` raises the dim channels to the
        brightest measured channel, so their unbalanced arithmetic mean is not
        the correct pre-WB acceptance signal.
        """

        return float(max(self.mean_bgr))

    @property
    def maximum_clipped_fraction(self) -> float:
        return float(max(self.clipped_fraction_bgr))

    @property
    def temporal_noise_rms_dn(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.temporal_noise_bgr))))

    @property
    def shutter_rate_hz(self) -> float:
        return 1.0e6 / float(self.exposure_us)

    @property
    def exposure_refresh_periods(self) -> float:
        """Number of complete panel periods integrated by this exposure."""

        return 1.0 / float(self.shutter_refresh_multiplier)


@dataclass(frozen=True)
class BlackLevelObservation:
    level: int
    mean_bgr: tuple[float, float, float]
    zero_fraction_bgr: tuple[float, float, float]
    temporal_noise_bgr: tuple[float, float, float]

    @property
    def maximum_zero_fraction(self) -> float:
        return float(max(self.zero_fraction_bgr))

    @property
    def temporal_noise_rms_dn(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.temporal_noise_bgr))))


def refresh_quantized_exposure_us(refresh_hz: float, multiplier: float) -> float:
    """Return an exposure aligned to whole or reciprocal panel periods.

    Integer factors ``1, 2, 3`` retain the original faster shutter rates.
    Reciprocal-integer factors ``1/2, 1/3`` integrate two or three complete
    panel refresh periods without ending at a fractional display cycle.
    """

    refresh_hz, multiplier = float(refresh_hz), float(multiplier)
    if refresh_hz <= 0 or multiplier <= 0:
        raise ValueError("Refresh rate and shutter-rate factor must be positive")
    nearest_integer = round(multiplier)
    reciprocal = 1.0 / multiplier
    nearest_reciprocal_integer = round(reciprocal)
    integer_rate = multiplier >= 1.0 and abs(multiplier - nearest_integer) <= 1.0e-9
    whole_periods = multiplier <= 1.0 and abs(reciprocal - nearest_reciprocal_integer) <= 1.0e-9
    if not (integer_rate or whole_periods):
        raise ValueError(
            "Shutter-rate factor must be an integer or reciprocal integer relative to panel refresh"
        )
    return 1.0e6 / (refresh_hz * multiplier)


def charuco_orientation_evidence(
    screen_to_camera_3x3: Sequence[Sequence[float]],
    probe_screen_xy: Sequence[float],
    probe_distance_px: float = 100.0,
) -> dict[str, Any]:
    """Measure app/display axes in camera pixels from the fitted ChArUco map."""

    x, y = map(float, probe_screen_xy)
    distance = max(1.0, float(probe_distance_px))
    screen_points = np.asarray(
        [[x, y], [x + distance, y], [x, y - distance]], dtype=np.float64
    )
    camera_points = transform_points(screen_points, screen_to_camera_3x3)
    app_right = camera_points[1] - camera_points[0]
    app_up = camera_points[2] - camera_points[0]
    right_norm = float(np.linalg.norm(app_right))
    up_norm = float(np.linalg.norm(app_up))
    if min(right_norm, up_norm) <= 1.0e-9:
        raise RuntimeError("ChArUco orientation probe is degenerate")
    app_right /= right_norm
    app_up /= up_norm
    clockwise_from_camera_up = math.degrees(math.atan2(float(app_up[0]), float(-app_up[1]))) % 360.0
    handedness = float(np.linalg.det(np.vstack([app_right, -app_up])))
    return {
        "source": "charuco_correspondences",
        "probe_screen_xy": [x, y],
        "probe_distance_screen_px": distance,
        "app_up_unit_vector_camera_xy": app_up.tolist(),
        "app_right_unit_vector_camera_xy": app_right.tolist(),
        "camera_up_to_app_up_clockwise_degrees": clockwise_from_camera_up,
        "camera_xy_handedness": "right_down",
        "app_axes_handedness_determinant": handedness,
        "adapter_output_up": "app_up",
        "adapter_output_right": "app_right",
    }


def white_statistics(image: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Measure BGR brightness and clipping inside one fixed white-area mask."""

    if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("White-balance image must be non-empty BGR")
    selected = np.asarray(mask) > 0
    if selected.shape != image.shape[:2] or np.count_nonzero(selected) < 64:
        raise ValueError("White-area mask has insufficient support")
    pixels = image[selected].astype(np.float64)
    return {
        "pixel_count": int(len(pixels)),
        "mean_bgr": [float(value) for value in np.mean(pixels, axis=0)],
        "clipped_fraction_bgr": [
            float(value) for value in np.mean(pixels >= 255.0, axis=0)
        ],
        "p95_bgr": [float(value) for value in np.percentile(pixels, 95, axis=0)],
    }


def temporal_white_statistics(
    images: Sequence[np.ndarray], mask: np.ndarray
) -> dict[str, Any]:
    """Measure brightness, clipping, and temporal noise on a fixed white mask."""

    frames = [np.asarray(image) for image in images]
    if len(frames) < 2:
        raise ValueError("At least two frames are required for temporal noise")
    if any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("Temporal-noise frames must have identical shapes")
    stack = np.stack(frames).astype(np.float64)
    selected = np.asarray(mask) > 0
    if selected.shape != stack.shape[1:3] or np.count_nonzero(selected) < 64:
        raise ValueError("White-area mask has insufficient support")
    pixels = stack[:, selected, :]
    return {
        "frame_count": len(frames),
        "mean_bgr": np.mean(pixels, axis=(0, 1)).tolist(),
        "clipped_fraction_bgr": np.mean(pixels >= 255.0, axis=(0, 1)).tolist(),
        "temporal_noise_bgr": np.mean(np.std(pixels, axis=0, ddof=1), axis=0).tolist(),
    }


def temporal_black_statistics(
    images: Sequence[np.ndarray], mask: np.ndarray
) -> dict[str, Any]:
    """Measure black pedestal, crushed pixels, and temporal noise."""

    frames = [np.asarray(image) for image in images]
    if len(frames) < 2:
        raise ValueError("At least two frames are required for temporal noise")
    if any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("Temporal-noise frames must have identical shapes")
    stack = np.stack(frames).astype(np.float64)
    selected = np.asarray(mask) > 0
    if selected.shape != stack.shape[1:3] or np.count_nonzero(selected) < 64:
        raise ValueError("Black-area mask has insufficient support")
    pixels = stack[:, selected, :]
    return {
        "frame_count": len(frames),
        "mean_bgr": np.mean(pixels, axis=(0, 1)).tolist(),
        "zero_fraction_bgr": np.mean(pixels <= 0.0, axis=(0, 1)).tolist(),
        "temporal_noise_bgr": np.mean(
            np.std(pixels, axis=0, ddof=1), axis=0
        ).tolist(),
    }


def choose_black_level(
    observations: Iterable[BlackLevelObservation],
    maximum_zero_fraction: float = 0.05,
) -> BlackLevelObservation:
    """Choose the lowest non-crushing black level, using noise as a tie-breaker."""

    rows = list(observations)
    if not rows:
        raise ValueError("At least one black-level observation is required")
    safe = [
        row
        for row in rows
        if row.maximum_zero_fraction <= float(maximum_zero_fraction)
    ]
    if safe:
        return min(safe, key=lambda row: (row.level, row.temporal_noise_rms_dn))
    return min(
        rows,
        key=lambda row: (
            row.maximum_zero_fraction,
            row.temporal_noise_rms_dn,
            row.level,
        ),
    )


def choose_exposure(
    observations: Iterable[ExposureObservation],
    target_fraction: float = 0.90,
    maximum_clipped_fraction: float = 0.05,
    acceptable_fraction: float = 0.02,
) -> ExposureObservation:
    """Choose a safe row using its predicted post-WB white level."""

    rows = list(observations)
    if not rows:
        raise ValueError("At least one exposure observation is required")
    target = 255.0 * float(target_fraction)
    tolerance = 255.0 * float(acceptable_fraction)
    safe = [
        row
        for row in rows
        if row.maximum_clipped_fraction <= float(maximum_clipped_fraction)
    ]
    if not safe:
        # Quality thresholds guide the best usable lock; they are not a reason
        # to make a functioning camera unavailable. The workflow records the
        # threshold miss as a warning.
        return min(
            rows,
            key=lambda row: (
                row.maximum_clipped_fraction,
                row.temporal_noise_rms_dn,
                row.gain,
            ),
        )
    acceptable = [
        row
        for row in safe
        if abs(row.white_balance_reference_brightness - target) <= tolerance
    ]
    if acceptable:
        return min(
            acceptable,
            key=lambda row: (row.temporal_noise_rms_dn, row.gain, row.exposure_us),
        )

    def penalty(row: ExposureObservation):
        clipping = max(
            0.0, row.maximum_clipped_fraction - float(maximum_clipped_fraction)
        )
        return (
            abs(row.white_balance_reference_brightness - target)
            + clipping * 2550.0,
            row.temporal_noise_rms_dn,
            row.gain,
            row.exposure_us,
        )

    return min(safe, key=penalty)


def manual_white_balance_ratios(
    image: np.ndarray,
    mask: np.ndarray,
    base_ratio: int = 1000,
    blur_sigma: float = 15.0,
) -> dict[str, Any]:
    """Return B/G/R ratios after strong blur suppresses display subpixel aliasing."""

    if blur_sigma <= 0:
        raise ValueError("White-balance blur sigma must be positive")
    blurred = cv2.GaussianBlur(image, (0, 0), float(blur_sigma))
    statistics = white_statistics(blurred, mask)
    means = np.asarray(statistics["mean_bgr"], dtype=np.float64)
    if np.any(means <= 1.0):
        raise RuntimeError("White-balance channel signal is too small")
    target = float(np.max(means))
    ratios = np.clip(np.rint(base_ratio * target / means), 1, 4095).astype(int)
    return {
        "base_ratio": int(base_ratio),
        "blur_sigma": float(blur_sigma),
        "source_mean_bgr": statistics["mean_bgr"],
        "ratios_bgr": [int(value) for value in ratios],
        "ratio_blue": int(ratios[0]),
        "ratio_green": int(ratios[1]),
        "ratio_red": int(ratios[2]),
    }


def camera_visible_screen_region(
    viewport_polygon_screen_xy: Sequence[Sequence[float]],
    screen_size_px: Sequence[int],
    margin_px: int = 8,
) -> dict[str, Any]:
    """Intersect the camera footprint with the screen and return a safe XYWH."""

    width, height = map(int, screen_size_px)
    viewport = np.asarray(viewport_polygon_screen_xy, dtype=np.float32).reshape(
        (-1, 1, 2)
    )
    screen = np.asarray(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    ).reshape((-1, 1, 2))
    area, intersection = cv2.intersectConvexConvex(viewport, screen)
    if intersection is None or float(area) <= 0:
        raise RuntimeError("Camera view does not intersect the phone display")
    polygon = intersection.reshape((-1, 2))
    margin = max(0, int(margin_px))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 1)
    if margin:
        mask = cv2.erode(
            mask,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2 * margin + 1, 2 * margin + 1)),
        )
    # Largest axis-aligned all-visible rectangle (histogram/monotonic-stack).
    heights = np.zeros(width, dtype=np.int32)
    best = (0, 0, 0, 0, 0)
    for bottom in range(height):
        heights = np.where(mask[bottom] > 0, heights + 1, 0)
        stack = []
        for column in range(width + 1):
            current = int(heights[column]) if column < width else 0
            start = column
            while stack and stack[-1][1] > current:
                left, bar_height = stack.pop()
                area_rect = bar_height * (column - left)
                if area_rect > best[0]:
                    best = (area_rect, left, bottom - bar_height + 1, column - left, bar_height)
                start = left
            if not stack or stack[-1][1] < current:
                stack.append((start, current))
    _, x, y, region_width, region_height = best
    if region_width < 32 or region_height < 32:
        raise RuntimeError("Camera-visible phone region is too small after margin")
    screen_area = float(width * height)
    viewport_area = abs(float(cv2.contourArea(viewport)))
    union = screen_area + viewport_area - float(area)
    return {
        "xywh": [int(x), int(y), int(region_width), int(region_height)],
        "intersection_polygon_screen_xy": polygon.astype(float).tolist(),
        "intersection_area_screen_px2": float(area),
        "screen_fraction": float(area / max(screen_area, 1.0)),
        "screen_view_iou": float(area / max(union, 1.0)),
        "margin_px": margin,
        "space": "canonical_phone_screen_px",
    }


def camera_roi_for_screen_region(
    screen_region_xywh: Sequence[int],
    screen_to_camera_3x3: Sequence[Sequence[float]],
    camera_size_px: Sequence[int],
    margin_px: int = 4,
) -> list[int]:
    x, y, width, height = map(int, screen_region_xywh)
    polygon = [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]
    camera_polygon = transform_points(polygon, np.asarray(screen_to_camera_3x3))
    left = int(math.floor(float(np.min(camera_polygon[:, 0])))) - int(margin_px)
    top = int(math.floor(float(np.min(camera_polygon[:, 1])))) - int(margin_px)
    right = int(math.ceil(float(np.max(camera_polygon[:, 0])))) + int(margin_px)
    bottom = int(math.ceil(float(np.max(camera_polygon[:, 1])))) + int(margin_px)
    camera_width, camera_height = map(int, camera_size_px)
    left, top = max(0, left), max(0, top)
    right, bottom = min(camera_width, right), min(camera_height, bottom)
    if right <= left or bottom <= top:
        raise RuntimeError("Phone region does not map to a valid camera ROI")
    return [left, top, right - left, bottom - top]


def compose_hardware_roi_homography(
    full_camera_to_output_3x3: Sequence[Sequence[float]],
    hardware_roi_xywh: Sequence[int],
) -> np.ndarray:
    """Adapt a full-sensor homography to frames whose origin is hardware ROI XY."""

    x, y, _, _ = map(int, hardware_roi_xywh)
    crop_to_full = np.asarray(
        [[1.0, 0.0, float(x)], [0.0, 1.0, float(y)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    matrix = np.asarray(full_camera_to_output_3x3, dtype=np.float64).dot(crop_to_full)
    return matrix / matrix[2, 2]


def laplacian_sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(np.var(cv2.Laplacian(gray, cv2.CV_64F)))


def _homogeneous_line(first: Sequence[float], second: Sequence[float]) -> np.ndarray:
    return np.cross(
        np.asarray([float(first[0]), float(first[1]), 1.0]),
        np.asarray([float(second[0]), float(second[1]), 1.0]),
    )


def _line_intersection(first: np.ndarray, second: np.ndarray) -> Optional[np.ndarray]:
    point = np.cross(first, second)
    if abs(float(point[2])) <= 1.0e-9:
        return None
    return point[:2] / point[2]


def _match_quad_to_expected(
    points_xy: Sequence[Sequence[float]], expected_xy: Sequence[Sequence[float]]
) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float64).reshape((4, 2))
    expected = np.asarray(expected_xy, dtype=np.float64).reshape((4, 2))
    candidates = []
    for value in (points, points[::-1]):
        for offset in range(4):
            candidate = np.roll(value, -offset, axis=0)
            candidates.append(candidate)
    return min(
        candidates,
        key=lambda candidate: float(np.mean(np.sum((candidate - expected) ** 2, axis=1))),
    )


def detect_focus_pose_frame(
    image: np.ndarray,
    expected_camera_quad_xy: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Detect the focus chart's white outer frame using threshold/contour primitives."""

    if image is None or image.size == 0:
        raise ValueError("Focus pose frame requires a non-empty camera image")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0.0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    expected = np.asarray(expected_camera_quad_xy, dtype=np.float64).reshape((4, 2))
    expected_area = abs(float(cv2.contourArea(expected.astype(np.float32))))
    if expected_area <= 64.0:
        raise ValueError("Expected focus frame is degenerate")
    candidates = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approximated = cv2.approxPolyDP(contour, max(2.0, perimeter * 0.015), True)
        if len(approximated) != 4 or not cv2.isContourConvex(approximated):
            continue
        area = abs(float(cv2.contourArea(approximated)))
        if area < expected_area * 0.35 or area > expected_area * 2.5:
            continue
        ordered = _match_quad_to_expected(approximated.reshape((4, 2)), expected)
        corner_error = float(np.mean(np.linalg.norm(ordered - expected, axis=1)))
        area_error = abs(math.log(max(area, 1.0) / expected_area))
        candidates.append((corner_error + 30.0 * area_error, ordered, area))
    if not candidates:
        raise RuntimeError("Complete focus pose frame was not detected")
    _, ordered, area = min(candidates, key=lambda row: row[0])
    return {
        "camera_quad_xy": ordered.astype(float).tolist(),
        "contour_area_camera_px2": float(area),
        "method": "otsu_threshold_external_contour_quad",
    }


def estimate_focus_target_pose(
    camera_quad_xy: Sequence[Sequence[float]],
    target_size_mm: Sequence[float],
    camera_size_px: Sequence[int],
    focal_length_px: Optional[float] = None,
) -> dict[str, Any]:
    """Estimate camera-to-planar-target pose from its known rectangular frame."""

    points = np.asarray(camera_quad_xy, dtype=np.float64).reshape((4, 2))
    target_width_mm, target_height_mm = map(float, target_size_mm)
    camera_width, camera_height = map(int, camera_size_px)
    if min(target_width_mm, target_height_mm, camera_width, camera_height) <= 0:
        raise ValueError("Target and camera sizes must be positive")
    center_x = (camera_width - 1.0) / 2.0
    center_y = (camera_height - 1.0) / 2.0
    inferred = focal_length_px is None
    if inferred:
        horizontal_vanishing = _line_intersection(
            _homogeneous_line(points[0], points[1]),
            _homogeneous_line(points[3], points[2]),
        )
        vertical_vanishing = _line_intersection(
            _homogeneous_line(points[0], points[3]),
            _homogeneous_line(points[1], points[2]),
        )
        if horizontal_vanishing is None or vertical_vanishing is None:
            raise RuntimeError("Target is too front-on to infer camera focal length")
        horizontal = horizontal_vanishing - [center_x, center_y]
        vertical = vertical_vanishing - [center_x, center_y]
        focal_squared = -float(np.dot(horizontal, vertical))
        if not np.isfinite(focal_squared) or focal_squared <= 0.0:
            raise RuntimeError("Orthogonal vanishing points do not yield a physical focal length")
        focal_length_px = math.sqrt(focal_squared)
        if not 0.25 * max(camera_width, camera_height) <= focal_length_px <= 20.0 * max(
            camera_width, camera_height
        ):
            raise RuntimeError("Inferred focal length is numerically unstable")
    focal = float(focal_length_px)
    camera_matrix = np.asarray(
        [[focal, 0.0, center_x], [0.0, focal, center_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    objects = np.asarray(
        [
            [-target_width_mm / 2.0, -target_height_mm / 2.0, 0.0],
            [target_width_mm / 2.0, -target_height_mm / 2.0, 0.0],
            [target_width_mm / 2.0, target_height_mm / 2.0, 0.0],
            [-target_width_mm / 2.0, target_height_mm / 2.0, 0.0],
        ],
        dtype=np.float64,
    )
    ok, rotation_vector, translation = cv2.solvePnP(
        objects,
        points,
        camera_matrix,
        np.zeros(5, dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("Focus target pose could not be fitted")
    rotation, _ = cv2.Rodrigues(rotation_vector)
    normal = rotation[:, 2]
    if normal[2] < 0.0:
        normal = -normal
    translation_xyz = translation.reshape(-1)
    yaw = math.degrees(math.atan2(float(normal[0]), float(normal[2])))
    pitch = math.degrees(
        math.atan2(
            float(-normal[1]),
            math.sqrt(float(normal[0] ** 2 + normal[2] ** 2)),
        )
    )
    top_midpoint = (points[0] + points[1]) * 0.5
    bottom_midpoint = (points[2] + points[3]) * 0.5
    app_up = top_midpoint - bottom_midpoint
    phone_rotation = math.degrees(
        math.atan2(float(app_up[0]), float(-app_up[1]))
    )
    projected, _ = cv2.projectPoints(
        objects, rotation_vector, translation, camera_matrix, np.zeros(5, dtype=np.float64)
    )
    errors = np.linalg.norm(projected.reshape((4, 2)) - points, axis=1)
    return {
        "pitch_deg": float(pitch),
        "yaw_deg": float(yaw),
        "phone_rotation_clockwise_from_camera_up_deg": float(phone_rotation),
        "lens_to_panel_distance_mm": abs(float(np.dot(normal, translation_xyz))),
        "optical_axis_depth_mm": abs(float(translation_xyz[2])),
        "focal_length_px": focal,
        "focal_length_source": (
            "orthogonal_target_vanishing_points" if inferred else "retained_target_estimate"
        ),
        "reprojection_p95_camera_px": float(np.percentile(errors, 95)),
        "method": "thresholded_known_rectangle_solvepnp",
    }
