"""Planar camera/screen calibration and optional ChArUco helpers."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .contracts import GeometryEstimate, matrix_3x3, points_xy


def _raster_extent(size: Sequence[int]) -> np.ndarray:
    width, height = map(float, size)
    if width <= 0 or height <= 0:
        raise ValueError("Raster size must be positive")
    return np.asarray(
        [
            [-0.5, -0.5],
            [width - 0.5, -0.5],
            [width - 0.5, height - 0.5],
            [-0.5, height - 0.5],
        ],
        dtype=np.float64,
    )


def transform_points(points: Sequence[Sequence[float]], transform: np.ndarray) -> np.ndarray:
    source = points_xy(points)
    matrix = matrix_3x3(transform)
    homogeneous = np.column_stack([source, np.ones(len(source), dtype=np.float64)])
    projected = homogeneous.dot(matrix.T)
    denominator = projected[:, 2]
    if np.any(np.abs(denominator) < 1.0e-12):
        raise ValueError("Transform projects points to infinity")
    return projected[:, :2] / denominator[:, None]


def _intersection_area(first: np.ndarray, second: np.ndarray) -> float:
    first32 = np.asarray(first, dtype=np.float32).reshape((-1, 1, 2))
    second32 = np.asarray(second, dtype=np.float32).reshape((-1, 1, 2))
    try:
        area, _ = cv2.intersectConvexConvex(first32, second32)
    except cv2.error:
        return 0.0
    return max(0.0, float(area))


def _polygon_area(polygon: np.ndarray) -> float:
    return abs(float(cv2.contourArea(np.asarray(polygon, dtype=np.float32))))


def _normalized_polygon(polygon: np.ndarray, size: Sequence[int]) -> np.ndarray:
    width, height = map(float, size)
    return np.asarray(polygon, dtype=np.float64) / np.asarray([width, height])


def visible_screen_mask(
    camera_size_px: Sequence[int],
    screen_size_px: Sequence[int],
    camera_to_screen_3x3: Sequence[Sequence[float]],
) -> np.ndarray:
    """Return screen pixels backed by real camera samples without interpolation."""

    camera_width, camera_height = map(int, camera_size_px)
    screen_width, screen_height = map(int, screen_size_px)
    if min(camera_width, camera_height, screen_width, screen_height) <= 0:
        raise ValueError("Camera and screen sizes must be positive")
    source = np.full((camera_height, camera_width), 255, dtype=np.uint8)
    return cv2.warpPerspective(
        source,
        matrix_3x3(camera_to_screen_3x3),
        (screen_width, screen_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def select_visible_quality_region(
    camera_size_px: Sequence[int],
    screen_size_px: Sequence[int],
    camera_to_screen_3x3: Sequence[Sequence[float]],
    required_region_screen_xy: Optional[Sequence[Sequence[float]]] = None,
    supported_region_screen_xy: Optional[Sequence[Sequence[float]]] = None,
    supported_region_margin_display_px: int = 0,
    margin_display_px: int = 8,
    minimum_size_display_px: int = 64,
    maximum_size_display_px: int = 640,
) -> Dict[str, Any]:
    """Choose a square quality patch wholly inside visible task-screen pixels.

    The ChArUco atlas establishes the camera-to-display transform first. This
    helper then intersects its camera-backed mask with the caller's task ROI and
    finds a conservative inscribed square. It therefore works when the camera
    sees only a small part of the display and never treats black warp borders as
    captured evidence.
    """

    screen_width, screen_height = map(int, screen_size_px)
    mask = visible_screen_mask(
        camera_size_px, screen_size_px, camera_to_screen_3x3
    )
    candidate = mask.copy()
    required_area = float(screen_width * screen_height)
    if required_region_screen_xy is not None:
        required = points_xy(required_region_screen_xy, minimum=3)
        if not cv2.isContourConvex(required.astype(np.float32).reshape((-1, 1, 2))):
            raise ValueError("Required region must be a convex polygon")
        required_mask = np.zeros_like(candidate)
        cv2.fillConvexPoly(
            required_mask, np.round(required).astype(np.int32), 255, cv2.LINE_8
        )
        candidate = cv2.bitwise_and(candidate, required_mask)
        required_area = max(_polygon_area(required), 1.0)
    if supported_region_screen_xy is not None:
        supported = points_xy(supported_region_screen_xy, minimum=3)
        support_mask = np.zeros_like(candidate)
        cv2.fillConvexPoly(
            support_mask, np.round(supported).astype(np.int32), 255, cv2.LINE_8
        )
        support_margin = max(0, int(supported_region_margin_display_px))
        if support_margin:
            support_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (support_margin * 2 + 1, support_margin * 2 + 1),
            )
            support_mask = cv2.dilate(support_mask, support_kernel, iterations=1)
        candidate = cv2.bitwise_and(candidate, support_mask)

    margin = max(0, int(margin_display_px))
    if margin:
        kernel_size = margin * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        safe = cv2.erode(candidate, kernel, iterations=1)
    else:
        safe = candidate
    if not np.any(safe):
        raise ValueError("No camera-visible pixels remain inside the required ROI")

    distance = cv2.distanceTransform((safe > 0).astype(np.uint8), cv2.DIST_L2, 5)
    _, radius, _, center = cv2.minMaxLoc(distance)
    half = int(np.floor(float(radius) / np.sqrt(2.0)))
    half = min(half, max(1, int(maximum_size_display_px) // 2))
    size = half * 2
    if size < int(minimum_size_display_px):
        raise ValueError(
            "Camera-visible task region is too small for quality measurement: "
            "{} px square available, {} px required".format(
                size, int(minimum_size_display_px)
            )
        )
    center_x, center_y = map(int, center)
    left = max(0, min(screen_width - size, center_x - half))
    top = max(0, min(screen_height - size, center_y - half))
    rect_mask = safe[top : top + size, left : left + size]
    if rect_mask.shape != (size, size) or not np.all(rect_mask > 0):
        raise RuntimeError("Cannot construct a fully camera-visible quality patch")

    visible_required_area = float(np.count_nonzero(candidate))
    return {
        "status": "available",
        "xywh": [int(left), int(top), int(size), int(size)],
        "space": "canonical_phone_screen_px",
        "selection": "largest_conservative_square_in_visible_required_region",
        "requires_detected_atlas_hull_support": supported_region_screen_xy is not None,
        "atlas_hull_support_margin_display_px": max(
            0, int(supported_region_margin_display_px)
        ),
        "margin_display_px": margin,
        "visible_required_fraction": float(
            np.clip(visible_required_area / required_area, 0.0, 1.0)
        ),
        "valid_pixel_count": int(np.count_nonzero(candidate)),
    }


def estimate_screen_geometry(
    camera_points_xy: Sequence[Sequence[float]],
    screen_points_xy: Sequence[Sequence[float]],
    camera_size_px: Sequence[int],
    screen_size_px: Sequence[int],
    required_region_screen_xy: Optional[Sequence[Sequence[float]]] = None,
    ransac_threshold_screen_px: float = 2.0,
) -> GeometryEstimate:
    """Fit camera-undistorted-pixel to canonical-screen-pixel geometry.

    The returned 3x3 matrix always maps camera input pixel centres to canonical
    phone-screen pixel centres. Coverage polygons use continuous raster extents.
    """

    camera_points = points_xy(camera_points_xy, minimum=4)
    screen_points = points_xy(screen_points_xy, minimum=4)
    if len(camera_points) != len(screen_points):
        raise ValueError("Camera and screen correspondence counts differ")
    if ransac_threshold_screen_px <= 0:
        raise ValueError("RANSAC threshold must be positive")

    transform, mask = cv2.findHomography(
        camera_points.astype(np.float64),
        screen_points.astype(np.float64),
        cv2.RANSAC,
        float(ransac_threshold_screen_px),
    )
    if transform is None or mask is None:
        raise RuntimeError("Cannot fit camera-to-screen homography")
    transform = matrix_3x3(transform)
    inverse = matrix_3x3(np.linalg.inv(transform))
    inliers = mask.reshape(-1).astype(bool)
    if np.count_nonzero(inliers) < 4:
        raise RuntimeError("Homography has fewer than four inliers")

    projected = transform_points(camera_points, transform)
    errors = np.linalg.norm(projected - screen_points, axis=1)
    screen_extent = _raster_extent(screen_size_px)
    camera_extent = _raster_extent(camera_size_px)
    viewport_in_screen = transform_points(camera_extent, transform)
    screen_in_camera = transform_points(screen_extent, inverse)

    screen_area = _polygon_area(screen_extent)
    camera_area = _polygon_area(camera_extent)
    visible_screen_area = _intersection_area(screen_extent, viewport_in_screen)
    useful_camera_area = _intersection_area(camera_extent, screen_in_camera)

    normalized_screen = _normalized_polygon(screen_extent, screen_size_px)
    normalized_viewport = _normalized_polygon(viewport_in_screen, screen_size_px)
    normalized_intersection = _intersection_area(normalized_screen, normalized_viewport)
    normalized_union = (
        _polygon_area(normalized_screen)
        + _polygon_area(normalized_viewport)
        - normalized_intersection
    )

    screen_coverage = visible_screen_area / max(screen_area, 1.0e-12)
    camera_utilization = useful_camera_area / max(camera_area, 1.0e-12)
    screen_view_iou = normalized_intersection / max(normalized_union, 1.0e-12)

    hull = cv2.convexHull(screen_points[inliers].astype(np.float32)).reshape((-1, 2))
    detected_hull_coverage = _intersection_area(screen_extent, hull) / max(
        screen_area, 1.0e-12
    )

    required_coverage = 1.0
    required_hull_coverage = 1.0
    if required_region_screen_xy is not None:
        required = points_xy(required_region_screen_xy, minimum=3)
        if not cv2.isContourConvex(required.astype(np.float32).reshape((-1, 1, 2))):
            raise ValueError("Required region must be a convex polygon")
        required_area = _polygon_area(required)
        if required_area <= 0:
            raise ValueError("Required region must have positive area")
        required_coverage = _intersection_area(required, viewport_in_screen) / required_area
        required_hull_coverage = _intersection_area(required, hull) / required_area

    inlier_errors = errors[inliers]
    rmse = float(np.sqrt(np.mean(inlier_errors ** 2)))
    inlier_ratio = float(np.mean(inliers))
    error_quality = float(np.exp(-rmse / max(ransac_threshold_screen_px, 1.0e-6)))
    count_quality = min(1.0, float(np.count_nonzero(inliers)) / 12.0)
    spatial_quality = min(1.0, detected_hull_coverage / 0.35)
    confidence = float(
        np.clip(
            0.35 * inlier_ratio
            + 0.25 * error_quality
            + 0.20 * count_quality
            + 0.20 * spatial_quality,
            0.0,
            1.0,
        )
    )

    warnings: List[str] = []
    if np.count_nonzero(inliers) < 12:
        warnings.append("fewer_than_12_inlier_corners")
    if detected_hull_coverage < 0.25:
        warnings.append("limited_detected_screen_hull")
    if required_coverage < 0.999:
        warnings.append("required_region_is_cropped")
    if required_hull_coverage < 0.999:
        warnings.append("required_region_requires_extrapolation")

    metrics = {
        "screen_coverage": float(np.clip(screen_coverage, 0.0, 1.0)),
        "camera_utilization": float(np.clip(camera_utilization, 0.0, 1.0)),
        "screen_view_iou": float(np.clip(screen_view_iou, 0.0, 1.0)),
        "required_region_coverage": float(np.clip(required_coverage, 0.0, 1.0)),
        "required_region_detected_hull_coverage": float(
            np.clip(required_hull_coverage, 0.0, 1.0)
        ),
        "detected_hull_screen_coverage": float(
            np.clip(detected_hull_coverage, 0.0, 1.0)
        ),
        "inlier_ratio": inlier_ratio,
        "reprojection_rmse_px": rmse,
        "reprojection_p95_px": float(np.percentile(inlier_errors, 95)),
    }
    return GeometryEstimate(
        matrix_3x3=transform,
        inverse_matrix_3x3=inverse,
        inlier_mask=inliers,
        reprojection_errors_px=errors,
        screen_polygon_input_xy=screen_in_camera,
        viewport_polygon_screen_xy=viewport_in_screen,
        metrics=metrics,
        confidence=confidence,
        warnings=tuple(warnings),
    )


def calibrate_intrinsics_from_views(
    camera_points_by_view: Sequence[Sequence[Sequence[float]]],
    screen_points_by_view: Sequence[Sequence[Sequence[float]]],
    camera_size_px: Sequence[int],
    flags: int = 0,
) -> Dict[str, Any]:
    """Estimate camera intrinsics from several tilted views of the screen plane."""

    if len(camera_points_by_view) != len(screen_points_by_view):
        raise ValueError("Camera and screen view counts differ")
    if len(camera_points_by_view) < 3:
        raise ValueError("At least three tilted calibration views are required")
    image_points = []
    object_points = []
    for camera_value, screen_value in zip(camera_points_by_view, screen_points_by_view):
        camera_points = points_xy(camera_value, minimum=4).astype(np.float32)
        screen_points = points_xy(screen_value, minimum=4).astype(np.float32)
        if len(camera_points) != len(screen_points):
            raise ValueError("Correspondence count differs within a calibration view")
        objects = np.column_stack(
            [screen_points, np.zeros(len(screen_points), dtype=np.float32)]
        ).astype(np.float32)
        image_points.append(camera_points.reshape((-1, 1, 2)))
        object_points.append(objects.reshape((-1, 1, 3)))

    width, height = map(int, camera_size_px)
    if width <= 0 or height <= 0:
        raise ValueError("Camera size must be positive")
    rms, camera_matrix, distortion, rotations, translations = cv2.calibrateCamera(
        object_points,
        image_points,
        (width, height),
        None,
        None,
        flags=int(flags),
    )
    per_view = []
    for objects, observed, rotation, translation in zip(
        object_points, image_points, rotations, translations
    ):
        projected, _ = cv2.projectPoints(
            objects, rotation, translation, camera_matrix, distortion
        )
        error = np.linalg.norm(
            projected.reshape((-1, 2)) - observed.reshape((-1, 2)), axis=1
        )
        per_view.append(
            {
                "rmse_px": float(np.sqrt(np.mean(error ** 2))),
                "p95_px": float(np.percentile(error, 95)),
            }
        )
    return {
        "model": "opencv_radtan",
        "camera_matrix_3x3": camera_matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
        "rms_px": float(rms),
        "per_view": per_view,
        "view_count": len(per_view),
        "camera_size_px": [width, height],
    }


def _aruco_module():
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        raise RuntimeError(
            "OpenCV ArUco/ChArUco support is unavailable; install "
            "opencv-contrib-python-headless for live ChArUco calibration"
        )
    return aruco


@dataclass(frozen=True)
class CharucoLayout:
    screen_size_px: Tuple[int, int]
    squares_x: int = 7
    squares_y: int = 11
    margin_px: Tuple[int, int] = (24, 24)
    marker_ratio: float = 0.72
    dictionary_name: str = "DICT_5X5_1000"

    def __post_init__(self) -> None:
        width, height = self.screen_size_px
        if width <= 0 or height <= 0:
            raise ValueError("Screen size must be positive")
        if self.squares_x < 3 or self.squares_y < 3:
            raise ValueError("ChArUco layout needs at least 3x3 squares")
        if not 0.1 < self.marker_ratio < 1.0:
            raise ValueError("Marker ratio must be within (0.1, 1.0)")
        if min(self.margin_px) < 0:
            raise ValueError("ChArUco margins must be non-negative")

    @property
    def square_px(self) -> float:
        width, height = self.screen_size_px
        margin_x, margin_y = self.margin_px
        available_width = width - 2 * margin_x
        available_height = height - 2 * margin_y
        square = min(
            available_width / float(self.squares_x),
            available_height / float(self.squares_y),
        )
        if square < 4.0:
            raise ValueError("Screen is too small for the requested ChArUco layout")
        return square

    @property
    def board_size_px(self) -> Tuple[int, int]:
        return (
            int(round(self.square_px * self.squares_x)),
            int(round(self.square_px * self.squares_y)),
        )

    @property
    def offset_xy(self) -> Tuple[float, float]:
        width, height = self.screen_size_px
        board_width, board_height = self.board_size_px
        return ((width - board_width) / 2.0, (height - board_height) / 2.0)


def _charuco_board(layout: CharucoLayout):
    aruco = _aruco_module()
    dictionary_id = getattr(aruco, layout.dictionary_name, None)
    if dictionary_id is None:
        raise ValueError("Unknown ArUco dictionary {}".format(layout.dictionary_name))
    dictionary = aruco.getPredefinedDictionary(dictionary_id)
    square_length = 1.0
    marker_length = float(layout.marker_ratio)
    if hasattr(aruco, "CharucoBoard_create"):
        board = aruco.CharucoBoard_create(
            layout.squares_x,
            layout.squares_y,
            square_length,
            marker_length,
            dictionary,
        )
    else:
        board = aruco.CharucoBoard(
            (layout.squares_x, layout.squares_y),
            square_length,
            marker_length,
            dictionary,
        )
    return aruco, dictionary, board


def _board_corners(board) -> np.ndarray:
    if hasattr(board, "getChessboardCorners"):
        return np.asarray(board.getChessboardCorners(), dtype=np.float64).reshape((-1, 3))
    return np.asarray(board.chessboardCorners, dtype=np.float64).reshape((-1, 3))


def charuco_board_metric_to_panel_pixels(
    board_points_square_xy: Sequence[Sequence[float]],
    layout: CharucoLayout,
    panel_size_px: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Map known ChArUco square coordinates into the presenter surface.

    This is deliberately independent of Android's reported physical DPI.  When
    ``layout.screen_size_px`` comes from the native SurfaceView, the resulting
    correspondences make that possibly anisotropic surface scale part of the
    fitted camera-to-panel homography.
    """

    board_points = points_xy(board_points_square_xy, minimum=1)
    nominal_size = np.asarray(layout.screen_size_px, dtype=np.float64)
    panel_size = np.asarray(
        panel_size_px if panel_size_px is not None else layout.screen_size_px,
        dtype=np.float64,
    )
    if panel_size.shape != (2,) or np.any(panel_size <= 0):
        raise ValueError("Panel size must contain two positive pixel dimensions")
    surface_scale = panel_size / nominal_size
    offset = np.asarray(layout.offset_xy, dtype=np.float64) * surface_scale
    board_width, board_height = layout.board_size_px
    nominal_scale = np.asarray(
        [
            board_width / float(layout.squares_x),
            board_height / float(layout.squares_y),
        ],
        dtype=np.float64,
    )
    scale = nominal_scale * surface_scale
    return board_points * scale + offset, {
        "unit": "charuco_square",
        "panel_px_per_square_xy": scale.tolist(),
        "origin_panel_px_xy": offset.tolist(),
        "panel_size_px": list(map(int, panel_size)),
        "target_raster_size_px": list(map(int, layout.screen_size_px)),
        "target_raster_to_panel_scale_xy": surface_scale.tolist(),
        "adb_physical_dpi_used": False,
    }


def generate_charuco_target(layout: CharucoLayout) -> np.ndarray:
    """Render a full-screen BGR ChArUco target for an external presenter."""

    _, _, board = _charuco_board(layout)
    board_size = layout.board_size_px
    if hasattr(board, "generateImage"):
        board_image = board.generateImage(board_size, marginSize=0, borderBits=1)
    else:
        board_image = board.draw(board_size, marginSize=0, borderBits=1)
    width, height = layout.screen_size_px
    canvas = np.full((height, width), 255, dtype=np.uint8)
    offset_x, offset_y = layout.offset_xy
    left, top = int(round(offset_x)), int(round(offset_y))
    board_height, board_width = board_image.shape[:2]
    canvas[top : top + board_height, left : left + board_width] = board_image
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def detect_charuco_correspondences(
    image: np.ndarray, layout: CharucoLayout
) -> Dict[str, Any]:
    """Detect ChArUco points and map their IDs to canonical screen pixels."""

    if image is None or image.size == 0:
        raise ValueError("ChArUco input image is empty")
    aruco, dictionary, board = _charuco_board(layout)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if hasattr(aruco, "ArucoDetector") and hasattr(aruco, "CharucoDetector"):
        marker_corners, marker_ids, rejected = aruco.ArucoDetector(
            dictionary
        ).detectMarkers(gray)
        if marker_ids is None or len(marker_ids) == 0:
            raise RuntimeError("No ChArUco markers were detected")
        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            aruco.CharucoDetector(board).detectBoard(
                gray, None, None, marker_corners, marker_ids
            )
        )
        count = 0 if charuco_ids is None else len(charuco_ids)
    else:
        marker_corners, marker_ids, rejected = aruco.detectMarkers(gray, dictionary)
        if marker_ids is None or len(marker_ids) == 0:
            raise RuntimeError("No ChArUco markers were detected")
        count, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, gray, board
        )
    if marker_ids is None or len(marker_ids) == 0:
        raise RuntimeError("No ChArUco markers were detected")
    if charuco_ids is None or charuco_corners is None or int(count) < 4:
        raise RuntimeError("Fewer than four ChArUco corners were detected")
    ids = charuco_ids.reshape(-1).astype(int)
    camera_points = charuco_corners.reshape((-1, 2)).astype(np.float64)
    board_points = _board_corners(board)[ids, :2]
    screen_points, board_metric = charuco_board_metric_to_panel_pixels(
        board_points, layout
    )
    return {
        "camera_points_xy": camera_points,
        "screen_points_xy": screen_points,
        "board_points_square_xy": board_points,
        "board_metric_unit": "charuco_square",
        "board_metric_to_screen_scale_xy": board_metric[
            "panel_px_per_square_xy"
        ],
        "board_metric_origin_screen_xy": board_metric["origin_panel_px_xy"],
        "board_metric_panel_size_px": board_metric["panel_size_px"],
        "corner_ids": ids,
        "marker_count": int(len(marker_ids)),
        "corner_count": int(len(ids)),
        "rejected_marker_count": int(len(rejected)),
    }
