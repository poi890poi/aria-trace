"""Phone-display stimuli constrained to the camera-visible screen region."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from ..image_quality import generate_slanted_edge_target


def _sizes(screen_size_px: Sequence[int], region_xywh: Sequence[int]):
    screen_width, screen_height = map(int, screen_size_px)
    x, y, width, height = map(int, region_xywh)
    if min(screen_width, screen_height, width, height) <= 0:
        raise ValueError("Screen and region dimensions must be positive")
    if x < 0 or y < 0 or x + width > screen_width or y + height > screen_height:
        raise ValueError("Pattern region lies outside the phone display")
    return screen_width, screen_height, x, y, width, height


def white_patch(
    screen_size_px: Sequence[int],
    region_xywh: Sequence[int],
    intensity_bgr: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Return a dark screen with white only where camera samples are useful."""

    sw, sh, x, y, width, height = _sizes(screen_size_px, region_xywh)
    canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
    canvas[y : y + height, x : x + width] = np.asarray(intensity_bgr, dtype=np.uint8)
    return canvas


def camera_white_mask(
    camera_size_px: Sequence[int],
    screen_size_px: Sequence[int],
    region_xywh: Sequence[int],
    screen_to_camera_3x3: Sequence[Sequence[float]],
    inset_screen_px: int = 8,
) -> np.ndarray:
    """Project the known white display patch into full-sensor camera pixels."""

    camera_width, camera_height = map(int, camera_size_px)
    screen_width, screen_height, x, y, width, height = _sizes(
        screen_size_px, region_xywh
    )
    inset = max(0, int(inset_screen_px))
    if width <= 2 * inset or height <= 2 * inset:
        raise ValueError("White region is too small for the requested inset")
    screen_mask = np.zeros((screen_height, screen_width), dtype=np.uint8)
    screen_mask[
        y + inset : y + height - inset,
        x + inset : x + width - inset,
    ] = 255
    return cv2.warpPerspective(
        screen_mask,
        np.asarray(screen_to_camera_3x3, dtype=np.float64),
        (camera_width, camera_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def focus_pattern(
    screen_size_px: Sequence[int],
    region_xywh: Sequence[int],
) -> np.ndarray:
    """Render a four-field ISO 12233-style slanted-edge focus chart."""

    sw, sh, x, y, width, height = _sizes(screen_size_px, region_xywh)
    canvas = np.full((sh, sw, 3), 24, dtype=np.uint8)
    frame_x, frame_y, frame_width, frame_height = focus_frame_rect(region_xywh)
    frame_thickness = max(4, int(round(min(width, height) * 0.012)))
    cv2.rectangle(
        canvas,
        (frame_x, frame_y),
        (frame_x + frame_width - 1, frame_y + frame_height - 1),
        (255, 255, 255),
        cv2.FILLED,
        cv2.LINE_8,
    )
    cv2.rectangle(
        canvas,
        (frame_x + frame_thickness, frame_y + frame_thickness),
        (
            frame_x + frame_width - 1 - frame_thickness,
            frame_y + frame_height - 1 - frame_thickness,
        ),
        (24, 24, 24),
        cv2.FILLED,
        cv2.LINE_8,
    )
    for edge in focus_edge_regions(region_xywh):
        edge_x, edge_y, edge_width, edge_height = edge["rect_screen_xywh"]
        target = generate_slanted_edge_target(
            (edge_width, edge_height),
            (0, 0, edge_width, edge_height),
            edge_angle_deg=float(edge["angle_deg"]),
        )
        canvas[
            edge_y : edge_y + edge_height,
            edge_x : edge_x + edge_width,
        ] = target
    return canvas


def focus_frame_rect(region_xywh: Sequence[int]) -> List[int]:
    """Return a centered pose rectangle with acquisition margin for rig movement."""

    x, y, width, height = map(int, region_xywh)
    frame_width = int(round(width * 0.62))
    frame_height = int(round(height * 0.62))
    if min(frame_width, frame_height) < 160:
        raise ValueError("Focus region is too small for a pose frame")
    return [
        x + (width - frame_width) // 2,
        y + (height - frame_height) // 2,
        frame_width,
        frame_height,
    ]


def focus_edge_regions(region_xywh: Sequence[int]) -> List[Dict[str, Any]]:
    """Return four field/direction ROIs used by both rendering and e-SFR."""

    x, y, width, height = focus_frame_rect(region_xywh)
    if min(width, height) <= 0:
        raise ValueError("Focus region dimensions must be positive")
    gap = max(12, int(round(min(width, height) * 0.04)))
    patch_width = (width - 3 * gap) // 2
    patch_height = (height - 3 * gap) // 2
    if min(patch_width, patch_height) < 64:
        raise ValueError("Camera-visible region is too small for four focus edges")
    positions = (
        ("top_left_horizontal", x + gap, y + gap, 5.0),
        ("top_right_vertical", x + 2 * gap + patch_width, y + gap, 85.0),
        ("bottom_left_vertical", x + gap, y + 2 * gap + patch_height, 95.0),
        (
            "bottom_right_horizontal",
            x + 2 * gap + patch_width,
            y + 2 * gap + patch_height,
            -5.0,
        ),
    )
    return [
        {
            "id": name,
            "rect_screen_xywh": [left, top, patch_width, patch_height],
            "angle_deg": angle,
            "method": "ISO_12233_slanted_edge_eSFR",
        }
        for name, left, top, angle in positions
    ]


def tinted(image: np.ndarray, color_bgr: Sequence[float], intensity: float) -> np.ndarray:
    """Apply a declared color/intensity condition without changing geometry."""

    factors = np.asarray(color_bgr, dtype=np.float64).reshape((1, 1, 3))
    factors /= max(float(np.max(factors)), 1.0e-9)
    return np.clip(image.astype(np.float64) * factors * float(intensity), 0, 255).astype(np.uint8)
