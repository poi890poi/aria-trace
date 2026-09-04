"""Explicit Android game-frame geometry relative to rotation-0 panel space."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from rig_runtime.domain.spaces import RigSpaceId


CANONICAL_ANDROID_SPACE_ID = RigSpaceId.ANDROID_PHONE_NATURAL
LOGICAL_ANDROID_SPACE_ID = RigSpaceId.ANDROID_LOGICAL_DISPLAY


def _matrix_list(value: np.ndarray) -> list:
    result = np.asarray(value, dtype=np.float64).copy()
    result[np.abs(result) < 1.0e-12] = 0.0
    return result.tolist()


def natural_to_logical_matrix(
    natural_size_px: Sequence[int], quarter_turns_clockwise: int
) -> np.ndarray:
    """Map rotation-0 Android pixel centers into the current logical raster."""

    width, height = map(int, natural_size_px)
    if min(width, height) <= 0:
        raise ValueError("Android natural raster dimensions must be positive")
    turns = int(quarter_turns_clockwise) % 4
    return np.asarray(
        (
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, -1, height - 1], [1, 0, 0], [0, 0, 1]],
            [[-1, 0, width - 1], [0, -1, height - 1], [0, 0, 1]],
            [[0, 1, 0], [-1, 0, width - 1], [0, 0, 1]],
        )[turns],
        dtype=np.float64,
    )


def android_game_image_space(
    *,
    natural_size_px: Sequence[int],
    surface_quarter_turns_clockwise: int,
    source_size_px: Sequence[int],
    roi_xywh: Sequence[int],
    stored_size_px: Sequence[int],
    orientation_source: str,
) -> dict:
    """Describe one ADB/scrcpy raster without inferring geometry downstream.

    The source raster is the complete current Android logical display. The
    stored raster may be a crop and may be resized. Matrices use pixel-center
    coordinates and therefore include OpenCV's half-pixel resize offset.
    """

    natural = list(map(int, natural_size_px))
    source = list(map(int, source_size_px))
    roi = list(map(int, roi_xywh))
    stored = list(map(int, stored_size_px))
    if len(natural) != 2 or len(source) != 2 or len(stored) != 2 or len(roi) != 4:
        raise ValueError("Android image-space sizes must be W,H and ROI must be XYWH")
    if min(natural + source + stored + roi[2:]) <= 0 or min(roi[:2]) < 0:
        raise ValueError("Android image-space dimensions and ROI are invalid")
    turns = int(surface_quarter_turns_clockwise) % 4
    expected_source = (
        [natural[1], natural[0]] if turns % 2 else list(natural)
    )
    if source != expected_source:
        raise ValueError(
            "Android logical source {} does not match rotation-0 raster {} at "
            "quarter-turn {}".format(source, natural, turns)
        )
    if roi[0] + roi[2] > source[0] or roi[1] + roi[3] > source[1]:
        raise ValueError("Android ROI {} exceeds logical raster {}".format(roi, source))

    scale_x = float(roi[2]) / float(stored[0])
    scale_y = float(roi[3]) / float(stored[1])
    local_to_logical = np.asarray(
        [
            [scale_x, 0.0, float(roi[0]) + 0.5 * scale_x - 0.5],
            [0.0, scale_y, float(roi[1]) + 0.5 * scale_y - 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    natural_to_logical = natural_to_logical_matrix(natural, turns)
    local_to_natural = np.linalg.inv(natural_to_logical).dot(local_to_logical)
    full_unscaled = roi == [0, 0] + source and stored == source
    canonical_identity = full_unscaled and turns == 0
    return {
        "schema_version": "1.0",
        "space_id": (
            CANONICAL_ANDROID_SPACE_ID
            if canonical_identity
            else RigSpaceId.ANDROID_GAME_CAPTURE
        ),
        "stored_size_px": stored,
        "color_order": "BGR",
        "orientation": "android_surface_up",
        "canonical_space_id": CANONICAL_ANDROID_SPACE_ID,
        "canonical_size_px": natural,
        "parent_space_id": CANONICAL_ANDROID_SPACE_ID,
        "source_logical_space_id": LOGICAL_ANDROID_SPACE_ID,
        "source_logical_size_px": source,
        "source_roi_xywh": roi,
        "surface_quarter_turns_clockwise_from_canonical": turns,
        "orientation_source": str(orientation_source),
        "local_to_source_logical_3x3": _matrix_list(local_to_logical),
        "source_logical_to_canonical_3x3": _matrix_list(
            np.linalg.inv(natural_to_logical)
        ),
        "local_to_canonical_3x3": _matrix_list(local_to_natural),
        "canonical_to_local_3x3": _matrix_list(np.linalg.inv(local_to_natural)),
        "resampling": (
            "none" if stored == roi[2:4] else "opencv_resize_pixel_center"
        ),
    }


def image_space_from_surface(
    surface: Mapping[str, object],
    *,
    source_size_px: Sequence[int],
    roi_xywh: Sequence[int],
    stored_size_px: Sequence[int],
) -> dict:
    """Build an image-space record from one probed capture surface."""

    return android_game_image_space(
        natural_size_px=surface["natural_size_px"],
        surface_quarter_turns_clockwise=int(
            surface["quarter_turns_clockwise_from_natural"]
        ),
        source_size_px=source_size_px,
        roi_xywh=roi_xywh,
        stored_size_px=stored_size_px,
        orientation_source=str(surface.get("source") or "unspecified"),
    )


__all__ = [
    "CANONICAL_ANDROID_SPACE_ID",
    "LOGICAL_ANDROID_SPACE_ID",
    "android_game_image_space",
    "image_space_from_surface",
    "natural_to_logical_matrix",
]
