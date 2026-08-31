"""Strict spatial records and review canvases for rig-camera evidence.

The rig producer owns image-space identity.  This module validates and
serializes that identity; it never infers a space from a filename or raster
dimensions.  Review canvases are derived products whose geometry is explicit.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from aria_trace.evidence.media_trace import raster_record
from aria_trace.services.calibration.rig.contracts import FrameSample


SYNTHETIC_BACKGROUND_BGR = (255, 0, 255)
SYNTHETIC_BACKGROUND_DARK_BGR = (96, 0, 96)
FULL_SENSOR_OUTLINE_BGR = (0, 255, 255)
PHONE_DISPLAY_OUTLINE_BGR = (0, 255, 64)
MAXIMUM_REVIEW_DIMENSION_PX = 4096
MAXIMUM_REVIEW_PIXELS = 12_000_000


def _size_wh(image: np.ndarray) -> list[int]:
    if image is None or image.size == 0:
        raise ValueError("Rig evidence image must be non-empty")
    return [int(image.shape[1]), int(image.shape[0])]


def _matrix_3x3(value: Sequence[Sequence[float]], name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("{} must be a finite 3x3 matrix".format(name))
    if abs(float(np.linalg.det(matrix))) < 1.0e-12:
        raise ValueError("{} must be nonsingular".format(name))
    return matrix


def validate_rig_image_space(
    image: np.ndarray,
    image_space: Mapping[str, Any],
    *,
    parent_size_px: Optional[Sequence[int]] = None,
) -> dict:
    """Validate producer-supplied rig raster metadata without repairing it."""

    if not isinstance(image_space, Mapping):
        raise ValueError("Rig frame producer did not supply image_space metadata")
    result = copy.deepcopy(dict(image_space))
    space_id = str(result.get("space_id") or "").strip()
    if not space_id:
        raise ValueError("Rig image_space requires space_id")
    actual_size = _size_wh(image)
    declared_size = list(map(int, result.get("stored_size_px") or []))
    if declared_size != actual_size:
        raise ValueError(
            "Rig image {} is {}, producer declared {}"
            .format(space_id, actual_size, declared_size)
        )
    if not str(result.get("orientation") or "").strip():
        raise ValueError("Rig image_space requires orientation")
    if not str(result.get("color_order") or "").strip():
        raise ValueError("Rig image_space requires color_order")

    parent_id = str(result.get("parent_space_id") or "").strip()
    roi = result.get("roi_in_parent_xywh")
    local_to_parent = result.get("local_to_parent_3x3")
    if parent_id:
        if roi is None or local_to_parent is None:
            raise ValueError(
                "Rig child raster {} requires ROI and local-to-parent transform"
                .format(space_id)
            )
        roi = list(map(int, roi))
        if len(roi) != 4 or roi[2:4] != actual_size or min(roi) < 0:
            raise ValueError(
                "Rig child raster {} has inconsistent ROI {} for image {}"
                .format(space_id, roi, actual_size)
            )
        matrix = _matrix_3x3(local_to_parent, "local_to_parent_3x3")
        expected = np.asarray(
            [[1.0, 0.0, roi[0]], [0.0, 1.0, roi[1]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        if not np.allclose(matrix, expected, atol=1.0e-9, rtol=0.0):
            raise ValueError(
                "Rig ROI and local-to-parent transform disagree for {}"
                .format(space_id)
            )
        declared_parent_size = result.get("parent_size_px") or parent_size_px
        if declared_parent_size is not None:
            parent = list(map(int, declared_parent_size))
            if len(parent) != 2 or min(parent) <= 0:
                raise ValueError("Rig parent raster size must be positive W,H")
            if roi[0] + roi[2] > parent[0] or roi[1] + roi[3] > parent[1]:
                raise ValueError(
                    "Rig ROI {} lies outside parent raster {}".format(roi, parent)
                )
            result["parent_size_px"] = parent
        result["roi_in_parent_xywh"] = roi
        result["local_to_parent_3x3"] = matrix.tolist()
    elif roi is not None or local_to_parent is not None:
        raise ValueError(
            "Rig root raster {} cannot declare parent-local ROI fields"
            .format(space_id)
        )
    return result


def validated_rig_sample(
    sample: FrameSample,
    *,
    parent_size_px: Optional[Sequence[int]] = None,
    copy_image: bool = False,
) -> FrameSample:
    """Return a sample whose producer-owned space has passed rig validation."""

    metadata = copy.deepcopy(dict(sample.metadata))
    metadata["image_space"] = validate_rig_image_space(
        sample.image,
        metadata.get("image_space"),
        parent_size_px=parent_size_px,
    )
    return FrameSample(
        image=sample.image.copy() if copy_image else sample.image,
        time_ns=int(sample.time_ns),
        clock_id=str(sample.clock_id),
        receive_time_ns=(
            int(sample.receive_time_ns)
            if sample.receive_time_ns is not None
            else None
        ),
        source_id=str(sample.source_id),
        metadata=metadata,
    )


def rig_sample_media_record(
    file: str,
    sample: FrameSample,
    *,
    operation: str,
    metadata_reference: str,
    notes: Optional[str] = None,
) -> dict:
    """Serialize one acquired rig sample without assigning a new space."""

    space = validate_rig_image_space(
        sample.image, (sample.metadata or {}).get("image_space")
    )
    parent_id = str(space.get("parent_space_id") or "").strip()
    if parent_id:
        source_space = parent_id
        source_region = {
            "kind": "hardware_roi",
            "xywh": list(map(int, space["roi_in_parent_xywh"])),
            "local_origin_in_source_xy": list(
                map(int, space["roi_in_parent_xywh"][:2])
            ),
        }
        transform = {
            "local_to_parent_3x3": copy.deepcopy(space["local_to_parent_3x3"])
        }
    else:
        source_space = str(space["space_id"])
        source_region = {
            "kind": "full_frame",
            "xywh": [0, 0] + _size_wh(sample.image),
        }
        transform = None
    return raster_record(
        file,
        media_type="image",
        stored_size_px=_size_wh(sample.image),
        space_id=str(space["space_id"]),
        operation=str(operation),
        source_space_id=source_space,
        source_region=source_region,
        orientation={"value": str(space["orientation"])},
        transform=transform,
        metadata_reference=metadata_reference,
        notes=notes,
    )


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float64).reshape((-1, 1, 2)), matrix
    ).reshape((-1, 2))


def _checkerboard(height: int, width: int, cell: int = 32) -> np.ndarray:
    rows, columns = np.indices((height, width))
    selected = ((rows // cell) + (columns // cell)) % 2 == 0
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[selected] = SYNTHETIC_BACKGROUND_BGR
    canvas[~selected] = SYNTHETIC_BACKGROUND_DARK_BGR
    return canvas


def _polyline_points(points: np.ndarray) -> np.ndarray:
    return np.rint(points).astype(np.int32).reshape((-1, 1, 2))


@dataclass(frozen=True)
class ExpandedRigReview:
    image: np.ndarray
    image_space: Mapping[str, Any]
    geometry: Mapping[str, Any]


def expanded_rig_camera_review(
    sample: FrameSample,
    *,
    full_sensor_size_px: Sequence[int],
    phone_display_size_px: Optional[Sequence[int]],
    phone_display_to_full_sensor_3x3: Optional[Sequence[Sequence[float]]],
    phone_display_quadrilateral_full_sensor_xy: Optional[
        Sequence[Sequence[float]]
    ] = None,
    title: str,
    overlay: Optional[np.ndarray] = None,
) -> ExpandedRigReview:
    """Render uncropped rig evidence on an explicit synthetic review canvas.

    The canvas contains the complete full-sensor rectangle and, when geometry
    exists, the complete phone-display quadrilateral.  Captured pixels are
    placed using producer-supplied local-to-parent geometry.
    """

    full_width, full_height = map(int, full_sensor_size_px)
    if min(full_width, full_height) <= 0:
        raise ValueError("Full-sensor size must be positive")
    checked = validated_rig_sample(
        sample, parent_size_px=[full_width, full_height], copy_image=False
    )
    space = dict(checked.metadata["image_space"])
    local_to_full = np.asarray(
        space.get("local_to_parent_3x3") or np.eye(3), dtype=np.float64
    )
    source_image = overlay if overlay is not None else checked.image
    if _size_wh(source_image) != _size_wh(checked.image):
        raise ValueError("Rig review overlay must match its source frame size")

    full_corners = np.asarray(
        [[0, 0], [full_width - 1, 0], [full_width - 1, full_height - 1], [0, full_height - 1]],
        dtype=np.float64,
    )
    phone_corners = None
    phone_to_full = None
    if phone_display_quadrilateral_full_sensor_xy is not None:
        phone_corners = np.asarray(
            phone_display_quadrilateral_full_sensor_xy, dtype=np.float64
        )
        if phone_corners.shape != (4, 2) or not np.all(np.isfinite(phone_corners)):
            raise ValueError(
                "phone_display_quadrilateral_full_sensor_xy must be four finite XY points"
            )
        if phone_display_to_full_sensor_3x3 is not None:
            phone_to_full = _matrix_3x3(
                phone_display_to_full_sensor_3x3,
                "phone_display_to_full_sensor_3x3",
            )
    elif phone_display_to_full_sensor_3x3 is not None:
        if phone_display_size_px is None:
            raise ValueError("Phone size is required with phone projection")
        phone_width, phone_height = map(int, phone_display_size_px)
        if min(phone_width, phone_height) <= 0:
            raise ValueError("Phone-display size must be positive")
        phone_to_full = _matrix_3x3(
            phone_display_to_full_sensor_3x3,
            "phone_display_to_full_sensor_3x3",
        )
        phone_corners = _transform_points(
            np.asarray(
                [[0, 0], [phone_width - 1, 0], [phone_width - 1, phone_height - 1], [0, phone_height - 1]],
                dtype=np.float64,
            ),
            phone_to_full,
        )

    bounds_points = full_corners if phone_corners is None else np.vstack([full_corners, phone_corners])
    minimum = np.floor(np.min(bounds_points, axis=0))
    maximum = np.ceil(np.max(bounds_points, axis=0))
    margin = max(32.0, 0.03 * max(full_width, full_height))
    extent_width = float(maximum[0] - minimum[0] + 1 + 2 * margin)
    extent_height = float(maximum[1] - minimum[1] + 1 + 2 * margin)
    scale = min(
        1.0,
        MAXIMUM_REVIEW_DIMENSION_PX / max(extent_width, extent_height),
        math.sqrt(MAXIMUM_REVIEW_PIXELS / max(1.0, extent_width * extent_height)),
    )
    header_height = 64
    canvas_width = max(1, int(math.ceil(extent_width * scale)))
    canvas_height = max(1, int(math.ceil(extent_height * scale))) + header_height
    full_to_review = np.asarray(
        [
            [scale, 0.0, (-minimum[0] + margin) * scale],
            [0.0, scale, (-minimum[1] + margin) * scale + header_height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    local_to_review = full_to_review @ local_to_full
    canvas = _checkerboard(canvas_height, canvas_width)
    cv2.rectangle(canvas, (0, 0), (canvas_width - 1, header_height - 1), (30, 30, 30), -1)

    warped = cv2.warpPerspective(
        source_image,
        local_to_review,
        (canvas_width, canvas_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    source_mask = np.full(source_image.shape[:2], 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(
        source_mask,
        local_to_review,
        (canvas_width, canvas_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    canvas[warped_mask > 0] = warped[warped_mask > 0]

    full_review = _transform_points(full_corners, full_to_review)
    cv2.polylines(
        canvas, [_polyline_points(full_review)], True, FULL_SENSOR_OUTLINE_BGR, 3, cv2.LINE_AA
    )
    cv2.putText(
        canvas,
        "FULL CAMERA SENSOR",
        tuple(np.rint(full_review[0] + [8, 24]).astype(int)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        FULL_SENSOR_OUTLINE_BGR,
        2,
        cv2.LINE_AA,
    )
    phone_review = None
    if phone_corners is not None:
        phone_review = _transform_points(phone_corners, full_to_review)
        cv2.polylines(
            canvas,
            [_polyline_points(phone_review)],
            True,
            PHONE_DISPLAY_OUTLINE_BGR,
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "PROJECTED PHONE DISPLAY",
            tuple(np.rint(phone_review[0] + [8, -10]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            PHONE_DISPLAY_OUTLINE_BGR,
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        str(title),
        (14, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.66,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    subtitle = (
        "magenta checker = synthetic outside-capture area"
        if phone_corners is not None
        else "magenta checker = synthetic; phone projection unavailable"
    )
    cv2.putText(
        canvas,
        subtitle,
        (14, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )

    geometry = {
        "source_space_id": str(space["space_id"]),
        "source_size_px": _size_wh(checked.image),
        "full_sensor_size_px": [full_width, full_height],
        "source_local_to_full_sensor_3x3": local_to_full.tolist(),
        "full_sensor_to_review_3x3": full_to_review.tolist(),
        "source_local_to_review_3x3": local_to_review.tolist(),
        "full_sensor_quadrilateral_review_xy": full_review.tolist(),
        "phone_display_quadrilateral_full_sensor_xy": (
            phone_corners.tolist() if phone_corners is not None else None
        ),
        "phone_display_quadrilateral_review_xy": (
            phone_review.tolist() if phone_review is not None else None
        ),
        "phone_display_to_full_sensor_3x3": (
            phone_to_full.tolist() if phone_to_full is not None else None
        ),
        "phone_projection_model": (
            "explicit_raw_sensor_quadrilateral"
            if phone_display_quadrilateral_full_sensor_xy is not None
            else (
                "projective_homography"
                if phone_to_full is not None
                else "unavailable"
            )
        ),
        "synthetic_background_bgr": list(SYNTHETIC_BACKGROUND_BGR),
        "synthetic_background_pattern": "magenta_checkerboard",
        "review_scale": float(scale),
        "header_height_px": int(header_height),
    }
    review_space = {
        "schema_version": "1.0",
        "space_id": "rig_expanded_camera_review_pixels",
        "stored_size_px": [canvas_width, canvas_height],
        "orientation": "diagnostic_canvas_top_left_x_right_y_down",
        "color_order": "BGR",
        "review_geometry": copy.deepcopy(geometry),
    }
    return ExpandedRigReview(canvas, review_space, geometry)


def expanded_review_media_record(
    file: str,
    review: ExpandedRigReview,
    *,
    metadata_reference: str,
) -> dict:
    """Serialize one expanded human-review canvas and its exact placements."""

    return raster_record(
        file,
        media_type="image",
        stored_size_px=_size_wh(review.image),
        space_id=str(review.image_space["space_id"]),
        operation="expand_rig_camera_evidence_without_cropping",
        source_space_id=str(review.geometry["source_space_id"]),
        source_region={
            "kind": "complete_source_raster",
            "size_px": list(review.geometry["source_size_px"]),
        },
        orientation={"value": str(review.image_space["orientation"])},
        transform={
            "source_local_to_review_3x3": copy.deepcopy(
                review.geometry["source_local_to_review_3x3"]
            ),
            "full_sensor_to_review_3x3": copy.deepcopy(
                review.geometry["full_sensor_to_review_3x3"]
            ),
            "phone_display_to_full_sensor_3x3": copy.deepcopy(
                review.geometry["phone_display_to_full_sensor_3x3"]
            ),
        },
        metadata_reference=metadata_reference,
        notes=(
            "Magenta checkerboard pixels are synthetic review background. "
            "Yellow is the full sensor; green is the projected phone display."
        ),
    )
