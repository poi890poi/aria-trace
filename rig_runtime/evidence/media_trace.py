"""Shared provenance records for persisted raster images and videos."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np


MEDIA_SUFFIXES = {
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".bmp": "image",
    ".tif": "image",
    ".tiff": "image",
    ".mkv": "video",
    ".mp4": "video",
    ".avi": "video",
    ".mov": "video",
}


def raster_record(
    file: str,
    *,
    media_type: str,
    stored_size_px: Sequence[int],
    space_id: str,
    operation: str,
    source_space_id: str,
    source_region: Mapping[str, Any],
    content_size_px: Optional[Sequence[int]] = None,
    orientation: Optional[Mapping[str, Any]] = None,
    transform: Optional[Mapping[str, Any]] = None,
    metadata_reference: Optional[str] = None,
    timing: Optional[Mapping[str, Any]] = None,
    notes: Optional[str] = None,
) -> dict:
    """Build one explicit, JSON/YAML-safe media provenance record."""

    stored = list(map(int, stored_size_px))
    content = list(map(int, content_size_px or stored))
    if media_type not in ("image", "video"):
        raise ValueError("media_type must be image or video")
    if len(stored) != 2 or len(content) != 2 or min(stored + content) <= 0:
        raise ValueError("Media sizes must be positive [width, height]")
    if not str(space_id).strip() or not str(source_space_id).strip():
        raise ValueError("Media output and source spaces are required")
    result = {
        "file": str(file).replace("\\", "/"),
        "media_type": media_type,
        "stored_size_px": stored,
        "content_size_px": content,
        "space": {
            "id": str(space_id),
            "coordinates": "pixel_center_xy",
            "origin": "top_left_pixel_center_[0,0]",
            "axes": "+X_right_+Y_down",
        },
        "provenance": {
            "operation": str(operation),
            "source_space_id": str(source_space_id),
            "source_region": dict(source_region),
        },
    }
    if orientation is not None:
        result["space"]["orientation"] = dict(orientation)
    if transform is not None:
        result["provenance"]["transform"] = dict(transform)
    if metadata_reference:
        result["metadata_reference"] = str(metadata_reference).replace("\\", "/")
    if timing is not None:
        result["timing"] = dict(timing)
    if notes:
        result["notes"] = str(notes)
    return result


def image_size_px(path: Path) -> list[int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("Cannot read persisted image {}".format(path))
    return [int(image.shape[1]), int(image.shape[0])]


def array_size_px(image: np.ndarray) -> list[int]:
    if image is None or image.size == 0:
        raise ValueError("Image is empty")
    return [int(image.shape[1]), int(image.shape[0])]


def validate_media_registry(root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Check registry coverage and stored raster size only.

    Domain producers must validate semantic coordinate metadata before building
    these records. In particular, rig code uses ``rig_spatial`` and must not
    treat this generic filesystem check as proof that a space assignment is
    geometrically correct.
    """

    root = Path(root)
    registered = [str(row["file"]).replace("\\", "/") for row in records]
    if len(set(registered)) != len(registered):
        raise RuntimeError("Media registry contains duplicate file entries")
    discovered = sorted(
        str(path.relative_to(root)).replace("\\", "/")
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES
    )
    missing = sorted(set(discovered) - set(registered))
    extra = sorted(set(registered) - set(discovered))
    if missing or extra:
        raise RuntimeError(
            "Media registry coverage mismatch; missing={}, nonexistent={}"
            .format(missing, extra)
        )
    for row in records:
        relative = str(row["file"]).replace("\\", "/")
        path = root / Path(relative)
        expected_type = MEDIA_SUFFIXES.get(path.suffix.lower())
        if expected_type != row.get("media_type"):
            raise RuntimeError("Media type mismatch for {}".format(relative))
        if expected_type == "image":
            actual_size = image_size_px(path)
            if list(map(int, row["stored_size_px"])) != actual_size:
                raise RuntimeError(
                    "Stored image size for {} is {}, registry says {}".format(
                        relative, actual_size, row["stored_size_px"]
                    )
                )
