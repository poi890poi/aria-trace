"""Encode review JPEGs consistently while preserving lossless machine images."""

from pathlib import Path

import cv2


def write_evidence_image(path, image):
    """Use quality 90 and full chroma resolution for JPEG plots/annotations."""
    options = []
    if Path(path).suffix.lower() in (".jpg", ".jpeg"):
        options = [cv2.IMWRITE_JPEG_QUALITY, 90]
        if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR"):
            options += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    return cv2.imwrite(str(path), image, options)
