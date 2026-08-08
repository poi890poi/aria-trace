"""Small replaceable visual descriptor used by the first replay POC."""

from typing import Dict, Optional

import cv2
import numpy as np


DEFAULT_DESCRIPTOR_CONFIG = {
    "type": "gray_gradient_thumbnail_v1",
    "width": 32,
    "height": 18,
}


def describe(config: Optional[dict] = None) -> Dict[str, object]:
    value = dict(DEFAULT_DESCRIPTOR_CONFIG)
    if config:
        value.update(config)
    if value.get("type") != "gray_gradient_thumbnail_v1":
        raise ValueError("Unsupported descriptor: {}".format(value.get("type")))
    value["width"] = int(value["width"])
    value["height"] = int(value["height"])
    if value["width"] <= 0 or value["height"] <= 0:
        raise ValueError("Descriptor dimensions must be positive")
    return value


def extract(image: np.ndarray, config: Optional[dict] = None) -> np.ndarray:
    """Return a brightness-normalized appearance and edge descriptor."""
    value = describe(config)
    if image is None or image.size == 0:
        raise ValueError("Cannot describe an empty image")
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.resize(
        gray,
        (value["width"], value["height"]),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    gray = (gray - float(gray.mean())) / (float(gray.std()) + 1.0e-6)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    descriptor = np.concatenate(
        (gray.reshape(-1), gradient_x.reshape(-1), gradient_y.reshape(-1))
    ).astype(np.float32)
    norm = float(np.linalg.norm(descriptor))
    if norm > 1.0e-8:
        descriptor /= norm
    return descriptor


def extract_many(images, config: Optional[dict] = None) -> np.ndarray:
    descriptors = [extract(image, config) for image in images]
    if not descriptors:
        value = describe(config)
        dimension = int(value["width"] * value["height"] * 3)
        return np.empty((0, dimension), dtype=np.float32)
    return np.stack(descriptors).astype(np.float32)

