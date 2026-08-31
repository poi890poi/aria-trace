"""Android-camera-backed virtual device implementing the HIK adapter contract."""

from .driver import (
    HikMvsCameraAdapter,
    VirtualHikCameraAdapter,
    create_camera_adapter,
)

__all__ = [
    "HikMvsCameraAdapter",
    "VirtualHikCameraAdapter",
    "create_camera_adapter",
]
