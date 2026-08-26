"""Standalone rig-calibration application integration surface.

Importing this package has no hardware, subprocess, socket, or GUI side
effects. Applications select adapters explicitly and start them explicitly.
"""

from .device_adapters import (
    AdbAdapter,
    CameraAdapter,
    CameraConfiguration,
    CameraDevice,
    NullAdbAdapter,
    OpenCvCameraAdapter,
    SubprocessAdbAdapter,
    create_adb_adapter,
    create_camera_adapter,
    load_adapter_factory,
)
from .phone_target import LocalPhoneTargetServer, PhoneTargetAdapter, Presentation

__all__ = [
    "AdbAdapter",
    "CameraAdapter",
    "CameraConfiguration",
    "CameraDevice",
    "LocalPhoneTargetServer",
    "NullAdbAdapter",
    "OpenCvCameraAdapter",
    "PhoneTargetAdapter",
    "Presentation",
    "SubprocessAdbAdapter",
    "create_adb_adapter",
    "create_camera_adapter",
    "load_adapter_factory",
]
