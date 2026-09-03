"""Public, replaceable camera and ADB boundaries for the desktop app."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import cv2
import numpy as np

from rig_runtime.services.calibration.rig.contracts import FrameSample


@dataclass(frozen=True)
class CameraDevice:
    """A selectable camera without requiring it to be opened."""

    device_id: str
    label: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CameraConfiguration:
    """Requested camera mode; adapters report the effective mode per frame."""

    device_id: str = "0"
    width_px: int = 1920
    height_px: int = 1080
    fps: float = 30.0
    backend: str = "auto"

    def __post_init__(self) -> None:
        if self.width_px <= 0 or self.height_px <= 0 or self.fps <= 0:
            raise ValueError("Camera width, height, and FPS must be positive")


class CameraAdapter(ABC):
    """Customization point for UVC, capture-card, or remote frame sources.

    Construction and ``devices`` must not claim a camera. ``open`` is called
    only after an explicit operator action. Timestamps should use a declared
    monotonic clock and be taken as close to frame availability as possible.
    """

    adapter_id = "custom"

    def devices(self, probe: bool = False) -> Sequence[CameraDevice]:
        """Return known devices; probing must happen only when requested."""

        return ()

    @abstractmethod
    def open(self, configuration: CameraConfiguration) -> Mapping[str, Any]:
        """Claim one device and return its effective configuration."""

    @abstractmethod
    def read(self) -> FrameSample:
        """Return the next timestamped BGR frame.

        Controlled target capture requires ``receive_time_ns`` in the host
        monotonic clock so device-clock timestamps can still be ordered against
        phone paint acknowledgements.
        """

    @abstractmethod
    def close(self) -> None:
        """Release the device; repeated calls must be safe."""

    def controls(self) -> Mapping[str, Any]:
        return {}

    def set_control(self, name: str, value: Any) -> bool:
        return False


class OpenCvCameraAdapter(CameraAdapter):
    """Built-in OpenCV adapter with explicit, bounded DirectShow probing."""

    adapter_id = "opencv"

    def __init__(self, maximum_probe_index: int = 8) -> None:
        self.maximum_probe_index = int(maximum_probe_index)
        self._capture: Optional[cv2.VideoCapture] = None
        self._configuration: Optional[CameraConfiguration] = None

    @staticmethod
    def _backend(name: str) -> int:
        normalized = str(name).strip().lower()
        if normalized == "dshow":
            return int(getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY))
        if normalized in ("msmf", "media_foundation"):
            return int(getattr(cv2, "CAP_MSMF", cv2.CAP_ANY))
        return int(cv2.CAP_ANY)

    @staticmethod
    def _device_value(device_id: str) -> Any:
        text = str(device_id).strip()
        return int(text) if text.isdigit() else text

    def devices(self, probe: bool = False) -> Sequence[CameraDevice]:
        if not probe:
            return ()
        found = []
        backend = self._backend("dshow")
        for index in range(max(0, self.maximum_probe_index)):
            capture = cv2.VideoCapture(index, backend)
            try:
                if capture.isOpened():
                    found.append(CameraDevice(str(index), "Camera {}".format(index)))
            finally:
                capture.release()
        return tuple(found)

    def open(self, configuration: CameraConfiguration) -> Mapping[str, Any]:
        self.close()
        backend = self._backend(configuration.backend)
        capture = cv2.VideoCapture(
            self._device_value(configuration.device_id), backend
        )
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                "Cannot open camera {} with backend {}".format(
                    configuration.device_id, configuration.backend
                )
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(configuration.width_px))
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(configuration.height_px))
        capture.set(cv2.CAP_PROP_FPS, float(configuration.fps))
        self._capture = capture
        self._configuration = configuration
        return {
            "adapter_id": self.adapter_id,
            "device_id": configuration.device_id,
            "backend": configuration.backend,
            "width_px": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
            "height_px": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            "fps": float(capture.get(cv2.CAP_PROP_FPS) or configuration.fps),
        }

    def read(self) -> FrameSample:
        capture = self._capture
        if capture is None or not capture.isOpened():
            raise RuntimeError("Camera is not open")
        ok, image = capture.read()
        receive_time_ns = time.monotonic_ns()
        if not ok or image is None or image.size == 0:
            raise RuntimeError("Camera returned no frame")
        configuration = self._configuration
        return FrameSample(
            image=image,
            time_ns=receive_time_ns,
            receive_time_ns=receive_time_ns,
            source_id="camera:{}".format(
                configuration.device_id if configuration else "unknown"
            ),
            metadata={
                "adapter_id": self.adapter_id,
                "effective_size_px": [int(image.shape[1]), int(image.shape[0])],
            },
        )

    def close(self) -> None:
        capture, self._capture = self._capture, None
        self._configuration = None
        if capture is not None:
            capture.release()

    def controls(self) -> Mapping[str, Any]:
        capture = self._capture
        if capture is None:
            return {}
        names = {
            "focus": cv2.CAP_PROP_FOCUS,
            "autofocus": cv2.CAP_PROP_AUTOFOCUS,
            "exposure": cv2.CAP_PROP_EXPOSURE,
            "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
            "gain": cv2.CAP_PROP_GAIN,
        }
        return {name: float(capture.get(code)) for name, code in names.items()}

    def set_control(self, name: str, value: Any) -> bool:
        capture = self._capture
        if capture is None:
            return False
        names = {
            "focus": cv2.CAP_PROP_FOCUS,
            "autofocus": cv2.CAP_PROP_AUTOFOCUS,
            "exposure": cv2.CAP_PROP_EXPOSURE,
            "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
            "gain": cv2.CAP_PROP_GAIN,
        }
        code = names.get(str(name))
        return bool(code is not None and capture.set(code, float(value)))


class AdbAdapter(ABC):
    """Optional phone-observation boundary.

    The GUI never calls an ADB method during startup. Alternative transports
    can implement this contract without importing app or camera internals.
    """

    adapter_id = "custom"

    @abstractmethod
    def available(self) -> bool:
        pass

    @abstractmethod
    def devices(self) -> Sequence[str]:
        pass

    @abstractmethod
    def capture_screen(self, serial: Optional[str] = None) -> FrameSample:
        pass

    def close(self) -> None:
        return None


class NullAdbAdapter(AdbAdapter):
    adapter_id = "disabled"

    def available(self) -> bool:
        return False

    def devices(self) -> Sequence[str]:
        return ()

    def capture_screen(self, serial: Optional[str] = None) -> FrameSample:
        raise RuntimeError("ADB integration is disabled")


class SubprocessAdbAdapter(AdbAdapter):
    """ADB reference-capture adapter; commands run only on button actions."""

    adapter_id = "adb_exec_out"

    def __init__(self, executable: str = "adb", timeout_seconds: float = 10.0) -> None:
        self.executable = str(executable)
        self.timeout_seconds = float(timeout_seconds)

    def _base(self, serial: Optional[str] = None) -> list[str]:
        command = [self.executable]
        if serial:
            command.extend(["-s", str(serial)])
        return command

    def available(self) -> bool:
        candidate = Path(self.executable)
        return candidate.is_file() or shutil.which(self.executable) is not None

    def devices(self) -> Sequence[str]:
        if not self.available():
            return ()
        result = subprocess.run(
            self._base() + ["devices"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=self.timeout_seconds,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        rows = result.stdout.decode("utf-8", errors="replace").splitlines()[1:]
        return tuple(
            row.split("\t", 1)[0]
            for row in rows
            if "\tdevice" in row
        )

    def capture_screen(self, serial: Optional[str] = None) -> FrameSample:
        if not self.available():
            raise RuntimeError("ADB executable is unavailable")
        result = subprocess.run(
            self._base(serial) + ["exec-out", "screencap", "-p"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            timeout=self.timeout_seconds,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        receive_time_ns = time.monotonic_ns()
        data = np.frombuffer(result.stdout, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise RuntimeError("ADB screenshot was empty or undecodable")
        return FrameSample(
            image=image,
            time_ns=receive_time_ns,
            receive_time_ns=receive_time_ns,
            source_id="adb:{}".format(serial or "default"),
            metadata={"adapter_id": self.adapter_id, "serial": serial},
        )


AdapterFactory = Callable[[], Any]


def load_adapter_factory(specification: str) -> AdapterFactory:
    """Load a zero-argument ``module:function`` adapter factory."""

    module_name, separator, attribute_name = str(specification).partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("Adapter factory must use module:function syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name)
    if not callable(factory):
        raise TypeError("Adapter factory {} is not callable".format(specification))
    return factory


def create_camera_adapter(specification: Optional[str] = None) -> CameraAdapter:
    adapter = (
        load_adapter_factory(specification)()
        if specification
        else OpenCvCameraAdapter()
    )
    if not isinstance(adapter, CameraAdapter):
        raise TypeError("Camera factory must return CameraAdapter")
    return adapter


def create_adb_adapter(
    specification: Optional[str] = None, executable: str = "adb"
) -> AdbAdapter:
    adapter = (
        load_adapter_factory(specification)()
        if specification
        else SubprocessAdbAdapter(executable)
    )
    if not isinstance(adapter, AdbAdapter):
        raise TypeError("ADB factory must return AdbAdapter")
    return adapter
