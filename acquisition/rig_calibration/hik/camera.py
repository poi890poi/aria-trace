"""HikCamera-compatible facade backed by a saved rectified calibration.

Usage is intentionally shaped like the common ``hik_camera.hik_camera`` module::

    import acquisition.rig_calibration.hik.camera as hikcam

    with hikcam.HikCamera(config={"calibration": "...json"}) as cam:
        rgb = cam.get_frame()

Set ``ARIA_HIK_CALIBRATION`` to omit the config argument. Construction does not
claim hardware; the context manager (or ``open``) owns the camera lifecycle.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import cv2
import numpy as np

from .driver import HikMvsCameraAdapter, RectifiedHikCamera


CalibrationPath = Union[str, Path]


def _calibration_from_arguments(
    ip: Optional[str], config: Mapping[str, Any]
) -> Path:
    configured = config.get("calibration") or os.environ.get("ARIA_HIK_CALIBRATION")
    if configured is None and ip:
        candidate = Path(str(ip))
        if candidate.is_file() or candidate.suffix.lower() == ".json":
            configured = candidate
    if configured is None:
        raise ValueError(
            "Saved calibration is required in config['calibration'], as the first "
            "argument, or in ARIA_HIK_CALIBRATION"
        )
    path = Path(configured).resolve()
    if not path.is_file():
        raise FileNotFoundError("HIK calibration does not exist: {}".format(path))
    return path


class HikCamera:
    """Drop-in high-level HIK camera shape returning rectified display frames.

    The supported compatibility surface is deliberately high-level. It does not
    impersonate the vendor's ctypes structures or status-code based ``MV_CC_*``
    ABI. Methods raise Python exceptions on failure.
    """

    TIMEOUT_MS = 40000

    def __init__(
        self,
        ip: Optional[str] = None,
        host_ip: Optional[str] = None,
        setting_items: Optional[Mapping[str, Any]] = None,
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.config: Dict[str, Any] = dict(config or {})
        ip_candidate = Path(str(ip)) if ip else None
        ip_is_calibration_path = bool(
            ip_candidate
            and (ip_candidate.is_file() or ip_candidate.suffix.lower() == ".json")
        )
        self.calibration_path = _calibration_from_arguments(ip, self.config)
        self.calibration = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        camera = self.calibration["camera"]
        self._ip = str(camera.get("device_id", ip or ""))
        if ip and not ip_is_calibration_path and str(ip) != self._ip:
            raise ValueError(
                "Requested camera {!r} does not match calibrated camera {!r}".format(
                    ip, self._ip
                )
            )
        self.host_ip = host_ip
        self.setting_items = dict(setting_items or {})
        self.is_open = False
        self.last_time_get_frame = 0.0
        self._reader = None
        self._last_frame = None
        self._color_order = str(self.config.get("color_order", "RGB")).upper()
        if self._color_order not in ("RGB", "BGR"):
            raise ValueError("config['color_order'] must be RGB or BGR")
        imaging = self.calibration["imaging"]
        self._exposure_us = float(imaging["exposure_us"])
        self._gain = float(imaging["gain"])
        self._black_level = (
            int(imaging["black_level"]) if imaging.get("black_level") is not None else None
        )
        self._white_balance = dict(imaging["white_balance"])
        self._balance_selector = "Red"
        self._fps = float(camera["full_sensor_mode"]["fps"])
        output_width, output_height = map(
            int, self.calibration["normalization"]["output_size_px"]
        )
        self.shape = (output_height, output_width, 3)
        self.bit = 24
        self.pixel_format = "RGB8Packed" if self._color_order == "RGB" else "BGR8Packed"

    @property
    def orientation(self) -> Dict[str, Any]:
        """Return the ChArUco orientation evidence applied to every frame."""

        return dict(self.calibration.get("normalization", {}).get("orientation", {}))

    @property
    def ip(self) -> str:
        return self._ip

    @property
    def is_raw(self) -> bool:
        return False

    @classmethod
    def get_all_ips(cls, sdk_python_path: Optional[str] = None) -> list[str]:
        """Return HIK device identifiers without opening any camera."""

        calibration = os.environ.get("ARIA_HIK_CALIBRATION")
        if calibration and Path(calibration).is_file():
            value = json.loads(Path(calibration).read_text(encoding="utf-8"))
            return [str(value["camera"]["device_id"])]
        adapter = HikMvsCameraAdapter(sdk_python_path=sdk_python_path)
        return [device.device_id for device in adapter.devices(probe=True)]

    @classmethod
    def get_cam(cls) -> "HikCamera":
        return cls()

    @classmethod
    def get_cams(cls, ips: Optional[Sequence[str]] = None) -> "MultiHikCamera":
        selected = list(ips or cls.get_all_ips())
        return MultiHikCamera({camera_id: cls(camera_id) for camera_id in selected})

    get_all_cams = get_cams

    def _new_reader(self):
        factory = self.config.get("reader_factory", RectifiedHikCamera)
        return factory(self.calibration_path)

    def open(self) -> "HikCamera":
        if self.is_open:
            return self
        reader = self._new_reader()
        try:
            opened = reader.open()
            self._reader = opened if opened is not None else reader
            self.is_open = True
            self.setting()
        except Exception:
            try:
                reader.release()
            except Exception:
                pass
            self._reader = None
            self.is_open = False
            raise
        return self

    def close(self) -> None:
        reader, self._reader = self._reader, None
        self.is_open = False
        if reader is not None:
            reader.release()

    release = close

    def __enter__(self) -> "HikCamera":
        return self.open()

    def __exit__(self, *_exc) -> None:
        self.close()

    def _require_reader(self):
        if not self.is_open or self._reader is None:
            raise RuntimeError("HikCamera is not open; use `with cam:` or cam.open()")
        return self._reader

    def _convert_output(self, bgr: np.ndarray) -> np.ndarray:
        if self._color_order == "RGB":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return bgr

    def get_frame(self) -> np.ndarray:
        reader = self._require_reader()
        ok, bgr = reader.read()
        if not ok or bgr is None:
            raise RuntimeError("HIK camera returned no rectified frame")
        frame = self._convert_output(bgr)
        self._last_frame = frame
        self.last_time_get_frame = time.time()
        self.shape = tuple(frame.shape)
        self.bit = int(frame.dtype.itemsize * 8 * (frame.shape[2] if frame.ndim == 3 else 1))
        return frame

    def get_frame_with_config(self) -> None:
        self._last_frame = self.get_frame()

    def robust_get_frame(self) -> np.ndarray:
        try:
            return self.get_frame()
        except Exception:
            self.reset()
            return self.get_frame()

    def read(self):
        """OpenCV-compatible alias returning ``(ok, frame)``."""

        try:
            return True, self.get_frame()
        except Exception:
            return False, None

    def get_shape(self):
        return self.shape

    def reset(self) -> None:
        self.close()
        self.open()

    def waite(self, timeout: int = 20) -> None:
        """Compatibility spelling: verify the configured camera can be opened."""

        deadline = time.monotonic() + float(timeout)
        last_error = None
        while time.monotonic() < deadline:
            try:
                self.open()
                return
            except Exception as exc:
                last_error = exc
                time.sleep(0.1)
        raise TimeoutError("Cannot open calibrated HIK camera: {}".format(last_error))

    def set_rgb(self) -> None:
        self._color_order = "RGB"
        self.pixel_format = "RGB8Packed"

    def set_bgr(self) -> None:
        self._color_order = "BGR"
        self.pixel_format = "BGR8Packed"

    def setting(self) -> None:
        """Apply the configured high-level settings after opening."""

        for key, value in self.setting_items.items():
            self.setitem(key, value)

    def set_raw(self, *args, **kwargs) -> None:
        raise NotImplementedError("Rectified calibrated output cannot be exposed as Bayer/raw")

    def get_exposure(self) -> float:
        return float(self._exposure_us)

    def set_exposure(self, exposure_us: float) -> None:
        if abs(float(exposure_us) - self._exposure_us) > max(2.0, self._exposure_us * 0.001):
            raise RuntimeError("Exposure is locked by the saved rig calibration")

    def get_exposure_by_second(self) -> float:
        return self.get_exposure() * 1.0e-6

    def set_exposure_by_second(self, seconds: float) -> None:
        self.set_exposure(float(seconds) * 1.0e6)

    def get_gain(self) -> float:
        return float(self._gain)

    def set_gain(self, gain: float) -> None:
        if abs(float(gain) - self._gain) > 1.0e-6:
            raise RuntimeError("Gain is locked by the saved rig calibration")

    def set_white_balance(self, red: int, green: int, blue: int) -> None:
        requested = {
            "ratio_red": int(red),
            "ratio_green": int(green),
            "ratio_blue": int(blue),
        }
        if requested != {
            key: int(self._white_balance[key]) for key in requested
        }:
            raise RuntimeError("White balance is locked by the saved rig calibration")

    def getitem(self, key: str) -> Any:
        normalized = str(key)
        width, height = self.calibration["normalization"]["output_size_px"]
        values = {
            "ExposureTime": self.get_exposure(),
            "Gain": self.get_gain(),
            "Width": int(width),
            "Height": int(height),
            "OffsetX": 0,
            "OffsetY": 0,
            "PixelFormat": self.pixel_format,
            "AcquisitionFrameRate": self._fps,
            "AcquisitionFrameRateEnable": True,
            "ExposureAuto": "Off",
            "GainAuto": "Off",
            "BalanceWhiteAuto": "Off",
            "BalanceRatioSelector": self._balance_selector,
            "BalanceRatio": int(
                self._white_balance[
                    "ratio_{}".format(self._balance_selector.lower())
                ]
            ),
            "TriggerMode": "Off",
        }
        if self._black_level is not None:
            values["BlackLevelEnable"] = True
            values["BlackLevel"] = self._black_level
        if normalized not in values:
            raise KeyError("Unsupported calibrated-camera setting {}".format(normalized))
        return values[normalized]

    def setitem(self, key: str, value: Any) -> None:
        normalized = str(key)
        if normalized == "ExposureTime":
            self.set_exposure(float(value))
            return
        if normalized == "Gain":
            self.set_gain(float(value))
            return
        if normalized == "BlackLevel":
            if self._black_level is None:
                raise KeyError("Saved calibration has no black-level setting")
            if int(value) != self._black_level:
                raise RuntimeError("Black level is locked by the saved rig calibration")
            return
        if normalized == "AcquisitionFrameRate":
            reader = self._require_reader()
            if not reader.adapter.set_control("frame_rate", float(value)):
                raise RuntimeError("HIK camera does not expose frame-rate control")
            self._fps = float(value)
            return
        if normalized == "BalanceRatioSelector":
            selector = str(value).title()
            if selector not in ("Red", "Green", "Blue"):
                raise ValueError("BalanceRatioSelector must be Red, Green, or Blue")
            self._balance_selector = selector
            return
        if normalized == "BalanceRatio":
            ratios = {
                color: int(self._white_balance["ratio_{}".format(color)])
                for color in ("red", "green", "blue")
            }
            ratios[self._balance_selector.lower()] = int(value)
            self.set_white_balance(ratios["red"], ratios["green"], ratios["blue"])
            return
        if normalized == "PixelFormat":
            if str(value) == "RGB8Packed":
                self.set_rgb()
                return
            if str(value) in ("BGR8Packed", "BGR8"):
                self.set_bgr()
                return
            raise ValueError("Rectified HikCamera supports RGB8Packed or BGR8Packed")
        if normalized in ("Width", "Height", "OffsetX", "OffsetY"):
            raise RuntimeError(
                "Calibrated ROI/shape is immutable; changing it invalidates rectification"
            )
        fixed_modes = {
            "AcquisitionFrameRateEnable": True,
            "ExposureAuto": "Off",
            "GainAuto": "Off",
            "BalanceWhiteAuto": "Off",
            "TriggerMode": "Off",
        }
        if self._black_level is not None:
            fixed_modes["BlackLevelEnable"] = True
        if normalized in fixed_modes:
            if value != fixed_modes[normalized]:
                raise RuntimeError(
                    "Calibrated HikCamera requires {}={!r}".format(
                        normalized, fixed_modes[normalized]
                    )
                )
            return
        raise KeyError("Unsupported calibrated-camera setting {}".format(normalized))

    __getitem__ = getitem
    __setitem__ = setitem


class MultiHikCamera(dict):
    """Small synchronous compatibility container for multiple HikCamera objects."""

    def __enter__(self):
        opened = []
        try:
            for camera in self.values():
                camera.open()
                opened.append(camera)
        except Exception:
            for camera in reversed(opened):
                camera.close()
            raise
        return self

    def __exit__(self, *_exc) -> None:
        for camera in self.values():
            camera.close()

    def get_frame(self):
        return {key: camera.get_frame() for key, camera in self.items()}

    def robust_get_frame(self):
        return {key: camera.robust_get_frame() for key, camera in self.items()}


Camera = HikCamera
get_all_ips = HikCamera.get_all_ips
get_cam = HikCamera.get_cam
get_cams = HikCamera.get_cams
get_all_cams = HikCamera.get_all_cams

__all__ = [
    "Camera",
    "HikCamera",
    "MultiHikCamera",
    "get_all_cams",
    "get_all_ips",
    "get_cam",
    "get_cams",
]
