"""Hikrobot MVS camera plugin and rectified OpenCV-like stream interface."""

from __future__ import annotations

import ctypes
import importlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import cv2
import numpy as np

from aria_trace.adapters.rig.devices import CameraAdapter, CameraConfiguration, CameraDevice
from ..contracts import FrameSample
from .algorithms import camera_adapter_roi_to_output_homography
from .spaces import RigCalibratedSpaceConverter


def _decode_c_string(value: Any) -> str:
    try:
        raw = bytes(value)
    except Exception:
        return str(value)
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace").strip()


class MvsPythonBackend:
    """Thin dynamic wrapper around Hikrobot's `MvCameraControl_class.py`.

    The vendor wrapper is intentionally imported only when this backend is
    instantiated. SDK version differences are surfaced as explicit errors.
    """

    ENUM_VALUES = {
        "ExposureAuto": {"Off": 0, "Once": 1, "Continuous": 2},
        "GainAuto": {"Off": 0, "Once": 1, "Continuous": 2},
        "BalanceWhiteAuto": {"Off": 0, "Continuous": 1, "Once": 2},
        "AutoFunctionAOISelector": {"AOI1": 0, "AOI2": 1},
        "BalanceRatioSelector": {"Red": 0, "Green": 1, "Blue": 2},
        "TriggerMode": {"Off": 0, "On": 1},
    }
    INTERFACE_NAMES = {
        0: "value",
        1: "base",
        2: "integer",
        3: "boolean",
        4: "command",
        5: "float",
        6: "string",
        7: "register",
        8: "category",
        9: "enumeration",
        10: "enum_entry",
        11: "port",
    }
    ACCESS_NAMES = {
        0: "not_implemented",
        1: "not_available",
        2: "write_only",
        3: "read_only",
        4: "read_write",
        5: "undefined",
        6: "cycle_detect",
    }

    @staticmethod
    def _python_wrapper_directories(sdk_python_path: Optional[str]) -> list[Path]:
        candidates = []
        for value in (sdk_python_path, os.environ.get("HIK_MVS_PYTHON_PATH")):
            if value:
                candidates.append(Path(value))
        drives = {Path.cwd().drive or "C:", "C:"}
        for drive in sorted(drives):
            for program_files in ("Program Files (x86)", "Program Files"):
                candidates.append(
                    Path(drive + "\\" + program_files)
                    / "MVS"
                    / "Development"
                    / "Samples"
                    / "Python"
                    / "MvImport"
                )
        result = []
        seen = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            normalized = str(resolved).lower()
            if (
                normalized not in seen
                and (resolved / "MvCameraControl_class.py").is_file()
            ):
                seen.add(normalized)
                result.append(resolved)
        return result

    @staticmethod
    def _runtime_directories(sdk_python_path: Optional[str]) -> list[Path]:
        """Resolve likely MVS runtime directories without trusting process PATH."""

        candidates = []
        for value in (
            os.environ.get("MVS_RUNTIME_PATH"),
            os.environ.get("MVCAM_COMMON_RUNENV"),
        ):
            if value:
                root = Path(value)
                candidates.extend([root, root / "Win64_x64", root / "Win32_i86"])
        drives = {Path.cwd().drive or "C:"}
        if sdk_python_path:
            sdk_path = Path(sdk_python_path).resolve()
            drives.add(sdk_path.drive or "C:")
            parents = list(sdk_path.parents)
            if len(parents) >= 4:
                mvs_root = parents[3]
                candidates.extend(
                    [
                        mvs_root / "Runtime" / "Win64_x64",
                        mvs_root / "Runtime" / "Win32_i86",
                        mvs_root / "Development" / "Libraries" / "win64",
                        mvs_root / "Development" / "Libraries" / "win32",
                    ]
                )
        drives.add("C:")
        architecture = "Win64_x64" if platform.architecture()[0] == "64bit" else "Win32_i86"
        for drive in sorted(drives):
            candidates.extend(
                [
                    Path(drive + "\\Program Files (x86)\\Common Files\\MVS\\Runtime") / architecture,
                    Path(drive + "\\Program Files\\Common Files\\MVS\\Runtime") / architecture,
                ]
            )
        result = []
        seen = set()
        for candidate in candidates:
            normalized = str(candidate).lower()
            if normalized not in seen and candidate.is_dir():
                seen.add(normalized)
                result.append(candidate)
        return result

    @classmethod
    def _prepare_runtime(cls, sdk_python_path: Optional[str]):
        directories = cls._runtime_directories(sdk_python_path)
        if not directories:
            return [], []
        current = [item for item in os.environ.get("PATH", "").split(os.pathsep) if item]
        known = {item.lower() for item in current}
        prepend = [str(item) for item in directories if str(item).lower() not in known]
        if prepend:
            os.environ["PATH"] = os.pathsep.join(prepend + current)
        handles = []
        add_directory = getattr(os, "add_dll_directory", None)
        if add_directory is not None:
            for directory in directories:
                handles.append(add_directory(str(directory)))
        return directories, handles

    def __init__(self, sdk_python_path: Optional[str] = None) -> None:
        wrappers = self._python_wrapper_directories(sdk_python_path)
        candidate = str(wrappers[0]) if wrappers else None
        if candidate:
            resolved = str(Path(candidate).resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
        self.runtime_directories, self._dll_directory_handles = self._prepare_runtime(
            candidate
        )
        try:
            self.sdk = importlib.import_module("MvCameraControl_class")
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "HIK MVS runtime is unavailable. Install MVS and pass --mvs-python-path "
                "pointing to Samples/Python/MvImport. Runtime directories checked: {}"
                .format([str(item) for item in self.runtime_directories])
            ) from exc
        self.sdk_python_path = candidate
        self.camera = None
        self._device_info = None
        self._device_list = None
        self._grabbing = False

    def _check(self, result: int, operation: str) -> None:
        if int(result) != 0:
            raise RuntimeError("{} failed with MVS status 0x{:08x}".format(operation, int(result) & 0xFFFFFFFF))

    def enumerate_devices(self) -> list[dict[str, Any]]:
        sdk = self.sdk
        device_list = sdk.MV_CC_DEVICE_INFO_LIST()
        transport = int(getattr(sdk, "MV_GIGE_DEVICE", 1)) | int(
            getattr(sdk, "MV_USB_DEVICE", 4)
        )
        self._check(sdk.MvCamera.MV_CC_EnumDevices(transport, device_list), "enumerate devices")
        self._device_list = device_list
        devices = []
        for index in range(int(device_list.nDeviceNum)):
            pointer = device_list.pDeviceInfo[index]
            info = ctypes.cast(pointer, ctypes.POINTER(sdk.MV_CC_DEVICE_INFO)).contents
            transport_type = int(info.nTLayerType)
            serial = model = user_name = ""
            if transport_type == int(getattr(sdk, "MV_GIGE_DEVICE", 1)):
                special = info.SpecialInfo.stGigEInfo
                model = _decode_c_string(special.chModelName)
                user_name = _decode_c_string(special.chUserDefinedName)
                serial = _decode_c_string(getattr(special, "chSerialNumber", b""))
                transport_name = "GigE"
            elif transport_type == int(getattr(sdk, "MV_USB_DEVICE", 4)):
                special = info.SpecialInfo.stUsb3VInfo
                model = _decode_c_string(special.chModelName)
                user_name = _decode_c_string(special.chUserDefinedName)
                serial = _decode_c_string(special.chSerialNumber)
                transport_name = "USB3"
            else:
                transport_name = "transport-{}".format(transport_type)
            device_id = serial or str(index)
            devices.append(
                {
                    "device_id": device_id,
                    "index": index,
                    "serial": serial,
                    "model": model,
                    "user_name": user_name,
                    "transport": transport_name,
                    "transport_type": transport_type,
                    "pointer": pointer,
                }
            )
        return devices

    def open(self, device_id: str) -> dict[str, Any]:
        self.close()
        devices = self.enumerate_devices()
        selected = next(
            (
                item
                for item in devices
                if str(item["device_id"]) == str(device_id)
                or str(item["index"]) == str(device_id)
            ),
            None,
        )
        if selected is None:
            raise RuntimeError("HIK camera {!r} was not found".format(device_id))
        camera = self.sdk.MvCamera()
        # The installed MVS Python wrapper takes a structure and performs
        # byref(stDevInfo) itself. Passing pDeviceInfo directly creates a
        # pointer-to-pointer and the SDK rejects it with MV_E_PARAMETER.
        device_info = ctypes.cast(
            selected["pointer"], ctypes.POINTER(self.sdk.MV_CC_DEVICE_INFO)
        ).contents
        self._check(camera.MV_CC_CreateHandle(device_info), "create camera handle")
        try:
            access = int(getattr(self.sdk, "MV_ACCESS_Exclusive", 1))
            self._check(camera.MV_CC_OpenDevice(access, 0), "open camera")
            if selected["transport"] == "GigE" and hasattr(camera, "MV_CC_GetOptimalPacketSize"):
                packet_size = int(camera.MV_CC_GetOptimalPacketSize())
                if packet_size > 0:
                    camera.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)
        except Exception:
            camera.MV_CC_DestroyHandle()
            raise
        self.camera = camera
        self._device_info = {key: value for key, value in selected.items() if key != "pointer"}
        return dict(self._device_info)

    def close(self) -> None:
        camera, self.camera = self.camera, None
        self._grabbing = False
        if camera is not None:
            try:
                camera.MV_CC_StopGrabbing()
            except Exception:
                pass
            try:
                camera.MV_CC_CloseDevice()
            finally:
                camera.MV_CC_DestroyHandle()

    def start(self) -> None:
        if self.camera is None:
            raise RuntimeError("HIK camera is not open")
        self._check(self.camera.MV_CC_StartGrabbing(), "start grabbing")
        self._grabbing = True

    def stop(self) -> None:
        if self.camera is not None and self._grabbing:
            self._check(self.camera.MV_CC_StopGrabbing(), "stop grabbing")
        self._grabbing = False

    def _float_value(self, node: str):
        value = self.sdk.MVCC_FLOATVALUE()
        ctypes.memset(ctypes.byref(value), 0, ctypes.sizeof(value))
        self._check(self.camera.MV_CC_GetFloatValue(str(node), value), "read {}".format(node))
        return value

    def get_float(self, node: str) -> float:
        return float(self._float_value(node).fCurValue)

    def float_range(self, node: str) -> dict[str, float]:
        value = self._float_value(node)
        return {"minimum": float(value.fMin), "maximum": float(value.fMax), "current": float(value.fCurValue)}

    def set_float(self, node: str, value: float) -> None:
        self._check(self.camera.MV_CC_SetFloatValue(str(node), float(value)), "set {}".format(node))

    def _int_type(self):
        value_type = getattr(self.sdk, "MVCC_INTVALUE_EX", None)
        return value_type or getattr(self.sdk, "MVCC_INTVALUE")

    def _int_value(self, node: str):
        value = self._int_type()()
        ctypes.memset(ctypes.byref(value), 0, ctypes.sizeof(value))
        getter = getattr(self.camera, "MV_CC_GetIntValueEx", None)
        getter = getter or getattr(self.camera, "MV_CC_GetIntValue")
        self._check(getter(str(node), value), "read {}".format(node))
        return value

    def get_int(self, node: str) -> int:
        return int(self._int_value(node).nCurValue)

    def int_range(self, node: str) -> dict[str, int]:
        value = self._int_value(node)
        increment = int(getattr(value, "nInc", 1) or 1)
        return {"minimum": int(value.nMin), "maximum": int(value.nMax), "increment": increment, "current": int(value.nCurValue)}

    def set_int(self, node: str, value: int) -> None:
        setter = getattr(self.camera, "MV_CC_SetIntValueEx", None)
        setter = setter or getattr(self.camera, "MV_CC_SetIntValue")
        self._check(setter(str(node), int(value)), "set {}".format(node))

    def set_bool(self, node: str, value: bool) -> None:
        self._check(self.camera.MV_CC_SetBoolValue(str(node), bool(value)), "set {}".format(node))

    def get_bool(self, node: str) -> bool:
        value = ctypes.c_bool()
        self._check(self.camera.MV_CC_GetBoolValue(str(node), value), "read {}".format(node))
        return bool(value.value)

    def get_enum(self, node: str) -> int:
        value_type = getattr(self.sdk, "MVCC_ENUMVALUE_EX", None)
        value_type = value_type or getattr(self.sdk, "MVCC_ENUMVALUE")
        value = value_type()
        getter = getattr(self.camera, "MV_CC_GetEnumValueEx", None)
        getter = getter or getattr(self.camera, "MV_CC_GetEnumValue")
        self._check(getter(str(node), value), "read {}".format(node))
        return int(value.nCurValue)

    def get_string(self, node: str) -> str:
        value = self.sdk.MVCC_STRINGVALUE()
        self._check(self.camera.MV_CC_GetStringValue(str(node), value), "read {}".format(node))
        return _decode_c_string(value.chCurValue)

    def feature_descriptor(self, node: str) -> dict[str, Any]:
        access = ctypes.c_int()
        interface = ctypes.c_int()
        self._check(
            self.camera.MV_XML_GetNodeAccessMode(str(node), access),
            "read {} access".format(node),
        )
        self._check(
            self.camera.MV_XML_GetNodeInterfaceType(str(node), interface),
            "read {} interface".format(node),
        )
        return {
            "name": str(node),
            "access_code": int(access.value),
            "access": self.ACCESS_NAMES.get(int(access.value), "unknown"),
            "interface_code": int(interface.value),
            "interface": self.INTERFACE_NAMES.get(int(interface.value), "unknown"),
        }

    def genicam_xml(self) -> bytes:
        """Return the complete camera feature tree used by Beginner/Guru views."""

        length = ctypes.c_uint()
        initial = self.camera.MV_XML_GetGenICamXML(None, 0, length)
        if int(length.value) <= 0:
            self._check(initial, "query GenICam XML size")
            raise RuntimeError("MVS returned an empty GenICam feature tree")
        buffer = (ctypes.c_ubyte * int(length.value))()
        self._check(
            self.camera.MV_XML_GetGenICamXML(buffer, len(buffer), length),
            "read GenICam XML",
        )
        return bytes(buffer[: int(length.value)])

    def get_feature(self, node: str) -> Any:
        interface = self.feature_descriptor(node)["interface"]
        if interface == "integer":
            return self.get_int(node)
        if interface == "boolean":
            return self.get_bool(node)
        if interface == "float":
            return self.get_float(node)
        if interface == "string":
            return self.get_string(node)
        if interface == "enumeration":
            return self.get_enum(node)
        raise TypeError("GenICam node {} has unsupported readable interface {}".format(node, interface))

    def set_feature(self, node: str, value: Any) -> None:
        interface = self.feature_descriptor(node)["interface"]
        if interface == "integer":
            self.set_int(node, int(value))
            return
        if interface == "boolean":
            self.set_bool(node, bool(value))
            return
        if interface == "float":
            self.set_float(node, float(value))
            return
        if interface == "string":
            self._check(
                self.camera.MV_CC_SetStringValue(str(node), str(value)),
                "set {}".format(node),
            )
            return
        if interface == "enumeration":
            if isinstance(value, str):
                self.set_enum(node, value)
            else:
                self._check(
                    self.camera.MV_CC_SetEnumValue(str(node), int(value)),
                    "set {}".format(node),
                )
            return
        if interface == "command":
            self._check(self.camera.MV_CC_SetCommandValue(str(node)), "execute {}".format(node))
            return
        raise TypeError("GenICam node {} has unsupported writable interface {}".format(node, interface))

    def set_enum(self, node: str, value: str) -> None:
        by_string = getattr(self.camera, "MV_CC_SetEnumValueByString", None)
        if by_string is not None:
            result = by_string(str(node), str(value))
            if int(result) == 0:
                return
        numeric = self.ENUM_VALUES.get(str(node), {}).get(str(value))
        if numeric is None:
            raise RuntimeError("No numeric fallback for {}={}".format(node, value))
        self._check(self.camera.MV_CC_SetEnumValue(str(node), int(numeric)), "set {}".format(node))

    def set_bayer_conversion(
        self,
        gamma: float,
        ccm_rgb_3x3: Sequence[Sequence[float]],
    ) -> Mapping[str, Any]:
        """Configure MVS's existing Bayer-to-BGR conversion once per handle."""

        gamma_value = float(gamma)
        matrix = np.asarray(ccm_rgb_3x3, dtype=np.float64)
        if not np.isfinite(gamma_value) or not 0.1 <= gamma_value <= 4.0:
            raise ValueError("MVS Bayer gamma must be in [0.1, 4.0]")
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("MVS Bayer CCM must be a finite RGB 3x3 matrix")
        quantized = np.round(matrix * 1024.0).astype(np.int64)
        if np.any(quantized < -8192) or np.any(quantized > 8192):
            raise ValueError("MVS Bayer CCM coefficients must be in [-8, 8]")
        gamma_setter = getattr(self.camera, "MV_CC_SetGammaValue", None)
        ccm_setter = getattr(self.camera, "MV_CC_SetBayerCCMParam", None)
        ccm_type = getattr(self.sdk, "MV_CC_CCM_PARAM", None)
        if gamma_setter is None or ccm_setter is None or ccm_type is None:
            raise RuntimeError(
                "Installed MVS SDK lacks fused Bayer gamma/CCM conversion"
            )
        source_pixel_type = self.get_enum("PixelFormat")
        self._check(
            gamma_setter(source_pixel_type, gamma_value),
            "set MVS Bayer conversion gamma",
        )
        parameter = ccm_type()
        ctypes.memset(ctypes.byref(parameter), 0, ctypes.sizeof(parameter))
        parameter.bCCMEnable = True
        for index, value in enumerate(quantized.reshape(-1).tolist()):
            parameter.nCCMat[index] = int(value)
        self._check(
            ccm_setter(parameter),
            "set MVS Bayer conversion CCM",
        )
        return {
            "gamma": gamma_value,
            "ccm_rgb_3x3": (quantized.astype(np.float64) / 1024.0).tolist(),
            "ccm_quantization_scale": 1024,
            "source_pixel_type": int(source_pixel_type),
            "additional_frame_passes": 0,
            "additional_frame_copies": 0,
        }

    def read_bgr(self, timeout_ms: int = 1000) -> tuple[np.ndarray, dict[str, Any]]:
        if self.camera is None or not self._grabbing:
            raise RuntimeError("HIK camera is not grabbing")
        width, height = self.get_int("Width"), self.get_int("Height")
        size = int(width * height * 3)
        buffer = (ctypes.c_ubyte * size)()
        frame_info = self.sdk.MV_FRAME_OUT_INFO_EX()
        ctypes.memset(ctypes.byref(frame_info), 0, ctypes.sizeof(frame_info))
        getter = getattr(self.camera, "MV_CC_GetImageForBGR", None)
        if getter is None:
            raise RuntimeError(
                "Installed MVS wrapper lacks MV_CC_GetImageForBGR; update the SDK "
                "or supply a custom HIK backend with BGR conversion."
            )
        self._check(getter(buffer, size, frame_info, int(timeout_ms)), "read BGR frame")
        frame_width = int(frame_info.nWidth or width)
        frame_height = int(frame_info.nHeight or height)
        image = np.ctypeslib.as_array(buffer)[: frame_width * frame_height * 3]
        image = image.reshape((frame_height, frame_width, 3)).copy()
        return image, {
            "frame_number": int(getattr(frame_info, "nFrameNum", 0)),
            "pixel_type": int(getattr(frame_info, "enPixelType", 0)),
            "device_timestamp_high": int(getattr(frame_info, "nDevTimeStampHigh", 0)),
            "device_timestamp_low": int(getattr(frame_info, "nDevTimeStampLow", 0)),
        }


class HikMvsCameraAdapter(CameraAdapter):
    """CameraAdapter plugin for HIK/Hikrobot MVS cameras."""

    adapter_id = "hik_mvs"

    CONTROL_NODES = {
        "exposure_us": "ExposureTime",
        "gain": "Gain",
        "frame_rate": "AcquisitionFrameRate",
    }
    CALIBRATION_FEATURE_NODES = (
        "DeviceVendorName",
        "DeviceModelName",
        "DeviceSerialNumber",
        "SensorWidth",
        "SensorHeight",
        "PixelFormat",
        "ADCBitDepth",
        "ExposureAuto",
        "GainAuto",
        "BalanceWhiteAuto",
        "AutoFunctionAOISelector",
        "AutoFunctionAOIWidth",
        "AutoFunctionAOIHeight",
        "AutoFunctionAOIOffsetX",
        "AutoFunctionAOIOffsetY",
        "AutoFunctionAOIUsageIntensity",
        "AutoFunctionAOIUsageWhiteBalance",
        "AutoExposureTimeLowerLimit",
        "AutoExposureTimeUpperLimit",
        "AutoGainLowerLimit",
        "AutoGainUpperLimit",
        "BlackLevelEnable",
        "BlackLevel",
        "GammaEnable",
        "Gamma",
        "AcquisitionFrameRateEnable",
    )

    def __init__(self, backend=None, sdk_python_path: Optional[str] = None) -> None:
        self._backend = backend
        self._sdk_python_path = sdk_python_path
        self.configuration: Optional[CameraConfiguration] = None
        self.metadata: dict[str, Any] = {}
        self._opened = False

    @property
    def backend(self):
        if self._backend is None:
            self._backend = MvsPythonBackend(self._sdk_python_path)
        return self._backend

    def devices(self, probe: bool = False) -> Sequence[CameraDevice]:
        if not probe:
            return ()
        return tuple(
            CameraDevice(
                str(item["device_id"]),
                "{} {} {}".format(item.get("transport", "HIK"), item.get("model", "camera"), item.get("serial", "")).strip(),
                {key: value for key, value in item.items() if key not in ("pointer",)},
            )
            for item in self.backend.enumerate_devices()
        )

    def open(self, configuration: CameraConfiguration) -> Mapping[str, Any]:
        self.close()
        try:
            metadata = dict(self.backend.open(configuration.device_id))
            self.backend.set_enum("TriggerMode", "Off")
            try:
                self.backend.set_bool("AcquisitionFrameRateEnable", True)
                frame_range = self.backend.float_range("AcquisitionFrameRate")
                requested = float(np.clip(configuration.fps, frame_range["minimum"], frame_range["maximum"]))
                self.backend.set_float("AcquisitionFrameRate", requested)
                requested = self.backend.get_float("AcquisitionFrameRate")
            except Exception:
                requested = configuration.fps
            self.backend.start()
        except Exception:
            if self._backend is not None:
                self._backend.close()
            raise
        self.configuration = configuration
        self._opened = True
        metadata.update(
            {
                "adapter_id": self.adapter_id,
                "device_id": configuration.device_id,
                "width_px": self.backend.get_int("Width"),
                "height_px": self.backend.get_int("Height"),
                "fps": float(requested),
            }
        )
        self.metadata = metadata
        return dict(metadata)

    def read(self) -> FrameSample:
        if not self._opened:
            raise RuntimeError("HIK camera is not open")
        image, frame_metadata = self.backend.read_bgr()
        # Acquisition sessions use perf_counter_ns as their common host clock.
        # Keep the raw HIK device counter in metadata, but put the receive time
        # on the same host timebase as scrcpy's Android clock mapping.
        received = time.perf_counter_ns()
        return FrameSample(
            image=image,
            time_ns=received,
            receive_time_ns=received,
            source_id="hik:{}".format(self.metadata.get("device_id", "unknown")),
            metadata={"adapter_id": self.adapter_id, **frame_metadata},
        )

    def close(self) -> None:
        try:
            if self._backend is not None:
                self._backend.close()
        finally:
            self.configuration = None
            self._opened = False

    def controls(self) -> Mapping[str, Any]:
        result = {}
        for public, node in self.CONTROL_NODES.items():
            try:
                result[public] = self.backend.float_range(node)
            except Exception as exc:
                result[public] = {"available": False, "error": str(exc)}
        for public, node in {
            "width": "Width",
            "height": "Height",
            "offset_x": "OffsetX",
            "offset_y": "OffsetY",
        }.items():
            try:
                result[public] = self.backend.int_range(node)
            except Exception as exc:
                result[public] = {"available": False, "error": str(exc)}
        features = {}
        for node in self.CALIBRATION_FEATURE_NODES:
            try:
                descriptor = dict(self.feature_descriptor(node))
                if descriptor["access"] in ("read_only", "read_write") and descriptor[
                    "interface"
                ] not in (
                    "command",
                    "category",
                    "register",
                    "port",
                    "base",
                    "value",
                    "enum_entry",
                ):
                    descriptor["value"] = self.get_feature(node)
                features[node] = descriptor
            except Exception as exc:
                features[node] = {"available": False, "error": str(exc)}
        result["genicam"] = features
        return result

    def set_control(self, name: str, value: Any) -> bool:
        node = self.CONTROL_NODES.get(str(name))
        if node is None:
            return False
        self.backend.set_float(node, float(value))
        return True

    def feature_descriptor(self, node: str) -> Mapping[str, Any]:
        """Describe any Beginner/Guru GenICam node exposed by the camera."""

        return self.backend.feature_descriptor(str(node))

    def get_feature(self, node: str) -> Any:
        return self.backend.get_feature(str(node))

    def set_feature(self, node: str, value: Any) -> None:
        self.backend.set_feature(str(node), value)

    def genicam_xml(self) -> bytes:
        return self.backend.genicam_xml()

    def set_manual_imaging(self, exposure_us: float, gain: float) -> Mapping[str, float]:
        self.backend.set_enum("ExposureAuto", "Off")
        self.backend.set_enum("GainAuto", "Off")
        if self.configuration is not None:
            requested_fps = min(
                float(self.configuration.fps), 1.0e6 / max(float(exposure_us), 1.0)
            )
            try:
                limits = self.backend.float_range("AcquisitionFrameRate")
                requested_fps = float(np.clip(requested_fps, limits["minimum"], limits["maximum"]))
                self.backend.set_float("AcquisitionFrameRate", requested_fps)
                self.metadata["fps"] = self.backend.get_float("AcquisitionFrameRate")
            except Exception:
                pass
        self.backend.set_float("ExposureTime", float(exposure_us))
        self.backend.set_float("Gain", float(gain))
        actual_exposure = self.backend.get_float("ExposureTime")
        actual_gain = self.backend.get_float("Gain")
        tolerance_us = max(2.0, abs(float(exposure_us)) * 0.001)
        if abs(actual_exposure - float(exposure_us)) > tolerance_us:
            raise RuntimeError(
                "HIK clamped exposure from {:.3f} to {:.3f} us; refresh quantization is not satisfied".format(
                    float(exposure_us), actual_exposure
                )
            )
        return {
            "exposure_us": float(actual_exposure),
            "gain": float(actual_gain),
            "fps": float(self.metadata.get("fps", self.configuration.fps if self.configuration else 0.0)),
        }

    def set_auto_imaging(self) -> Mapping[str, float]:
        """Enable camera auto controls only for pre-geometry target visibility."""

        self.backend.set_enum("ExposureAuto", "Continuous")
        self.backend.set_enum("GainAuto", "Continuous")
        return {
            "exposure_us": self.backend.get_float("ExposureTime"),
            "gain": self.backend.get_float("Gain"),
        }

    @staticmethod
    def _enum_name(node: str, numeric: int) -> str:
        values = MvsPythonBackend.ENUM_VALUES.get(str(node), {})
        for name, value in values.items():
            if int(value) == int(numeric):
                return str(name)
        return "Unknown({})".format(int(numeric))

    def auto_imaging_modes(self) -> Mapping[str, str]:
        """Read the three HIK one-shot auto state machines without scanning Guru nodes."""

        return {
            node: self._enum_name(node, self.backend.get_enum(node))
            for node in ("ExposureAuto", "GainAuto", "BalanceWhiteAuto")
        }

    def white_balance_state(self) -> Mapping[str, int]:
        effective = {}
        for selector in ("Red", "Green", "Blue"):
            self.backend.set_enum("BalanceRatioSelector", selector)
            effective["ratio_{}".format(selector.lower())] = self.backend.get_int(
                "BalanceRatio"
            )
        return effective

    def set_once_auto_imaging(self) -> Mapping[str, Any]:
        """Start HIK's one-shot exposure, gain, and white-balance routines."""

        self.backend.set_enum("BalanceWhiteAuto", "Once")
        self.backend.set_enum("ExposureAuto", "Once")
        self.backend.set_enum("GainAuto", "Once")
        return {
            **dict(self.imaging_state()),
            "modes": dict(self.auto_imaging_modes()),
        }

    @staticmethod
    def _align_integer(value: int, limits: Mapping[str, Any]) -> int:
        minimum = int(limits["minimum"])
        maximum = int(limits["maximum"])
        increment = max(1, int(limits.get("increment", 1)))
        clipped = int(np.clip(int(value), minimum, maximum))
        return minimum + ((clipped - minimum) // increment) * increment

    def configure_auto_function_roi(
        self, camera_roi_xywh: Sequence[int]
    ) -> Mapping[str, Any]:
        """Bind HIK AE/gain AOI1 and AWB AOI2 to the controlled phone patch."""

        requested_x, requested_y, requested_width, requested_height = map(
            int, camera_roi_xywh
        )
        if min(requested_width, requested_height) <= 0:
            raise ValueError("Auto-function ROI dimensions must be positive")
        results = {}
        for selector in ("AOI1", "AOI2"):
            self.backend.set_enum("AutoFunctionAOISelector", selector)
            # Offsets must return to zero before dimensions can grow or change.
            self.backend.set_int("AutoFunctionAOIOffsetX", 0)
            self.backend.set_int("AutoFunctionAOIOffsetY", 0)
            width = self._align_integer(
                requested_width, self.backend.int_range("AutoFunctionAOIWidth")
            )
            height = self._align_integer(
                requested_height, self.backend.int_range("AutoFunctionAOIHeight")
            )
            self.backend.set_int("AutoFunctionAOIWidth", width)
            self.backend.set_int("AutoFunctionAOIHeight", height)
            x = self._align_integer(
                requested_x, self.backend.int_range("AutoFunctionAOIOffsetX")
            )
            y = self._align_integer(
                requested_y, self.backend.int_range("AutoFunctionAOIOffsetY")
            )
            self.backend.set_int("AutoFunctionAOIOffsetX", x)
            self.backend.set_int("AutoFunctionAOIOffsetY", y)
            if selector == "AOI1":
                try:
                    self.backend.set_bool("AutoFunctionAOIUsageIntensity", True)
                except Exception:
                    pass
            results[selector] = {
                "xywh": [
                    self.backend.get_int("AutoFunctionAOIOffsetX"),
                    self.backend.get_int("AutoFunctionAOIOffsetY"),
                    self.backend.get_int("AutoFunctionAOIWidth"),
                    self.backend.get_int("AutoFunctionAOIHeight"),
                ],
                "usage_intensity": self.backend.get_bool(
                    "AutoFunctionAOIUsageIntensity"
                ),
                "usage_white_balance": self.backend.get_bool(
                    "AutoFunctionAOIUsageWhiteBalance"
                ),
            }
        return {
            "requested_camera_roi_xywh": list(map(int, camera_roi_xywh)),
            "selectors": results,
            "source": "camera_genicam_auto_function_aoi",
        }

    def configure_once_auto_limits(
        self, maximum_exposure_us: float, maximum_gain: float
    ) -> Mapping[str, Any]:
        """Constrain HIK one-shot AE/gain to the requested operating envelope."""

        exposure_limits = self.backend.int_range("AutoExposureTimeUpperLimit")
        exposure_upper = self._align_integer(
            int(round(float(maximum_exposure_us))), exposure_limits
        )
        self.backend.set_int("AutoExposureTimeUpperLimit", exposure_upper)
        gain_limits = self.backend.float_range("AutoGainUpperLimit")
        gain_upper = float(
            np.clip(
                float(maximum_gain),
                float(gain_limits["minimum"]),
                float(gain_limits["maximum"]),
            )
        )
        self.backend.set_float("AutoGainUpperLimit", gain_upper)
        return {
            "exposure_upper_us": self.backend.get_int("AutoExposureTimeUpperLimit"),
            "gain_upper": self.backend.get_float("AutoGainUpperLimit"),
            "source": "camera_genicam_one_shot_auto_limits",
        }

    def imaging_state(self) -> Mapping[str, float]:
        return {
            "exposure_us": self.backend.get_float("ExposureTime"),
            "gain": self.backend.get_float("Gain"),
        }

    def set_white_balance(self, red: int, green: int, blue: int) -> Mapping[str, int]:
        self.backend.set_enum("BalanceWhiteAuto", "Off")
        effective = {}
        for selector, value in (("Red", red), ("Green", green), ("Blue", blue)):
            self.backend.set_enum("BalanceRatioSelector", selector)
            self.backend.set_int("BalanceRatio", int(value))
            effective["ratio_{}".format(selector.lower())] = self.backend.get_int("BalanceRatio")
        return effective

    def set_bayer_conversion(
        self,
        gamma: float,
        ccm_rgb_3x3: Sequence[Sequence[float]],
    ) -> Mapping[str, Any]:
        """Set fused MVS gamma+CCM without adding a streaming frame pass."""

        return self.backend.set_bayer_conversion(gamma, ccm_rgb_3x3)

    def black_level_control(self) -> Mapping[str, Any]:
        """Return the one black-pedestal control used by calibration, if writable."""

        try:
            limits = dict(self.backend.int_range("BlackLevel"))
            descriptor = dict(self.feature_descriptor("BlackLevel"))
            enabled = True
            try:
                enabled = bool(self.get_feature("BlackLevelEnable"))
            except Exception:
                pass
            return {
                "available": descriptor.get("access") == "read_write",
                "enabled": enabled,
                **limits,
            }
        except Exception as exc:
            return {"available": False, "error": str(exc)}

    def set_black_level(self, value: int) -> int:
        limits = self.black_level_control()
        if not limits.get("available"):
            raise RuntimeError("Writable HIK BlackLevel is unavailable")
        try:
            self.set_feature("BlackLevelEnable", True)
        except Exception:
            pass
        aligned = self._aligned(int(value), limits, lower=True)
        self.backend.set_int("BlackLevel", aligned)
        return self.backend.get_int("BlackLevel")

    @staticmethod
    def _aligned(value: int, limits: Mapping[str, Any], lower: bool = True) -> int:
        minimum = int(limits["minimum"])
        maximum = int(limits["maximum"])
        increment = max(1, int(limits.get("increment", 1)))
        value = int(np.clip(value, minimum, maximum))
        steps = (value - minimum) / float(increment)
        steps = np.floor(steps) if lower else np.ceil(steps)
        return int(np.clip(minimum + int(steps) * increment, minimum, maximum))

    def align_roi(self, roi_xywh: Sequence[int]) -> list[int]:
        x, y, width, height = map(int, roi_xywh)
        controls = self.controls()
        # Offset maxima are frequently dynamic and report zero while full Width/
        # Height is active. Derive the static sensor allowance from dimension
        # maxima/minima, retaining the offset node's declared increment.
        offset_x_limits = dict(controls["offset_x"])
        offset_y_limits = dict(controls["offset_y"])
        offset_x_limits["maximum"] = max(
            int(offset_x_limits.get("maximum", 0)),
            int(controls["width"]["maximum"]) - int(controls["width"]["minimum"]),
        )
        offset_y_limits["maximum"] = max(
            int(offset_y_limits.get("maximum", 0)),
            int(controls["height"]["maximum"]) - int(controls["height"]["minimum"]),
        )
        x = self._aligned(x, offset_x_limits, lower=True)
        y = self._aligned(y, offset_y_limits, lower=True)
        width = self._aligned(width, controls["width"], lower=False)
        height = self._aligned(height, controls["height"], lower=False)
        width_limits = dict(controls["width"])
        height_limits = dict(controls["height"])
        width_limits["maximum"] = int(controls["width"]["maximum"]) - x
        height_limits["maximum"] = int(controls["height"]["maximum"]) - y
        width = self._aligned(width, width_limits, lower=True)
        height = self._aligned(height, height_limits, lower=True)
        return [x, y, width, height]

    def set_roi(self, roi_xywh: Sequence[int]) -> list[int]:
        x, y, width, height = self.align_roi(roi_xywh)
        was_open = self._opened
        if was_open:
            self.backend.stop()
        # GenICam commonly requires offsets reset before changing dimensions.
        try:
            self.backend.set_int("OffsetX", 0)
            self.backend.set_int("OffsetY", 0)
            self.backend.set_int("Width", width)
            self.backend.set_int("Height", height)
            self.backend.set_int("OffsetX", x)
            self.backend.set_int("OffsetY", y)
        finally:
            if was_open:
                self.backend.start()
        effective = [
            self.backend.get_int("OffsetX"),
            self.backend.get_int("OffsetY"),
            self.backend.get_int("Width"),
            self.backend.get_int("Height"),
        ]
        return effective

    def reset_full_sensor_roi(self) -> list[int]:
        """Clear persistent HIK crop state before full-sensor rig calibration."""

        was_open = self._opened
        if was_open:
            self.backend.stop()
        try:
            # HIK stores Width/Height/Offset nodes in the device. Offsets must
            # be cleared before the dynamic Width/Height maxima expose the
            # complete sensor again.
            self.backend.set_int("OffsetX", 0)
            self.backend.set_int("OffsetY", 0)
            width_limits = dict(self.backend.int_range("Width"))
            height_limits = dict(self.backend.int_range("Height"))
            sensor_width = int(self.backend.get_int("SensorWidth"))
            sensor_height = int(self.backend.get_int("SensorHeight"))
            width = self._aligned(
                min(sensor_width, int(width_limits["maximum"])),
                width_limits,
                lower=True,
            )
            height = self._aligned(
                min(sensor_height, int(height_limits["maximum"])),
                height_limits,
                lower=True,
            )
            self.backend.set_int("Width", width)
            self.backend.set_int("Height", height)
            self.backend.set_int("OffsetX", 0)
            self.backend.set_int("OffsetY", 0)
        finally:
            if was_open:
                self.backend.start()
        effective = [
            self.backend.get_int("OffsetX"),
            self.backend.get_int("OffsetY"),
            self.backend.get_int("Width"),
            self.backend.get_int("Height"),
        ]
        if effective[:2] != [0, 0]:
            raise RuntimeError(
                "HIK full-sensor reset retained non-zero offsets: {}".format(
                    effective
                )
            )
        return effective


def create_camera_adapter() -> HikMvsCameraAdapter:
    """Zero-argument factory compatible with the desktop app plugin loader."""

    return HikMvsCameraAdapter()


class RectifiedHikCamera:
    """Small `cv2.VideoCapture`-like reader backed by a saved HIK calibration."""

    def __init__(
        self,
        calibration_file: Union[str, Path],
        adapter: Optional[HikMvsCameraAdapter] = None,
        rectify: bool = True,
    ) -> None:
        self.path = Path(calibration_file)
        self.config = json.loads(self.path.read_text(encoding="utf-8"))
        self.adapter = adapter or HikMvsCameraAdapter(
            sdk_python_path=self.config.get("mvs_python_path")
        )
        self._opened = False
        self._rectify_enabled = bool(rectify)
        self._matrix = None
        self._output_size = None
        self._map_x = None
        self._map_y = None
        self._dense_path = None
        self._effective_roi = None

    def is_calibrated(self) -> bool:
        """Return whether the saved bundle is sufficient for adapter streaming."""

        try:
            camera = self.config["camera"]
            mode = camera["full_sensor_mode"]
            roi = list(map(int, camera["hardware_roi_xywh"]))
            imaging = self.config["imaging"]
            white_balance = imaging["white_balance"]
            normalization = self.config["normalization"]
            output_size = list(map(int, normalization["output_size_px"]))
            matrix = np.asarray(
                normalization["full_sensor_camera_to_output_3x3"], np.float64
            )
            return bool(
                str(camera["device_id"])
                and int(mode["width_px"]) > 0
                and int(mode["height_px"]) > 0
                and float(mode["fps"]) > 0
                and len(roi) == 4
                and min(roi[2:]) > 0
                and float(imaging["exposure_us"]) > 0
                and np.isfinite(float(imaging["gain"]))
                and all(
                    int(white_balance[name]) > 0
                    for name in ("ratio_red", "ratio_green", "ratio_blue")
                )
                and len(output_size) == 2
                and min(output_size) > 0
                and matrix.shape == (3, 3)
                and np.isfinite(matrix).all()
                and abs(float(np.linalg.det(matrix))) > 1.0e-12
            )
        except (KeyError, TypeError, ValueError, OSError):
            return False

    def open(self) -> "RectifiedHikCamera":
        if self._opened:
            return self
        if not self.is_calibrated():
            raise RuntimeError(
                "Saved HIK bundle is incomplete and cannot configure the camera adapter"
            )
        camera = self.config["camera"]
        mode = camera["full_sensor_mode"]
        self.adapter.open(
            CameraConfiguration(
                device_id=str(camera["device_id"]),
                width_px=int(mode["width_px"]),
                height_px=int(mode["height_px"]),
                fps=float(mode["fps"]),
                backend="hik_mvs",
            )
        )
        imaging = self.config["imaging"]
        if imaging.get("black_level") is not None:
            self.adapter.set_black_level(int(imaging["black_level"]))
        self.adapter.set_manual_imaging(imaging["exposure_us"], imaging["gain"])
        wb = imaging["white_balance"]
        self.adapter.set_white_balance(wb["ratio_red"], wb["ratio_green"], wb["ratio_blue"])
        effective_roi = self.adapter.set_roi(camera["hardware_roi_xywh"])
        normalization = self.config["normalization"]
        # Keep the calibrated mapping available even for the minimum-latency
        # hardware-ROI stream.  It is used only by explicit evidence checks;
        # ordinary rectify=False reads still return the untouched camera ROI.
        self._matrix = camera_adapter_roi_to_output_homography(
            self.config, effective_roi
        )
        calibrated_output_size = tuple(
            map(int, normalization["output_size_px"])
        )
        dense_file = normalization.get("dense_map_file")
        if dense_file:
            dense_path = self.path.parent / str(dense_file)
            if dense_path.is_file():
                self._dense_path = dense_path
        self._effective_roi = list(map(int, effective_roi))
        if self._rectify_enabled:
            self._load_rectification_maps()
        self._output_size = (
            calibrated_output_size
            if self._rectify_enabled
            else (int(effective_roi[2]), int(effective_roi[3]))
        )
        self._opened = True
        return self

    @property
    def orientation(self) -> Mapping[str, Any]:
        return dict(self.config.get("normalization", {}).get("orientation", {}))

    def space_converter(
        self, adb_surface_quarter_turns_clockwise_from_natural: int = 0
    ) -> RigCalibratedSpaceConverter:
        """Return the authoritative adapter/ADB coordinate converter."""

        if not self._rectify_enabled:
            raise RuntimeError(
                "Adapter/ADB conversion requires the rig-rectified output space"
            )
        return RigCalibratedSpaceConverter(
            self.config, adb_surface_quarter_turns_clockwise_from_natural
        )

    def camera_adapter_to_adb_points(
        self,
        points_xy: Sequence[Sequence[float]],
        adb_surface_quarter_turns_clockwise_from_natural: int = 0,
    ) -> np.ndarray:
        return self.space_converter(
            adb_surface_quarter_turns_clockwise_from_natural
        ).camera_adapter_to_adb_points(points_xy)

    def adb_to_camera_adapter_points(
        self,
        points_xy: Sequence[Sequence[float]],
        adb_surface_quarter_turns_clockwise_from_natural: int = 0,
    ) -> np.ndarray:
        return self.space_converter(
            adb_surface_quarter_turns_clockwise_from_natural
        ).adb_to_camera_adapter_points(points_xy)

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if not self._opened:
            return False, None
        try:
            sample = self.adapter.read()
            rectified = self._rectify(sample.image)
            return True, rectified
        except Exception:
            return False, None

    def read_sample(self) -> FrameSample:
        if not self._opened:
            raise RuntimeError("Rectified HIK stream is not open")
        sample = self.adapter.read()
        rectified = self._rectify(sample.image)
        return FrameSample(
            rectified,
            sample.time_ns,
            clock_id=sample.clock_id,
            receive_time_ns=sample.receive_time_ns,
            source_id=sample.source_id
            + (":rectified" if self._rectify_enabled else ":hardware-roi"),
            metadata={
                **dict(sample.metadata),
                "rectified": self._rectify_enabled,
                "hardware_roi_output": not self._rectify_enabled,
            },
        )

    def release(self) -> None:
        self.adapter.close()
        self._opened = False

    def _rectify(self, image: np.ndarray) -> np.ndarray:
        if not self._rectify_enabled:
            return image
        return self.rectify_for_evidence(image)

    def rectify_for_evidence(self, image: np.ndarray) -> np.ndarray:
        """Rectify one ROI image for diagnostics without changing stream mode."""

        self._load_rectification_maps()
        calibrated_size = tuple(
            map(int, self.config["normalization"]["output_size_px"])
        )
        if self._map_x is not None and self._map_y is not None:
            return cv2.remap(
                image,
                self._map_x,
                self._map_y,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        return cv2.warpPerspective(
            image,
            self._matrix,
            calibrated_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def _load_rectification_maps(self) -> None:
        if self._map_x is not None or self._dense_path is None:
            return
        with np.load(str(self._dense_path)) as dense:
            self._map_x = (
                np.asarray(dense["map_x"], dtype=np.float32)
                - float(self._effective_roi[0])
            )
            self._map_y = (
                np.asarray(dense["map_y"], dtype=np.float32)
                - float(self._effective_roi[1])
            )

    def __enter__(self) -> "RectifiedHikCamera":
        return self.open()

    def __exit__(self, *_exc) -> None:
        self.release()
