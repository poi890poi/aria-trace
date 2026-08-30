"""Device-neutral source selection for the Workbench application.

The factory is an application composition boundary: it selects concrete capture
adapters from configuration, but owns no recording, calibration, or tracking
policy.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import List

from aria_trace.adapters.android.capture import (
    AndroidRoiFrameSource,
    AndroidRoiSpec,
    ScrcpyCaptureHub,
    find_scrcpy_server,
)
from aria_trace.adapters.hik.capture import CalibratedHikFrameSource, NativeHikFrameSource
from aria_trace.adapters.rig.dual_capture import (
    build_calibrated_rig_recording_bundle,
    single_source_recording_bundle,
)
from aria_trace.adapters.sources import (
    AdbClockMapper,
    AdbGetEventSource,
    AdbScreenshotFrameSource,
    OpenCvCameraFrameSource,
)
from aria_trace.adapters.windows import (
    WindowsKeyboardMouseSource,
    WindowsRawKeyboardMouseSource,
    WindowsWindowFrameSource,
    WindowsXInputSource,
)


def parse_adb_devices(output: str) -> List[dict]:
    """Parse ``adb devices -l`` without treating unavailable devices as targets."""
    devices = []
    for line in str(output or "").splitlines():
        fields = line.strip().split()
        if len(fields) < 2 or fields[0] == "List":
            continue
        properties = {}
        for field in fields[2:]:
            if ":" in field:
                key, value = field.split(":", 1)
                properties[key] = value
        devices.append(
            {
                "serial": fields[0],
                "status": fields[1],
                "available": fields[1] == "device",
                "model": properties.get("model"),
                "product": properties.get("product"),
                "device": properties.get("device"),
            }
        )
    return devices


class SourceFactory:
    """Construct recorder sources from neutral adapter configuration."""

    FRAME_ADAPTERS = (
        {"adapter": "windows_window", "label": "Windows game window", "status": "pc_mvp"},
        {
            "adapter": "android_scrcpy",
            "label": "Android device (scrcpy)",
            "status": "available",
        },
        {
            "adapter": "hik_mvs",
            "label": "HIK camera (native sensor)",
            "status": "available",
        },
        {
            "adapter": "hik_rig_calibrated",
            "label": "Calibrated rig (ADB + HIK)",
            "status": "available",
        },
        {"adapter": "uvc", "label": "UVC camera", "status": "available"},
        {"adapter": "adb_screenshot", "label": "ADB screenshot", "status": "available"},
    )
    INPUT_ADAPTERS = (
        {
            "adapter": "windows_xinput",
            "label": "Windows XInput gamepad",
            "status": "recommended_pc_mvp",
            "fidelity": "buttons, triggers, locomotion, camera axes, and timing",
        },
        {
            "adapter": "windows_raw_keyboard_mouse",
            "label": "Windows raw keyboard and mouse",
            "status": "recommended_pc_mvp",
            "fidelity": "key transitions, scan codes, buttons, wheel, relative camera motion, and timing",
        },
        {
            "adapter": "windows_keyboard_mouse",
            "label": "Windows keyboard and cursor state (legacy)",
            "status": "limited",
            "fidelity": "no raw relative mouse motion",
        },
        {
            "adapter": "adb_getevent",
            "label": "Android getevent",
            "status": "available",
            "fidelity": "raw device input events",
        },
        {
            "adapter": "none",
            "label": "No input source",
            "status": "available",
            "fidelity": "visual evidence only",
        },
    )

    def __init__(
        self,
        desktop_api=None,
        xinput_api=None,
        raw_input_api=None,
        hik_adapter_factory=None,
        rig_bundle_builder=None,
        workspace_root=None,
    ) -> None:
        self.desktop_api = desktop_api
        self.xinput_api = xinput_api
        self.raw_input_api = raw_input_api
        self.hik_adapter_factory = hik_adapter_factory
        self.rig_bundle_builder = (
            rig_bundle_builder or build_calibrated_rig_recording_bundle
        )
        self.workspace_root = Path(workspace_root or Path(__file__).resolve().parents[3])

    def _bundled_tool(self, name: str) -> Path:
        return self.workspace_root / ".tools" / "scrcpy-win64-v4.1" / name

    def _adb(self, config: dict) -> Path:
        value = config.get("adb") or shutil.which("adb")
        if value:
            return Path(value)
        bundled = self._bundled_tool("adb.exe" if os.name == "nt" else "adb")
        if bundled.is_file():
            return bundled
        raise RuntimeError("ADB is not on PATH and no executable was configured")

    def android_devices(self) -> List[dict]:
        adb = self._adb({})
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        output = subprocess.check_output(
            [str(adb), "devices", "-l"],
            timeout=5,
            universal_newlines=True,
            creationflags=creationflags,
        )
        return parse_adb_devices(output)

    def hik_devices(self) -> List[dict]:
        """Enumerate HIK devices without opening or changing camera state."""
        if self.hik_adapter_factory is None:
            from acquisition.rig_calibration.hik.driver import HikMvsCameraAdapter

            adapter = HikMvsCameraAdapter()
        else:
            adapter = self.hik_adapter_factory()
        try:
            return [
                {
                    "camera_id": str(device.device_id),
                    "label": str(device.label),
                    "metadata": dict(device.metadata),
                    "available": True,
                }
                for device in adapter.devices(probe=True)
            ]
        finally:
            close = getattr(adapter, "close", None)
            if callable(close):
                close()

    def capture_sources(self, frame_config: dict, input_config: dict):
        """Build one live frame source and its synchronized input source."""
        adapter = frame_config.get("adapter")
        if adapter == "hik_rig_calibrated":
            bundle = self.recording_bundle(frame_config, {"adapter": "none"})
            primary = next(
                source
                for source in bundle.frame_sources
                if source.stream_id == bundle.primary_stream_id
            )
            for source in bundle.frame_sources:
                if source is not primary:
                    source.stop()
            return primary, self.input(input_config)
        if adapter != "android_scrcpy":
            return self.frame(frame_config), self.input(input_config)
        serial = str(frame_config.get("serial") or "").strip()
        if not serial:
            raise ValueError("Choose a connected Android device")
        adb = self._adb(frame_config)
        clock = AdbClockMapper(adb, serial)
        server_config = frame_config.get("scrcpy_server")
        bundled_server = self._bundled_tool("scrcpy-server")
        if not server_config and bundled_server.is_file():
            server_config = bundled_server
        hub = ScrcpyCaptureHub(
            adb,
            find_scrcpy_server(server_config),
            serial=serial,
            ffmpeg=frame_config.get("ffmpeg"),
            clock=clock,
            bit_rate=int(frame_config.get("bit_rate") or 16_000_000),
            max_fps=float(frame_config.get("max_fps") or 60.0),
        )
        frame_source = AndroidRoiFrameSource(
            hub,
            AndroidRoiSpec("main", 0, 0, 0, 0),
        )
        if input_config.get("adapter") == "adb_getevent":
            input_source = AdbGetEventSource(
                adb,
                serial=serial,
                source_id=input_config.get("source_id", "android-input"),
                clock=clock,
            )
        else:
            input_source = self.input(input_config)
        return frame_source, input_source

    def recording_bundle(self, frame_config: dict, input_config: dict):
        """Build the complete source set owned by one recording adapter."""
        if frame_config.get("adapter") != "hik_rig_calibrated":
            frame_source, input_source = self.capture_sources(frame_config, input_config)
            return single_source_recording_bundle(frame_source, input_source)
        calibration = str(frame_config.get("calibration") or "").strip()
        if not calibration:
            raise ValueError("Choose a HIK rig calibration")
        server_config = frame_config.get("scrcpy_server")
        bundled_server = self._bundled_tool("scrcpy-server")
        if not server_config and bundled_server.is_file():
            server_config = bundled_server
        return self.rig_bundle_builder(
            Path(calibration),
            adb=self._adb(frame_config),
            scrcpy_server=server_config,
            ffmpeg=frame_config.get("ffmpeg"),
            input_adapter=input_config.get("adapter", "none"),
            input_source_id=input_config.get("source_id", "android-input"),
            bit_rate=int(frame_config.get("bit_rate") or 16_000_000),
            max_fps=float(frame_config.get("max_fps") or 60.0),
        )

    def frame(self, config: dict):
        adapter = config.get("adapter")
        if adapter == "windows_window":
            return WindowsWindowFrameSource(
                config.get("window_title", ""),
                stream_id=config.get("stream_id", "main"),
                fps=float(config.get("fps", 30.0)),
                exact_title=bool(config.get("exact_title", True)),
                api=self.desktop_api,
            )
        if adapter == "uvc":
            return OpenCvCameraFrameSource(
                int(config.get("device", 0)),
                stream_id=config.get("stream_id", "main"),
                width=config.get("width"),
                height=config.get("height"),
                fps=config.get("fps"),
            )
        if adapter == "adb_screenshot":
            return AdbScreenshotFrameSource(
                self._adb(config),
                serial=config.get("serial"),
                stream_id=config.get("stream_id", "main"),
                fps=float(config.get("fps", 2.0)),
            )
        if adapter == "hik_mvs":
            camera_id = str(config.get("camera_id") or "").strip()
            if not camera_id:
                raise ValueError("Choose a connected HIK camera")
            return NativeHikFrameSource(
                camera_id,
                stream_id=config.get("stream_id", "main"),
                width_px=int(config.get("width_px") or 2448),
                height_px=int(config.get("height_px") or 2048),
                fps=float(config.get("fps") or 30.0),
                sdk_python_path=config.get("mvs_python_path"),
            )
        if adapter == "hik_rig_calibrated":
            calibration = str(config.get("calibration") or "").strip()
            if not calibration:
                raise ValueError("Choose a HIK rig calibration")
            return CalibratedHikFrameSource(
                Path(calibration),
                stream_id=config.get("stream_id", "main"),
                rectify=True,
                output_quarter_turns_clockwise=int(
                    config.get("output_quarter_turns_clockwise") or 0
                ),
            )
        if adapter == "android_scrcpy":
            raise RuntimeError(
                "Android scrcpy capture must be created with its synchronized input source"
            )
        raise ValueError("Unsupported frame adapter: {}".format(adapter))

    def input(self, config: dict):
        adapter = config.get("adapter", "none")
        if adapter == "windows_xinput":
            return WindowsXInputSource(
                config.get("window_title", ""),
                poll_hz=float(config.get("poll_hz", 250.0)),
                user_index=int(config.get("user_index", 0)),
                exact_title=bool(config.get("exact_title", True)),
                desktop_api=self.desktop_api,
                xinput_api=self.xinput_api,
            )
        if adapter == "windows_raw_keyboard_mouse":
            return WindowsRawKeyboardMouseSource(
                config.get("window_title", ""),
                exact_title=bool(config.get("exact_title", True)),
                desktop_api=self.desktop_api,
                raw_input_api=self.raw_input_api,
            )
        if adapter == "windows_keyboard_mouse":
            return WindowsKeyboardMouseSource(
                config.get("window_title", ""),
                poll_hz=float(config.get("poll_hz", 125.0)),
                exact_title=bool(config.get("exact_title", True)),
                api=self.desktop_api,
            )
        if adapter == "adb_getevent":
            return AdbGetEventSource(
                self._adb(config),
                serial=config.get("serial"),
                source_id=config.get("source_id", "android-input"),
            )
        if adapter == "none":
            return None
        raise ValueError("Unsupported input adapter: {}".format(adapter))

    def descriptor(self) -> dict:
        return {
            "frame_adapters": list(self.FRAME_ADAPTERS),
            "input_adapters": list(self.INPUT_ADAPTERS),
        }
