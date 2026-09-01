"""Configure a HIK camera from a saved calibration and stream rectified frames."""

from __future__ import annotations

import argparse
from collections import deque
import importlib
import json
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import cv2

from aria_trace.apps.rig_presentation import console_print as print

from aria_trace.adapters.hik.compat import Camera, HikCamera
from aria_trace.adapters.hik.capture import NativeHikFrameSource
from aria_trace.adapters.hik.driver import HikMvsCameraAdapter, RectifiedHikCamera
from aria_trace.adapters.hik.game_camera import ProfiledHikGameCamera
from aria_trace.adapters.android.phone import AdbPhoneSession


class LiveStreamTelemetry:
    """Measured GUI throughput and latency without mixing clock domains."""

    def __init__(self, history: int = 60) -> None:
        self._ends = deque(maxlen=max(2, int(history)))
        self.fps = 0.0
        self.read_latency_ms = 0.0
        self.frame_age_ms: Optional[float] = None

    def observe(
        self,
        read_started_ns: int,
        read_finished_ns: int,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        finished = int(read_finished_ns)
        self._ends.append(finished)
        self.read_latency_ms = max(
            0.0, (finished - int(read_started_ns)) / 1.0e6
        )
        if len(self._ends) >= 2:
            elapsed = (self._ends[-1] - self._ends[0]) / 1.0e9
            self.fps = (
                (len(self._ends) - 1) / elapsed if elapsed > 0.0 else 0.0
            )
        values = dict(metadata or {})
        clock_id = str(values.get("host_timestamp_clock_id") or "")
        capture_ns = values.get("host_capture_time_ns")
        if clock_id in (
            "host_perf_counter_ns",
            "host_monotonic",
            "host_monotonic_ns",
        ) and capture_ns is not None:
            age = (finished - int(capture_ns)) / 1.0e6
            self.frame_age_ms = age if age >= 0.0 else None
        else:
            self.frame_age_ms = None

    def label(self) -> str:
        age = (
            "{:.1f} ms".format(self.frame_age_ms)
            if self.frame_age_ms is not None
            else "n/a"
        )
        return "FPS {:.1f} | read {:.1f} ms | age {}".format(
            self.fps, self.read_latency_ms, age
        )


def overlay_stream_telemetry(frame, telemetry: LiveStreamTelemetry):
    rendered = frame.copy()
    label = telemetry.label()
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
    )
    cv2.rectangle(
        rendered,
        (4, 4),
        (14 + text_width, 14 + text_height + baseline),
        (12, 12, 12),
        -1,
    )
    cv2.putText(
        rendered,
        label,
        (9, 10 + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return rendered


def _last_stream_metadata(camera, stream_name: str) -> Mapping[str, object]:
    getter = getattr(camera, "get_iris_frame_metadata", None)
    if not callable(getter):
        getter = getattr(camera, "get_aria_frame_metadata", None)
    if not callable(getter):
        return {}
    try:
        value = getter(stream_name)
    except TypeError:
        value = getter()
    if not isinstance(value, Mapping):
        return {}
    streams = value.get("streams")
    if isinstance(streams, Mapping):
        selected = streams.get(stream_name)
        return dict(selected) if isinstance(selected, Mapping) else {}
    return dict(value)


def adapter_hik_camera_class():
    """Import the public drop-in adapter exactly as application code does."""

    module = importlib.import_module("hikcam")
    camera = getattr(module, "HikCamera", None)
    if camera is None:
        raise RuntimeError("The hikcam adapter module does not export HikCamera")
    return camera


def open_native_mvs_source(
    camera_id: Optional[str], mvs_python_path: Optional[str]
) -> NativeHikFrameSource:
    """Open an uncalibrated full-sensor stream through Hikrobot's MVS SDK."""

    adapter = HikMvsCameraAdapter(sdk_python_path=mvs_python_path)
    selected_camera_id = str(camera_id).strip() if camera_id else ""
    if not selected_camera_id:
        devices = list(adapter.devices(probe=True))
        if not devices:
            raise RuntimeError("No HIK MVS camera was found")
        if len(devices) != 1:
            raise RuntimeError(
                "Multiple HIK MVS cameras are connected; pass --camera-id: {}"
                .format(", ".join(str(device.device_id) for device in devices))
            )
        selected_camera_id = str(devices[0].device_id)
    source = NativeHikFrameSource(selected_camera_id, adapter=adapter)
    source.start()
    return source


class PhoneDisplayPowerSession:
    """Manage only Android panel power; never mutate settings or launch an app."""

    def __init__(
        self,
        serial: str,
        adb_executable: str = "adb",
        phone: Optional[AdbPhoneSession] = None,
    ) -> None:
        self.phone = phone or AdbPhoneSession(serial, adb_executable=adb_executable)
        self._opened = False
        self.last_error: Optional[str] = None

    def open(self) -> "PhoneDisplayPowerSession":
        if self._opened:
            return self
        self._opened = True
        try:
            self.phone.shell("input", "keyevent", "KEYCODE_WAKEUP")
        except Exception as exc:
            self.last_error = str(exc)
        return self

    def close(self) -> None:
        if not self._opened:
            return
        try:
            self.phone.shell("input", "keyevent", "KEYCODE_SLEEP")
        except Exception as exc:
            self.last_error = str(exc)
        finally:
            self._opened = False

    def __enter__(self) -> "PhoneDisplayPowerSession":
        return self.open()

    def __exit__(self, *_exc) -> None:
        self.close()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Stream HIK video through either IRIS's calibrated drop-in "
            "adapter or Hikrobot's native MVS Python SDK."
        )
    )
    value.add_argument(
        "--camera-library",
        choices=("adapter", "native"),
        default="adapter",
        help=(
            "import IRIS's drop-in adapter (default) or the independently "
            "installed Hikrobot MVS SDK (MvCameraControl_class)"
        ),
    )
    value.add_argument(
        "--diagnostic-calibration-override",
        type=Path,
        help="explicit rig JSON for a diagnostic run; production uses the registry",
    )
    value.add_argument(
        "--diagnostic-rig-game-profile-override",
        type=Path,
        help="explicit immutable rig-game profile for a diagnostic run",
    )
    value.add_argument(
        "--mode",
        choices=("minimap", "full", "dual"),
        default="full",
        help=(
            "adapter stream mode; native MVS verification supports full only"
        ),
    )
    value.add_argument("--mvs-python-path")
    value.add_argument("--profile-root", type=Path)
    value.add_argument("--game-id")
    value.add_argument("--camera-id")
    value.add_argument("--phone-serial")
    value.add_argument("--color-order", choices=("RGB", "BGR"), default="BGR")
    value.add_argument(
        "--color-policy",
        choices=("auto", "rig_locked", "game_matched", "unadjusted"),
        default="auto",
    )
    value.add_argument("--minimap-margin-px", type=int, default=6)
    value.add_argument("--gui", action="store_true", help="Show live frames; Q/Esc closes")
    value.add_argument(
        "--no-rectify",
        action="store_true",
        help="Adapter only: show hardware-ROI frames without remap/warp",
    )
    value.add_argument(
        "--manage-phone-display",
        action="store_true",
        help="Wake the calibrated phone display for the GUI session and sleep it on exit",
    )
    value.add_argument("--adb", default="adb", help="ADB executable used only for display power")
    return value


def open_camera(
    diagnostic_calibration_override: Optional[Path] = None,
    mvs_python_path: Optional[str] = None,
    rectify: bool = True,
    diagnostic_rig_game_profile_override: Optional[Path] = None,
    mode: str = "full",
    profile_root: Optional[Path] = None,
    game_id: Optional[str] = None,
    camera_id: Optional[str] = None,
    phone_serial: Optional[str] = None,
    color_order: str = "BGR",
    color_policy: str = "auto",
    minimap_margin_px: int = 6,
):
    """Public UVC-like constructor for application code (read/release/isOpened)."""

    if diagnostic_calibration_override is None:
        if diagnostic_rig_game_profile_override is not None:
            raise ValueError(
                "A diagnostic rig-game profile requires a diagnostic rig calibration"
            )
        return HikCamera(
            config={
                "profile_root": profile_root,
                "game_id": game_id,
                "camera_id": camera_id,
                "phone_id": phone_serial,
                "mode": mode,
                "rectify": rectify,
                "color_order": color_order,
                "color_policy": color_policy,
                "minimap_margin_px": minimap_margin_px,
                "mvs_python_path": mvs_python_path,
            }
        ).open()
    if diagnostic_rig_game_profile_override is None and mode != "full":
        raise ValueError(
            "Diagnostic --mode {} requires --diagnostic-rig-game-profile-override"
            .format(mode)
        )
    adapter = HikMvsCameraAdapter(sdk_python_path=mvs_python_path)
    if diagnostic_rig_game_profile_override is not None:
        return ProfiledHikGameCamera(
            diagnostic_calibration_override,
            diagnostic_rig_game_profile_override,
            mode=mode,
            rectify_minimap=rectify,
            adapter=adapter,
        ).open()
    return RectifiedHikCamera(
        diagnostic_calibration_override, adapter=adapter, rectify=rectify
    ).open()


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.manage_phone_display and not arguments.gui:
        parser().error("--manage-phone-display requires --gui")
    if arguments.camera_library == "native":
        if arguments.mode != "full":
            parser().error("--camera-library native supports only --mode full")
        if arguments.diagnostic_calibration_override is not None:
            parser().error(
                "--camera-library native does not use a rig calibration override"
            )
        if arguments.diagnostic_rig_game_profile_override is not None:
            parser().error(
                "--camera-library native does not use a rig-game profile override"
            )
        if arguments.manage_phone_display:
            parser().error(
                "--camera-library native does not identify a calibrated phone; "
                "remove --manage-phone-display"
            )
    if (
        arguments.diagnostic_calibration_override is not None
        and arguments.diagnostic_rig_game_profile_override is None
        and arguments.mode != "full"
    ):
        parser().error(
            "Diagnostic --mode {} requires "
            "--diagnostic-rig-game-profile-override".format(arguments.mode)
        )
    if (
        arguments.diagnostic_calibration_override is None
        and arguments.diagnostic_rig_game_profile_override is not None
    ):
        parser().error(
            "--diagnostic-rig-game-profile-override requires "
            "--diagnostic-calibration-override"
        )
    display = None
    camera = None
    windows = []
    try:
        if arguments.camera_library == "native":
            camera = open_native_mvs_source(
                arguments.camera_id, arguments.mvs_python_path
            )
            configuration = None
        elif arguments.diagnostic_calibration_override is None:
            selected_camera_class = adapter_hik_camera_class()
            camera = selected_camera_class(
                config={
                    "profile_root": arguments.profile_root,
                    "game_id": arguments.game_id,
                    "camera_id": arguments.camera_id,
                    "phone_id": arguments.phone_serial,
                    "mode": arguments.mode,
                    "rectify": not arguments.no_rectify,
                    "color_order": arguments.color_order,
                    "color_policy": arguments.color_policy,
                    "minimap_margin_px": arguments.minimap_margin_px,
                    "mvs_python_path": arguments.mvs_python_path,
                }
            )
            configuration = camera.calibration
        else:
            print(
                "Warning: diagnostic calibration override bypasses automatic "
                "profile selection.",
                flush=True,
            )
            configuration = json.loads(
                arguments.diagnostic_calibration_override.read_text(encoding="utf-8")
            )
        if arguments.manage_phone_display:
            serial = str(configuration.get("phone", {}).get("serial", "")).strip()
            if serial:
                try:
                    display = PhoneDisplayPowerSession(
                        serial, adb_executable=arguments.adb
                    )
                    print(
                        "Turning on Android display {} (best effort).".format(serial),
                        flush=True,
                    )
                    display.open()
                    if display.last_error:
                        print(
                            "Display wake warning: {}. Camera startup continues."
                            .format(display.last_error),
                            flush=True,
                        )
                except Exception as exc:
                    display = None
                    print(
                        "Display wake warning: {}. Camera startup continues."
                        .format(exc),
                        flush=True,
                    )
        if camera is None:
            camera = open_camera(
                arguments.diagnostic_calibration_override,
                arguments.mvs_python_path,
                rectify=not arguments.no_rectify,
                diagnostic_rig_game_profile_override=(
                    arguments.diagnostic_rig_game_profile_override
                ),
                mode=arguments.mode,
                color_order=arguments.color_order,
                color_policy=arguments.color_policy,
                minimap_margin_px=arguments.minimap_margin_px,
            )
        elif arguments.camera_library == "adapter" and not camera.is_open:
            camera.open()
        if not arguments.gui:
            label = (
                "Native Hikrobot MVS stream"
                if arguments.camera_library == "native"
                else "Rectified stream"
            )
            print(
                "{} configured. Pass --gui to verify live frames.".format(label)
            )
            return 0
        profiled = bool(
            arguments.camera_library == "adapter"
            and (
                arguments.diagnostic_rig_game_profile_override is not None
                or getattr(camera, "config", {}).get("minimap_calibration")
            )
        )
        stream_names = (
            ["full", "minimap"]
            if profiled and arguments.mode == "dual"
            else [arguments.mode if profiled else "full"]
        )
        windows = ["HIK {}".format(name) for name in stream_names]
        print(
            "Live {} stream opened. Press Q/Esc or close a window to exit."
            .format(
                arguments.mode
                if profiled
                else (
                    "native Hikrobot MVS"
                    if arguments.camera_library == "native"
                    else "rectified phone"
                )
            ),
            flush=True,
        )
        for window in windows:
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        telemetry = LiveStreamTelemetry()
        while True:
            read_started_ns = time.perf_counter_ns()
            native_packet = None
            if profiled and arguments.mode == "dual":
                if hasattr(camera, "get_frames"):
                    displayed = camera.get_frames()
                else:
                    frame_set = camera.read_streams()
                    displayed = frame_set.streams
            else:
                if arguments.camera_library == "native":
                    packet = camera.read()
                    native_packet = packet
                    frame = None if packet is None else packet.image
                    displayed = {stream_names[0]: frame} if frame is not None else {}
                elif hasattr(camera, "get_frame"):
                    frame = camera.get_frame()
                    displayed = {stream_names[0]: frame} if frame is not None else {}
                else:
                    ok, frame = camera.read()
                    displayed = {stream_names[0]: frame} if ok and frame is not None else {}
            read_finished_ns = time.perf_counter_ns()
            representative_name = next(iter(displayed), None)
            if representative_name is not None:
                metadata = (
                    {
                        "host_capture_time_ns": int(
                            native_packet.host_capture_time_ns
                        ),
                        "host_timestamp_clock_id": "host_perf_counter_ns",
                    }
                    if native_packet is not None
                    else _last_stream_metadata(camera, representative_name)
                )
                telemetry.observe(
                    read_started_ns, read_finished_ns, metadata
                )
            for name, frame in displayed.items():
                cv2.imshow(
                    "HIK {}".format(name),
                    overlay_stream_telemetry(frame, telemetry),
                )
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                return 0
            if any(
                cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
                for window in windows
            ):
                return 0
    finally:
        try:
            if camera is not None and arguments.camera_library == "native":
                camera.stop()
            elif camera is not None:
                camera.release()
        finally:
            if arguments.gui:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass
            if display is not None:
                print("Turning off Android display.", flush=True)
                display.close()


if __name__ == "__main__":
    raise SystemExit(main())
