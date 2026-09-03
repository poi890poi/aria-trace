"""Configure a HIK camera from a saved calibration and stream rectified frames."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import importlib
import json
import subprocess
import time
from pathlib import Path
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np

from rig_runtime.apps.rig_presentation import console_print as print

from rig_runtime.adapters.hik.compat import Camera, HikCamera
from rig_runtime.adapters.hik.capture import NativeHikFrameSource
from rig_runtime.adapters.hik.driver import HikMvsCameraAdapter, RectifiedHikCamera
from rig_runtime.adapters.hik.game_camera import (
    MinimapRoiUnavailableError,
    ProfiledHikGameCamera,
)
from rig_runtime.adapters.android.phone import AdbPhoneSession
from rig_runtime.adapters.android.phone import resolve_adb_executable
from rig_runtime.adapters.android.game_launcher import (
    foreground_component,
    launch_android_game,
)


def capture_adb_screenshot(
    adb_executable: str, serial: str, timeout_seconds: float = 10.0
):
    """Capture one canonical full ADB raster without operating the phone UI."""

    adb = resolve_adb_executable(adb_executable)
    command = [adb, "-s", str(serial), "exec-out", "screencap", "-p"]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=float(timeout_seconds),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "ADB screencap failed: {}".format(
                completed.stderr.decode("utf-8", errors="replace").strip()
                or "exit {}".format(completed.returncode)
            )
        )
    image = cv2.imdecode(np.frombuffer(completed.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("ADB screencap returned no decodable PNG image")
    return image


def wait_for_foreground_game_surface(
    phone: AdbPhoneSession,
    package: str,
    *,
    timeout_seconds: float = 15.0,
    stable_probes: int = 3,
) -> Mapping[str, object]:
    """Wait until the foreground game's Android surface rotation is stable."""

    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    stable = 0
    last_signature = None
    last_component = None
    last_surface = None
    while time.monotonic() < deadline:
        last_component = foreground_component(phone)
        observed_package = (
            last_component.split("/", 1)[0] if last_component else None
        )
        if observed_package != str(package):
            stable = 0
            time.sleep(0.25)
            continue
        last_surface = phone.capture_surface()
        signature = (
            int(last_surface["quarter_turns_clockwise_from_natural"]),
            tuple(map(int, last_surface["logical_size_px"])),
        )
        stable = stable + 1 if signature == last_signature else 1
        last_signature = signature
        if stable >= max(1, int(stable_probes)):
            return {
                "foreground_component": last_component,
                "foreground_package": observed_package,
                "surface": dict(last_surface),
                "stable_probes": stable,
            }
        time.sleep(0.25)
    raise RuntimeError(
        "Game package {} did not reach a stable foreground Android surface; "
        "last activity {}, last surface {}".format(
            package, last_component or "unknown", last_surface or "unknown"
        )
    )


class LiveStreamTelemetry:
    """Measured GUI throughput and latency without mixing clock domains."""

    def __init__(self, history: int = 60, display_interval_ms: float = 500.0) -> None:
        self._ends = deque(maxlen=max(2, int(history)))
        self._display_interval_ns = max(0, int(float(display_interval_ms) * 1.0e6))
        self._last_display_refresh_ns: Optional[int] = None
        self.fps = 0.0
        self.read_latency_ms = 0.0
        self.frame_age_ms: Optional[float] = None
        self._display_fps = 0.0
        self._display_read_latency_ms = 0.0
        self._display_frame_age_ms: Optional[float] = None

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
        if (
            len(self._ends) <= 2
            or self._last_display_refresh_ns is None
            or finished - self._last_display_refresh_ns >= self._display_interval_ns
        ):
            self._display_fps = self.fps
            self._display_read_latency_ms = self.read_latency_ms
            self._display_frame_age_ms = self.frame_age_ms
            self._last_display_refresh_ns = finished

    def label(self) -> str:
        age = (
            "{:.1f} ms".format(self._display_frame_age_ms)
            if self._display_frame_age_ms is not None
            else "n/a"
        )
        return "FPS {:.1f} avg | read {:.1f} ms | age {}".format(
            self._display_fps, self._display_read_latency_ms, age
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


@dataclass
class GeometryOverlayState:
    enabled: bool = True
    minimap_boundary: bool = True
    cursor: bool = True

    def handle_key(self, key: int) -> Optional[str]:
        if key in (ord("g"), ord("G")):
            self.enabled = not self.enabled
        elif key in (ord("b"), ord("B")):
            self.minimap_boundary = not self.minimap_boundary
            self.enabled = True
        elif key in (ord("c"), ord("C")):
            self.cursor = not self.cursor
            self.enabled = True
        else:
            return None
        return "Geometry overlay: {} (boundary {}, cursor {})".format(
            "on" if self.enabled else "off",
            "on" if self.minimap_boundary else "off",
            "on" if self.cursor else "off",
        )


def _space_matches_frame(geometry: Mapping[str, object], frame) -> bool:
    image_space = geometry.get("image_space")
    if not isinstance(image_space, Mapping):
        return False
    size = image_space.get("stored_size_px")
    return (
        isinstance(size, Sequence)
        and len(size) == 2
        and [int(size[0]), int(size[1])] == [int(frame.shape[1]), int(frame.shape[0])]
    )


def _runtime_geometry(camera, method_name: str, stream_name: str) -> Mapping[str, object]:
    getter = getattr(camera, method_name, None)
    if not callable(getter):
        return {}
    try:
        value = getter(stream_name)
    except (RuntimeError, TypeError, ValueError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def overlay_stream_geometry(
    frame,
    camera,
    stream_name: str,
    state: GeometryOverlayState,
):
    """Draw only geometry explicitly converted into this runtime image space."""

    rendered = frame.copy()
    if not state.enabled:
        return rendered
    if state.minimap_boundary:
        geometry = _runtime_geometry(camera, "get_minimap_geometry", stream_name)
        if (
            geometry.get("available_in_stream_space")
            and _space_matches_frame(geometry, frame)
        ):
            center = tuple(int(round(value)) for value in geometry["center_xy_px"])
            size = geometry.get("boundary_size_xy_px") or [
                2.0 * float(geometry["radius_px"]),
                2.0 * float(geometry["radius_px"]),
            ]
            axes = tuple(max(1, int(round(float(value) / 2.0))) for value in size)
            cv2.ellipse(
                rendered,
                center,
                axes,
                0.0,
                0.0,
                360.0,
                (255, 210, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                rendered,
                "mini-map boundary",
                (max(4, center[0] - axes[0]), max(18, center[1] - axes[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 210, 0),
                1,
                cv2.LINE_AA,
            )
    if state.cursor:
        geometry = _runtime_geometry(camera, "get_cursor_geometry", stream_name)
        if (
            geometry.get("available_in_stream_space")
            and _space_matches_frame(geometry, frame)
        ):
            center = tuple(int(round(value)) for value in geometry["center_xy_px"])
            size = geometry.get("rotating_cursor_envelope_size_xy_px")
            if isinstance(size, Sequence) and len(size) == 2:
                axes = tuple(max(1, int(round(float(value) / 2.0))) for value in size)
                cv2.ellipse(
                    rendered,
                    center,
                    axes,
                    0.0,
                    0.0,
                    360.0,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            arm = 7
            cv2.line(
                rendered,
                (center[0] - arm, center[1]),
                (center[0] + arm, center[1]),
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.line(
                rendered,
                (center[0], center[1] - arm),
                (center[0], center[1] + arm),
                (255, 0, 255),
                2,
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

    def close(self, *, turn_display_off: bool = True) -> None:
        if not self._opened:
            return
        try:
            if turn_display_off:
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
    value.add_argument(
        "--launch-game",
        action="store_true",
        help=(
            "Demo only: wake the calibrated phone, bring the selected game to "
            "foreground, wait for a stable surface, and compose game-up before streaming"
        ),
    )
    value.add_argument(
        "--android-package",
        help="Explicit package used with --launch-game; otherwise profile/game mapping is used",
    )
    value.add_argument("--camera-id")
    value.add_argument("--phone-serial")
    value.add_argument("--color-order", choices=("RGB", "BGR"), default="BGR")
    value.add_argument(
        "--color-policy",
        choices=("auto", "rig_locked", "game_matched", "unadjusted"),
        default="auto",
    )
    value.add_argument("--minimap-margin-px", type=int, default=6)
    value.add_argument(
        "--mask-policy",
        choices=("none", "minimap_circle"),
        default="none",
        help=(
            "Adapter only: precompose the calibrated mini-map circle into the "
            "rectification map (requires rectification and minimap/dual mode)"
        ),
    )
    value.add_argument(
        "--gui",
        action="store_true",
        help=(
            "Show live frames; G/B/C toggles geometry, O corrects game-up "
            "from fresh ADB/HIK evidence, and Q/Esc closes"
        ),
    )
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
    value.add_argument(
        "--adb",
        default="adb",
        help="ADB executable used for display power and GUI O orientation correction",
    )
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
    mask_policy: str = "none",
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
                "mask_policy": mask_policy,
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
            mask_policy=mask_policy,
        ).open()
    return RectifiedHikCamera(
        diagnostic_calibration_override, adapter=adapter, rectify=rectify
    ).open()


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.manage_phone_display and not arguments.gui:
        parser().error("--manage-phone-display requires --gui")
    if arguments.launch_game and not arguments.gui:
        parser().error("--launch-game requires --gui")
    if arguments.mask_policy != "none":
        if arguments.camera_library == "native":
            parser().error(
                "--mask-policy is available only with --camera-library adapter"
            )
        if arguments.mode == "full":
            parser().error("--mask-policy minimap_circle requires minimap or dual mode")
        if arguments.no_rectify:
            parser().error("--mask-policy minimap_circle requires rectification")
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
        if arguments.launch_game:
            parser().error(
                "--launch-game orientation composition requires --camera-library adapter"
            )
    if arguments.launch_game and arguments.diagnostic_calibration_override is not None:
        parser().error(
            "--launch-game requires automatic profile-managed adapter selection"
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
    stream_started = False
    effective_mode = arguments.mode
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
                    "mask_policy": arguments.mask_policy,
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
        if arguments.manage_phone_display or arguments.launch_game:
            serial = str(
                arguments.phone_serial
                or (configuration or {}).get("phone", {}).get("serial", "")
            ).strip()
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
        if arguments.launch_game:
            try:
                serial = str(
                    arguments.phone_serial
                    or (configuration or {}).get("phone", {}).get("serial", "")
                ).strip()
                if not serial:
                    raise RuntimeError(
                        "no phone serial is available from --phone-serial or the active rig"
                    )
                phone = AdbPhoneSession(serial, adb_executable=arguments.adb)
                profile_package = str(
                    (((getattr(camera, "resolved_config", {}).get("context") or {}).get(
                        "game"
                    ) or {}).get("package") or "")
                ).strip()
                profile_game_id = str(
                    (((getattr(camera, "resolved_config", {}).get("context") or {}).get(
                        "game"
                    ) or {}).get("id") or "")
                ).strip()
                explicit_package = arguments.android_package or profile_package or None
                selected_game_id = (
                    arguments.game_id
                    or profile_game_id
                    or arguments.android_package
                )
                if not selected_game_id:
                    raise RuntimeError(
                        "no game is selected by auto configuration, --game-id, "
                        "or --android-package"
                    )
                launch = launch_android_game(
                    phone,
                    selected_game_id,
                    explicit_package=explicit_package,
                )
                package = str(launch.get("package") or "").strip()
                if not package:
                    raise RuntimeError(
                        "no Android package is known for {!r}; pass --android-package"
                        .format(selected_game_id)
                    )
                ready = wait_for_foreground_game_surface(phone, package)
                compose = getattr(
                    camera, "correct_game_orientation_from_android_surface", None
                )
                if not callable(compose):
                    raise RuntimeError(
                        "the loaded camera adapter has no Android-surface "
                        "orientation composition function"
                    )
                orientation = compose(
                    ready["surface"]["quarter_turns_clockwise_from_natural"],
                    foreground_package=ready["foreground_package"],
                )
                print(
                    "Game ready: {} at Android surface {} degrees; adapter game-up "
                    "is {} degrees clockwise from rig-calibration-display-up."
                    .format(
                        ready["foreground_package"],
                        int(ready["surface"][
                            "quarter_turns_clockwise_from_natural"
                        ]) * 90,
                        int(orientation[
                            "selected_camera_adapter_image_degrees_clockwise_from_calibration_display"
                        ]),
                    ),
                    flush=True,
                )
                for warning in orientation.get("warnings") or []:
                    print("Orientation warning: {}".format(warning), flush=True)
            except Exception as exc:
                print(
                    "Game preparation/orientation skipped: {}: {}. The demo will "
                    "continue with the saved adapter orientation; press O later "
                    "for four-orientation ADB/HIK image verification."
                    .format(type(exc).__name__, exc),
                    flush=True,
                )
        if camera is None:
            try:
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
                    mask_policy=arguments.mask_policy,
                )
            except MinimapRoiUnavailableError as exc:
                if arguments.mode == "full" or arguments.camera_library != "adapter":
                    raise
                print(
                    "Mini-map ROI unavailable after checking all four game "
                    "orientations: {}. Continuing the demo with the full "
                    "rig-calibrated phone stream; mini-map output is skipped."
                    .format(exc),
                    flush=True,
                )
                camera = open_camera(
                    arguments.diagnostic_calibration_override,
                    arguments.mvs_python_path,
                    rectify=not arguments.no_rectify,
                    diagnostic_rig_game_profile_override=(
                        arguments.diagnostic_rig_game_profile_override
                    ),
                    mode="full",
                    color_order=arguments.color_order,
                    color_policy=arguments.color_policy,
                    minimap_margin_px=arguments.minimap_margin_px,
                    mask_policy="none",
                )
                effective_mode = "full"
        elif arguments.camera_library == "adapter" and not camera.is_open:
            try:
                camera.open()
            except MinimapRoiUnavailableError as exc:
                if arguments.mode == "full":
                    raise
                print(
                    "Mini-map ROI unavailable after checking all four game "
                    "orientations: {}. Continuing the demo with the full "
                    "rig-calibrated phone stream; mini-map output is skipped."
                    .format(exc),
                    flush=True,
                )
                camera.config["mode"] = "full"
                adapter_plan = getattr(camera, "resolved_config", {}).get(
                    "adapter_plan"
                )
                if isinstance(adapter_plan, dict):
                    adapter_plan["mode"] = "full"
                camera.open()
                effective_mode = "full"
        stream_started = True
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
            if profiled and effective_mode == "dual"
            else [effective_mode if profiled else "full"]
        )
        windows = ["HIK {}".format(name) for name in stream_names]
        print(
            "Live {} stream opened. G toggles geometry, B boundary, C cursor; "
            "O matches all four orientations against a fresh ADB screenshot; "
            "Q/Esc or close a window exits."
            .format(
                effective_mode
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
        geometry_overlay = GeometryOverlayState()
        while True:
            read_started_ns = time.perf_counter_ns()
            native_packet = None
            if profiled and effective_mode == "dual":
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
                rendered = overlay_stream_geometry(
                    frame, camera, name, geometry_overlay
                )
                cv2.imshow(
                    "HIK {}".format(name),
                    overlay_stream_telemetry(rendered, telemetry),
                )
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                return 0
            if key in (ord("o"), ord("O")):
                correction = getattr(camera, "correct_game_orientation", None)
                if arguments.camera_library != "adapter" or not callable(correction):
                    print(
                        "Orientation correction unavailable: use the IRIS adapter, "
                        "not the native HIK stream.",
                        flush=True,
                    )
                elif "full" not in displayed:
                    print(
                        "Orientation correction needs the full stream; restart the "
                        "demo with --mode full or --mode dual.",
                        flush=True,
                    )
                else:
                    serial = str(
                        arguments.phone_serial
                        or (configuration or {}).get("phone", {}).get("serial", "")
                    ).strip()
                    if not serial:
                        print(
                            "Orientation correction needs --phone-serial because "
                            "the active rig calibration has no phone serial.",
                            flush=True,
                        )
                    else:
                        try:
                            print(
                                "Checking four game-up orientations from fresh "
                                "ADB/HIK image evidence...",
                                flush=True,
                            )
                            adb_image = capture_adb_screenshot(arguments.adb, serial)
                            result = correction(adb_image, displayed["full"])
                            if result.get("applied"):
                                print(
                                    "Game-up corrected to {} degrees clockwise "
                                    "from calibration-display-up (confidence {:.3f}, "
                                    "margin {:.3f}); rectification maps rebuilt."
                                    .format(
                                        int(result[
                                            "selected_camera_adapter_image_degrees_clockwise_from_calibration_display"
                                        ]),
                                        float(result["selected_confidence"]),
                                        float(result.get("confidence_margin") or 0.0),
                                    ),
                                    flush=True,
                                )
                            else:
                                print(
                                    "Game-up was not changed: image evidence is "
                                    "ambiguous (confidence {:.3f}, margin {:.3f}; "
                                    "required {:.3f}/{:.3f})."
                                    .format(
                                        float(result["selected_confidence"]),
                                        float(result.get("confidence_margin") or 0.0),
                                        float(result["preferred_confidence"]),
                                        float(result["preferred_margin"]),
                                    ),
                                    flush=True,
                                )
                        except Exception as exc:
                            print(
                                "Orientation correction failed: {}: {}".format(
                                    type(exc).__name__, exc
                                ),
                                flush=True,
                            )
            message = geometry_overlay.handle_key(key)
            if message:
                print(message, flush=True)
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
                if stream_started:
                    print("Turning off Android display.", flush=True)
                else:
                    print(
                        "Demo startup did not complete; leaving Android display "
                        "power unchanged.",
                        flush=True,
                    )
                display.close(turn_display_off=stream_started)


if __name__ == "__main__":
    raise SystemExit(main())
