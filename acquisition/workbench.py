"""Zero-interruption acquisition workbench for repeated route demonstrations."""

import argparse
import json
import math
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import urlopen

import cv2

from replay.alignment import align_session
from replay.package import compile_replay_package
from replay.route_tracking import RouteTrackingPackage
from replay.route_similarity import write_live_route_similarity

from .annotations import AnnotationStore
from .android_capture import (
    AndroidRoiFrameSource,
    AndroidRoiSpec,
    ScrcpyCaptureHub,
    find_scrcpy_server,
)
from .map_stitching import load_localization_reference_candidates, stitch_map_session
from .map_layers import LayeredGlobalLocalizer, build_map_atlas
from .map_layer_references import transition_endpoint_references
from .minimap_transition_analysis import analyze_transition_session
from .cursor_pose import CursorPoseEstimator
from .frame_pump import LatestFramePump
from .hik_capture import CalibratedHikFrameSource, NativeHikFrameSource
from .live_tracker import (
    GlobalMapLocalizer,
    MinimapExtractor,
    TwoRateRealtimeTracker,
    render_map_overlay,
    render_minimap_route_overlay,
)
from .live_tracking_evidence import LiveTrackingEvidenceRecorder
from .minimap_calibration import (
    ORDINARY_MOTION_SEGMENT_LABELS,
    calibrate_segment_sessions,
    calibrate_session,
)
from .minimap_verification import verify_forward_session
from .poc_evidence import build_poc_evidence_index
from .profiles import ProfileCatalog
from .recorder import AcquisitionRecorder
from .route_compilation import compile_route_session
from .route_tracker import RouteCandidateAdvisor
from .session import SessionReader, input_capture_health
from .scene_yaw_calibration import calibrate_scene_yaw_session
from .sources import (
    AdbClockMapper,
    AdbGetEventSource,
    AdbScreenshotFrameSource,
    OpenCvCameraFrameSource,
)
from .tracking_profiles import resolve_tracking_profile
from .teleport_analysis import analyze_teleport_session
from .windows import (
    WindowsDesktopApi,
    WindowsKeyboardMouseSource,
    WindowsRawKeyboardMouseSource,
    WindowsWindowFrameSource,
    WindowsXInputSource,
    select_window,
)


WORKBENCH_SERVICE = "aria-trace-workbench"


def _strict_json_value(value):
    """Return browser-compatible JSON data without NaN or infinity tokens."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    return value


def _require_ready_map_localization(stitch: dict, action: str) -> dict:
    """Return a usable localization layer or explain its recorded quality."""

    localization = stitch.get("localization") or {}
    if localization.get("status") == "ready":
        return localization
    quality = localization.get("quality") or {}
    details = []
    for key, label, digits, suffix in (
        ("gradient_correlation_score", "correlation", 3, ""),
        ("gradient_correlation_margin", "margin", 3, ""),
        ("reprojection_p95_original_map_px", "reprojection p95", 2, " px"),
    ):
        value = quality.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            details.append(
                "{} {:.{}f}{}".format(label, float(value), digits, suffix)
            )
    status = str(localization.get("status") or "missing").replace("_", " ")
    summary = " ({})".format(", ".join(details)) if details else ""
    raise ValueError(
        "The selected map stitch localization is {}{}; rebuild it with compatible "
        "mini-map evidence before {}".format(status, summary, action)
    )


def parse_adb_devices(output: str) -> List[dict]:
    """Parse `adb devices -l` without treating unavailable devices as targets."""
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


def is_client_disconnect(exc: BaseException) -> bool:
    """Return whether a request ended because the HTTP client went away."""
    return isinstance(
        exc,
        (BrokenPipeError, ConnectionAbortedError, ConnectionResetError),
    )


class WorkbenchHttpServer(ThreadingHTTPServer):
    """Threaded HTTP server that treats browser disconnects as routine."""

    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        if is_client_disconnect(sys.exc_info()[1]):
            return
        super().handle_error(request, client_address)


def _connect_host(host: str) -> str:
    return "127.0.0.1" if host in ("", "0.0.0.0", "::") else host


def discover_workbench_instance(host: str, port: int, timeout: float = 0.75):
    """Identify a Workbench already listening on host/port, including old builds."""
    base = "http://{}:{}".format(_connect_host(host), int(port))
    try:
        with urlopen(base + "/api/instance", timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
        if value.get("service") == WORKBENCH_SERVICE:
            return value
    except HTTPError as exc:
        if exc.code != 404:
            return None
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return None

    # Workbench builds predating /api/instance can still be distinguished from
    # unrelated services so a duplicate launch produces useful guidance. Check
    # the static shell before the more expensive full-state descriptor.
    try:
        with urlopen(base + "/", timeout=timeout) as response:
            body = response.read(16384).decode("utf-8", errors="replace")
        if "<title>AriaTrace Recorder</title>" in body:
            return {
                "service": WORKBENCH_SERVICE,
                "legacy": True,
                "url": base + "/",
            }
    except (OSError, URLError, ValueError):
        return None

    try:
        with urlopen(base + "/api/state", timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
        if "session_labels" in value and "sources" in value:
            return {
                "service": WORKBENCH_SERVICE,
                "legacy": True,
                "url": base + "/",
            }
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return None
    return None


def occupied_port_message(host: str, port: int, existing) -> str:
    endpoint = "http://{}:{}/".format(_connect_host(host), int(port))
    if not existing:
        return (
            "Cannot start the Workbench at {} because the address is already in "
            "use by another process. This command did not replace or stop it."
        ).format(endpoint)
    if existing.get("legacy"):
        return (
            "An older AriaTrace Workbench is already running at {}. Stop it with "
            "Ctrl+C in the terminal that started it, then launch this version. "
            "This command did not replace or stop it."
        ).format(endpoint)
    details = ["PID {}".format(existing.get("process_id", "unknown"))]
    if existing.get("started_utc"):
        details.append("started {}".format(existing["started_utc"]))
    if existing.get("session_root"):
        details.append("sessions {}".format(existing["session_root"]))
    return (
        "AriaTrace Workbench instance {instance_id} is already running at {url} "
        "({details}). Stop it with Ctrl+C in its owning terminal if you intend to "
        "restart it. This command did not replace or stop it."
    ).format(
        instance_id=existing.get("instance_id", "unknown"),
        url=existing.get("url") or endpoint,
        details=", ".join(details),
    )


def safe_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", (value or "").strip()).strip("-.")
    if not normalized:
        raise ValueError("An experiment ID is required")
    return normalized[:80]


def nearest_frame_index(frames: List[dict], host_time_ns: int) -> int:
    if not frames:
        raise ValueError("Cannot map an event without recorded frames")
    return min(
        range(len(frames)),
        key=lambda index: abs(
            int(frames[index]["host_capture_time_ns"]) - int(host_time_ns)
        ),
    )


def input_packet_is_active(packet) -> bool:
    """Return whether one live input packet represents intentional control."""
    payload = packet.payload or {}
    if not payload.get("foreground", True):
        return False
    if packet.kind == "pc_raw_keyboard":
        # Raw Input can report injected/synthetic packets with no device and
        # key-release traffic from switching back to the game. Neither is a
        # physical press, so do not let it start the recording clock.
        if not int(payload.get("device_handle") or 0):
            return False
        if not payload.get("pressed", False):
            return False
        return True
    if packet.kind == "pc_raw_mouse":
        if not int(payload.get("device_handle") or 0):
            return False
        return bool(
            int(payload.get("delta_x", 0))
            or int(payload.get("delta_y", 0))
            or payload.get("button_transitions")
            or int(payload.get("wheel_delta", 0))
            or int(payload.get("horizontal_wheel_delta", 0))
        )
    if packet.kind == "pc_input_state":
        return bool(payload.get("keys") or payload.get("mouse_buttons"))
    if packet.kind == "pc_xinput_state":
        state = payload.get("state") or {}
        axes = list(state.get("left_stick") or []) + list(
            state.get("right_stick") or []
        )
        return bool(
            state.get("buttons")
            or float(state.get("left_trigger", 0.0)) > 0.02
            or float(state.get("right_trigger", 0.0)) > 0.02
            or any(abs(float(value)) > 0.08 for value in axes)
        )
    return not packet.kind.endswith("_error")


def capture_diagnostic_snapshot(reader, input_health: dict, active: dict) -> dict:
    """Summarize a capture before a failed session directory is discarded."""
    manifest = reader.manifest
    inputs = list(reader.inputs)
    raw_input = None
    for source in manifest.get("input_sources") or ():
        candidate = source.get("raw_input_diagnostics")
        if candidate is not None:
            raw_input = dict(candidate)
            break
    foreground_events = 0
    background_events = 0
    physical_device_handles = set()
    for event in inputs:
        payload = event.get("payload") or {}
        if payload.get("foreground", True):
            foreground_events += 1
        else:
            background_events += 1
        device_handle = int(payload.get("device_handle") or 0)
        if device_handle:
            physical_device_handles.add(device_handle)
    frames = sum(len(items) for items in reader.frames_by_stream.values())
    recording_start = manifest.get("recording_start") or {}
    return {
        "attempted_utc": datetime.now(timezone.utc).isoformat(),
        "run_index": int(active["run_index"]),
        "status": manifest.get("status"),
        "recording_started": bool(recording_start.get("started")),
        "first_input_kind": active.get("first_input_kind"),
        "duration_s": round(float(manifest.get("duration_ns") or 0) / 1.0e9, 3),
        "frames": frames,
        "persisted_input_events": len(inputs),
        "foreground_input_events": foreground_events,
        "background_input_events": background_events,
        "physical_device_handles": sorted(physical_device_handles),
        "raw_input": raw_input,
        "input_capture": input_health,
    }


def input_failure_message(diagnostics: dict) -> str:
    """Explain an input failure using measured receiver and event evidence."""
    health = diagnostics["input_capture"]
    raw = diagnostics.get("raw_input") or {}
    evidence = health.get("evidence") or {}
    missing = ", ".join(health.get("missing") or ()) or "required gameplay input"
    if health.get("adapter") != "windows_raw_keyboard_mouse":
        return (
            "Input capture failed validation for {} (missing: {}). Persisted {}, "
            "meaningful {}, error events {}. The failed session was discarded."
        ).format(
            health.get("adapter") or "unknown adapter",
            missing,
            int(diagnostics.get("persisted_input_events") or 0),
            int(evidence.get("meaningful_events") or 0),
            int(health.get("error_events") or 0),
        )
    return (
        "Input capture failed validation (missing: {}). Raw Input received {}, "
        "accepted {}, persisted {}, physical {}, meaningful {}, foreground {}. "
        "The failed session was discarded."
    ).format(
        missing,
        int(raw.get("packets_received") or 0),
        int(raw.get("packets_accepted") or 0),
        int(diagnostics.get("persisted_input_events") or 0),
        int(evidence.get("physical_events") or 0),
        int(evidence.get("meaningful_events") or 0),
        int(diagnostics.get("foreground_input_events") or 0),
    )


def automatic_take_bounds(
    frames: List[dict],
    inputs: List[dict],
    fallback_host_time_ns: Optional[int] = None,
):
    """Find the first observed human control; retain the complete captured tail."""
    if not frames:
        raise ValueError("A take contains no frames")
    active_times = []
    previous_cursor = None
    for event in inputs:
        payload = event.get("payload") or {}
        if not payload.get("foreground", True):
            continue
        kind = event.get("kind")
        active = False
        if kind == "pc_input_state":
            cursor = tuple(payload.get("cursor_client") or ())
            active = bool(payload.get("keys") or payload.get("mouse_buttons"))
            if previous_cursor is not None and cursor and cursor != previous_cursor:
                active = True
            if cursor:
                previous_cursor = cursor
        elif kind == "pc_xinput_state":
            state = payload.get("state") or {}
            axes = list(state.get("left_stick") or []) + list(
                state.get("right_stick") or []
            )
            active = bool(
                state.get("buttons")
                or float(state.get("left_trigger", 0.0)) > 0.02
                or float(state.get("right_trigger", 0.0)) > 0.02
                or any(abs(float(value)) > 0.08 for value in axes)
            )
        elif kind == "pc_raw_mouse":
            active = bool(
                int(payload.get("delta_x", 0))
                or int(payload.get("delta_y", 0))
                or payload.get("button_transitions")
                or int(payload.get("wheel_delta", 0))
                or int(payload.get("horizontal_wheel_delta", 0))
            )
        elif kind == "pc_raw_keyboard":
            active = True
        elif kind not in (
            "pc_input_error",
            "pc_xinput_error",
            "pc_raw_input_error",
        ):
            active = True
        if active:
            active_times.append(int(event["host_time_ns"]))
    if active_times:
        start_index = nearest_frame_index(frames, active_times[0])
    elif fallback_host_time_ns is not None:
        start_index = nearest_frame_index(frames, fallback_host_time_ns)
    else:
        start_index = 0
    return start_index, len(frames) - 1


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


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
            "label": "HIK camera (rig calibrated)",
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
    ) -> None:
        self.desktop_api = desktop_api
        self.xinput_api = xinput_api
        self.raw_input_api = raw_input_api
        self.hik_adapter_factory = hik_adapter_factory

    @staticmethod
    def _adb(config: dict) -> Path:
        value = config.get("adb") or shutil.which("adb")
        if value:
            return Path(value)
        bundled = (
            Path(__file__).resolve().parents[1]
            / ".tools"
            / "scrcpy-win64-v4.1"
            / ("adb.exe" if os.name == "nt" else "adb")
        )
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
            from .rig_calibration.hik.driver import HikMvsCameraAdapter

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
        """Build a synchronized frame/input pair for one recorder lifetime."""
        if frame_config.get("adapter") != "android_scrcpy":
            return self.frame(frame_config), self.input(input_config)
        serial = str(frame_config.get("serial") or "").strip()
        if not serial:
            raise ValueError("Choose a connected Android device")
        adb = self._adb(frame_config)
        clock = AdbClockMapper(adb, serial)
        server_config = frame_config.get("scrcpy_server")
        bundled_server = (
            Path(__file__).resolve().parents[1]
            / ".tools"
            / "scrcpy-win64-v4.1"
            / "scrcpy-server"
        )
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

class AcquisitionWorkbench:
    INPUT_SETTLE_DELAY_S = 1.5
    STATE_SCHEMA_VERSION = "1.0"
    CAPTURE_KINDS = (
        "route",
        "game_profile",
        "game_behavior",
        "full_map",
        "minimap_calibration",
        "minimap_cruise",
        "scene_yaw_calibration",
    )
    SESSION_LABELS = (
        {"value": "", "label": "Unlabeled"},
        {
            "value": "ordinary_cruise",
            "label": "Ordinary movement + camera",
            "capture_kind": "game_profile",
            "workflow_stage_id": "control-cruise",
            "capture_id": "genshin-control-cruise",
        },
        {
            "value": "rotation_only",
            "label": "Rotate camera, stand still",
            "capture_kind": "minimap_calibration",
            "workflow_stage_id": "minimap-rotation-only",
            "capture_id": "genshin-minimap-rotation-only",
        },
        {
            "value": "scene_rotation_360",
            "label": "Slow horizontal scene turn (360°+)",
            "capture_kind": "scene_yaw_calibration",
            "workflow_stage_id": "scene-rotation-360",
            "capture_id": "genshin-scene-rotation-360",
        },
        {
            "value": "movement_only",
            "label": "Move without turning camera",
            "capture_kind": "minimap_calibration",
            "workflow_stage_id": "minimap-movement-only",
            "capture_id": "genshin-minimap-movement-only",
        },
        {
            "value": "forward_no_turn",
            "label": "Straight forward, no camera turn",
            "capture_kind": "minimap_calibration",
            "workflow_stage_id": "minimap-forward-no-turn",
            "capture_id": "genshin-minimap-forward-no-turn",
        },
        {
            "value": "full_map",
            "label": "Full-map coverage",
            "capture_kind": "full_map",
            "workflow_stage_id": "full-map",
            "capture_id": "genshin-full-map",
        },
        {
            "value": "teleportation",
            "label": "Teleport behavior",
            "capture_kind": "game_behavior",
            "workflow_stage_id": "teleport-behavior",
            "capture_id": "genshin-teleport-behavior",
        },
        {
            "value": "minimap_transition",
            "label": "Mini-map scale/detail transition",
            "capture_kind": "game_behavior",
            "workflow_stage_id": "minimap-transition",
            "capture_id": "genshin-minimap-transition",
        },
        {
            "value": "route",
            "label": "Route demonstration",
            "capture_kind": "route",
            "workflow_stage_id": "route",
            "capture_id": "genshin-poc-short-route",
        },
    )
    SESSION_METADATA_FILENAME = "session_metadata.json"

    def __init__(
        self,
        session_root: Path,
        artifact_root: Path,
        profiles: Optional[ProfileCatalog] = None,
        desktop_api=None,
        xinput_api=None,
        raw_input_api=None,
        folder_opener=None,
    ) -> None:
        self.session_root = Path(session_root)
        self.artifact_root = Path(artifact_root)
        self.session_root.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.profiles = profiles or ProfileCatalog()
        self.desktop_api = desktop_api
        self._folder_opener = folder_opener
        self.sources = SourceFactory(
            desktop_api=desktop_api,
            xinput_api=xinput_api,
            raw_input_api=raw_input_api,
        )
        self._lock = threading.RLock()
        self._armed = None
        self._active = None
        self._last_error = None
        self._last_capture_diagnostics = None
        self._compile_state = "not_ready"
        self._compile_result = None
        self._analysis_jobs = {}
        self._live_tracker = None
        self._live_tracker_engine = None
        self._live_tracker_mosaic = None
        self._live_tracker_route_points = None
        self._android_control = None
        self._hud_notice = None
        self._session_summary_cache = {}
        self._instance = {
            "service": WORKBENCH_SERVICE,
            "schema_version": "1.0",
            "instance_id": uuid.uuid4().hex[:12],
            "process_id": os.getpid(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "workspace_root": str(Path(__file__).resolve().parents[1]),
            "session_root": str(self.session_root.resolve()),
            "artifact_root": str(self.artifact_root.resolve()),
            "host": None,
            "port": None,
            "url": None,
            "lifecycle_owner": "terminal",
        }
        self._hud_runtime = {
            "available": False,
            "enabled": False,
            "capture_exclusion": False,
            "error": None,
        }
        self._hud_toggle = None
        if not self._restore_state_file():
            self._armed = self._recover_latest_experiment()
            if self._armed is not None:
                self._persist_state()
        for game_profile_id, game in self.profiles.games.items():
            if game.get("poc_workflow"):
                self._refresh_poc_evidence_index(game_profile_id)

    def configure_server_endpoint(self, host: str, port: int) -> None:
        """Publish the bound endpoint without transferring lifecycle ownership."""
        with self._lock:
            self._instance.update(
                {
                    "host": host,
                    "port": int(port),
                    "url": "http://{}:{}/".format(_connect_host(host), int(port)),
                }
            )

    def instance_descriptor(self) -> dict:
        with self._lock:
            return dict(self._instance)

    def _state_path(self) -> Path:
        return self.artifact_root / "workbench_state.json"

    def _persist_state(self) -> None:
        _write_json_atomic(
            self._state_path(),
            {
                "schema_version": self.STATE_SCHEMA_VERSION,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "armed": self._armed,
            },
        )

    def _restore_state_file(self) -> bool:
        path = self._state_path()
        if not path.is_file():
            return False
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            armed = value.get("armed")
            if armed is not None:
                experiment_id = safe_id(armed.get("experiment_id"))
                if experiment_id != armed.get("experiment_id"):
                    raise ValueError("Persisted experiment ID is not canonical")
                if not isinstance(armed.get("frame_source"), dict):
                    raise ValueError("Persisted frame source is missing")
                if not isinstance(armed.get("input_source"), dict):
                    raise ValueError("Persisted input source is missing")
                stage = armed.get("workflow_stage") or {}
                armed.setdefault("segment_label", stage.get("segment_label"))
                armed.setdefault(
                    "segment_semantics", stage.get("segment_semantics")
                )
                armed.setdefault(
                    "start_trigger", stage.get("start_trigger") or "first_input"
                )
                armed.setdefault(
                    "input_requirement",
                    stage.get("input_requirement")
                    or (
                        "required"
                        if armed["start_trigger"] == "first_input"
                        else "optional"
                    ),
                )
                armed.setdefault(
                    "start_delay_s",
                    float(stage.get("start_delay_s") or self.INPUT_SETTLE_DELAY_S),
                )
            self._armed = armed
            return True
        except Exception as exc:
            self._last_error = "Could not restore workbench state: {}: {}".format(
                type(exc).__name__, exc
            )
            return False

    def _recover_latest_experiment(self) -> Optional[dict]:
        candidates = [
            path
            for path in self.session_root.glob("*/run_*/manifest.json")
            if re.fullmatch(r"run_\d+", path.parent.name)
        ]
        if not candidates:
            return None
        manifest_path = max(candidates, key=lambda path: path.stat().st_mtime_ns)
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            context = manifest.get("context") or {}
            experiment_id = safe_id(context.get("experiment_id"))
            if experiment_id != manifest_path.parent.parent.name:
                return None
            game_id = context.get("game_profile_id")
            route_profile_id = context.get("route_profile_id")
            game = self.profiles.game(game_id) if game_id else None
            route = self.profiles.route(route_profile_id) if route_profile_id else None
            stage_id = context.get("workflow_stage_id")
            stage = next(
                (
                    dict(item)
                    for item in (game or {}).get("poc_workflow", [])
                    if item.get("stage_id") == stage_id
                ),
                context.get("workflow_stage"),
            )
            frame_description = (manifest.get("frame_sources") or [{}])[0]
            input_description = (manifest.get("input_sources") or [{}])[0]
            frame_config = dict((game or {}).get("default_frame_source") or {})
            frame_config["adapter"] = context.get("frame_adapter") or frame_config.get(
                "adapter", "windows_window"
            )
            window_title = frame_description.get(
                "matched_window_title"
            ) or frame_description.get("window_title_query")
            if window_title:
                frame_config.update(window_title=window_title, exact_title=True)
            if frame_description.get("requested_fps"):
                frame_config["fps"] = frame_description["requested_fps"]
            input_config = dict((game or {}).get("default_input_source") or {})
            input_config["adapter"] = context.get("input_adapter") or input_config.get(
                "adapter", "none"
            )
            if window_title and input_config["adapter"].startswith("windows_"):
                input_config.update(window_title=window_title, exact_title=True)
            if input_description.get("poll_hz"):
                input_config["poll_hz"] = input_description["poll_hz"]
            target_runs = int(
                (stage or {}).get("target_runs")
                or (route or {}).get("target_runs")
                or context.get("run_index")
                or 1
            )
            capture_duration_s = float(
                (stage or {}).get("capture_duration_s")
                or (route or {}).get("capture_duration_s")
                or max(5.0, manifest.get("duration_ns", 0) / 1.0e9)
            )
            capture_kind = context.get("capture_kind") or "route"
            capture_id = safe_id(
                context.get("capture_id") or context.get("route_id")
            )
            return {
                "experiment_id": experiment_id,
                "game_profile_id": game_id,
                "route_profile_id": route_profile_id,
                "route_id": context.get("route_id") or capture_id,
                "capture_kind": capture_kind,
                "capture_id": capture_id,
                "workflow_stage_id": stage_id,
                "workflow_stage": stage,
                "segment_label": context.get("segment_label")
                or (stage or {}).get("segment_label"),
                "segment_semantics": context.get("segment_semantics")
                or (stage or {}).get("segment_semantics"),
                "start_trigger": context.get("start_trigger")
                or (stage or {}).get("start_trigger")
                or "first_input",
                "input_requirement": context.get("input_requirement")
                or (stage or {}).get("input_requirement")
                or "required",
                "start_delay_s": float(
                    context.get("start_delay_s")
                    or (stage or {}).get("start_delay_s")
                    or self.INPUT_SETTLE_DELAY_S
                ),
                "game_profile_draft": context.get("game_profile_draft"),
                "confirmation_label": (stage or {}).get("confirmation_label")
                or ("full route boundary" if capture_kind == "route" else "useful capture"),
                "target_runs": target_runs,
                "capture_duration_s": capture_duration_s,
                "frame_source": frame_config,
                "input_source": input_config,
                "armed_utc": context.get("created_utc") or manifest.get("created_utc"),
                "restored_from_session": str(manifest_path.parent),
            }
        except Exception as exc:
            self._last_error = "Could not recover latest experiment: {}: {}".format(
                type(exc).__name__, exc
            )
            return None

    def set_hud_runtime(
        self,
        enabled: bool,
        capture_exclusion: bool = False,
        error: Optional[str] = None,
        available: Optional[bool] = None,
    ) -> None:
        with self._lock:
            self._hud_runtime = {
                "available": (
                    bool(available)
                    if available is not None
                    else bool(self._hud_runtime.get("available"))
                ),
                "enabled": bool(enabled),
                "capture_exclusion": bool(capture_exclusion),
                "error": error or None,
            }

    def configure_hud_control(self, toggle) -> None:
        """Attach the process-level HUD controller to the HTTP workbench state."""
        with self._lock:
            self._hud_toggle = toggle
            self._hud_runtime["available"] = toggle is not None

    def set_hud_enabled(self, enabled: bool) -> dict:
        with self._lock:
            toggle = self._hud_toggle
        if toggle is None:
            raise RuntimeError("The in-game overlay is unavailable")
        try:
            toggle(bool(enabled))
        except Exception as exc:
            self.set_hud_runtime(
                enabled=False,
                capture_exclusion=False,
                error="{}: {}".format(type(exc).__name__, exc),
                available=True,
            )
            raise RuntimeError("Could not change the overlay: {}".format(exc))
        self.set_hud_runtime(
            enabled=bool(enabled),
            capture_exclusion=bool(enabled),
            available=True,
        )
        return self.descriptor()

    def hud_descriptor(self) -> dict:
        """Return a lightweight status contract for the in-game overlay."""
        with self._lock:
            armed = self._armed
            active = self._active
            notice = self._hud_notice
            tracker = self._live_tracker
            if tracker and tracker.get("status") in ("starting", "running"):
                latest = tracker.get("latest") or {}
                pose = latest.get("pose") or {}
                global_fix = latest.get("global_fix") or {}
                detail = "Searching the observed map for the first absolute fix."
                if pose:
                    detail = (
                        "x {x:.1f} · y {y:.1f} · yaw {yaw:+.1f}° · "
                        "global {global_score:.3f} · local {local_score:.3f}"
                    ).format(
                        x=float(pose.get("x") or 0),
                        y=float(pose.get("y") or 0),
                        yaw=float(pose.get("yaw_deg") or 0),
                        global_score=float(global_fix.get("score") or 0),
                        local_score=float(
                            (latest.get("local_motion") or {}).get("response") or 0
                        ),
                    )
                descriptor = {
                    "visible": True,
                    "title": "ARIATRACE LIVE TRACKER",
                    "window_title": (tracker.get("frame_source") or {}).get(
                        "window_title"
                    ),
                    "state": "live_tracker",
                    "status": str(latest.get("mode") or "LOCALIZING"),
                    "detail": detail,
                    "color": "#72dfa3" if pose else "#ffd166",
                    "map_overlay_url": "/api/tracker/overlay?compact=1",
                }
                if (
                    pose
                    and self._live_tracker_route_points is not None
                    and tracker.get("frame_size_wh")
                    and tracker.get("minimap_crop_xywh")
                ):
                    descriptor.update(
                        {
                            "minimap_route_overlay_url": (
                                "/api/tracker/minimap-route-overlay"
                            ),
                            "minimap_crop_xywh": list(
                                tracker["minimap_crop_xywh"]
                            ),
                            "frame_size_wh": list(tracker["frame_size_wh"]),
                            "minimap_route_role": "visual-guidance-only",
                        }
                    )
                return descriptor
            if armed is None:
                return {"visible": False}
            stage = armed.get("workflow_stage") or {}
            stage_name = stage.get("display_name") or armed.get("capture_id")
            run_index = active.get("run_index") if active else (
                notice or {}
            ).get("run_index")
            title = "ARIATRACE"
            if run_index:
                title += " · SESSION {}/{}".format(
                    run_index, int(armed.get("target_runs") or 1)
                )
            title += " · {}".format(stage_name)
            common = {
                "visible": True,
                "title": title.upper(),
                "window_title": armed.get("frame_source", {}).get("window_title"),
            }
            if active is not None:
                if active["phase"] == "arming_sources":
                    common.update(
                        {
                            "state": "arming_sources",
                            "status": "ARMING",
                            "detail": "Preparing frame and input capture.",
                            "color": "#ffd166",
                        }
                    )
                    return common
                if active["phase"] == "settling_queue_input":
                    remaining_s = max(
                        0.0,
                        (
                            active["input_eligible_host_time_ns"]
                            - time.perf_counter_ns()
                        )
                        / 1.0e9,
                    )
                    common.update(
                        {
                            "state": "settling_queue_input",
                            "status": "SWITCH TO GAME",
                            "detail": (
                                "Recording starts automatically in {:.1f}s".format(
                                    remaining_s
                                )
                                if armed.get("start_trigger") == "settled_timer"
                                else "Ignoring queue-click input · ready in {:.1f}s".format(
                                    remaining_s
                                )
                            ),
                            "color": "#ffd166",
                        }
                    )
                    return common
                if active["phase"] == "waiting_for_first_input":
                    common.update(
                        {
                            "state": "waiting_for_first_input",
                            "status": "PLAY TO START",
                            "detail": "Begin the sample naturally; the first control becomes time zero.",
                            "color": "#59d7e8",
                        }
                    )
                    return common
                if active["phase"] == "finalizing_capture":
                    common.update(
                        {
                            "state": "finalizing_capture",
                            "status": "CAPTURE ENDED",
                            "detail": "Finalizing files; wait for the completion signal.",
                            "color": "#ffd166",
                        }
                    )
                    return common
                deadline = active.get("recording_deadline_host_time_ns")
                remaining_s = (
                    max(0, math.ceil((deadline - time.perf_counter_ns()) / 1.0e9))
                    if deadline
                    else 0
                )
                common.update(
                    {
                        "state": "recording",
                        "remaining_s": remaining_s,
                        "status": "REC · {:02d}:{:02d}".format(
                            remaining_s // 60,
                            remaining_s % 60,
                        ),
                        "detail": "Inputs recorded: {} · keep playing naturally.".format(
                            active.get("recorded_input_events", 0)
                        ),
                        "color": "#ff7b84",
                    }
                )
                return common
            if notice:
                if notice.get("state") == "complete":
                    common.update(
                        {
                            "state": "complete",
                            "status": "CAPTURE COMPLETE",
                            "detail": "Return to the workbench and confirm this capture.",
                            "color": "#6ee7a1",
                        }
                    )
                else:
                    common.update(
                        {
                            "state": "failed",
                            "status": "CAPTURE FAILED",
                            "detail": notice.get("detail")
                            or "Return to the workbench and rerecord.",
                            "color": "#ff7b84",
                        }
                    )
                return common
            return {"visible": False}

    def _profile_draft_path(self, game_profile_id: str) -> Path:
        return (
            self.artifact_root
            / "game_profiles"
            / safe_id(game_profile_id)
            / "draft.json"
        )

    def _poc_evidence_path(self, game_profile_id: str) -> Path:
        return (
            self.artifact_root
            / "poc_evidence"
            / safe_id(game_profile_id)
            / "evidence_index.json"
        )

    def _refresh_poc_evidence_index(self, game_profile_id: str) -> Optional[dict]:
        game = self.profiles.game(game_profile_id)
        if not game.get("poc_workflow"):
            return None
        draft_path = self._profile_draft_path(game_profile_id)
        profile_draft = None
        if draft_path.is_file():
            try:
                profile_draft = json.loads(draft_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                profile_draft = {
                    "error": "{}: {}".format(type(exc).__name__, exc)
                }
        index = build_poc_evidence_index(
            self.session_root,
            game,
            profile_draft=profile_draft,
        )
        _write_json_atomic(self._poc_evidence_path(game_profile_id), index)
        return index

    def _poc_evidence_indexes(self) -> dict:
        values = {}
        for game_profile_id, game in self.profiles.games.items():
            if not game.get("poc_workflow"):
                continue
            path = self._poc_evidence_path(game_profile_id)
            try:
                values[game_profile_id] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                values[game_profile_id] = {
                    "game_profile_id": game_profile_id,
                    "error": "{}: {}".format(type(exc).__name__, exc),
                }
        return values


    def _minimap_calibration_root(self, game_profile_id: str) -> Path:
        return self.artifact_root / "minimap_calibrations" / safe_id(game_profile_id)

    def _minimap_calibrations(self) -> dict:
        values = {}
        root = self.artifact_root / "minimap_calibrations"
        if not root.is_dir():
            return values
        for path in sorted(root.glob("*/*/calibration.json")):
            game_profile_id = path.parent.parent.name
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                item["calibration_id"] = path.parent.name
                item["artifact_relative_path"] = str(path.parent.relative_to(self.artifact_root))
                values.setdefault(game_profile_id, []).append(item)
            except (OSError, json.JSONDecodeError) as exc:
                values.setdefault(game_profile_id, []).append({"calibration_id": path.parent.name, "status": "invalid", "error": "{}: {}".format(type(exc).__name__, exc)})
        for items in values.values():
            items.sort(key=lambda item: item.get("generated_utc") or "", reverse=True)
        return values

    @staticmethod
    def _analysis_candidate_score(session: dict, role: str) -> float:
        evidence = (session.get("input_capture") or {}).get("evidence") or {}
        duration = float(session.get("duration_s") or 0.0)
        frames = int(session.get("frames") or 0)
        dropped = int(session.get("dropped_frames") or 0)
        score = duration + min(frames, 1800) / 60.0 - dropped * 20.0
        mouse = int(evidence.get("raw_mouse_events") or 0)
        movement = int(evidence.get("movement_key_events") or 0)
        keys = {str(item).upper() for item in evidence.get("key_names") or ()}
        if role == "rotation_only":
            score += min(mouse, 1000) / 50.0 - min(movement, 100) / 5.0
        elif role == "scene_rotation_360":
            score += min(mouse, 4000) / 50.0 - min(movement, 100) / 5.0
            score += min(duration, 90.0) / 3.0
        elif role == "movement_only":
            score += min(movement, 1000) / 50.0 - min(mouse, 100) / 5.0
        elif role == "forward_no_turn":
            score += min(movement, 1000) / 50.0
            if keys and keys.issubset({"W"}):
                score += 20.0
            score -= min(mouse, 100) / 5.0
        elif role == "full_map":
            score += min(mouse, 2000) / 100.0 + min(duration, 300.0) / 10.0
        return round(score, 4)

    def analysis_candidates(self) -> dict:
        """Return ranked retained sessions for each label-driven analysis role."""
        roles = {
            "rotation_only",
            "scene_rotation_360",
            "movement_only",
            "forward_no_turn",
            "full_map",
            "ordinary_cruise",
            "route",
            "teleportation",
            "minimap_transition",
        }
        values = {}
        for session in self.sessions():
            role = session.get("label")
            game_profile_id = session.get("game_profile_id")
            if (
                role not in roles
                or not game_profile_id
                or session.get("status") != "ready"
                or int(session.get("frames") or 0) <= 0
            ):
                continue
            candidate = dict(session)
            candidate["quality_score"] = self._analysis_candidate_score(
                session, role
            )
            values.setdefault(game_profile_id, {}).setdefault(role, []).append(
                candidate
            )
        for game in values.values():
            for candidates in game.values():
                candidates.sort(
                    key=lambda item: (
                        float(item["quality_score"]),
                        item.get("finished_utc") or "",
                    ),
                    reverse=True,
                )
                for index, candidate in enumerate(candidates):
                    candidate["recommended"] = index == 0
        return values

    def _queue_analysis(self, kind: str, value: dict, runner) -> dict:
        """Start one observable analysis job without blocking the HTTP request."""
        with self._lock:
            if self._active is not None:
                raise RuntimeError("Wait for the active take to finish")
            if self._tracker_running():
                raise RuntimeError("Stop live tracking before running analysis")
            if any(
                item.get("status") in {"queued", "running"}
                for item in self._analysis_jobs.values()
            ):
                raise RuntimeError("Wait for the active analysis task to finish")
            job_id = "{}-{}".format(kind, time.time_ns())
            request = {
                key: item
                for key, item in dict(value).items()
                if isinstance(item, (str, int, float, bool)) or item is None
            }
            self._analysis_jobs[kind] = {
                "job_id": job_id,
                "kind": kind,
                "status": "queued",
                "queued_utc": datetime.now(timezone.utc).isoformat(),
                "started_utc": None,
                "finished_utc": None,
                "request": request,
                "error": None,
                "message": "Waiting for the background worker",
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            }

        def report(message: str) -> None:
            with self._lock:
                job = self._analysis_jobs.get(kind)
                if job and job.get("job_id") == job_id:
                    job["message"] = str(message)
                    job["updated_utc"] = datetime.now(timezone.utc).isoformat()

        def work() -> None:
            with self._lock:
                job = self._analysis_jobs.get(kind)
                if not job or job.get("job_id") != job_id:
                    return
                job["status"] = "running"
                job["started_utc"] = datetime.now(timezone.utc).isoformat()
                job["message"] = "Starting analysis"
                job["updated_utc"] = job["started_utc"]
            try:
                runner(dict(value), report)
            except Exception as exc:
                with self._lock:
                    job = self._analysis_jobs.get(kind)
                    if job and job.get("job_id") == job_id:
                        job["status"] = "failed"
                        job["error"] = "{}: {}".format(
                            type(exc).__name__, exc
                        )
                        job["message"] = "Analysis failed"
                        job["finished_utc"] = datetime.now(
                            timezone.utc
                        ).isoformat()
                        job["updated_utc"] = job["finished_utc"]
            else:
                with self._lock:
                    job = self._analysis_jobs.get(kind)
                    if job and job.get("job_id") == job_id:
                        job["status"] = "complete"
                        job["message"] = "Analysis complete"
                        job["finished_utc"] = datetime.now(
                            timezone.utc
                        ).isoformat()
                        job["updated_utc"] = job["finished_utc"]

        threading.Thread(
            target=work,
            name="acquisition-workbench-{}".format(kind),
            daemon=True,
        ).start()
        return self.descriptor()

    def queue_minimap_calibration(self, value: dict) -> dict:
        return self._queue_analysis(
            "minimap_calibration", value, self.run_minimap_calibration
        )

    def queue_pose_verification(self, value: dict) -> dict:
        return self._queue_analysis(
            "pose_verification", value, self.run_pose_verification
        )

    def queue_map_stitch(self, value: dict) -> dict:
        return self._queue_analysis("map_stitching", value, self.run_map_stitch)

    def queue_map_atlas(self, value: dict) -> dict:
        return self._queue_analysis("map_atlas", value, self.run_map_atlas)

    def queue_route_tracking_compile(self, value: dict) -> dict:
        return self._queue_analysis(
            "route_tracking_compile", value, self.run_route_tracking_compile
        )

    def queue_scene_yaw_calibration(self, value: dict) -> dict:
        return self._queue_analysis(
            "scene_yaw_calibration", value, self.run_scene_yaw_calibration
        )

    def queue_teleport_analysis(self, value: dict) -> dict:
        return self._queue_analysis(
            "teleport_analysis", value, self.run_teleport_analysis
        )

    def run_minimap_calibration(self, value: dict, progress=None) -> dict:
        if progress:
            progress("Checking the selected calibration sessions")
        with self._lock:
            if self._active is not None:
                raise RuntimeError("Wait for the active take to finish")
            game_profile_id = str(value.get("game_profile_id") or "")
            if not game_profile_id:
                raise ValueError("Choose a game profile")
            game = self.profiles.game(game_profile_id)
            calibration_config = game.get("minimap_calibration")
            if not calibration_config:
                raise ValueError("The selected game has no mini-map calibration profile")
            session_root = self.session_root.resolve()

            def checked_session(relative_path: str):
                if not relative_path:
                    raise ValueError("Choose every required calibration segment")
                path = (session_root / relative_path).resolve()
                try:
                    path.relative_to(session_root)
                except ValueError:
                    raise ValueError("Calibration session must stay inside the session root")
                manifest_path = path / "manifest.json"
                if not manifest_path.is_file():
                    raise ValueError("Calibration session has no manifest")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                context = manifest.get("context") or {}
                if context.get("game_profile_id") != game_profile_id:
                    raise ValueError("Calibration session belongs to another game")
                return path, manifest, context

            legacy = bool(value.get("session_relative_path"))
            if not legacy:
                candidates = self.analysis_candidates().get(game_profile_id, {})

                def choose(role: str, field: str):
                    relative = str(value.get(field) or "")
                    if not relative:
                        ranked = candidates.get(role) or []
                        relative = str(ranked[0]["session_key"]) if ranked else ""
                    if not relative:
                        raise ValueError(
                            "No ready {} session is available".format(role)
                        )
                    path, manifest, context = checked_session(relative)
                    metadata = self._session_metadata(path)
                    actual_role = metadata.get("label") or context.get(
                        "segment_label"
                    )
                    if actual_role != role:
                        raise ValueError(
                            "Expected a {} session, got {}".format(
                                role, actual_role or "unlabeled"
                            )
                        )
                    return relative, path, manifest, context

                def choose_optional_motion(field: str):
                    relative = str(value.get(field) or "")
                    if not relative and field not in value:
                        ranked = candidates.get("ordinary_cruise") or []
                        relative = str(ranked[0]["session_key"]) if ranked else ""
                    if not relative:
                        return None, None, None, None, None
                    path, manifest, context = checked_session(relative)
                    metadata = self._session_metadata(path)
                    actual_role = metadata.get("label") or context.get(
                        "segment_label"
                    )
                    if actual_role not in ORDINARY_MOTION_SEGMENT_LABELS:
                        raise ValueError(
                            "Expected an {} session, got {}".format(
                                " or ".join(ORDINARY_MOTION_SEGMENT_LABELS),
                                actual_role or "unlabeled",
                            )
                        )
                    return relative, path, manifest, context, actual_role

                rotation_key, rotation_path, rotation_manifest, _ = choose(
                    "rotation_only", "rotation_session_relative_path"
                )
                movement_key, movement_path, movement_manifest, _ = choose(
                    "movement_only", "movement_session_relative_path"
                )
                (
                    ordinary_key,
                    ordinary_path,
                    ordinary_manifest,
                    _,
                    ordinary_recorded_label,
                ) = choose_optional_motion(
                    "ordinary_session_relative_path"
                )
                calibration_id = safe_id(
                    "segments-{}-{}".format(
                        str(rotation_manifest.get("session_id") or "rotation")[:12],
                        str(movement_manifest.get("session_id") or "movement")[:12],
                    )
                )
                output = self._minimap_calibration_root(game_profile_id) / calibration_id
                source_sessions = {
                    "rotation_only": {
                        "session_key": rotation_key,
                        "session_id": rotation_manifest.get("session_id"),
                    },
                    "movement_only": {
                        "session_key": movement_key,
                        "session_id": movement_manifest.get("session_id"),
                    },
                }
                if ordinary_manifest is not None:
                    source_sessions["ordinary_cruise"] = {
                        "session_key": ordinary_key,
                        "session_id": ordinary_manifest.get("session_id"),
                        "recorded_label": ordinary_recorded_label,
                    }
            else:
                session_key = str(value.get("session_relative_path") or "")
                session_path, manifest, context = checked_session(session_key)
                if context.get("workflow_stage_id") != "game-profile":
                    raise ValueError("Legacy calibration requires a basic gameplay sample")
                try:
                    segments = {
                        "rotation_only": [
                            float(value.get("rotation_start_s")),
                            float(value.get("rotation_end_s")),
                        ],
                        "movement_only": [
                            float(value.get("movement_start_s")),
                            float(value.get("movement_end_s")),
                        ],
                    }
                except (TypeError, ValueError):
                    raise ValueError("Enter all four segment boundaries in seconds")
                calibration_id = safe_id("{}-run{:02d}".format(context.get("experiment_id") or manifest.get("session_id"), int(context.get("run_index") or 1)))
                output = self._minimap_calibration_root(game_profile_id) / calibration_id
                source_sessions = {
                    "legacy_segmented_session": {
                        "session_key": session_key,
                        "session_id": manifest.get("session_id"),
                    }
                }

        if legacy:
            result = calibrate_session(
                session_path,
                output,
                segments,
                calibration_config,
                progress=progress,
            )
        else:
            result = calibrate_segment_sessions(
                rotation_path,
                movement_path,
                output,
                calibration_config,
                progress=progress,
                ordinary_session_path=ordinary_path,
            )

        if progress:
            progress("Saving the calibration result and evidence index")
        with self._lock:
            result["source_sessions"] = source_sessions
            result["calibration_id"] = calibration_id
            result["artifact_relative_path"] = str(output.relative_to(self.artifact_root))
            _write_json_atomic(output / "calibration.json", result)
            self._last_error = None
            return self.descriptor()

    def _scene_yaw_root(self, game_profile_id: str) -> Path:
        return self.artifact_root / "scene_yaw_calibrations" / safe_id(game_profile_id)

    def _scene_yaw_calibrations(self) -> dict:
        values = {}
        root = self.artifact_root / "scene_yaw_calibrations"
        if not root.is_dir():
            return values
        for path in sorted(root.glob("*/*/scene_yaw_calibration.json")):
            game_profile_id = path.parent.parent.name
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                item.pop("focal_search", None)
                item["calibration_id"] = path.parent.name
                item["artifact_relative_path"] = str(
                    path.parent.relative_to(self.artifact_root)
                )
                values.setdefault(game_profile_id, []).append(item)
            except (OSError, json.JSONDecodeError) as exc:
                values.setdefault(game_profile_id, []).append(
                    {
                        "calibration_id": path.parent.name,
                        "status": "invalid",
                        "error": "{}: {}".format(type(exc).__name__, exc),
                    }
                )
        for items in values.values():
            items.sort(key=lambda item: item.get("generated_utc") or "", reverse=True)
        return values

    def run_scene_yaw_calibration(self, value: dict, progress=None) -> dict:
        if progress:
            progress("Checking the selected full-turn scene session")
        with self._lock:
            if self._active is not None:
                raise RuntimeError("Wait for the active take to finish")
            game_profile_id = str(value.get("game_profile_id") or "")
            if not game_profile_id:
                raise ValueError("Choose a game profile")
            game = self.profiles.game(game_profile_id)
            config = game.get("scene_yaw_calibration")
            if not config:
                raise ValueError("The selected game has no scene-yaw calibration profile")
            relative = str(value.get("session_relative_path") or "")
            if not relative:
                candidates = (
                    self.analysis_candidates()
                    .get(game_profile_id, {})
                    .get("scene_rotation_360", [])
                )
                relative = str(candidates[0]["session_key"]) if candidates else ""
            if not relative:
                raise ValueError("No ready scene_rotation_360 session is available")
            path = self._session_path(relative)
            described = self._describe_session(path)
            if described.get("game_profile_id") != game_profile_id:
                raise ValueError("Scene-yaw session belongs to another game")
            if described.get("label") != "scene_rotation_360":
                raise ValueError("Expected a scene_rotation_360 session")
            calibration_id = safe_id(described.get("session_id") or path.name)
            output = self._scene_yaw_root(game_profile_id) / calibration_id

        result = calibrate_scene_yaw_session(
            path,
            output,
            config=config,
            progress=progress,
        )
        with self._lock:
            result["source_session_key"] = relative
            result["calibration_id"] = calibration_id
            result["artifact_relative_path"] = str(
                output.relative_to(self.artifact_root)
            )
            _write_json_atomic(output / "scene_yaw_calibration.json", result)
            self._last_error = None
            return self.descriptor()

    def scene_yaw_image(
        self, game_profile_id: str, calibration_id: str, name: str
    ) -> bytes:
        if Path(name).name != name or not name.lower().endswith(".png"):
            raise ValueError("Invalid scene-yaw evidence image name")
        root = self._scene_yaw_root(game_profile_id) / safe_id(calibration_id)
        descriptor = json.loads(
            (root / "scene_yaw_calibration.json").read_text(encoding="utf-8")
        )
        declared = {item.get("name") for item in descriptor.get("evidence", [])}
        if name not in declared:
            raise ValueError("Unknown scene-yaw evidence image")
        return (root / name).read_bytes()

    def minimap_calibration_image(self, game_profile_id: str, calibration_id: str, name: str) -> bytes:
        if Path(name).name != name or not name.lower().endswith(".png"):
            raise ValueError("Invalid evidence image name")
        root = self._minimap_calibration_root(game_profile_id) / safe_id(calibration_id)
        descriptor = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
        declared = {item.get("name") for item in descriptor.get("evidence", [])}
        if name not in declared:
            raise ValueError("Unknown calibration evidence image")
        return (root / name).read_bytes()

    def run_pose_verification(self, value: dict, progress=None) -> dict:
        """Cross-check the existing pose model against one forward session."""
        if progress:
            progress("Checking the calibration and forward session")
        with self._lock:
            if self._active is not None:
                raise RuntimeError("Wait for the active take to finish")
            game_profile_id = str(value.get("game_profile_id") or "")
            if not game_profile_id:
                raise ValueError("Choose a game profile")
            self.profiles.game(game_profile_id)
            calibration_id = str(value.get("calibration_id") or "")
            if not calibration_id or safe_id(calibration_id) != calibration_id:
                raise ValueError("Choose a calibration for the selected sessions")
            calibration_root = (
                self._minimap_calibration_root(game_profile_id) / calibration_id
            )
            calibration_file = calibration_root / "calibration.json"
            if not calibration_file.is_file():
                raise ValueError("Calibration artifact does not exist")
            calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
            source_sessions = calibration.get("source_sessions") or {}
            expected_rotation = str(value.get("rotation_session_relative_path") or "")
            expected_movement = str(value.get("movement_session_relative_path") or "")
            if expected_rotation and (
                (source_sessions.get("rotation_only") or {}).get("session_key")
                != expected_rotation
            ):
                raise ValueError("Calibration does not use the selected rotation session")
            if expected_movement and (
                (source_sessions.get("movement_only") or {}).get("session_key")
                != expected_movement
            ):
                raise ValueError("Calibration does not use the selected movement session")
            forward_key = str(value.get("forward_session_relative_path") or "")
            if not forward_key:
                candidates = (
                    self.analysis_candidates()
                    .get(game_profile_id, {})
                    .get("forward_no_turn", [])
                )
                forward_key = str(candidates[0]["session_key"]) if candidates else ""
            if not forward_key:
                raise ValueError("No ready forward_no_turn session is available")
            forward_path = self._session_path(forward_key)
            forward = self._describe_session(forward_path)
            if forward.get("game_profile_id") != game_profile_id:
                raise ValueError("Forward session belongs to another game")
            if forward.get("label") != "forward_no_turn":
                raise ValueError("Expected a forward_no_turn session")

        verification = verify_forward_session(
            forward_path,
            calibration_root,
            calibration_root,
            progress=progress,
        )

        if progress:
            progress("Saving pose verification evidence")
        with self._lock:
            verification["calibration_id"] = calibration_id
            verification["source_session_key"] = forward_key
            calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
            forward_names = {
                "forward_start.png",
                "forward_end.png",
                "forward_shift_mask.png",
                "forward_registration_overlay.png",
                "forward_pose_shift.png",
            }
            calibration["evidence"] = [
                item
                for item in calibration.get("evidence") or []
                if item.get("name") not in forward_names
            ]
            calibration["evidence"].extend(verification.get("evidence") or [])
            calibration["forward_verification"] = verification
            _write_json_atomic(calibration_file, calibration)
            self._last_error = None
            return self.descriptor()

    def _map_stitch_root(self, game_profile_id: str) -> Path:
        return self.artifact_root / "map_stitches" / safe_id(game_profile_id)

    def _map_stitches(self) -> dict:
        values = {}
        root = self.artifact_root / "map_stitches"
        if not root.is_dir():
            return values
        for path in sorted(root.glob("*/*/map_stitch.json")):
            game_profile_id = path.parent.parent.name
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                item.pop("registrations", None)
                provenance = dict(item.get("provenance") or {})
                provenance.pop("source_frame_records", None)
                item["provenance"] = provenance
                item["stitch_id"] = path.parent.name
                item["artifact_relative_path"] = str(
                    path.parent.relative_to(self.artifact_root)
                )
                values.setdefault(game_profile_id, []).append(item)
            except (OSError, json.JSONDecodeError) as exc:
                values.setdefault(game_profile_id, []).append(
                    {
                        "stitch_id": path.parent.name,
                        "status": "invalid",
                        "error": "{}: {}".format(type(exc).__name__, exc),
                    }
                )
        for items in values.values():
            items.sort(
                key=lambda item: item.get("generated_utc") or "", reverse=True
            )
        return values

    def run_map_stitch(self, value: dict, progress=None) -> dict:
        if progress:
            progress("Checking the selected full-map session")
        with self._lock:
            if self._active is not None:
                raise RuntimeError("Wait for the active take to finish")
            game_profile_id = str(value.get("game_profile_id") or "")
            if not game_profile_id:
                raise ValueError("Choose a game profile")
            self.profiles.game(game_profile_id)
            relative = str(value.get("session_relative_path") or "")
            if not relative:
                candidates = (
                    self.analysis_candidates()
                    .get(game_profile_id, {})
                    .get("full_map", [])
                )
                relative = str(candidates[0]["session_key"]) if candidates else ""
            if not relative:
                raise ValueError("No ready full_map session is available")
            path = self._session_path(relative)
            described = self._describe_session(path)
            if described.get("game_profile_id") != game_profile_id:
                raise ValueError("Map session belongs to another game")
            if described.get("label") != "full_map":
                raise ValueError("Expected a full_map session")
            calibration_id = str(value.get("minimap_calibration_id") or "")
            if not calibration_id or safe_id(calibration_id) != calibration_id:
                raise ValueError(
                    "Choose a reviewed mini-map calibration before rebuilding the map"
                )
            calibration_root = (
                self._minimap_calibration_root(game_profile_id) / calibration_id
            )
            calibration_file = calibration_root / "calibration.json"
            if not calibration_file.is_file():
                raise ValueError("Mini-map calibration artifact does not exist")
            calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
            verification = calibration.get("forward_verification") or {}
            declared_evidence = {
                item.get("name") for item in verification.get("evidence") or []
            }
            if "forward_start.png" not in declared_evidence:
                raise ValueError(
                    "Run and review forward pose verification before rebuilding the map"
                )
            candidates = load_localization_reference_candidates(
                calibration_root, calibration
            )
            if not candidates:
                raise ValueError("Forward scale-reference images cannot be decoded")
            primary = candidates[0]
            localization_reference = {
                "image": primary["image"],
                "calibration": calibration,
                "calibration_id": calibration_id,
                "source_image_name": primary["source_image_name"],
                "candidates": candidates,
            }
            stitch_id = safe_id(described.get("session_id") or path.name)
            output = self._map_stitch_root(game_profile_id) / stitch_id

        result = stitch_map_session(
            path,
            output,
            progress=progress,
            localization_reference=localization_reference,
        )

        if progress:
            progress("Saving the stitched-map result and evidence index")
        with self._lock:
            result["source_session_key"] = relative
            result["source_minimap_calibration_id"] = calibration_id
            result["stitch_id"] = stitch_id
            result["artifact_relative_path"] = str(
                output.relative_to(self.artifact_root)
            )
            _write_json_atomic(output / "map_stitch.json", result)
            self._last_error = None
            return self.descriptor()

    def map_stitch_image(
        self, game_profile_id: str, stitch_id: str, name: str
    ) -> bytes:
        if Path(name).name != name or not name.lower().endswith(".png"):
            raise ValueError("Invalid map-stitch evidence image name")
        root = self._map_stitch_root(game_profile_id) / safe_id(stitch_id)
        descriptor = json.loads(
            (root / "map_stitch.json").read_text(encoding="utf-8")
        )
        declared = {item.get("name") for item in descriptor.get("evidence", [])}
        if name not in declared:
            raise ValueError("Unknown map-stitch evidence image")
        return (root / name).read_bytes()

    def _map_atlas_root(self, game_profile_id: str) -> Path:
        return self.artifact_root / "map_atlases" / safe_id(game_profile_id)

    def _map_atlases(self) -> dict:
        values = {}
        root = self.artifact_root / "map_atlases"
        if not root.is_dir():
            return values
        for path in sorted(root.glob("*/*/map_atlas.json")):
            game_profile_id = path.parent.parent.name
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                item["atlas_id"] = path.parent.name
                item["artifact_relative_path"] = str(
                    path.parent.relative_to(self.artifact_root)
                )
                values.setdefault(game_profile_id, []).append(item)
            except (OSError, json.JSONDecodeError) as exc:
                values.setdefault(game_profile_id, []).append(
                    {
                        "atlas_id": path.parent.name,
                        "status": "invalid",
                        "error": "{}: {}".format(type(exc).__name__, exc),
                    }
                )
        for items in values.values():
            items.sort(
                key=lambda item: item.get("generated_utc") or "", reverse=True
            )
        return values

    def run_map_atlas(self, value: dict, progress=None) -> dict:
        if progress:
            progress("Checking the selected rendered map layers")
        with self._lock:
            if self._active is not None:
                raise RuntimeError("Wait for the active take to finish")
            game_profile_id = str(value.get("game_profile_id") or "")
            if not game_profile_id:
                raise ValueError("Choose a game profile")
            game = self.profiles.game(game_profile_id)
            layers = [dict(item) for item in value.get("layers") or ()]
            if not layers:
                map_stitch_id = value.get("map_stitch_id") or value.get(
                    "world_stitch_id"
                )
                layers = [
                    {
                        "mode_id": "world",
                        "stitch_id": map_stitch_id,
                        "display_name": "World overview",
                    },
                    {
                        "mode_id": "town",
                        "stitch_id": value.get("town_stitch_id") or map_stitch_id,
                        "display_name": "Town detail",
                    },
                ]
            stitch_root = self._map_stitch_root(game_profile_id)
            resolved_layers = []
            for layer in layers:
                stitch_id = str(layer.get("stitch_id") or "")
                if not stitch_id or safe_id(stitch_id) != stitch_id:
                    raise ValueError("Choose a valid stitched map for every atlas layer")
                source = stitch_root / stitch_id
                if not (source / "map_stitch.json").is_file():
                    raise ValueError("Map stitch does not exist: {}".format(stitch_id))
                resolved = dict(layer)
                resolved["stitch_root"] = source
                resolved_layers.append(resolved)
            canonical_mode_id = str(value.get("canonical_mode_id") or "world")
            atlas_id = safe_id(
                str(value.get("atlas_id") or uuid.uuid4())
            )
            output = self._map_atlas_root(game_profile_id) / atlas_id
            transition_relative = str(
                value.get("transition_session_relative_path") or ""
            )
            transition_path = None
            transition_extractor = None
            if transition_relative:
                transition_path = self._session_path(transition_relative)
                described = self._describe_session(transition_path)
                if described.get("game_profile_id") != game_profile_id:
                    raise ValueError("Transition session belongs to another game")
                if described.get("label") != "minimap_transition":
                    raise ValueError("Expected a minimap_transition session")
                calibration_id = str(value.get("minimap_calibration_id") or "")
                calibration = self._read_tracker_artifact(
                    self._minimap_calibration_root(game_profile_id),
                    "calibration.json",
                    calibration_id,
                )
                transition_extractor = MinimapExtractor(
                    game["minimap_calibration"]["crop_xywh"], calibration
                )

        endpoint_provenance = None
        if transition_path is not None:
            if progress:
                progress("Selecting stable mini-map references at both transition ends")
            endpoints = transition_endpoint_references(
                transition_path, transition_extractor
            )
            source_mode_id = str(value.get("source_mode_id") or "world")
            target_mode_id = str(value.get("target_mode_id") or "town")
            references = {
                source_mode_id: endpoints["source"],
                target_mode_id: endpoints["target"],
            }
            for layer in resolved_layers:
                reference = references.get(str(layer["mode_id"]))
                if reference is None:
                    continue
                layer["minimap_reference"] = reference["image"]
                layer["minimap_reference_mask"] = reference["mask"]
            endpoint_provenance = {
                "source_session_key": transition_relative,
                "source_mode_id": source_mode_id,
                "target_mode_id": target_mode_id,
                "source_frame_index": endpoints["source"]["source_frame_index"],
                "target_frame_index": endpoints["target"]["source_frame_index"],
                "source_laplacian_variance": endpoints["source"][
                    "laplacian_variance"
                ],
                "target_laplacian_variance": endpoints["target"][
                    "laplacian_variance"
                ],
            }
        result = build_map_atlas(
            resolved_layers,
            output,
            canonical_mode_id=canonical_mode_id,
            atlas_id=atlas_id,
        )
        transition_model = None
        if transition_path is not None:
            mode_scales = {
                str(item["mode_id"]): float(
                    item["map_pixels_per_minimap_pixel"]
                )
                for item in result["layers"]
                if item.get("map_pixels_per_minimap_pixel") is not None
            }
            if progress:
                progress("Learning the recorded scale-transition behavior")
            transition_model = analyze_transition_session(
                transition_path,
                transition_extractor,
                cv2.imread(str(output / result["canonical_mosaic_file"])),
                cv2.imread(
                    str(output / result["canonical_coverage_file"]),
                    cv2.IMREAD_GRAYSCALE,
                ),
                mode_scales,
                output,
                source_mode_id=source_mode_id,
                target_mode_id=target_mode_id,
            )
        with self._lock:
            result["game_profile_id"] = game_profile_id
            result["source_layers"] = [
                {
                    "mode_id": item["mode_id"],
                    "stitch_id": item["stitch_id"],
                }
                for item in layers
            ]
            source_stitch_ids = {
                str(item["stitch_id"]) for item in layers
            }
            result["source_map_stitch_id"] = (
                next(iter(source_stitch_ids))
                if len(source_stitch_ids) == 1
                else None
            )
            result["transition_reference"] = endpoint_provenance
            result["transition_model"] = transition_model
            result["artifact_relative_path"] = str(
                output.relative_to(self.artifact_root)
            )
            _write_json_atomic(output / "map_atlas.json", result)
            self._last_error = None
            return self.descriptor()

    def map_atlas_image(
        self, game_profile_id: str, atlas_id: str, name: str
    ) -> bytes:
        if Path(name).name != name or not name.lower().endswith(".png"):
            raise ValueError("Invalid map-atlas evidence image name")
        root = self._map_atlas_root(game_profile_id) / safe_id(atlas_id)
        descriptor = json.loads(
            (root / "map_atlas.json").read_text(encoding="utf-8")
        )
        declared = {
            descriptor.get("canonical_mosaic_file"),
            descriptor.get("canonical_coverage_file"),
        }
        declared.update(
            item.get("alignment_evidence_file")
            for item in descriptor.get("layers") or []
        )
        declared.update(
            item.get("minimap_reference_file")
            for item in descriptor.get("layers") or []
        )
        declared.add((descriptor.get("transition_model") or {}).get("evidence_file"))
        if name not in declared:
            raise ValueError("Unknown map-atlas evidence image")
        return (root / name).read_bytes()

    def _route_tracking_root(self, game_profile_id: str) -> Path:
        return self.artifact_root / "route_tracking" / safe_id(game_profile_id)

    def _route_tracking_packages(self) -> dict:
        values = {}
        root = self.artifact_root / "route_tracking"
        if not root.is_dir():
            return values
        for path in sorted(root.glob("*/*/manifest.json")):
            game_profile_id = path.parent.parent.name
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                item["package_id"] = path.parent.name
                item["artifact_relative_path"] = str(
                    path.parent.relative_to(self.artifact_root)
                )
                values.setdefault(game_profile_id, []).append(item)
            except (OSError, json.JSONDecodeError) as exc:
                values.setdefault(game_profile_id, []).append(
                    {
                        "package_id": path.parent.name,
                        "status": "invalid",
                        "error": "{}: {}".format(type(exc).__name__, exc),
                    }
                )
        for items in values.values():
            items.sort(
                key=lambda item: item.get("generated_utc") or "", reverse=True
            )
        return values

    def run_route_tracking_compile(self, value: dict, progress=None) -> dict:
        if progress:
            progress("Checking the route, atlas, and mini-map geometry")
        with self._lock:
            if self._active is not None:
                raise RuntimeError("Wait for the active take to finish")
            game_profile_id = str(value.get("game_profile_id") or "")
            if not game_profile_id:
                raise ValueError("Choose a game profile")
            game = self.profiles.game(game_profile_id)
            relative = str(value.get("session_relative_path") or "")
            if not relative:
                candidates = (
                    self.analysis_candidates().get(game_profile_id, {}).get("route", [])
                )
                relative = str(candidates[0]["session_key"]) if candidates else ""
            if not relative:
                raise ValueError("No ready route session is available")
            session_path = self._session_path(relative)
            described = self._describe_session(session_path)
            if described.get("game_profile_id") != game_profile_id:
                raise ValueError("Route session belongs to another game")
            if described.get("label") != "route":
                raise ValueError("Expected a route session")
            atlas_id = str(value.get("map_atlas_id") or "")
            atlas_path = self._map_atlas_root(game_profile_id) / safe_id(atlas_id)
            atlas_manifest = atlas_path / "map_atlas.json"
            if not atlas_id or not atlas_manifest.is_file():
                raise ValueError("Choose a ready multi-scale map atlas")
            calibration_id = str(value.get("minimap_calibration_id") or "")
            minimap = self._read_tracker_artifact(
                self._minimap_calibration_root(game_profile_id),
                "calibration.json",
                calibration_id,
            )
            route_id = str(
                value.get("route_id")
                or described.get("route_id")
                or described.get("capture_id")
                or session_path.name
            )
            package_id = safe_id(
                "{}-{}".format(
                    described.get("session_id") or session_path.name,
                    uuid.uuid4().hex[:8],
                )
            )
            output = self._route_tracking_root(game_profile_id) / package_id

        result = compile_route_session(
            session_path,
            output,
            stream_id=str(value.get("stream_id") or "main"),
            route_id=route_id,
            atlas_path=atlas_path,
            minimap_config=game["minimap_calibration"],
            minimap_calibration=minimap,
            reference_rate_hz=float(value.get("reference_rate_hz") or 5.0),
            max_step_px=float(value.get("max_step_px") or 80.0),
            corridor_radius_px=float(value.get("corridor_radius_px") or 35.0),
            progress=progress,
        )
        with self._lock:
            result["package_id"] = package_id
            result["game_profile_id"] = game_profile_id
            result["source_session_key"] = relative
            result["minimap_calibration_id"] = calibration_id
            result["artifact_relative_path"] = str(
                output.relative_to(self.artifact_root)
            )
            _write_json_atomic(output / "manifest.json", result)
            self._last_error = None
            return self.descriptor()

    def _teleport_behavior_root(self, game_profile_id: str) -> Path:
        return self.artifact_root / "teleport_behaviors" / safe_id(game_profile_id)

    def _teleport_behaviors(self) -> dict:
        values = {}
        root = self.artifact_root / "teleport_behaviors"
        if not root.is_dir():
            return values
        for path in sorted(root.glob("*/*/teleport.analysis.json")):
            game_profile_id = path.parent.parent.name
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                item["behavior_id"] = path.parent.name
                item["artifact_relative_path"] = str(
                    path.parent.relative_to(self.artifact_root)
                )
                values.setdefault(game_profile_id, []).append(item)
            except (OSError, json.JSONDecodeError) as exc:
                values.setdefault(game_profile_id, []).append(
                    {
                        "behavior_id": path.parent.name,
                        "status": "invalid",
                        "error": "{}: {}".format(type(exc).__name__, exc),
                    }
                )
        for items in values.values():
            items.sort(
                key=lambda item: (
                    (item.get("provenance") or {}).get("generated_utc") or ""
                ),
                reverse=True,
            )
        return values

    def live_tracking_image(
        self,
        game_profile_id: str,
        tracking_id: str,
        fix_id: str,
        name: str,
    ):
        if safe_id(game_profile_id) != game_profile_id:
            raise ValueError("Invalid game profile ID")
        if safe_id(tracking_id) != tracking_id:
            raise ValueError("Invalid live-tracking ID")
        if safe_id(fix_id) != fix_id:
            raise ValueError("Invalid global-fix ID")
        root = (
            self._live_tracking_root(game_profile_id)
            / tracking_id
            / "global_fixes"
            / fix_id
        )
        descriptor = json.loads(
            (root / "global_fix.json").read_text(encoding="utf-8")
        )
        if name not in set(descriptor.get("evidence") or ()):
            raise ValueError("Unknown live-tracking evidence image")
        content_type = "image/jpeg" if name.lower().endswith(".jpg") else "image/png"
        return content_type, (root / name).read_bytes()

    def run_teleport_analysis(self, value: dict, progress=None) -> dict:
        if progress:
            progress("Checking teleport session and spatial artifacts")
        with self._lock:
            if self._active is not None:
                raise RuntimeError("Wait for the active take to finish")
            game_profile_id = str(value.get("game_profile_id") or "")
            if not game_profile_id:
                raise ValueError("Choose a game profile")
            game = self.profiles.game(game_profile_id)
            minimap_config = game.get("minimap_calibration")
            if not minimap_config:
                raise ValueError("The selected game has no mini-map profile")

            relative = str(value.get("session_relative_path") or "")
            if not relative:
                candidates = (
                    self.analysis_candidates()
                    .get(game_profile_id, {})
                    .get("teleportation", [])
                )
                relative = str(candidates[0]["session_key"]) if candidates else ""
            if not relative:
                raise ValueError("No ready teleportation session is available")
            session_path = self._session_path(relative)
            described = self._describe_session(session_path)
            if described.get("game_profile_id") != game_profile_id:
                raise ValueError("Teleport session belongs to another game")
            if described.get("label") != "teleportation":
                raise ValueError("Expected a teleportation session")

            calibration_id = str(value.get("minimap_calibration_id") or "")
            if not calibration_id or safe_id(calibration_id) != calibration_id:
                raise ValueError("Choose a mini-map calibration")
            calibration_root = (
                self._minimap_calibration_root(game_profile_id) / calibration_id
            )
            calibration_file = calibration_root / "calibration.json"
            if not calibration_file.is_file():
                raise ValueError("Mini-map calibration artifact does not exist")
            calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
            calibration["calibration_id"] = calibration_id

            stitch_id = str(value.get("map_stitch_id") or "")
            if not stitch_id or safe_id(stitch_id) != stitch_id:
                raise ValueError("Choose a stitched global map")
            stitch_root = self._map_stitch_root(game_profile_id) / stitch_id
            stitch_file = stitch_root / "map_stitch.json"
            if not stitch_file.is_file():
                raise ValueError("Map-stitch artifact does not exist")
            stitch = json.loads(stitch_file.read_text(encoding="utf-8"))
            stitch["stitch_id"] = stitch_id
            if stitch.get("source_minimap_calibration_id") != calibration_id:
                raise ValueError(
                    "Map stitch and teleport destination use different mini-map calibrations"
                )
            _require_ready_map_localization(stitch, "teleport analysis")
            behavior_id = safe_id(described.get("session_id") or session_path.name)
            output = self._teleport_behavior_root(game_profile_id) / behavior_id

        result = analyze_teleport_session(
            session_path,
            output,
            game_profile_id=game_profile_id,
            minimap_config=minimap_config,
            minimap_calibration=calibration,
            map_stitch=stitch,
            map_stitch_root=stitch_root,
            progress=progress,
        )
        with self._lock:
            result["source_session_key"] = relative
            result["behavior_id"] = behavior_id
            result["map_stitch_id"] = stitch_id
            result["minimap_calibration_id"] = calibration_id
            result["artifact_relative_path"] = str(
                output.relative_to(self.artifact_root)
            )
            _write_json_atomic(output / "teleport.analysis.json", result)
            self._last_error = None
            return self.descriptor()

    def teleport_behavior_image(
        self, game_profile_id: str, behavior_id: str, name: str
    ) -> bytes:
        if Path(name).name != name or not name.lower().endswith(".png"):
            raise ValueError("Invalid teleport evidence image name")
        root = self._teleport_behavior_root(game_profile_id) / safe_id(behavior_id)
        descriptor = json.loads(
            (root / "teleport.analysis.json").read_text(encoding="utf-8")
        )
        declared = {item.get("name") for item in descriptor.get("evidence_files", [])}
        if name not in declared:
            raise ValueError("Unknown teleport evidence image")
        return (root / name).read_bytes()

    def _live_tracker_descriptor(self) -> Optional[dict]:
        if self._live_tracker is None:
            return None
        return {
            key: value
            for key, value in self._live_tracker.items()
            if key not in ("thread", "stop", "evidence_recorder")
        }

    def _live_tracking_root(self, game_profile_id: str) -> Path:
        return self.artifact_root / "live_tracking" / safe_id(game_profile_id)

    def _live_tracking_runs(self) -> dict:
        values = {}
        root = self.artifact_root / "live_tracking"
        if not root.is_dir():
            return values
        for path in sorted(root.glob("*/*/live_tracking.json")):
            game_profile_id = path.parent.parent.name
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                item["tracking_id"] = path.parent.name
                item["artifact_relative_path"] = str(
                    path.parent.relative_to(self.artifact_root)
                )
                recent_fixes = []
                for fix_path in sorted(
                    (path.parent / "global_fixes").glob("*/global_fix.json"),
                    reverse=True,
                )[:6]:
                    try:
                        fix = json.loads(fix_path.read_text(encoding="utf-8"))
                        fix["fix_id"] = fix_path.parent.name
                        recent_fixes.append(fix)
                    except (OSError, json.JSONDecodeError):
                        continue
                item["recent_global_fixes"] = recent_fixes
                values.setdefault(game_profile_id, []).append(item)
            except (OSError, json.JSONDecodeError) as exc:
                values.setdefault(game_profile_id, []).append(
                    {
                        "tracking_id": path.parent.name,
                        "status": "invalid",
                        "error": "{}: {}".format(type(exc).__name__, exc),
                    }
                )
        for items in values.values():
            items.sort(key=lambda item: item.get("started_utc") or "", reverse=True)
        return values

    def _tracker_running(self) -> bool:
        return bool(
            self._live_tracker
            and self._live_tracker.get("status") in ("starting", "running")
        )

    @staticmethod
    def _read_tracker_artifact(root: Path, filename: str, artifact_id: str) -> dict:
        if not artifact_id or safe_id(artifact_id) != artifact_id:
            raise ValueError("Choose a valid {} artifact".format(filename))
        path = root / artifact_id / filename
        if not path.is_file():
            raise ValueError("Tracker artifact does not exist: {}".format(path))
        return json.loads(path.read_text(encoding="utf-8"))

    def start_live_tracker(self, value: dict) -> dict:
        """Start user-controlled live capture with asynchronous absolute fixes."""
        with self._lock:
            if self._active is not None:
                raise RuntimeError("Stop the active recording before live tracking")
            if self._tracker_running():
                raise RuntimeError("The live tracker is already running")
            if any(
                job.get("status") in ("queued", "running")
                for job in self._analysis_jobs.values()
            ):
                raise RuntimeError("Wait for background analysis to finish")
            game_profile_id = str(value.get("game_profile_id") or "")
            if not game_profile_id:
                raise ValueError("Choose a game profile")
            game = self.profiles.game(game_profile_id)
            minimap_config = game.get("minimap_calibration")
            if not minimap_config:
                raise ValueError(
                    "This game profile has no verified mini-map geometry; "
                    "live tracking cannot substitute another platform's calibration"
                )
            calibration_id = str(value.get("minimap_calibration_id") or "")
            scene_yaw_id = str(value.get("scene_yaw_calibration_id") or "")
            stitch_id = str(value.get("map_stitch_id") or "")
            atlas_id = str(value.get("map_atlas_id") or "")
            route_package_id = str(value.get("route_package_id") or "")
            requested_tracking_mode = str(
                value.get("tracking_mode") or "free-roam"
            )
            # Preserve old clients and saved requests while making the runtime
            # contract explicit: a demonstrated route is an acceleration hint,
            # never a source of pose authority.
            tracking_mode = (
                "route-assisted"
                if requested_tracking_mode == "route-locked"
                else requested_tracking_mode
            )
            if tracking_mode not in ("free-roam", "route-assisted"):
                raise ValueError("Tracking mode must be free-roam or route-assisted")
            minimap = self._read_tracker_artifact(
                self._minimap_calibration_root(game_profile_id),
                "calibration.json",
                calibration_id,
            )
            scene_yaw = self._read_tracker_artifact(
                self._scene_yaw_root(game_profile_id),
                "scene_yaw_calibration.json",
                scene_yaw_id,
            )
            route_package = None
            if atlas_id or tracking_mode == "route-assisted":
                if tracking_mode == "route-assisted":
                    package_root = (
                        self._route_tracking_root(game_profile_id) / safe_id(route_package_id)
                    )
                    if not route_package_id or not (package_root / "manifest.json").is_file():
                        raise ValueError("Choose a compiled demonstrated route")
                    route_package = RouteTrackingPackage(package_root)
                    if atlas_id and atlas_id != route_package.manifest["atlas_id"]:
                        raise ValueError("Route package and selected map atlas differ")
                    atlas_id = str(route_package.manifest["atlas_id"])
                atlas_root = self._map_atlas_root(game_profile_id) / safe_id(atlas_id)
                atlas = self._read_tracker_artifact(
                    self._map_atlas_root(game_profile_id),
                    "map_atlas.json",
                    atlas_id,
                )
                mosaic = cv2.imread(
                    str(atlas_root / atlas["canonical_mosaic_file"]),
                    cv2.IMREAD_COLOR,
                )
                if mosaic is None:
                    raise ValueError("Could not decode the canonical map-atlas mosaic")
                if tracking_mode == "route-assisted" and (
                    route_package.manifest["coordinate_space_id"]
                    != atlas["coordinate_space_id"]
                ):
                    raise ValueError("Route package uses another canonical map space")
                localizer = LayeredGlobalLocalizer(atlas_root)
            else:
                stitch_root = self._map_stitch_root(game_profile_id)
                stitch = self._read_tracker_artifact(
                    stitch_root, "map_stitch.json", stitch_id
                )
                mosaic_path = stitch_root / stitch_id / "mosaic.png"
                declared = {item.get("name") for item in stitch.get("evidence") or []}
                if "mosaic.png" not in declared or not mosaic_path.is_file():
                    raise ValueError("The selected map stitch has no declared mosaic image")
                mosaic = cv2.imread(str(mosaic_path), cv2.IMREAD_COLOR)
                if mosaic is None:
                    raise ValueError("Could not decode the selected map mosaic")
                localization = _require_ready_map_localization(
                    stitch, "starting live tracking"
                )
                if localization.get("source_minimap_calibration_id") != calibration_id:
                    raise ValueError(
                        "The selected map localization raster was built for another "
                        "mini-map calibration; rebuild the full map"
                    )
                localization_names = {
                    localization.get("mosaic_file"), localization.get("coverage_file")
                }
                if not localization_names.issubset(declared):
                    raise ValueError(
                        "Map localization raster files are not declared as review evidence"
                    )
                localization_root = stitch_root / stitch_id
                localization_mosaic = cv2.imread(
                    str(localization_root / localization["mosaic_file"]),
                    cv2.IMREAD_COLOR,
                )
                localization_coverage = cv2.imread(
                    str(localization_root / localization["coverage_file"]),
                    cv2.IMREAD_GRAYSCALE,
                )
                if localization_mosaic is None or localization_coverage is None:
                    raise ValueError("Could not decode the map localization raster")
                localizer = GlobalMapLocalizer(
                    localization_mosaic,
                    localization_coverage,
                    localization.get("localization_to_original_map_3x3"),
                )
            tracking_profile_name = str(value.get("tracking_profile") or "real-time")
            profile_overrides = {}
            if value.get("cursor_pose_method"):
                profile_overrides["cursor_pose_method"] = str(
                    value["cursor_pose_method"]
                )
            if value.get("global_interval_s") is not None:
                profile_overrides["global_interval_s"] = float(
                    value["global_interval_s"]
                )
            resolved_profile = resolve_tracking_profile(
                tracking_profile_name, profile_overrides
            )
            gaussian_fit_method = str(resolved_profile["cursor_pose_method"])
            if gaussian_fit_method not in CursorPoseEstimator.GAUSSIAN_FIT_METHODS:
                raise ValueError(
                    "Cursor pose method must be one of: {}".format(
                        ", ".join(CursorPoseEstimator.GAUSSIAN_FIT_METHODS)
                    )
                )
            cursor_pose_process_config = {
                "calibration_path": (
                    self._minimap_calibration_root(game_profile_id)
                    / calibration_id
                ),
                "gaussian_fit_method": gaussian_fit_method,
                "validation_policy": str(
                    resolved_profile["cursor_validation_policy"]
                ),
                "opencv_threads": int(
                    resolved_profile["cursor_opencv_threads"]
                ),
                "calibration_metadata": minimap,
            }
            frame_config = dict(value.get("frame_source") or {})
            adapter = frame_config.get("adapter")
            if adapter not in (
                "windows_window",
                "android_scrcpy",
                "hik_mvs",
                "hik_rig_calibrated",
            ):
                raise ValueError(
                    "Choose a Windows window, Android phone, or HIK camera source"
                )
            if adapter == "windows_window" and not frame_config.get("window_title"):
                raise ValueError("Choose the exact game window for live tracking")
            if adapter == "android_scrcpy" and not frame_config.get("serial"):
                raise ValueError("Choose an Android device for live tracking")
            if adapter == "hik_mvs" and not frame_config.get("camera_id"):
                raise ValueError("Choose a HIK camera for live tracking")
            if adapter == "hik_rig_calibrated":
                rig = self._checked_rig_calibration(
                    frame_config.get("calibration")
                )
                frame_config["calibration"] = rig["path"]
            global_interval_s = float(resolved_profile["global_interval_s"])
            frame_source, _ = self.sources.capture_sources(
                frame_config, {"adapter": "none"}
            )
            engine = TwoRateRealtimeTracker(
                mosaic,
                minimap_config,
                minimap,
                scene_yaw,
                global_interval_s=global_interval_s,
                localizer=localizer,
                cursor_pose_process_config=cursor_pose_process_config,
                cursor_interval_s=float(resolved_profile["cursor_interval_s"]),
                temporal_pose_search=bool(
                    resolved_profile["temporal_pose_search"]
                ),
                pose_confidence_min=float(
                    resolved_profile["pose_confidence_min"]
                ),
                global_candidate_advisor=(
                    RouteCandidateAdvisor(route_package)
                    if tracking_mode == "route-assisted"
                    else None
                ),
            )
            stop = threading.Event()
            runtime = {
                "status": "starting",
                "detail": "Starting the selected capture source.",
                "game_profile_id": game_profile_id,
                "minimap_calibration_id": calibration_id,
                "scene_yaw_calibration_id": scene_yaw_id,
                "map_stitch_id": stitch_id,
                "map_atlas_id": atlas_id or None,
                "route_package_id": route_package_id or None,
                "tracking_mode": tracking_mode,
                "route_policy": (
                    "candidate-acceleration-only"
                    if tracking_mode == "route-assisted"
                    else None
                ),
                "frame_source": frame_config,
                "frame_size_wh": None,
                "minimap_crop_xywh": list(minimap_config["crop_xywh"]),
                "global_interval_s": global_interval_s,
                "cursor_pose_method": gaussian_fit_method,
                "tracking_profile": tracking_profile_name,
                "resolved_tracking_profile": resolved_profile,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "latest": None,
                "high_rate_fps": 0.0,
                "processed_frames": 0,
                "error": None,
                "route_similarity": None,
                "stop": stop,
                "thread": None,
            }
            tracking_id = "{}-{}".format(
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
                uuid.uuid4().hex[:8],
            )
            evidence_output = self._live_tracking_root(game_profile_id) / tracking_id
            evidence_recorder = LiveTrackingEvidenceRecorder(
                evidence_output,
                {
                    "tracking_id": tracking_id,
                    "game_profile_id": game_profile_id,
                    "minimap_calibration_id": calibration_id,
                    "scene_yaw_calibration_id": scene_yaw_id,
                    "map_stitch_id": stitch_id,
                    "map_atlas_id": atlas_id or None,
                    "route_package_id": route_package_id or None,
                    "tracking_mode": tracking_mode,
                    "route_policy": (
                        "candidate-acceleration-only"
                        if tracking_mode == "route-assisted"
                        else None
                    ),
                    "tracking_profile": tracking_profile_name,
                    "resolved_tracking_profile": resolved_profile,
                    "frame_source": frame_config,
                },
            )
            runtime["tracking_id"] = tracking_id
            runtime["artifact_relative_path"] = str(
                evidence_output.relative_to(self.artifact_root)
            )
            runtime["evidence"] = evidence_recorder.summary
            runtime["evidence_recorder"] = evidence_recorder
            self._live_tracker = runtime
            self._live_tracker_engine = engine
            self._live_tracker_mosaic = mosaic
            self._live_tracker_route_points = (
                [state["canonical_xy"] for state in route_package.states]
                if route_package is not None
                else None
            )
            self._last_error = None

        def work() -> None:
            recent_times = []
            frame_pump = LatestFramePump(frame_source)
            run_error = None
            try:
                frame_pump.start()
                with self._lock:
                    runtime["status"] = "running"
                    runtime["detail"] = (
                        "High-rate visual tracking is active; absolute map searches "
                        "run independently at low rate."
                    )
                while not stop.is_set():
                    packet = frame_pump.read_latest(timeout_s=0.25)
                    if packet is None:
                        continue
                    latest = engine.update(
                        packet.image, packet.host_capture_time_ns
                    )
                    take_diagnostics = getattr(
                        engine, "take_global_diagnostics", None
                    )
                    diagnostics = (
                        take_diagnostics() if callable(take_diagnostics) else None
                    )
                    extractor = getattr(engine, "extractor", None)
                    minimap = None
                    if extractor is not None:
                        minimap, _ = extractor.extract(packet.image)
                    if latest.get("global_fix_fresh"):
                        diagnostics = dict(diagnostics or {})
                        diagnostics["map_overlay"] = render_map_overlay(
                            mosaic,
                            latest,
                            route_points=self._live_tracker_route_points,
                        )
                    evidence_recorder.record(
                        packet.image, minimap, latest, diagnostics
                    )
                    now = time.perf_counter()
                    recent_times.append(now)
                    recent_times = [item for item in recent_times if now - item <= 2.0]
                    high_rate_fps = (
                        (len(recent_times) - 1) / (recent_times[-1] - recent_times[0])
                        if len(recent_times) >= 2
                        and recent_times[-1] > recent_times[0]
                        else 0.0
                    )
                    with self._lock:
                        runtime["latest"] = latest
                        runtime["processed_frames"] = latest["sequence"]
                        runtime["high_rate_fps"] = high_rate_fps
                        runtime["capture_dropped_before_processing"] = (
                            frame_pump.dropped_before_processing
                        )
                        runtime["frame_size_wh"] = [
                            int(packet.image.shape[1]),
                            int(packet.image.shape[0]),
                        ]
                        if latest["sequence"] % 30 == 0:
                            runtime["evidence"] = evidence_recorder.summary
                with self._lock:
                    runtime["status"] = "stopped"
                    runtime["detail"] = "Live tracking stopped by the user."
            except Exception as exc:
                error = "{}: {}".format(type(exc).__name__, exc)
                run_error = error
                with self._lock:
                    runtime["status"] = "failed"
                    runtime["detail"] = error
                    runtime["error"] = error
                    self._last_error = "Live tracker failed: {}".format(error)
            finally:
                stop.set()
                try:
                    frame_pump.stop()
                except Exception:
                    pass
                engine.close()
                evidence_summary = evidence_recorder.close(
                    status="failed" if run_error else "stopped",
                    error=run_error,
                    processed_frames=runtime.get("processed_frames"),
                )
                route_similarity = None
                if route_package is not None:
                    try:
                        route_similarity = write_live_route_similarity(
                            evidence_output, route_package
                        )
                    except Exception as exc:
                        route_similarity = {
                            "status": "failed",
                            "role": "post-run-review-only",
                            "feeds_tracker": False,
                            "error": "{}: {}".format(type(exc).__name__, exc),
                        }
                with self._lock:
                    runtime["evidence"] = evidence_summary
                    runtime["route_similarity"] = route_similarity

        thread = threading.Thread(
            target=work, name="acquisition-live-tracker", daemon=True
        )
        with self._lock:
            runtime["thread"] = thread
        thread.start()
        return self.descriptor()

    def stop_live_tracker(self) -> dict:
        with self._lock:
            runtime = self._live_tracker
            if not runtime or runtime.get("status") not in ("starting", "running"):
                raise RuntimeError("The live tracker is not running")
            runtime["detail"] = "Stopping live capture and tracker."
            runtime["stop"].set()
        return self.descriptor()

    def live_tracker_overlay_image(self, compact: bool = False) -> bytes:
        with self._lock:
            if self._live_tracker_mosaic is None or self._live_tracker is None:
                raise ValueError("No live tracker overlay is available")
            latest = dict(self._live_tracker.get("latest") or {})
            mosaic = self._live_tracker_mosaic
            route_points = self._live_tracker_route_points
        image = render_map_overlay(
            mosaic,
            latest,
            size=(360, 240) if compact else (520, 360),
            route_points=route_points,
        )
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("Could not encode the live tracker overlay")
        return encoded.tobytes()

    def live_tracker_minimap_route_overlay_image(self) -> bytes:
        with self._lock:
            if self._live_tracker is None or self._live_tracker_route_points is None:
                raise ValueError("No demonstrated route guide is available")
            latest = dict(self._live_tracker.get("latest") or {})
            calibration_id = self._live_tracker.get("minimap_calibration_id")
            game_profile_id = self._live_tracker.get("game_profile_id")
            crop_xywh = self._live_tracker.get("minimap_crop_xywh")
            route_points = self._live_tracker_route_points
        calibration = self._read_tracker_artifact(
            self._minimap_calibration_root(game_profile_id),
            "calibration.json",
            calibration_id,
        )
        image = render_minimap_route_overlay(
            route_points, latest, calibration, crop_xywh
        )
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("Could not encode the mini-map route guide")
        return encoded.tobytes()

    def _profile_drafts(self) -> dict:
        values = {}
        for game_profile_id in self.profiles.games:
            path = self._profile_draft_path(game_profile_id)
            if path.is_file():
                try:
                    values[game_profile_id] = json.loads(
                        path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    values[game_profile_id] = {
                        "error": "{}: {}".format(type(exc).__name__, exc)
                    }
        return values

    def save_profile_draft(self, value: dict) -> dict:
        with self._lock:
            if self._active is not None:
                raise RuntimeError("Wait for the active take to finish")
            game_profile_id = value.get("game_profile_id")
            if not game_profile_id:
                raise ValueError("Choose a game profile")
            game = self.profiles.game(game_profile_id)
            controls = value.get("controls") or {}
            if not isinstance(controls, dict):
                raise ValueError("Profile controls must be an object")
            allowed_controls = {
                str(item.get("id"))
                for item in game.get("profile_editor", {}).get("controls", [])
                if item.get("id")
            }
            if allowed_controls:
                unknown = sorted(set(controls) - allowed_controls)
                if unknown:
                    raise ValueError(
                        "Unknown control profile fields: {}".format(", ".join(unknown))
                    )
            cleaned = {}
            for key, item in controls.items():
                if not isinstance(item, dict):
                    raise ValueError("Each control field must be an object")
                cleaned[key] = {
                    "binding": str(item.get("binding") or "").strip(),
                    "activation": str(item.get("activation") or "unknown").strip(),
                    "status": str(item.get("status") or "human_confirmed").strip(),
                }
            draft = {
                "schema_version": "1.0",
                "game_profile_id": game_profile_id,
                "source_profile_file": game.get("source_file"),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "controls": cleaned,
                "behavior_notes": str(value.get("behavior_notes") or "").strip(),
                "map_viewer_notes": str(value.get("map_viewer_notes") or "").strip(),
            }
            _write_json_atomic(self._profile_draft_path(game_profile_id), draft)
            self._refresh_poc_evidence_index(game_profile_id)
            self._last_error = None
        return self.descriptor()

    def _desktop(self):
        if self.desktop_api is None:
            self.desktop_api = WindowsDesktopApi()
            self.sources.desktop_api = self.desktop_api
        return self.desktop_api

    def _windows(self) -> List[dict]:
        try:
            return [
                {"handle": handle, "title": title}
                for handle, title in self._desktop().list_windows()
            ]
        except Exception as exc:
            return [{"error": "{}: {}".format(type(exc).__name__, exc)}]

    def _preflight_input_integrity(
        self, window_title: Optional[str], input_adapter: str
    ) -> Optional[dict]:
        """Reject an unreadable Windows input target before creating a session."""
        if input_adapter != "windows_raw_keyboard_mouse":
            return None
        desktop = self._desktop()
        if not hasattr(desktop, "input_integrity_status"):
            return None
        hwnd, matched_title = select_window(
            desktop.list_windows(), str(window_title or ""), exact=True
        )
        status = dict(desktop.input_integrity_status(hwnd))
        status["matched_window_title"] = matched_title
        if not status.get("matched", True):
            target = "elevated" if status.get("target_elevated") else "not elevated"
            recorder = (
                "elevated" if status.get("recorder_elevated") else "not elevated"
            )
            raise RuntimeError(
                "Input privilege mismatch: target {!r} is {}, but the Workbench "
                "serving this page is {}. Stop the Workbench currently using "
                "this port, then launch it at the same privilege level as the "
                "game; a second elevated instance cannot replace a server that "
                "already owns the port.".format(matched_title, target, recorder)
            )
        return status

    def arm(self, value: dict) -> dict:
        with self._lock:
            if self._active is not None:
                raise RuntimeError("Cannot change configuration while a take is active")
            game_id = value.get("game_profile_id")
            route_profile_id = value.get("route_profile_id")
            game = self.profiles.game(game_id) if game_id else None
            route = self.profiles.route(route_profile_id) if route_profile_id else None
            if route and game and route["game_profile_id"] != game["profile_id"]:
                raise ValueError("Selected route does not belong to the game profile")

            frame_config = dict((game or {}).get("default_frame_source", {}))
            input_config = dict((game or {}).get("default_input_source", {}))
            frame_config.update(value.get("frame_source") or {})
            input_config.update(value.get("input_source") or {})
            if not frame_config.get("adapter"):
                raise ValueError("Choose a frame source adapter")
            if not input_config.get("adapter"):
                input_config["adapter"] = "none"
            if frame_config.get("adapter") == "windows_window":
                window_title = value.get("window_title") or frame_config.get("window_title")
                if not window_title:
                    raise ValueError("Choose a visible Windows game window")
                frame_config.update(window_title=window_title, exact_title=True)
                if input_config.get("adapter") in (
                    "windows_xinput",
                    "windows_raw_keyboard_mouse",
                    "windows_keyboard_mouse",
                ):
                    input_config.update(window_title=window_title, exact_title=True)
            elif frame_config.get("adapter") == "android_scrcpy":
                serial = str(frame_config.get("serial") or "").strip()
                if not serial:
                    raise ValueError("Choose a connected Android device")
                if input_config.get("adapter") not in ("adb_getevent", "none"):
                    raise ValueError(
                        "Android display capture supports Android getevent or no input capture"
                    )
                frame_config["serial"] = serial
                if input_config.get("adapter") == "adb_getevent":
                    input_config["serial"] = serial
            elif frame_config.get("adapter") in (
                "hik_mvs",
                "hik_rig_calibrated",
            ):
                if frame_config["adapter"] == "hik_mvs" and not str(
                    frame_config.get("camera_id") or ""
                ).strip():
                    raise ValueError("Choose a connected HIK camera")
                if frame_config["adapter"] == "hik_rig_calibrated" and not str(
                    frame_config.get("calibration") or ""
                ).strip():
                    raise ValueError("Choose a HIK rig calibration")
                if input_config.get("adapter") not in ("adb_getevent", "none"):
                    raise ValueError(
                        "HIK phone capture supports Android getevent or no input capture"
                    )
                if input_config.get("adapter") == "adb_getevent":
                    serial = str(input_config.get("serial") or "").strip()
                    if not serial:
                        raise ValueError("Choose the Android phone used by the HIK rig")
                    input_config["serial"] = serial

            capture_kind = str(value.get("capture_kind") or "route")
            if capture_kind not in self.CAPTURE_KINDS:
                raise ValueError("Unsupported capture kind: {}".format(capture_kind))
            capture_id = value.get("capture_id") or value.get("route_id") or (
                route or {}
            ).get("route_id")
            if not capture_id:
                raise ValueError("A capture ID is required")
            capture_id = safe_id(capture_id)
            workflow_stage_id = value.get("workflow_stage_id") or None
            workflow_stage = None
            if workflow_stage_id:
                if game is None:
                    raise ValueError("A workflow stage requires a game profile")
                workflow_stage = next(
                    (
                        dict(item)
                        for item in game.get("poc_workflow", [])
                        if item.get("stage_id") == workflow_stage_id
                    ),
                    None,
                )
                if workflow_stage is None:
                    raise ValueError(
                        "Unknown workflow stage for {}: {}".format(
                            game["profile_id"], workflow_stage_id
                        )
                    )
                if workflow_stage.get("capture_kind") != capture_kind:
                    raise ValueError("Workflow stage and capture kind disagree")
                if safe_id(workflow_stage.get("capture_id")) != capture_id:
                    raise ValueError("Workflow stage and capture ID disagree")
            route_id = (
                value.get("route_id") or (route or {}).get("route_id") or capture_id
            )
            if capture_kind == "route" and not route_id:
                raise ValueError("A route ID is required")
            experiment_id = safe_id(value.get("experiment_id") or capture_id)
            target_runs = int(value.get("target_runs") or (route or {}).get("target_runs", 3))
            capture_duration_s = float(
                value.get("capture_duration_s")
                or (route or {}).get("capture_duration_s", 30.0)
            )
            if target_runs < 1 or target_runs > 20:
                raise ValueError("Target runs must be between 1 and 20")
            if capture_duration_s < 5.0 or capture_duration_s > 600.0:
                raise ValueError("Capture duration must be between 5 and 600 seconds")

            segment_label = value.get("segment_label") or (
                workflow_stage or {}
            ).get("segment_label")
            segment_semantics = value.get("segment_semantics") or (
                workflow_stage or {}
            ).get("segment_semantics")
            start_trigger = str(
                value.get("start_trigger")
                or (workflow_stage or {}).get("start_trigger")
                or "first_input"
            )
            if start_trigger not in ("first_input", "settled_timer"):
                raise ValueError("Unsupported recording start trigger: {}".format(start_trigger))
            input_requirement = str(
                value.get("input_requirement")
                or (workflow_stage or {}).get("input_requirement")
                or ("required" if start_trigger == "first_input" else "optional")
            )
            if input_requirement not in ("required", "optional", "none"):
                raise ValueError(
                    "Input requirement must be required, optional, or none"
                )
            if input_requirement == "required" and input_config.get("adapter") == "none":
                raise ValueError("This capture requires an input adapter")
            if start_trigger == "first_input" and input_config.get("adapter") == "none":
                raise ValueError("First-input start requires an input adapter")
            start_delay_s = float(
                value.get("start_delay_s")
                or (workflow_stage or {}).get("start_delay_s")
                or self.INPUT_SETTLE_DELAY_S
            )
            if start_delay_s < 0.0 or start_delay_s > 30.0:
                raise ValueError("Recording start delay must be between 0 and 30 seconds")

            self._armed = {
                "experiment_id": experiment_id,
                "game_profile_id": game_id,
                "route_profile_id": route_profile_id,
                "route_id": route_id,
                "capture_kind": capture_kind,
                "capture_id": capture_id,
                "workflow_stage_id": workflow_stage_id,
                "workflow_stage": workflow_stage,
                "segment_label": segment_label,
                "segment_semantics": segment_semantics,
                "start_trigger": start_trigger,
                "input_requirement": input_requirement,
                "start_delay_s": start_delay_s,
                "game_profile_draft": self._profile_drafts().get(game_id),
                "confirmation_label": value.get("confirmation_label")
                or ("full route boundary" if capture_kind == "route" else "useful capture"),
                "target_runs": target_runs,
                "capture_duration_s": capture_duration_s,
                "frame_source": frame_config,
                "input_source": input_config,
                "armed_utc": datetime.now(timezone.utc).isoformat(),
            }
            self._last_error = None
            self._compile_state = "not_ready"
            self._compile_result = None
            self._hud_notice = None
            self._persist_state()
        return self.descriptor()

    def disarm(self) -> dict:
        with self._lock:
            if self._active is not None:
                raise RuntimeError("Cancel the active take before changing configuration")
            self._armed = None
            self._hud_notice = None
            self._persist_state()
        return self.descriptor()

    def _experiment_root(self) -> Path:
        if self._armed is None:
            raise RuntimeError("Configure and arm an experiment first")
        return self.session_root / self._armed["experiment_id"]

    def _run_path(self, run_index: int) -> Path:
        return self._experiment_root() / "run_{:02d}".format(run_index)

    def _session_key(self, path: Path) -> str:
        return path.resolve().relative_to(self.session_root.resolve()).as_posix()

    def _session_path(self, session_key: str, require_manifest: bool = True) -> Path:
        pure = PurePosixPath(str(session_key or ""))
        if (
            pure.is_absolute()
            or len(pure.parts) != 2
            or any(part in ("", ".", "..") for part in pure.parts)
            or not re.fullmatch(r"run_\d+", pure.parts[1])
        ):
            raise ValueError("Invalid session identifier")
        root = self.session_root.resolve()
        path = (root / Path(*pure.parts)).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            raise ValueError("Session must stay inside the session root")
        if require_manifest and not (path / "manifest.json").is_file():
            raise ValueError("Unknown recorded session")
        return path

    def _session_metadata(self, path: Path) -> dict:
        metadata_path = path / self.SESSION_METADATA_FILENAME
        if not metadata_path.is_file():
            return {}
        try:
            value = json.loads(metadata_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _label_definition(self, label: str) -> dict:
        for item in self.SESSION_LABELS:
            if item["value"] == label:
                return dict(item)
        raise ValueError("Unknown session label")

    def _describe_session(self, path: Path) -> dict:
        signature = []
        for name in (
            "manifest.json",
            "annotations.jsonl",
            self.SESSION_METADATA_FILENAME,
            "frames.jsonl",
            "inputs.jsonl",
        ):
            item_path = path / name
            if item_path.is_file():
                stat = item_path.stat()
                signature.append((name, stat.st_mtime_ns, stat.st_size))
        cache_key = str(path.resolve())
        cached = self._session_summary_cache.get(cache_key)
        if cached and cached["signature"] == signature:
            return dict(cached["value"])
        reader = SessionReader(path)
        annotations = AnnotationStore(path).list()
        kinds = [item["kind"] for item in annotations]
        input_health = input_capture_health(reader.manifest, reader.inputs)
        metadata = self._session_metadata(path)
        context = reader.manifest.get("context") or {}
        if "route_failed" in kinds or "capture_failed" in kinds:
            status = "failed"
        elif not input_health["healthy"]:
            status = "failed"
        elif (
            "route_start" in kinds and "route_complete" in kinds
        ) or (
            "capture_start" in kinds and "capture_complete" in kinds
        ):
            status = "ready"
        elif "take_start" in kinds and "take_end" in kinds:
            status = "recorded"
        else:
            status = "incomplete"
        label = metadata.get("label")
        if label is None:
            label = context.get("segment_label") or ""
        value = {
            "session_key": self._session_key(path),
            "session_id": reader.manifest.get("session_id"),
            "experiment_id": context.get("experiment_id") or path.parent.name,
            "run_index": context.get("run_index"),
            "game_profile_id": context.get("game_profile_id"),
            "window_title": (
                (reader.manifest.get("frame_sources") or [{}])[0].get(
                    "matched_window_title"
                )
                or (reader.manifest.get("frame_sources") or [{}])[0].get(
                    "window_title_query"
                )
            ),
            "created_utc": reader.manifest.get("created_utc"),
            "finished_utc": reader.manifest.get("finished_utc"),
            "duration_s": reader.manifest.get("duration_ns", 0) / 1.0e9,
            "frames": len(reader.frames_by_stream.get("main", [])),
            "input_events": len(reader.inputs),
            "dropped_frames": reader.manifest.get("dropped_frames", {}).get(
                "main", 0
            ),
            "status": status,
            "label": label,
            "markers": kinds,
            "input_capture": input_health,
        }
        self._session_summary_cache[cache_key] = {
            "signature": signature,
            "value": dict(value),
        }
        return value

    def sessions(self) -> List[dict]:
        values = []
        paths = []
        active_path = (
            Path(self._active["path"]).resolve()
            if self._active is not None
            else None
        )
        for manifest_path in self.session_root.glob("*/run_*/manifest.json"):
            if (
                re.fullmatch(r"run_\d+", manifest_path.parent.name)
                and manifest_path.parent.resolve() != active_path
            ):
                paths.append(manifest_path.parent)
        for path in paths:
            try:
                values.append(self._describe_session(path))
            except Exception as exc:
                values.append(
                    {
                        "session_key": self._session_key(path),
                        "status": "invalid",
                        "label": "",
                        "error": "{}: {}".format(type(exc).__name__, exc),
                    }
                )
        values.sort(
            key=lambda item: item.get("finished_utc")
            or item.get("created_utc")
            or item.get("session_key", ""),
            reverse=True,
        )
        if self._active is not None:
            active_key = self._session_key(active_path)
            values.insert(
                0,
                {
                    "session_key": active_key,
                    "experiment_id": active_path.parent.name,
                    "run_index": self._active["run_index"],
                    "game_profile_id": (self._armed or {}).get(
                        "game_profile_id"
                    ),
                    "created_utc": None,
                    "duration_s": None,
                    "frames": None,
                    "input_events": self._active.get("recorded_input_events", 0),
                    "dropped_frames": None,
                    "status": self._active["phase"],
                    "label": "",
                },
            )
        return values

    @staticmethod
    def _archive_existing(path: Path) -> None:
        if not path.exists():
            return
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        candidate = path.with_name(path.name + ".previous-" + suffix)
        counter = 1
        while candidate.exists():
            candidate = path.with_name(
                path.name + ".previous-{}-{}".format(suffix, counter)
            )
            counter += 1
        path.rename(candidate)

    def _slot(self, run_index: int) -> dict:
        path = self._run_path(run_index)
        if self._active is not None and self._active["run_index"] == run_index:
            return {
                "run_index": run_index,
                "status": self._active["phase"],
                "path": str(path),
            }
        if not (path / "manifest.json").is_file():
            return {"run_index": run_index, "status": "empty", "path": str(path)}
        try:
            reader = SessionReader(path)
            annotations = AnnotationStore(path).list()
            kinds = [item["kind"] for item in annotations]
            input_health = input_capture_health(reader.manifest, reader.inputs)
            if "route_failed" in kinds or "capture_failed" in kinds:
                status = "needs_rerecord"
            elif not input_health["healthy"]:
                status = "needs_rerecord"
            elif (
                "route_start" in kinds and "route_complete" in kinds
            ) or (
                "capture_start" in kinds and "capture_complete" in kinds
            ):
                status = "ready"
            elif "take_start" in kinds and "take_end" in kinds:
                status = "captured_needs_confirmation"
            else:
                status = "invalid"
            return {
                "run_index": run_index,
                "status": status,
                "path": str(path),
                "frames": len(reader.frames_by_stream.get("main", [])),
                "duration_s": reader.manifest.get("duration_ns", 0) / 1.0e9,
                "dropped_frames": reader.manifest.get("dropped_frames", {}).get(
                    "main", 0
                ),
                "input_events": len(reader.inputs),
                "control_input_events": input_health["control_events"],
                "input_capture": input_health,
                "markers": kinds,
            }
        except Exception as exc:
            return {
                "run_index": run_index,
                "status": "invalid",
                "path": str(path),
                "error": "{}: {}".format(type(exc).__name__, exc),
            }

    def _runs(self) -> List[dict]:
        if self._armed is None:
            return []
        indexes = set()
        root = self._experiment_root()
        if root.is_dir():
            for path in root.glob("run_*"):
                match = re.fullmatch(r"run_(\d+)", path.name)
                if match:
                    indexes.add(int(match.group(1)))
        if self._active is not None:
            indexes.add(int(self._active["run_index"]))
        return [self._slot(index) for index in sorted(indexes)]

    def descriptor(self) -> dict:
        with self._lock:
            runs = self._runs()
            all_ready = bool(runs) and all(run["status"] == "ready" for run in runs)
            return {
                "schema_version": "1.2",
                "instance": dict(self._instance),
                "profiles": self.profiles.descriptor(),
                "game_profile_drafts": self._profile_drafts(),
                "poc_evidence_indexes": self._poc_evidence_indexes(),
                "minimap_calibrations": self._minimap_calibrations(),
                "scene_yaw_calibrations": self._scene_yaw_calibrations(),
                "map_stitches": self._map_stitches(),
                "map_atlases": self._map_atlases(),
                "route_tracking_packages": self._route_tracking_packages(),
                "teleport_behaviors": self._teleport_behaviors(),
                "live_tracking_runs": self._live_tracking_runs(),
                "analysis_candidates": self.analysis_candidates(),
                "analysis_jobs": {
                    key: dict(value) for key, value in self._analysis_jobs.items()
                },
                "live_tracker": self._live_tracker_descriptor(),
                "hud_runtime": dict(self._hud_runtime),
                "sources": self.sources.descriptor(),
                "visible_windows": self._windows(),
                "armed": self._armed,
                "runs": runs,
                "sessions": self.sessions(),
                "session_labels": [dict(item) for item in self.SESSION_LABELS],
                "all_runs_ready": all_ready,
                "active_run": self._active["run_index"] if self._active else None,
                "active_phase": self._active["phase"] if self._active else None,
                "android_control": (
                    dict(self._android_control) if self._android_control else None
                ),
                "capture_policy": {
                    "in_game_controls": "none",
                    "start": "sources_ready_then_queue_input_settles_then_first_active_input",
                    "end": "fixed_duration_or_user_cancel",
                    "route_bounds": "post_take_confirmation",
                    "stages": "derived_or_annotated_after_recording",
                },
                "last_error": self._last_error,
                "last_capture_diagnostics": self._last_capture_diagnostics,
                "compile_state": (
                    self._compile_state
                    if all_ready and (self._armed or {}).get("capture_kind") == "route"
                    else "not_applicable"
                    if self._armed and self._armed.get("capture_kind") != "route"
                    else "not_ready"
                ),
                "compile_result": self._compile_result,
            }

    def android_devices(self) -> dict:
        """Enumerate devices only on explicit request from the Android UI."""
        try:
            return {"devices": self.sources.android_devices(), "error": None}
        except Exception as exc:
            return {
                "devices": [],
                "error": "{}: {}".format(type(exc).__name__, exc),
            }

    def _rig_artifact_root(self) -> Path:
        """Return the workspace artifact root without assuming one deployment layout."""

        return (
            self.artifact_root.parent
            if self.artifact_root.name == "workbench"
            else self.artifact_root
        )

    def rig_calibrations(self) -> List[dict]:
        """Describe saved HIK rig calibrations without opening camera hardware."""

        values = []
        root = self._rig_artifact_root()
        if not root.is_dir():
            return values
        for directory in sorted(
            root.glob("hik-calibration-*"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        ):
            path = directory / "hik_camera_calibration.json"
            if not path.is_file():
                continue
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
                camera = document.get("camera") or {}
                phone = document.get("phone") or {}
                normalization = document.get("normalization") or {}
                dense_name = str(normalization.get("dense_map_file") or "")
                dense_path = directory / dense_name if dense_name else None
                valid_mask = directory / "valid_screen_mask.png"
                usable = bool(
                    camera.get("device_id")
                    and phone.get("serial")
                    and dense_path is not None
                    and dense_path.is_file()
                    and valid_mask.is_file()
                )
                metadata = camera.get("metadata") or {}
                values.append(
                    {
                        "calibration_id": directory.name,
                        "path": str(path.resolve()),
                        "camera_id": str(camera.get("device_id") or ""),
                        "camera_model": str(
                            metadata.get("model")
                            or camera.get("model")
                            or "HIK camera"
                        ),
                        "phone_serial": str(phone.get("serial") or ""),
                        "phone_model": str(phone.get("model") or "Android phone"),
                        "usable": usable,
                        "status": "adapter_ready" if usable else "incomplete",
                        "updated_utc": datetime.fromtimestamp(
                            path.stat().st_mtime, timezone.utc
                        ).isoformat(),
                    }
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                values.append(
                    {
                        "calibration_id": directory.name,
                        "path": str(path.resolve()),
                        "usable": False,
                        "status": "invalid",
                        "error": "{}: {}".format(type(exc).__name__, exc),
                    }
                )
        return values

    def capture_source_inventory(self) -> dict:
        """Enumerate selectable phones, HIK cameras, and calibrated HIK rigs."""

        android = self.android_devices()
        try:
            hik_devices = self.sources.hik_devices()
            hik_error = None
        except Exception as exc:
            hik_devices = []
            hik_error = "{}: {}".format(type(exc).__name__, exc)
        return {
            "android_devices": android["devices"],
            "android_error": android["error"],
            "hik_cameras": hik_devices,
            "hik_error": hik_error,
            "rig_calibrations": self.rig_calibrations(),
        }

    def _checked_rig_calibration(self, value: object) -> dict:
        selected = Path(str(value or "")).resolve()
        for calibration in self.rig_calibrations():
            if Path(calibration["path"]).resolve() == selected:
                if not calibration.get("usable"):
                    raise ValueError("The selected HIK rig calibration is incomplete")
                return calibration
        raise ValueError("Choose a HIK rig calibration from this workspace")

    def run_android_straight_forward(self, value: dict) -> dict:
        """Hold one exact touchscreen vector through ADB motion events."""
        try:
            start_x = int(value.get("start_x"))
            start_y = int(value.get("start_y"))
            end_x = int(value.get("end_x"))
            end_y = int(value.get("end_y"))
            duration_s = float(value.get("duration_s"))
        except (TypeError, ValueError):
            raise ValueError("Enter integer start/end coordinates and a hold duration")
        if min(start_x, start_y, end_x, end_y) < 0:
            raise ValueError("Touch coordinates cannot be negative")
        if (start_x, start_y) == (end_x, end_y):
            raise ValueError("Straight-forward touch needs a non-zero movement vector")
        if duration_s < 0.25 or duration_s > 120.0:
            raise ValueError("Straight-forward hold must be between 0.25 and 120 seconds")

        with self._lock:
            active = self._active
            armed = self._armed or {}
            frame_config = armed.get("frame_source") or {}
            if active is None:
                raise RuntimeError("Start an Android recording first")
            if frame_config.get("adapter") != "android_scrcpy":
                raise RuntimeError("Straight-forward touch is available only for Android capture")
            if active.get("phase") not in (
                "waiting_for_first_input",
                "recording_uninterrupted_take",
            ):
                raise RuntimeError("Wait until Android input is ready")
            if self._android_control and self._android_control.get("status") == "running":
                raise RuntimeError("A straight-forward touch is already running")
            serial = str(frame_config.get("serial") or "")
            adb = self.sources._adb(frame_config)
            control = {
                "status": "running",
                "kind": "straight_forward_touch",
                "serial": serial,
                "start_xy": [start_x, start_y],
                "end_xy": [end_x, end_y],
                "requested_duration_s": duration_s,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "detail": "Holding the configured Android movement vector.",
            }
            self._android_control = control

        def motion(action: str, x: int, y: int) -> None:
            command = [str(adb), "-s", serial, "shell", "input", "touchscreen", "motionevent", action, str(x), str(y)]
            subprocess.check_call(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )

        def work() -> None:
            down = False
            timing = {}
            error = None
            try:
                timing["down_host_time_ns"] = time.perf_counter_ns()
                motion("DOWN", start_x, start_y)
                down = True
                timing["move_host_time_ns"] = time.perf_counter_ns()
                motion("MOVE", end_x, end_y)
                active["stop"].wait(duration_s)
            except Exception as exc:
                error = "{}: {}".format(type(exc).__name__, exc)
            finally:
                if down:
                    try:
                        timing["up_host_time_ns"] = time.perf_counter_ns()
                        motion("UP", end_x, end_y)
                    except Exception as exc:
                        if error is None:
                            error = "{}: {}".format(type(exc).__name__, exc)
                result = dict(control)
                result.update(timing)
                result["finished_utc"] = datetime.now(timezone.utc).isoformat()
                result["status"] = "failed" if error else "complete"
                result["detail"] = error or "Straight-forward touch released."
                with self._lock:
                    active.setdefault("android_control_events", []).append(result)
                    self._android_control = result
                    if error:
                        self._last_error = "Android control failed: {}".format(error)

        control_thread = threading.Thread(
            target=work,
            name="android-straight-forward-control",
            daemon=True,
        )
        with self._lock:
            active["android_control_thread"] = control_thread
        control_thread.start()
        return self.descriptor()

    def _next_run_index(self) -> int:
        if self._armed is None:
            raise RuntimeError("Configure and arm an experiment first")
        indexes = []
        root = self._experiment_root()
        if root.is_dir():
            for path in root.glob("run_*"):
                match = re.fullmatch(r"run_(\d+)", path.name)
                if match:
                    indexes.append(int(match.group(1)))
        return max(indexes or [0]) + 1

    def queue_next_take(self) -> dict:
        return self.queue_take(self._next_run_index())

    def start_session(self, value: dict) -> dict:
        """Apply the simple recorder settings and append one new session."""
        with self._lock:
            if self._tracker_running():
                raise RuntimeError("Stop live tracking before recording a session")
        game_id = value.get("game_profile_id") or None
        experiment_id = value.get("experiment_id") or "recordings-{}".format(
            game_id or "custom"
        )
        input_adapter = str(value.get("input_adapter") or "none")
        game = self.profiles.game(game_id) if game_id else None
        capture_adapter = str(
            value.get("capture_adapter")
            or (game or {}).get("default_frame_source", {}).get("adapter")
            or "windows_window"
        )
        if capture_adapter not in (
            "windows_window",
            "android_scrcpy",
            "hik_mvs",
            "hik_rig_calibrated",
        ):
            raise ValueError("Unsupported recorder capture adapter: {}".format(capture_adapter))
        window_title = value.get("window_title") or (
            (game or {}).get("default_frame_source", {}).get("window_title")
        )
        serial = str(value.get("serial") or "").strip()
        if capture_adapter == "android_scrcpy":
            if not serial:
                raise ValueError("Choose a connected Android device")
            if input_adapter not in ("adb_getevent", "none"):
                raise ValueError(
                    "Android sessions support Android getevent or no input capture"
                )
            android_fps = float(value.get("fps") or 60)
            frame_source = {
                "adapter": "android_scrcpy",
                "serial": serial,
                "fps": android_fps,
                "max_fps": android_fps,
                "bit_rate": int(value.get("bit_rate") or 16_000_000),
            }
            input_source = {"adapter": input_adapter, "serial": serial}
            window_title = None
        elif capture_adapter in ("hik_mvs", "hik_rig_calibrated"):
            if input_adapter not in ("adb_getevent", "none"):
                raise ValueError(
                    "HIK phone sessions support Android getevent or no input capture"
                )
            fps = float(value.get("fps") or 30)
            if capture_adapter == "hik_mvs":
                camera_id = str(value.get("camera_id") or "").strip()
                if not camera_id:
                    raise ValueError("Choose a connected HIK camera")
                frame_source = {
                    "adapter": "hik_mvs",
                    "camera_id": camera_id,
                    "fps": fps,
                }
            else:
                rig = self._checked_rig_calibration(value.get("rig_calibration"))
                frame_source = {
                    "adapter": "hik_rig_calibrated",
                    "camera_id": rig["camera_id"],
                    "calibration": rig["path"],
                    "fps": fps,
                }
                if not serial:
                    serial = rig["phone_serial"]
            if input_adapter == "adb_getevent" and not serial:
                raise ValueError("Choose the Android phone used by the HIK rig")
            input_source = {"adapter": input_adapter}
            if input_adapter == "adb_getevent":
                input_source["serial"] = serial
            window_title = None
        else:
            self._preflight_input_integrity(window_title, input_adapter)
            frame_source = {
                "adapter": "windows_window",
                "fps": float(value.get("fps") or 30),
            }
            input_source = {
                "adapter": input_adapter,
                "poll_hz": 250 if input_adapter == "windows_xinput" else 125,
            }
        self.arm(
            {
                "game_profile_id": game_id,
                "experiment_id": experiment_id,
                "capture_kind": "game_profile",
                "capture_id": "unlabeled-session",
                "route_id": "unlabeled-session",
                "target_runs": 1,
                "capture_duration_s": value.get("capture_duration_s") or 30,
                "window_title": window_title,
                "frame_source": frame_source,
                "input_source": input_source,
                "start_trigger": (
                    "settled_timer" if input_adapter == "none" else "first_input"
                ),
                "start_delay_s": float(value.get("start_delay_s") or 3),
                "input_requirement": (
                    "none" if input_adapter == "none" else "required"
                ),
                "confirmation_label": "labeled session",
            }
        )
        return self.queue_next_take()

    def queue_take(self, run_index: int) -> dict:
        with self._lock:
            if self._armed is None:
                raise RuntimeError("Configure and arm an experiment first")
            if self._active is not None:
                raise RuntimeError("A take is already active")
            if self._tracker_running():
                raise RuntimeError("Stop live tracking before recording a session")
            if run_index < 1:
                raise ValueError("Run index must be positive")
            if run_index > int(self._armed.get("target_runs") or 1):
                self._armed["target_runs"] = run_index
                self._persist_state()
            config = dict(self._armed)
            config["frame_source"] = dict(self._armed["frame_source"])
            config["input_source"] = dict(self._armed["input_source"])
            if (
                config.get("start_trigger", "first_input") == "first_input"
                and config["input_source"].get("adapter", "none") == "none"
            ):
                raise RuntimeError(
                    "Choose an input adapter; first-input session start cannot "
                    "work with input capture disabled"
                )
            self._preflight_input_integrity(
                config["frame_source"].get("window_title"),
                config["input_source"].get("adapter", "none"),
            )
            active = {
                "run_index": run_index,
                "path": self._run_path(run_index),
                "phase": "arming_sources",
                "cancel": threading.Event(),
                "stop": threading.Event(),
                "recording_started_host_time_ns": None,
                "recording_deadline_host_time_ns": None,
                "input_eligible_host_time_ns": None,
                "recorded_input_events": 0,
                "first_input_kind": None,
                "android_control_events": [],
                "android_control_thread": None,
            }
            self._active = active
            self._android_control = None
            self._hud_notice = None
            self._last_error = None
            self._last_capture_diagnostics = None

        def work() -> None:
            keep_session = False
            capture_diagnostics = None
            hud_result = {
                "state": "failed",
                "run_index": run_index,
                "detail": "Return to the workbench and rerecord.",
            }
            try:
                self._archive_existing(active["path"])
                frame_source, input_source = self.sources.capture_sources(
                    config["frame_source"], config["input_source"]
                )
                if input_source is not None and hasattr(
                    input_source, "disable_foreground_filter"
                ):
                    input_source.disable_foreground_filter()
                recorder = AcquisitionRecorder(
                    active["path"],
                    [frame_source],
                    [input_source] if input_source is not None else [],
                    queue_size=8192,
                    video_encoding="h264",
                    video_fps=float(config["frame_source"].get("fps", 30.0)),
                    session_context={
                        "experiment_id": config["experiment_id"],
                        "game_profile_id": config["game_profile_id"],
                        "route_profile_id": config["route_profile_id"],
                        "route_id": config["route_id"],
                        "capture_kind": config["capture_kind"],
                        "capture_id": config["capture_id"],
                        "workflow_stage_id": config.get("workflow_stage_id"),
                        "workflow_stage": config.get("workflow_stage"),
                        "segment_label": config.get("segment_label"),
                        "segment_semantics": config.get("segment_semantics"),
                        "start_trigger": config.get("start_trigger"),
                        "input_requirement": config.get("input_requirement"),
                        "start_delay_s": config.get("start_delay_s"),
                        "game_profile_draft": config.get("game_profile_draft"),
                        "run_index": run_index,
                        "frame_adapter": config["frame_source"]["adapter"],
                        "input_adapter": config["input_source"].get(
                            "adapter", "none"
                        ),
                        "capture_policy": (
                            "settled_active_lifetime_first_input_start_v5"
                            if config.get("start_trigger") == "first_input"
                            else "settled_timer_start_segment_v1"
                        ),
                    },
                )
                def sources_started() -> None:
                    with self._lock:
                        if self._active is active:
                            active["input_eligible_host_time_ns"] = (
                                time.perf_counter_ns()
                                + int(config["start_delay_s"] * 1.0e9)
                            )
                            active["phase"] = "settling_queue_input"

                def input_eligible() -> None:
                    with self._lock:
                        if self._active is active:
                            active["phase"] = "waiting_for_first_input"

                def recording_started(packet) -> None:
                    with self._lock:
                        if self._active is active:
                            started_ns = (
                                packet.host_time_ns
                                if packet is not None
                                else active["input_eligible_host_time_ns"]
                            )
                            active["recording_started_host_time_ns"] = started_ns
                            active["recording_deadline_host_time_ns"] = (
                                started_ns
                                + int(config["capture_duration_s"] * 1.0e9)
                            )
                            active["first_input_kind"] = (
                                packet.kind if packet is not None else None
                            )
                            active["phase"] = "recording_uninterrupted_take"

                def input_recorded(_packet) -> None:
                    with self._lock:
                        if self._active is active:
                            active["recorded_input_events"] += 1

                manifest = recorder.run(
                    duration_s=config["capture_duration_s"],
                    external_stop=active["stop"],
                    on_sources_started=sources_started,
                    start_on_input=config.get("start_trigger") == "first_input",
                    start_after_delay_s=(
                        config["start_delay_s"]
                        if config.get("start_trigger") == "settled_timer"
                        else None
                    ),
                    input_start_delay_s=config["start_delay_s"],
                    input_start_predicate=input_packet_is_active,
                    on_input_eligible=input_eligible,
                    on_recording_started=recording_started,
                    on_input_recorded=input_recorded,
                )
                active["stop"].set()
                control_thread = active.get("android_control_thread")
                if control_thread is not None:
                    control_thread.join(timeout=6)
                with self._lock:
                    if self._active is active:
                        active["phase"] = "finalizing_capture"

                reader = SessionReader(active["path"])
                input_health = input_capture_health(reader.manifest, reader.inputs)
                capture_diagnostics = capture_diagnostic_snapshot(
                    reader, input_health, active
                )
                with self._lock:
                    self._last_capture_diagnostics = capture_diagnostics
                session_started = bool(
                    (manifest.get("recording_start") or {}).get("started")
                )
                frames = reader.frames_by_stream.get("main", [])
                duration_ns = int(manifest.get("duration_ns") or 0)
                empty_input = not input_health["healthy"]
                failed = bool(
                    active["cancel"].is_set()
                    or not session_started
                    or empty_input
                    or manifest.get("status") != "complete"
                    or duration_ns <= 0
                    or not frames
                )
                if not failed:
                    if active.get("android_control_events"):
                        _write_json_atomic(
                            active["path"] / "android_control.json",
                            {
                                "schema_version": "1.0",
                                "events": list(active["android_control_events"]),
                            },
                        )
                    self._finalize_take(
                        active["path"],
                        config["route_id"],
                        False,
                        active["recording_started_host_time_ns"],
                        capture_kind=config["capture_kind"],
                        capture_id=config["capture_id"],
                    )
                    keep_session = True
                if not session_started:
                    with self._lock:
                        self._last_error = (
                            "No gameplay input was received; the session never "
                            "started. The partial capture was discarded. Check "
                            "the input adapter and try again."
                        )
                elif empty_input:
                    with self._lock:
                        self._last_error = input_failure_message(
                            capture_diagnostics
                        )
                elif active["cancel"].is_set():
                    with self._lock:
                        self._last_error = (
                            "Take was canceled; the partial capture was discarded"
                        )
                elif manifest.get("status") != "complete":
                    with self._lock:
                        self._last_error = (
                            "Recording did not complete; the partial capture was "
                            "discarded"
                        )
                elif duration_ns <= 0 or not frames:
                    with self._lock:
                        self._last_error = (
                            "Recording contained no usable video duration or "
                            "frames; the partial capture was discarded"
                        )
                else:
                    hud_result["state"] = "complete"
            except Exception as exc:
                hud_result["detail"] = "{}: {}".format(type(exc).__name__, exc)
                with self._lock:
                    self._last_error = "{}: {}".format(type(exc).__name__, exc)
            finally:
                active["stop"].set()
                if capture_diagnostics is None and active["path"].is_dir():
                    try:
                        failed_reader = SessionReader(active["path"])
                        failed_health = input_capture_health(
                            failed_reader.manifest, failed_reader.inputs
                        )
                        capture_diagnostics = capture_diagnostic_snapshot(
                            failed_reader, failed_health, active
                        )
                        with self._lock:
                            self._last_capture_diagnostics = capture_diagnostics
                    except Exception:
                        # The original recorder error remains authoritative when
                        # no readable manifest was produced.
                        pass
                if not keep_session:
                    try:
                        if active["path"].exists():
                            shutil.rmtree(active["path"])
                        self._session_summary_cache.pop(
                            str(active["path"].resolve()), None
                        )
                    except Exception as exc:
                        with self._lock:
                            cleanup_error = "Could not discard failed capture: {}: {}".format(
                                type(exc).__name__, exc
                            )
                            self._last_error = (
                                "{}; {}".format(self._last_error, cleanup_error)
                                if self._last_error
                                else cleanup_error
                            )
                try:
                    if config.get("game_profile_id"):
                        self._refresh_poc_evidence_index(
                            config["game_profile_id"]
                        )
                except Exception as exc:
                    with self._lock:
                        if self._last_error is None:
                            self._last_error = (
                                "POC evidence index: {}: {}".format(
                                    type(exc).__name__, exc
                                )
                            )
                with self._lock:
                    if self._active is active:
                        self._active = None
                    self._hud_notice = hud_result
        active["thread"] = threading.Thread(
            target=work,
            name="acquisition-uninterrupted-take",
            daemon=True,
        )
        active["thread"].start()
        return self.descriptor()

    def cancel_active_take(self) -> dict:
        with self._lock:
            if self._active is None:
                return self.descriptor()
            self._active["cancel"].set()
            self._active["stop"].set()
        return self.descriptor()

    def label_session(self, session_key: str, label: str) -> dict:
        with self._lock:
            if self._active is not None and self._session_key(
                Path(self._active["path"])
            ) == session_key:
                raise RuntimeError("Wait for the active recording to finish")
            definition = self._label_definition(str(label or ""))
            path = self._session_path(session_key)
            reader = SessionReader(path)
            context = reader.manifest.get("context") or {}
            game_id = context.get("game_profile_id")
            if definition["value"] and game_id:
                game = self.profiles.game(game_id)
                stage = next(
                    (
                        item
                        for item in game.get("poc_workflow") or ()
                        if item.get("segment_label") == definition["value"]
                    ),
                    None,
                )
                if stage is not None:
                    definition.update(
                        capture_kind=stage.get("capture_kind"),
                        workflow_stage_id=stage.get("stage_id"),
                        capture_id=stage.get("capture_id"),
                    )
            metadata = {
                "schema_version": "1.0",
                "label": definition["value"],
                "label_display": definition["label"],
                "capture_kind": definition.get("capture_kind"),
                "workflow_stage_id": definition.get("workflow_stage_id"),
                "capture_id": definition.get("capture_id"),
                "updated_utc": datetime.now(timezone.utc).isoformat(),
            }
            _write_json_atomic(path / self.SESSION_METADATA_FILENAME, metadata)

            # Choosing a non-empty label is the single post-capture review action.
            # It promotes a successfully recorded take to usable evidence.
            if definition["value"]:
                annotations = AnnotationStore(path).list()
                by_kind = {item["kind"]: item for item in annotations}
                if "take_start" in by_kind and "take_end" in by_kind:
                    pair = (
                        ("route_start", "route_complete")
                        if definition.get("capture_kind") == "route"
                        else ("capture_start", "capture_complete")
                    )
                    store = AnnotationStore(path)
                    for marker, source_name in zip(
                        pair, ("take_start", "take_end")
                    ):
                        if marker not in by_kind:
                            source = by_kind[source_name]
                            store.add(
                                marker,
                                source["session_time_ns"],
                                source["stream_id"],
                                source["frame_index"],
                                route_id=(
                                    definition.get("capture_id")
                                    if marker.startswith("route_")
                                    else None
                                ),
                                note="session_manager_label:{}".format(
                                    definition["value"]
                                ),
                            )
            game_profile_id = context.get("game_profile_id")
            if game_profile_id:
                self._refresh_poc_evidence_index(game_profile_id)
            self._last_error = None
        return self.descriptor()

    def delete_session(self, session_key: str) -> dict:
        """Remove a session from the Workbench by moving it to recoverable trash."""
        with self._lock:
            path = self._session_path(session_key)
            if self._active is not None and path.resolve() == Path(
                self._active["path"]
            ).resolve():
                raise RuntimeError("Cancel the active recording before deleting it")
            manifest = json.loads(
                (path / "manifest.json").read_text(encoding="utf-8")
            )
            context = manifest.get("context") or {}
            trash = self.session_root / ".trash"
            trash.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            target = trash / "{}-{}-{}".format(
                stamp, safe_id(path.parent.name), path.name
            )
            path.rename(target)
            self._session_summary_cache.pop(str(path.resolve()), None)
            game_profile_id = context.get("game_profile_id")
            if game_profile_id:
                self._refresh_poc_evidence_index(game_profile_id)
            self._hud_notice = None
            self._last_error = None
        return self.descriptor()

    def open_session_folder(self, session_key: str) -> dict:
        """Open one validated retained-session directory in the OS file manager."""
        path = self._session_path(session_key)
        if self._folder_opener is not None:
            self._folder_opener(str(path))
        elif os.name == "nt" and hasattr(os, "startfile"):
            os.startfile(str(path))
        else:
            raise RuntimeError("Opening session folders is unavailable on this platform")
        return self.descriptor()

    @staticmethod
    def _finalize_take(
        path: Path,
        route_id: str,
        failed: bool,
        focus_acquired_host_time_ns: Optional[int] = None,
        capture_kind: str = "route",
        capture_id: Optional[str] = None,
        failure_note: Optional[str] = None,
    ) -> None:
        reader = SessionReader(path)
        frames = reader.frames_by_stream.get("main", [])
        start_index, end_index = automatic_take_bounds(
            frames,
            reader.inputs,
            fallback_host_time_ns=focus_acquired_host_time_ns,
        )
        store = AnnotationStore(path)
        for kind, index, note in (
            ("take_start", start_index, "automatic:first_retained_control"),
            ("take_end", end_index, "automatic:capture_end"),
        ):
            frame = frames[index]
            store.add(
                kind,
                frame["session_time_ns"],
                "main",
                frame["frame_index"],
                route_id=route_id if capture_kind == "route" else None,
                note=note,
            )
        if failed:
            frame = frames[end_index]
            store.add(
                "route_failed" if capture_kind == "route" else "capture_failed",
                frame["session_time_ns"],
                "main",
                frame["frame_index"],
                route_id=route_id if capture_kind == "route" else None,
                note=failure_note
                or "automatic:{}:focus_lost_or_user_canceled".format(
                    capture_id or capture_kind
                ),
            )

    def confirm_take(self, run_index: int) -> dict:
        with self._lock:
            if self._active is not None:
                raise RuntimeError("Wait for the active take to finish")
            slot = self._slot(run_index)
            if slot["status"] != "captured_needs_confirmation":
                raise RuntimeError("Run is not waiting for post-take confirmation")
            path = self._run_path(run_index)
            annotations = AnnotationStore(path).list()
            take_start = next(
                item for item in annotations if item["kind"] == "take_start"
            )
            take_end = next(
                item for item in annotations if item["kind"] == "take_end"
            )
            store = AnnotationStore(path)
            capture_kind = self._armed.get("capture_kind", "route")
            marker_pair = (
                (("route_start", take_start), ("route_complete", take_end))
                if capture_kind == "route"
                else (("capture_start", take_start), ("capture_complete", take_end))
            )
            for kind, source in marker_pair:
                store.add(
                    kind,
                    source["session_time_ns"],
                    source["stream_id"],
                    source["frame_index"],
                    route_id=(
                        self._armed["route_id"] if capture_kind == "route" else None
                    ),
                    note="post_take_confirmation:{}:{}".format(
                        self._armed.get("capture_id"),
                        self._armed.get("confirmation_label"),
                    ),
                )
            if self._armed.get("game_profile_id"):
                self._refresh_poc_evidence_index(
                    self._armed["game_profile_id"]
                )
            self._hud_notice = None
        return self.descriptor()
    def compile_and_evaluate(self) -> dict:
        with self._lock:
            if self._armed is None or self._armed.get("capture_kind") != "route":
                raise RuntimeError("Only route captures can be compiled and evaluated")
            runs = self._runs()
            if len(runs) < 2 or not all(run["status"] == "ready" for run in runs):
                raise RuntimeError(
                    "Confirm at least two uninterrupted takes before compiling"
                )
            if self._compile_state == "running":
                return self.descriptor()
            self._compile_state = "running"
            self._compile_result = None
            self._last_error = None
            config = dict(self._armed)
            experiment_artifacts = self.artifact_root / config["experiment_id"]

        def work() -> None:
            try:
                self._archive_existing(experiment_artifacts)
                experiment_artifacts.mkdir(parents=True, exist_ok=True)
                package_path = experiment_artifacts / "package"
                compile_replay_package(
                    self._run_path(1),
                    package_path,
                    "main",
                    config["route_id"],
                    reference_rate_hz=5.0,
                )
                alignments = []
                for run_index in range(2, config["target_runs"] + 1):
                    alignments.append(
                        align_session(
                            package_path,
                            self._run_path(run_index),
                            experiment_artifacts
                            / "alignment_run_{:02d}".format(run_index),
                            query_rate_hz=5.0,
                            distance_threshold=0.45,
                        )
                    )
                result = {
                    "package": str(package_path),
                    "alignments": alignments,
                    "ready_for_live_runner": all(
                        value["final_progress"] == 1.0
                        and value["accepted_fraction"] >= 0.8
                        and (value["stage_label_accuracy"] or 0.0) >= 0.8
                        for value in alignments
                    ),
                }
                _write_json_atomic(
                    experiment_artifacts / "workbench_summary.json", result
                )
                with self._lock:
                    self._compile_result = result
                    self._compile_state = "complete"
            except Exception as exc:
                with self._lock:
                    self._compile_state = "failed"
                    self._last_error = "{}: {}".format(type(exc).__name__, exc)

        threading.Thread(
            target=work,
            name="acquisition-workbench-compile",
            daemon=True,
        ).start()
        return self.descriptor()

    def close(self) -> None:
        with self._lock:
            active = self._active
            tracker = self._live_tracker
        if active is not None:
            active["cancel"].set()
            active["stop"].set()
            active["thread"].join(timeout=5)
        if tracker:
            if tracker.get("status") in ("starting", "running"):
                tracker["stop"].set()
            # The worker publishes "stopped" before its evidence and post-run
            # report files are finalized. Always join a still-live worker so
            # callers can safely release or remove the artifact directory.
            thread = tracker.get("thread")
            if thread is not None and thread.is_alive():
                thread.join(timeout=5)


def make_handler(state: AcquisitionWorkbench):
    static_path = Path(__file__).resolve().parent / "static" / "recorder.html"

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, content_type: str, body: bytes) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                # Browsers routinely cancel obsolete polling and image requests.
                # The response no longer has a recipient, so there is nothing to
                # retry or report as a Workbench failure.
                self.close_connection = True

        def _json(self, status: int, value: object) -> None:
            self._send(
                status,
                "application/json",
                json.dumps(
                    _strict_json_value(value), allow_nan=False
                ).encode("utf-8"),
            )

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > 65536:
                raise ValueError("Invalid request size")
            return (
                json.loads(self.rfile.read(length).decode("utf-8"))
                if length
                else {}
            )

        def do_GET(self):
            try:
                parsed = urlparse(self.path)
                path = parsed.path
                if path == "/":
                    self._send(
                        200,
                        "text/html; charset=utf-8",
                        static_path.read_bytes(),
                    )
                elif path == "/api/state":
                    self._json(200, state.descriptor())
                elif path == "/api/instance":
                    self._json(200, state.instance_descriptor())
                elif path == "/api/android/devices":
                    self._json(200, state.android_devices())
                elif path == "/api/capture-sources":
                    self._json(200, state.capture_source_inventory())
                elif path == "/api/hud":
                    self._json(200, state.hud_descriptor())
                elif path == "/api/minimap-calibration/image":
                    query = parse_qs(parsed.query)
                    body = state.minimap_calibration_image(query.get("game_id", [""])[0], query.get("calibration_id", [""])[0], query.get("name", [""])[0])
                    self._send(200, "image/png", body)
                elif path == "/api/map-stitch/image":
                    query = parse_qs(parsed.query)
                    body = state.map_stitch_image(
                        query.get("game_id", [""])[0],
                        query.get("stitch_id", [""])[0],
                        query.get("name", [""])[0],
                    )
                    self._send(200, "image/png", body)
                elif path == "/api/scene-yaw/image":
                    query = parse_qs(parsed.query)
                    body = state.scene_yaw_image(
                        query.get("game_id", [""])[0],
                        query.get("calibration_id", [""])[0],
                        query.get("name", [""])[0],
                    )
                    self._send(200, "image/png", body)
                elif path == "/api/map-atlas/image":
                    query = parse_qs(parsed.query)
                    body = state.map_atlas_image(
                        query.get("game_id", [""])[0],
                        query.get("atlas_id", [""])[0],
                        query.get("name", [""])[0],
                    )
                    self._send(200, "image/png", body)
                elif path == "/api/teleport-analysis/image":
                    query = parse_qs(parsed.query)
                    body = state.teleport_behavior_image(
                        query.get("game_id", [""])[0],
                        query.get("behavior_id", [""])[0],
                        query.get("name", [""])[0],
                    )
                    self._send(200, "image/png", body)
                elif path == "/api/tracker/overlay":
                    query = parse_qs(parsed.query)
                    self._send(
                        200,
                        "image/png",
                        state.live_tracker_overlay_image(
                            compact=query.get("compact", [""])[0] == "1"
                        ),
                    )
                elif path == "/api/tracker/minimap-route-overlay":
                    self._send(
                        200,
                        "image/png",
                        state.live_tracker_minimap_route_overlay_image(),
                    )
                elif path == "/api/live-tracking/image":
                    query = parse_qs(parsed.query)
                    content_type, body = state.live_tracking_image(
                        query.get("game_id", [""])[0],
                        query.get("tracking_id", [""])[0],
                        query.get("fix_id", [""])[0],
                        query.get("name", [""])[0],
                    )
                    self._send(200, content_type, body)
                else:
                    self._send(404, "text/plain; charset=utf-8", b"Not found")
            except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
                self._send(400, "text/plain; charset=utf-8", str(exc).encode("utf-8"))
            except Exception as exc:
                self._send(
                    500,
                    "text/plain; charset=utf-8",
                    str(exc).encode("utf-8"),
                )

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                value = self._body()
                if path == "/api/arm":
                    result = state.arm(value)
                elif path == "/api/disarm":
                    result = state.disarm()
                elif path == "/api/session/start":
                    result = state.start_session(value)
                elif path == "/api/android/straight-forward":
                    result = state.run_android_straight_forward(value)
                elif path == "/api/session/label":
                    result = state.label_session(
                        str(value.get("session_key") or ""),
                        str(value.get("label") or ""),
                    )
                elif path == "/api/session/delete":
                    result = state.delete_session(
                        str(value.get("session_key") or "")
                    )
                elif path == "/api/session/open-folder":
                    result = state.open_session_folder(
                        str(value.get("session_key") or "")
                    )
                elif path == "/api/hud/toggle":
                    result = state.set_hud_enabled(bool(value.get("enabled")))
                elif path == "/api/take/queue":
                    result = state.queue_next_take()
                elif path == "/api/take/cancel":
                    result = state.cancel_active_take()
                elif path == "/api/take/confirm":
                    result = state.confirm_take(int(value["run_index"]))
                elif path == "/api/compile":
                    result = state.compile_and_evaluate()
                elif path == "/api/profile/draft":
                    result = state.save_profile_draft(value)
                elif path == "/api/minimap-calibration/run":
                    result = state.queue_minimap_calibration(value)
                elif path == "/api/pose-verification/run":
                    result = state.queue_pose_verification(value)
                elif path == "/api/map-stitch/run":
                    result = state.queue_map_stitch(value)
                elif path == "/api/map-atlas/run":
                    result = state.queue_map_atlas(value)
                elif path == "/api/route-tracking/compile":
                    result = state.queue_route_tracking_compile(value)
                elif path == "/api/scene-yaw/run":
                    result = state.queue_scene_yaw_calibration(value)
                elif path == "/api/teleport-analysis/run":
                    result = state.queue_teleport_analysis(value)
                elif path == "/api/tracker/start":
                    result = state.start_live_tracker(value)
                elif path == "/api/tracker/stop":
                    result = state.stop_live_tracker()
                else:
                    self._send(404, "text/plain; charset=utf-8", b"Not found")
                    return
                self._json(200, result)
            except (ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
                self._send(
                    400,
                    "text/plain; charset=utf-8",
                    str(exc).encode("utf-8"),
                )
            except Exception as exc:
                self._send(
                    500,
                    "text/plain; charset=utf-8",
                    str(exc).encode("utf-8"),
                )

        def log_message(self, format, *args):
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session-root",
        type=Path,
        default=Path("sessions") / "workbench",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts") / "workbench",
    )
    parser.add_argument("--profile-root", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-hud",
        action="store_true",
        help="start with the capture-safe Windows status HUD hidden",
    )
    args = parser.parse_args()
    catalog = ProfileCatalog(args.profile_root) if args.profile_root else ProfileCatalog()
    state = AcquisitionWorkbench(
        args.session_root,
        args.artifact_root,
        profiles=catalog,
    )
    try:
        server = WorkbenchHttpServer((args.host, args.port), make_handler(state))
    except OSError:
        existing = discover_workbench_instance(args.host, args.port)
        state.close()
        raise SystemExit(
            occupied_port_message(args.host, args.port, existing)
        ) from None
    state.configure_server_endpoint(args.host, server.server_address[1])
    hud = None
    hud_error = None
    if os.name == "nt":
        try:
            from .hud_process import WorkbenchHudProcess

            hud_host = (
                "127.0.0.1"
                if args.host in ("0.0.0.0", "::", "")
                else args.host
            )
            hud = WorkbenchHudProcess(
                "http://{}:{}/api/hud".format(
                    hud_host,
                    server.server_address[1],
                )
            )
            def toggle_hud(enabled: bool) -> None:
                if enabled:
                    hud.start()
                else:
                    hud.stop()

            state.configure_hud_control(toggle_hud)
            if not args.no_hud:
                toggle_hud(True)
                state.set_hud_runtime(
                    enabled=True,
                    capture_exclusion=True,
                    available=True,
                )
            else:
                state.set_hud_runtime(
                    enabled=False,
                    capture_exclusion=False,
                    available=True,
                )
        except Exception as exc:
            hud_error = "{}: {}".format(type(exc).__name__, exc)
            if hud is not None:
                hud.stop()
            state.set_hud_runtime(
                enabled=False,
                capture_exclusion=False,
                error=hud_error,
                available=hud is not None,
            )
    print("AriaTrace Acquisition Workbench")
    instance = state.instance_descriptor()
    print("Instance {} (PID {})".format(instance["instance_id"], instance["process_id"]))
    print("Open {}".format(instance["url"]))
    print("Stop this instance with Ctrl+C in this terminal")
    if state.descriptor()["hud_runtime"]["enabled"]:
        print("In-game HUD enabled (click-through and excluded from capture)")
    elif hud_error:
        print("In-game HUD unavailable: {}".format(hud_error))
    elif hud is not None:
        print("In-game HUD hidden (it can be shown from the Workbench)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if hud is not None:
            hud.stop()
        server.server_close()
        state.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
