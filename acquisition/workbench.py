"""Zero-interruption acquisition workbench for repeated route demonstrations."""

import argparse
import json
import math
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

from replay.alignment import align_session
from replay.package import compile_replay_package

from .annotations import AnnotationStore
from .android_capture import (
    AndroidRoiFrameSource,
    AndroidRoiSpec,
    ScrcpyCaptureHub,
    find_scrcpy_server,
)
from .map_stitching import stitch_map_session
from .minimap_calibration import calibrate_segment_sessions, calibrate_session
from .minimap_verification import verify_forward_session
from .poc_evidence import build_poc_evidence_index
from .profiles import ProfileCatalog
from .recorder import AcquisitionRecorder
from .session import SessionReader, input_capture_health
from .sources import (
    AdbClockMapper,
    AdbGetEventSource,
    AdbScreenshotFrameSource,
    OpenCvCameraFrameSource,
)
from .windows import (
    WindowsDesktopApi,
    WindowsKeyboardMouseSource,
    WindowsRawKeyboardMouseSource,
    WindowsWindowFrameSource,
    WindowsXInputSource,
    select_window,
)


WORKBENCH_SERVICE = "aria-trace-workbench"


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

    def __init__(self, desktop_api=None, xinput_api=None, raw_input_api=None) -> None:
        self.desktop_api = desktop_api
        self.xinput_api = xinput_api
        self.raw_input_api = raw_input_api

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
        "full_map",
        "minimap_calibration",
        "minimap_cruise",
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
            "movement_only",
            "forward_no_turn",
            "full_map",
            "ordinary_cruise",
            "route",
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

                rotation_key, rotation_path, rotation_manifest, _ = choose(
                    "rotation_only", "rotation_session_relative_path"
                )
                movement_key, movement_path, movement_manifest, _ = choose(
                    "movement_only", "movement_session_relative_path"
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
            stitch_id = safe_id(described.get("session_id") or path.name)
            output = self._map_stitch_root(game_profile_id) / stitch_id

        result = stitch_map_session(path, output, progress=progress)

        if progress:
            progress("Saving the stitched-map result and evidence index")
        with self._lock:
            result["source_session_key"] = relative
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
                "map_stitches": self._map_stitches(),
                "analysis_candidates": self.analysis_candidates(),
                "analysis_jobs": {
                    key: dict(value) for key, value in self._analysis_jobs.items()
                },
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
        if capture_adapter not in ("windows_window", "android_scrcpy"):
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
        if active is not None:
            active["cancel"].set()
            active["stop"].set()
            active["thread"].join(timeout=5)


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
                json.dumps(value).encode("utf-8"),
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
    main()
