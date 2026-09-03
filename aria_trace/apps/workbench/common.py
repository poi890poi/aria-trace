"""Zero-interruption acquisition workbench for repeated route demonstrations."""

import argparse
import json
import math
import multiprocessing
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import cv2

from aria_trace.apps.workbench.server import (
    WORKBENCH_SERVICE,
    WorkbenchHttpServer,
    connect_host as _connect_host,
    discover_workbench_instance,
    is_client_disconnect,
    occupied_port_message,
)
from aria_trace.apps.workbench.sources import SourceFactory, parse_adb_devices
from aria_trace.apps.workbench.jobs import AnalysisJobManager
from aria_trace.apps.workbench.catalog import (
    SessionCatalog,
    archive_existing,
    session_primary_stream_id,
)

from replay.alignment import align_session
from replay.package import compile_replay_package
from replay.route_tracking import RouteTrackingPackage
from replay.route_similarity import write_live_route_similarity

from rig_runtime.adapters.filesystem.annotations import AnnotationStore
from aria_trace.services.mapping.stitching import load_localization_reference_candidates, stitch_map_session
from aria_trace.services.mapping.layers import LayeredGlobalLocalizer, build_map_atlas
from aria_trace.services.mapping.references import transition_endpoint_references
from rig_runtime.services.calibration.minimap.transition_analysis import analyze_transition_session
from rig_runtime.services.calibration.cursor.pose import CursorPoseEstimator
from rig_runtime.services.capture.frame_pump import LatestFramePump
from aria_trace.services.tracking.runtime import (
    GlobalMapLocalizer,
    MinimapExtractor,
    TwoRateRealtimeTracker,
    render_map_overlay,
    render_minimap_route_overlay,
)
from aria_trace.evidence.tracking import LiveTrackingEvidenceRecorder
from rig_runtime.services.calibration.minimap.calibration import (
    ORDINARY_MOTION_SEGMENT_LABELS,
    calibrate_segment_sessions,
    calibrate_session,
)
from rig_runtime.services.calibration.minimap.verification import verify_forward_session
from aria_trace.evidence.poc_catalog import build_poc_evidence_index
from rig_runtime.adapters.filesystem.profiles import ProfileCatalog
from rig_runtime.workflows.recording import AcquisitionRecorder
from aria_trace.workflows.route import compile_route_session
from aria_trace.services.localization.route.tracker import RouteCandidateAdvisor, RouteVisualTracker
from rig_runtime.adapters.filesystem.session import SessionReader, input_capture_health
from rig_runtime.services.calibration.scene_yaw import calibrate_scene_yaw_session
from aria_trace.services.tracking.profiles import resolve_tracking_profile
from aria_trace.workflows.teleport import analyze_teleport_session
from aria_trace.adapters.windows import (
    WindowsDesktopApi,
    select_window,
)




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
