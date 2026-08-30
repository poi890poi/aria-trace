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

from aria_trace.adapters.filesystem.annotations import AnnotationStore
from aria_trace.services.mapping.stitching import load_localization_reference_candidates, stitch_map_session
from aria_trace.services.mapping.layers import LayeredGlobalLocalizer, build_map_atlas
from aria_trace.services.mapping.references import transition_endpoint_references
from aria_trace.services.calibration.minimap.transition_analysis import analyze_transition_session
from aria_trace.services.calibration.cursor.pose import CursorPoseEstimator
from aria_trace.services.capture.frame_pump import LatestFramePump
from aria_trace.services.tracking.runtime import (
    GlobalMapLocalizer,
    MinimapExtractor,
    TwoRateRealtimeTracker,
    render_map_overlay,
    render_minimap_route_overlay,
)
from aria_trace.evidence.tracking import LiveTrackingEvidenceRecorder
from aria_trace.services.calibration.minimap.calibration import (
    ORDINARY_MOTION_SEGMENT_LABELS,
    calibrate_segment_sessions,
    calibrate_session,
)
from aria_trace.services.calibration.minimap.verification import verify_forward_session
from aria_trace.evidence.poc_catalog import build_poc_evidence_index
from aria_trace.adapters.filesystem.profiles import ProfileCatalog
from aria_trace.workflows.recording import AcquisitionRecorder
from aria_trace.workflows.route import compile_route_session
from aria_trace.services.localization.route.tracker import RouteCandidateAdvisor, RouteVisualTracker
from aria_trace.adapters.filesystem.session import SessionReader, input_capture_health
from aria_trace.services.calibration.scene_yaw import calibrate_scene_yaw_session
from aria_trace.services.tracking.profiles import resolve_tracking_profile
from aria_trace.workflows.teleport import analyze_teleport_session
from aria_trace.adapters.windows import (
    WindowsDesktopApi,
    select_window,
)



from .common import *  # noqa: F401,F403
from .common import _write_json_atomic


class WorkbenchStateMixin:
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
