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


class WorkbenchCaptureMixin:
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
        return self._session_catalog.key(path)

    def _session_path(self, session_key: str, require_manifest: bool = True) -> Path:
        return self._session_catalog.resolve(session_key, require_manifest)

    def _session_metadata(self, path: Path) -> dict:
        return self._session_catalog.metadata(path)

    def _label_definition(self, label: str) -> dict:
        return self._session_catalog.label_definition(label)

    def _describe_session(self, path: Path) -> dict:
        return self._session_catalog.describe(path)

    def sessions(self) -> List[dict]:
        active = None if self._active is None else dict(self._active)
        if active is not None:
            active["game_profile_id"] = (self._armed or {}).get("game_profile_id")
        return self._session_catalog.list(active)

    @staticmethod
    def _archive_existing(path: Path) -> None:
        archive_existing(path)

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
            primary_stream_id = session_primary_stream_id(reader)
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
                "frames": len(reader.frames_by_stream.get(primary_stream_id, [])),
                "duration_s": reader.manifest.get("duration_ns", 0) / 1.0e9,
                "dropped_frames": reader.manifest.get("dropped_frames", {}).get(
                    primary_stream_id, 0
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
                "analysis_jobs": self._analysis_jobs.snapshot(),
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
                source_bundle = self.sources.recording_bundle(
                    config["frame_source"], config["input_source"]
                )
                for input_source in source_bundle.input_sources:
                    if hasattr(input_source, "disable_foreground_filter"):
                        input_source.disable_foreground_filter()
                recorder = AcquisitionRecorder(
                    active["path"],
                    source_bundle.frame_sources,
                    source_bundle.input_sources,
                    queue_size=8192,
                    video_encoding="h264",
                    video_fps=float(config["frame_source"].get("fps", 30.0)),
                    frame_processors=source_bundle.frame_processors,
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
                        **source_bundle.session_context,
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
                source_bundle.finalize(active["path"], manifest)
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
                frames = reader.frames_by_stream.get(
                    source_bundle.primary_stream_id, []
                )
                missing_streams = [
                    stream_id
                    for stream_id in source_bundle.required_stream_ids
                    if not reader.frames_by_stream.get(stream_id)
                ]
                duration_ns = int(manifest.get("duration_ns") or 0)
                empty_input = not input_health["healthy"]
                failed = bool(
                    active["cancel"].is_set()
                    or not session_started
                    or empty_input
                    or manifest.get("status") != "complete"
                    or duration_ns <= 0
                    or not frames
                    or missing_streams
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
                        stream_id=source_bundle.primary_stream_id,
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
                elif duration_ns <= 0 or not frames or missing_streams:
                    with self._lock:
                        self._last_error = (
                            "Recording contained no usable duration or required "
                            "frame streams{}; the partial capture was discarded"
                        ).format(
                            " ({})".format(", ".join(missing_streams))
                            if missing_streams
                            else ""
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
                        self._session_catalog.invalidate(active["path"])
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
            self._session_catalog.invalidate(path)
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
        stream_id: str = "main",
    ) -> None:
        reader = SessionReader(path)
        stream_id = str(stream_id)
        frames = reader.frames_by_stream.get(stream_id, [])
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
                stream_id,
                frame["frame_index"],
                route_id=route_id if capture_kind == "route" else None,
                note=note,
            )
        if failed:
            frame = frames[end_index]
            store.add(
                "route_failed" if capture_kind == "route" else "capture_failed",
                frame["session_time_ns"],
                stream_id,
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
