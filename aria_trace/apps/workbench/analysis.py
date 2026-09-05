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



from .common import *  # noqa: F401,F403
from .common import _require_ready_map_localization, _write_json_atomic


class WorkbenchAnalysisMixin:
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
            "route_repeatability",
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
        def preflight() -> None:
            if self._active is not None:
                raise RuntimeError("Wait for the active take to finish")
            if self._tracker_running():
                raise RuntimeError("Stop live tracking before running analysis")
        self._analysis_jobs.queue(kind, value, runner, preflight=preflight)
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
            current_source_ids = {}
            for item in items:
                source_key = str(item.get("source_session_key") or "")
                if not source_key:
                    item["source_current"] = None
                    continue
                if source_key not in current_source_ids:
                    try:
                        source_path = self._session_path(source_key)
                        current_source_ids[source_key] = str(
                            self._describe_session(source_path).get("session_id") or ""
                        )
                    except (OSError, ValueError, KeyError):
                        current_source_ids[source_key] = ""
                artifact_source_id = str(
                    (item.get("provenance") or {}).get("source_session_id") or ""
                )
                current_source_id = current_source_ids[source_key]
                item["source_current"] = (
                    artifact_source_id == current_source_id
                    if artifact_source_id and current_source_id
                    else None
                )
                item["history_only"] = item["source_current"] is False
            selectable = [
                item
                for item in items
                if item.get("status") not in ("failed", "invalid")
                and not item.get("history_only")
            ]
            if not selectable:
                selectable = [
                    item
                    for item in items
                    if item.get("status") not in ("failed", "invalid")
                ]
            if selectable:
                recommended = max(selectable, key=self._map_stitch_selection_score)
                recommended["recommended_for_atlas"] = True
                recommended["recommendation_reason"] = (
                    "largest observed world coverage at the available source detail"
                )
        return values

    @staticmethod
    def _map_stitch_selection_score(item: dict):
        size = item.get("mosaic_size_wh") or [0, 0]
        width = float(size[0]) if len(size) > 0 else 0.0
        height = float(size[1]) if len(size) > 1 else 0.0
        coverage = max(0.0, float(item.get("observed_canvas_coverage") or 0.0))
        localization = item.get("localization") or {}
        scale = max(
            1.0e-9,
            float(localization.get("map_pixels_per_minimap_pixel") or 0.0),
        )
        # Convert observed source pixels back to a comparable mini-map-space
        # extent. This keeps a detailed but tiny scan from outranking a master
        # that covers substantially more of the world.
        comparable_world_area = width * height * coverage / (scale * scale)
        return (
            comparable_world_area,
            scale,
            localization.get("status") == "ready",
            str(item.get("generated_utc") or ""),
        )

    def _recommended_map_stitch(self, game_profile_id: str) -> Optional[dict]:
        items = self._map_stitches().get(game_profile_id, [])
        return next(
            (item for item in items if item.get("recommended_for_atlas")),
            None,
        )

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
            transition_relative = str(
                value.get("transition_session_relative_path") or ""
            )
            layers = [dict(item) for item in value.get("layers") or ()]
            if not layers:
                world_stitch_id = value.get("world_stitch_id") or value.get(
                    "map_stitch_id"
                )
                if not world_stitch_id:
                    recommended = self._recommended_map_stitch(game_profile_id)
                    world_stitch_id = (
                        recommended.get("stitch_id") if recommended else None
                    )
                if not world_stitch_id:
                    raise ValueError(
                        "Build a usable map stitch before creating an atlas"
                    )
                town_stitch_id = value.get("town_stitch_id")
                if town_stitch_id:
                    # Compatibility for previously saved/API-authored requests.
                    layers = [
                        {"mode_id": "world", "stitch_id": world_stitch_id,
                         "display_name": "World overview"},
                        {"mode_id": "town", "stitch_id": town_stitch_id,
                         "display_name": "Town detail"},
                    ]
                elif transition_relative:
                    layers = [
                        {"mode_id": "world", "stitch_id": world_stitch_id,
                         "display_name": "World scale"},
                        {"mode_id": "town", "stitch_id": world_stitch_id,
                         "display_name": "Town scale"},
                    ]
                else:
                    layers = [
                        {"mode_id": "world", "stitch_id": world_stitch_id,
                         "display_name": "Single scale"},
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
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.suffix.lower() != ".png"
        ):
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
        declared.update(
            item.get("localization_mosaic_file")
            for item in descriptor.get("layers") or []
        )
        declared.add((descriptor.get("transition_model") or {}).get("evidence_file"))
        if name not in declared:
            raise ValueError("Unknown map-atlas evidence image")
        return (root / relative).read_bytes()

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
