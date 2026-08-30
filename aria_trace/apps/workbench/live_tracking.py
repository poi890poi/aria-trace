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
from .common import _require_ready_map_localization


class WorkbenchLiveTrackingMixin:
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
            if self._analysis_jobs.has_active_job():
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
                representation_interval_s=float(
                    resolved_profile["representation_interval_s"]
                ),
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
                route_visual_tracker=(
                    RouteVisualTracker(
                        route_package,
                        localizer,
                        score_min=float(resolved_profile["route_map_score_min"]),
                    )
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
                    "current-frame-map-correlation"
                    if tracking_mode == "route-assisted"
                    else None
                ),
                "frame_source": frame_config,
                "frame_size_wh": None,
                "minimap_crop_xywh": list(minimap_config["crop_xywh"]),
                "global_interval_s": global_interval_s,
                "representation_interval_s": float(
                    resolved_profile["representation_interval_s"]
                ),
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
                        "current-frame-map-correlation"
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
