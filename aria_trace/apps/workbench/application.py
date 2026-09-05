"""Zero-interruption acquisition workbench for repeated route demonstrations."""

import argparse
import multiprocessing
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

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
)
from rig_runtime.adapters.filesystem.profiles import ProfileCatalog


















from .common import *  # noqa: F401,F403
from .state import WorkbenchStateMixin
from .analysis import WorkbenchAnalysisMixin
from .live_tracking import WorkbenchLiveTrackingMixin
from .capture import WorkbenchCaptureMixin


class AcquisitionWorkbench(WorkbenchStateMixin, WorkbenchAnalysisMixin, WorkbenchLiveTrackingMixin, WorkbenchCaptureMixin):
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
        "benchmark",
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
        {
            "value": "route_repeatability",
            "label": "Repeated route laps / back-and-forth",
            "capture_kind": "benchmark",
            "workflow_stage_id": "route-repeatability",
            "capture_id": "route-repeatability",
            "segment_semantics": {
                "benchmark_type": "localization_repeatability",
                "repetition_unit": "repeated_complete_state_visit",
                "supported_direction_patterns": [
                    "same_direction_laps",
                    "alternating_forward_reverse",
                ],
                "minimum_complete_passes": 3,
                "pass_correspondence_source": (
                    "post_run_reference_state_recurrence"
                ),
                "state_identity": ["canonical_xy", "heading_deg", "map_mode"],
                "lap_boundary_requirement": "none",
                "equal_path_or_timing_required": False,
                "path_variation_policy": (
                    "subtract_reference_displacement_before_repeatability_scoring"
                ),
                "input_role": "optional_turn_timing_and_behavior_evidence_only",
                "position_truth_role": "none",
                "label_source": "post_capture_user_confirmation",
            },
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
        self._analysis_jobs = AnalysisJobManager(lock=self._lock)
        # Live control must never wait behind the catalog/session scans performed
        # by the one-second Workbench state refresh.
        self._live_tracker_lock = threading.RLock()
        self._live_tracker = None
        self._live_tracker_engine = None
        self._live_tracker_mosaic = None
        self._live_tracker_route_points = None
        self._android_control = None
        self._hud_notice = None
        self._session_catalog = SessionCatalog(
            self.session_root,
            self.SESSION_LABELS,
            self.SESSION_METADATA_FILENAME,
        )
        self._instance = {
            "service": WORKBENCH_SERVICE,
            "schema_version": "1.0",
            "instance_id": uuid.uuid4().hex[:12],
            "process_id": os.getpid(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "workspace_root": str(Path(__file__).resolve().parents[3]),
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

































































































from .api import make_handler


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
            from aria_trace.apps.workbench.hud_process import WorkbenchHudProcess

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
