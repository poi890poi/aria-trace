"""Zero-interruption acquisition workbench for repeated route demonstrations."""

import argparse
import json
import os
import re
import shutil
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

from replay.alignment import align_session
from replay.package import compile_replay_package

from .annotations import AnnotationStore
from .profiles import ProfileCatalog
from .recorder import AcquisitionRecorder
from .session import SessionReader
from .sources import AdbGetEventSource, AdbScreenshotFrameSource, OpenCvCameraFrameSource
from .windows import (
    WindowsDesktopApi,
    WindowsKeyboardMouseSource,
    WindowsRawKeyboardMouseSource,
    WindowsWindowFrameSource,
    WindowsXInputSource,
    select_window,
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


class SourceFactory:
    """Construct recorder sources from neutral adapter configuration."""

    FRAME_ADAPTERS = (
        {"adapter": "windows_window", "label": "Windows game window", "status": "pc_mvp"},
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
        if not value:
            raise RuntimeError("ADB is not on PATH and no executable was configured")
        return Path(value)

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
    def __init__(
        self,
        session_root: Path,
        artifact_root: Path,
        profiles: Optional[ProfileCatalog] = None,
        desktop_api=None,
        xinput_api=None,
        raw_input_api=None,
    ) -> None:
        self.session_root = Path(session_root)
        self.artifact_root = Path(artifact_root)
        self.session_root.mkdir(parents=True, exist_ok=True)
        self.profiles = profiles or ProfileCatalog()
        self.desktop_api = desktop_api
        self.sources = SourceFactory(
            desktop_api=desktop_api,
            xinput_api=xinput_api,
            raw_input_api=raw_input_api,
        )
        self._lock = threading.RLock()
        self._armed = None
        self._active = None
        self._last_error = None
        self._compile_state = "not_ready"
        self._compile_result = None

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

            route_id = value.get("route_id") or (route or {}).get("route_id")
            if not route_id:
                raise ValueError("A route ID is required")
            experiment_id = safe_id(value.get("experiment_id") or route_id)
            target_runs = int(value.get("target_runs") or (route or {}).get("target_runs", 3))
            capture_duration_s = float(
                value.get("capture_duration_s")
                or (route or {}).get("capture_duration_s", 30.0)
            )
            focus_wait_timeout_s = float(value.get("focus_wait_timeout_s") or 60.0)
            if target_runs < 1 or target_runs > 20:
                raise ValueError("Target runs must be between 1 and 20")
            if capture_duration_s < 5.0 or capture_duration_s > 600.0:
                raise ValueError("Capture duration must be between 5 and 600 seconds")
            if focus_wait_timeout_s < 5.0 or focus_wait_timeout_s > 600.0:
                raise ValueError("Focus wait must be between 5 and 600 seconds")

            self._armed = {
                "experiment_id": experiment_id,
                "game_profile_id": game_id,
                "route_profile_id": route_profile_id,
                "route_id": route_id,
                "target_runs": target_runs,
                "capture_duration_s": capture_duration_s,
                "focus_wait_timeout_s": focus_wait_timeout_s,
                "frame_source": frame_config,
                "input_source": input_config,
                "armed_utc": datetime.now(timezone.utc).isoformat(),
            }
            self._last_error = None
            self._compile_state = "not_ready"
            self._compile_result = None
        return self.descriptor()

    def disarm(self) -> dict:
        with self._lock:
            if self._active is not None:
                raise RuntimeError("Cancel the active take before changing configuration")
            self._armed = None
        return self.descriptor()

    def _experiment_root(self) -> Path:
        if self._armed is None:
            raise RuntimeError("Configure and arm an experiment first")
        return self.session_root / self._armed["experiment_id"]

    def _run_path(self, run_index: int) -> Path:
        return self._experiment_root() / "run_{:02d}".format(run_index)

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
            if "route_failed" in kinds:
                status = "needs_rerecord"
            elif "route_start" in kinds and "route_complete" in kinds:
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
        return [
            self._slot(index)
            for index in range(1, self._armed["target_runs"] + 1)
        ]

    def descriptor(self) -> dict:
        with self._lock:
            runs = self._runs()
            all_ready = bool(runs) and all(run["status"] == "ready" for run in runs)
            return {
                "schema_version": "1.1",
                "profiles": self.profiles.descriptor(),
                "sources": self.sources.descriptor(),
                "visible_windows": self._windows(),
                "armed": self._armed,
                "runs": runs,
                "all_runs_ready": all_ready,
                "active_run": self._active["run_index"] if self._active else None,
                "active_phase": self._active["phase"] if self._active else None,
                "capture_policy": {
                    "in_game_controls": "none",
                    "start": "sources_prearmed_then_clock_starts_on_first_game_focus",
                    "end": "fixed_duration_or_focus_loss",
                    "route_bounds": "post_take_confirmation",
                    "stages": "derived_or_annotated_after_recording",
                },
                "last_error": self._last_error,
                "compile_state": self._compile_state if all_ready else "not_ready",
                "compile_result": self._compile_result,
            }

    def _selected_window_foreground(self, frame_config: dict) -> bool:
        if frame_config.get("adapter") != "windows_window":
            return True
        try:
            desktop = self._desktop()
            window = select_window(
                desktop.list_windows(),
                frame_config["window_title"],
                exact=True,
            )
            if hasattr(desktop, "is_foreground"):
                return bool(desktop.is_foreground(window[0]))
            return bool(desktop.input_snapshot(window[0])["foreground"])
        except (RuntimeError, ValueError):
            return False

    def _next_run_index(self) -> int:
        if self._armed is None:
            raise RuntimeError("Configure and arm an experiment first")
        for run_index in range(1, self._armed["target_runs"] + 1):
            if self._slot(run_index)["status"] != "ready":
                return run_index
        raise RuntimeError("All requested runs are already complete")
    def queue_next_take(self) -> dict:
        return self.queue_take(self._next_run_index())

    def queue_take(self, run_index: int) -> dict:
        with self._lock:
            if self._armed is None:
                raise RuntimeError("Configure and arm an experiment first")
            if self._active is not None:
                raise RuntimeError("A take is already active")
            if run_index < 1 or run_index > self._armed["target_runs"]:
                raise ValueError("Run index is outside the configured experiment")
            config = dict(self._armed)
            config["frame_source"] = dict(self._armed["frame_source"])
            config["input_source"] = dict(self._armed["input_source"])
            active = {
                "run_index": run_index,
                "path": self._run_path(run_index),
                "phase": "waiting_for_game_focus",
                "cancel": threading.Event(),
                "stop": threading.Event(),
                "focus_interrupted": False,
                "focus_timeout": False,
                "focus_acquired_host_time_ns": None,
            }
            self._active = active
            self._last_error = None

        def work() -> None:
            guard = None
            try:
                self._archive_existing(active["path"])
                frame_source = self.sources.frame(config["frame_source"])
                input_source = self.sources.input(config["input_source"])
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
                        "run_index": run_index,
                        "frame_adapter": config["frame_source"]["adapter"],
                        "input_adapter": config["input_source"].get(
                            "adapter", "none"
                        ),
                        "capture_policy": "prearmed_uninterrupted_take_v2",
                    },
                )
                sources_started = threading.Event()

                def guard_focus() -> None:
                    while not sources_started.wait(0.02):
                        if active["stop"].is_set():
                            return
                    focus_deadline = (
                        time.monotonic() + config["focus_wait_timeout_s"]
                    )
                    end_time = None
                    lost_since = None
                    while not active["stop"].is_set():
                        now = time.monotonic()
                        foreground = self._selected_window_foreground(
                            config["frame_source"]
                        )
                        if end_time is None:
                            if foreground:
                                active["focus_acquired_host_time_ns"] = (
                                    time.perf_counter_ns()
                                )
                                end_time = now + config["capture_duration_s"]
                                with self._lock:
                                    if self._active is active:
                                        active["phase"] = (
                                            "recording_uninterrupted_take"
                                        )
                            elif now >= focus_deadline:
                                active["focus_timeout"] = True
                                active["stop"].set()
                                return
                        else:
                            if now >= end_time:
                                active["stop"].set()
                                return
                            if foreground:
                                lost_since = None
                            elif lost_since is None:
                                lost_since = now
                            elif now - lost_since >= 0.75:
                                active["focus_interrupted"] = True
                                active["stop"].set()
                                return
                        active["stop"].wait(0.02)

                guard = threading.Thread(
                    target=guard_focus,
                    name="acquisition-focus-guard",
                    daemon=True,
                )
                guard.start()
                recorder.run(
                    external_stop=active["stop"],
                    started_event=sources_started,
                )

                failed = bool(
                    active["cancel"].is_set()
                    or active["focus_interrupted"]
                    or active["focus_timeout"]
                    or active["focus_acquired_host_time_ns"] is None
                )
                self._finalize_take(
                    active["path"],
                    config["route_id"],
                    failed,
                    active["focus_acquired_host_time_ns"],
                )
                if active["focus_timeout"]:
                    with self._lock:
                        self._last_error = (
                            "Take never received game focus; rerecord it"
                        )
                elif failed:
                    with self._lock:
                        self._last_error = (
                            "Take was canceled or game focus was lost; rerecord it"
                        )
            except Exception as exc:
                with self._lock:
                    self._last_error = "{}: {}".format(type(exc).__name__, exc)
            finally:
                active["stop"].set()
                if guard is not None:
                    guard.join(timeout=1)
                with self._lock:
                    if self._active is active:
                        self._active = None
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

    @staticmethod
    def _finalize_take(
        path: Path,
        route_id: str,
        failed: bool,
        focus_acquired_host_time_ns: Optional[int] = None,
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
            ("take_start", start_index, "automatic:first_observed_control_or_game_focus"),
            ("take_end", end_index, "automatic:capture_end"),
        ):
            frame = frames[index]
            store.add(
                kind,
                frame["session_time_ns"],
                "main",
                frame["frame_index"],
                route_id=route_id,
                note=note,
            )
        if failed:
            frame = frames[end_index]
            store.add(
                "route_failed",
                frame["session_time_ns"],
                "main",
                frame["frame_index"],
                route_id=route_id,
                note="automatic:focus_lost_or_user_canceled",
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
            for kind, source in (
                ("route_start", take_start),
                ("route_complete", take_end),
            ):
                store.add(
                    kind,
                    source["session_time_ns"],
                    source["stream_id"],
                    source["frame_index"],
                    route_id=self._armed["route_id"],
                    note="post_take_confirmation:full_take_boundary",
                )
        return self.descriptor()
    def compile_and_evaluate(self) -> dict:
        with self._lock:
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
    static_path = Path(__file__).resolve().parent / "static" / "workbench.html"

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

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
                path = urlparse(self.path).path
                if path == "/":
                    self._send(
                        200,
                        "text/html; charset=utf-8",
                        static_path.read_bytes(),
                    )
                elif path == "/api/state":
                    self._json(200, state.descriptor())
                else:
                    self._send(404, "text/plain; charset=utf-8", b"Not found")
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
                elif path == "/api/take/queue":
                    result = state.queue_next_take()
                elif path == "/api/take/cancel":
                    result = state.cancel_active_take()
                elif path == "/api/take/confirm":
                    result = state.confirm_take(int(value["run_index"]))
                elif path == "/api/compile":
                    result = state.compile_and_evaluate()
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
    args = parser.parse_args()
    catalog = ProfileCatalog(args.profile_root) if args.profile_root else ProfileCatalog()
    state = AcquisitionWorkbench(
        args.session_root,
        args.artifact_root,
        profiles=catalog,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print("AriaTrace Acquisition Workbench")
    print("Open http://{}:{}/".format(args.host, server.server_address[1]))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state.close()


if __name__ == "__main__":
    main()