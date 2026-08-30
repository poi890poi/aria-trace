import ctypes
import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2

from acquisition.annotations import AnnotationStore
from acquisition.hud import _HudWindow
from acquisition.models import FramePacket, InputPacket
from acquisition.poc_evidence import build_poc_evidence_index
from acquisition.profiles import ProfileCatalog
from acquisition.session import SessionReader, SessionWriter, input_capture_health
from acquisition.workbench import (
    AcquisitionWorkbench,
    SourceFactory,
    WorkbenchHttpServer,
    automatic_take_bounds,
    discover_workbench_instance,
    input_packet_is_active,
    is_client_disconnect,
    make_handler,
    nearest_frame_index,
    occupied_port_message,
    parse_adb_devices,
    safe_id,
)
from acquisition.windows import (
    WindowsKeyboardMouseSource,
    WindowsRawKeyboardMouseSource,
    WindowsWindowFrameSource,
    WindowsXInputSource,
    _RawInput,
    decode_raw_input,
)

import numpy as np


class ArbitraryDesktop:
    def list_windows(self):
        return [(17, "Popular Game A"), (18, "Another Game")]

    def input_snapshot(self, hwnd):
        return {
            "foreground": hwnd == 17,
            "keys": [],
            "buttons": [],
            "cursor_client": (0, 0),
            "cursor_normalized": (0.0, 0.0),
        }

    def capture_client(self, hwnd):
        return np.full((32, 48, 3), 64, dtype=np.uint8), (0, 0, 48, 32)


class NeverForegroundDesktop(ArbitraryDesktop):
    def input_snapshot(self, hwnd):
        snapshot = super().input_snapshot(hwnd)
        snapshot["foreground"] = False
        return snapshot


class IntegrityMismatchDesktop(ArbitraryDesktop):
    def input_integrity_status(self, hwnd):
        return {
            "target_process_id": 991,
            "target_elevated": True,
            "recorder_elevated": False,
            "matched": False,
        }


class FailingCaptureDesktop(ArbitraryDesktop):
    def capture_client(self, hwnd):
        raise RuntimeError("synthetic frame failure")


class FakeXInputApi:
    def __init__(self):
        self.packet = 0

    def read_state(self, user_index):
        self.packet += 1
        return {
            "packet_number": self.packet,
            "buttons_mask": 0x1000,
            "buttons": ["a"],
            "left_trigger_raw": 0,
            "right_trigger_raw": 128,
            "left_trigger": 0.0,
            "right_trigger": 128.0 / 255.0,
            "left_stick_raw": [12000, -5000],
            "right_stick_raw": [8000, 4000],
            "left_stick": [0.36, -0.15],
            "right_stick": [0.24, 0.12],
        }


class FakeRawInputApi:
    def __init__(self):
        self.stopped = threading.Event()

    def run(self, emit, ready_event):
        ready_event.set()
        now = time.perf_counter_ns()
        emit(
            {
                "kind": "pc_raw_mouse",
                "host_time_ns": now,
                "payload": {
                    "device_handle": 55,
                    "movement_mode": "relative",
                    "delta_x": 14,
                    "delta_y": -7,
                    "button_transitions": ["left_down"],
                    "wheel_delta": 0,
                    "horizontal_wheel_delta": 0,
                },
            }
        )
        emit(
            {
                "kind": "pc_raw_keyboard",
                "host_time_ns": now + 1,
                "payload": {
                    "virtual_key": 87,
                    "key_name": "W",
                    "scan_code": 17,
                    "pressed": True,
                },
            }
        )
        self.stopped.wait(1)

    def stop(self):
        self.stopped.set()


class SettlingRawInputApi(FakeRawInputApi):
    def run(self, emit, ready_event):
        ready_event.set()
        now = time.perf_counter_ns()
        emit(
            {
                "kind": "pc_raw_mouse",
                "host_time_ns": now,
                "payload": {
                    "movement_mode": "relative",
                    "delta_x": 3,
                    "delta_y": 0,
                    "button_transitions": ["left_up"],
                    "wheel_delta": 0,
                    "horizontal_wheel_delta": 0,
                },
            }
        )
        if not self.stopped.wait(0.15):
            emit(
                {
                    "kind": "pc_raw_keyboard",
                    "host_time_ns": time.perf_counter_ns(),
                    "payload": {
                        "device_handle": 55,
                        "virtual_key": 87,
                        "key_name": "W",
                        "scan_code": 17,
                        "pressed": True,
                    },
                }
            )
            emit(
                {
                    "kind": "pc_raw_mouse",
                    "host_time_ns": time.perf_counter_ns(),
                    "payload": {
                        "device_handle": 55,
                        "movement_mode": "relative",
                        "delta_x": 9,
                        "delta_y": -2,
                        "button_transitions": [],
                        "wheel_delta": 0,
                        "horizontal_wheel_delta": 0,
                    },
                }
            )
        self.stopped.wait(1)


class DescribedSource:
    stream_id = "main"

    def describe(self):
        return {"type": "test", "stream_id": "main"}


def write_catalog(root):
    games = root / "games"
    routes = root / "routes"
    games.mkdir(parents=True)
    routes.mkdir()
    (games / "game_a.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "profile_id": "game-a",
                "display_name": "Popular Game A",
                "platform": "windows_pc",
                "default_frame_source": {
                    "adapter": "windows_window",
                    "window_title": "Popular Game A",
                    "fps": 30,
                },
                "default_input_source": {
                    "adapter": "windows_xinput",
                    "poll_hz": 250,
                },
                "poc_workflow": [
                    {
                        "stage_id": "full-map",
                        "display_name": "Record full map",
                        "capture_kind": "full_map",
                        "capture_id": "game-a-full-map",
                        "target_runs": 1,
                        "capture_duration_s": 20,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (routes / "route_a.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "route_profile_id": "route-a-profile",
                "display_name": "Short route A",
                "game_profile_id": "game-a",
                "route_id": "short-route-a",
                "target_runs": 3,
                "capture_duration_s": 20,
                "setup_steps": ["Stand at the same start point."],
            }
        ),
        encoding="utf-8",
    )


class WorkbenchTests(unittest.TestCase):
    def test_client_disconnects_are_not_server_failures(self):
        self.assertTrue(is_client_disconnect(BrokenPipeError()))
        self.assertTrue(is_client_disconnect(ConnectionResetError()))
        self.assertTrue(is_client_disconnect(ConnectionAbortedError(10053, "aborted")))
        self.assertFalse(is_client_disconnect(ValueError("bad request")))

        class AbortedWriter:
            def write(self, body):
                raise ConnectionAbortedError(10053, "aborted")

        handler = object.__new__(make_handler(None))
        handler.requestline = "GET /api/hud HTTP/1.1"
        handler.request_version = "HTTP/1.1"
        handler.command = "GET"
        handler.wfile = AbortedWriter()
        handler.close_connection = False
        handler._send(200, "application/json", b"{}")
        self.assertTrue(handler.close_connection)

    def test_instance_discovery_recognizes_older_workbench_shell(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def read(self, limit=-1):
                return b"<title>AriaTrace Recorder</title>"

        not_found = urllib.error.HTTPError(
            "http://127.0.0.1:8080/api/instance", 404, "Not found", {}, None
        )
        with patch(
            "acquisition.workbench.urlopen", side_effect=[not_found, Response()]
        ):
            instance = discover_workbench_instance("127.0.0.1", 8080)
        self.assertTrue(instance["legacy"])
        self.assertEqual(instance["url"], "http://127.0.0.1:8080/")

    def test_hud_visibility_requires_exact_target_foreground(self):
        class FakeUser32:
            foreground = 17

            @staticmethod
            def FindWindowW(_class_name, title):
                return 17 if title == "Popular Game A" else 0

            def GetForegroundWindow(self):
                return self.foreground

        user32 = FakeUser32()
        self.assertTrue(
            _HudWindow._target_is_foreground(user32, "Popular Game A")
        )
        user32.foreground = 18
        self.assertFalse(
            _HudWindow._target_is_foreground(user32, "Popular Game A")
        )
        self.assertFalse(_HudWindow._target_is_foreground(user32, None))

    def test_first_input_trigger_ignores_controls_outside_game_focus(self):
        background = InputPacket(
            "pc-raw-input",
            "pc_raw_keyboard",
            time.perf_counter_ns(),
            {
                "foreground": False,
                "device_handle": 55,
                "key_name": "W",
                "pressed": True,
            },
        )
        foreground = InputPacket(
            "pc-raw-input",
            "pc_raw_keyboard",
            time.perf_counter_ns(),
            {
                "foreground": True,
                "device_handle": 55,
                "key_name": "W",
                "pressed": True,
            },
        )
        self.assertFalse(input_packet_is_active(background))
        self.assertTrue(input_packet_is_active(foreground))

    def test_first_input_trigger_requires_physical_pressed_gameplay_input(self):
        def packet(kind, **payload):
            return InputPacket("pc-raw-input", kind, time.perf_counter_ns(), payload)

        self.assertFalse(
            input_packet_is_active(
                packet(
                    "pc_raw_keyboard",
                    foreground=True,
                    device_handle=0,
                    key_name="W",
                    pressed=True,
                )
            )
        )
        self.assertFalse(
            input_packet_is_active(
                packet(
                    "pc_raw_keyboard",
                    foreground=True,
                    device_handle=55,
                    key_name="W",
                    pressed=False,
                )
            )
        )
        self.assertTrue(
            input_packet_is_active(
                packet(
                    "pc_raw_keyboard",
                    foreground=True,
                    device_handle=55,
                    key_name="Tab",
                    pressed=True,
                )
            )
        )
        self.assertFalse(
            input_packet_is_active(
                packet(
                    "pc_raw_mouse",
                    foreground=True,
                    device_handle=0,
                    delta_x=8,
                )
            )
        )

    def test_required_raw_input_accepts_any_physical_user_action(self):
        manifest = {
            "context": {
                "input_adapter": "windows_raw_keyboard_mouse",
                "input_requirement": "required",
                "capture_kind": "game_profile",
            },
            "input_counts": {"pc-raw-input:pc_raw_mouse": 1},
        }
        mouse_only = [
            {
                "kind": "pc_raw_mouse",
                "payload": {
                    "device_handle": 65616,
                    "delta_x": 1,
                    "delta_y": 0,
                },
            }
        ]
        health = input_capture_health(manifest, mouse_only)
        self.assertTrue(health["healthy"])
        self.assertEqual(health["missing"], [])

    def test_simple_input_recording_arms_for_first_game_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(),
                desktop_api=ArbitraryDesktop(),
            )
            try:
                state.queue_next_take = state.descriptor
                descriptor = state.start_session(
                    {
                        "experiment_id": "first-input-session",
                        "window_title": "Popular Game A",
                        "input_adapter": "windows_raw_keyboard_mouse",
                        "capture_duration_s": 20,
                    }
                )
                self.assertEqual(descriptor["armed"]["start_trigger"], "first_input")
                self.assertEqual(descriptor["armed"]["input_requirement"], "required")
            finally:
                state.close()

    def test_poc_evidence_ignores_recoverable_trash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session_root = root / "sessions"
            writer = SessionWriter(
                session_root / ".trash" / "deleted-session",
                [DescribedSource()],
                [],
                video_encoding="mjpeg",
                session_context={
                    "experiment_id": "deleted",
                    "game_profile_id": "genshin-impact-pc",
                    "capture_kind": "game_profile",
                    "capture_id": "genshin-control-cruise",
                    "workflow_stage_id": "control-cruise",
                    "input_adapter": "none",
                    "input_requirement": "none",
                },
            )
            writer.close()

            index = build_poc_evidence_index(
                session_root, ProfileCatalog().game("genshin-impact-pc")
            )
            self.assertEqual(index["unassigned_sessions"], [])
            self.assertTrue(all(not stage["sessions"] for stage in index["stages"]))

    def test_simple_session_rejects_privilege_mismatch_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(),
                desktop_api=IntegrityMismatchDesktop(),
            )
            try:
                with self.assertRaisesRegex(
                    RuntimeError, "second elevated instance cannot replace"
                ):
                    state.start_session(
                        {
                            "experiment_id": "must-not-exist",
                            "window_title": "Popular Game A",
                            "input_adapter": "windows_raw_keyboard_mouse",
                            "capture_duration_s": 5,
                        }
                    )
                self.assertFalse((root / "sessions" / "must-not-exist").exists())
                self.assertIsNone(state.descriptor()["armed"])
            finally:
                state.close()

    def test_recorder_failure_discards_partial_session_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(),
                desktop_api=FailingCaptureDesktop(),
            )
            try:
                state.start_session(
                    {
                        "experiment_id": "failed-capture",
                        "window_title": "Popular Game A",
                        "input_adapter": "none",
                        "capture_duration_s": 5,
                        "start_delay_s": 0.01,
                    }
                )
                deadline = time.time() + 5
                while state.descriptor()["active_run"] is not None:
                    self.assertLess(time.time(), deadline)
                    time.sleep(0.02)

                self.assertFalse(
                    (root / "sessions" / "failed-capture" / "run_01").exists()
                )
                descriptor = state.descriptor()
                self.assertEqual(descriptor["sessions"], [])
                self.assertIn("synthetic frame failure", descriptor["last_error"])
            finally:
                state.close()

    def test_restart_restores_persisted_armed_experiment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_root = root / "profiles"
            write_catalog(profile_root)
            profiles = ProfileCatalog(profile_root)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=profiles,
                desktop_api=ArbitraryDesktop(),
            )
            state.arm(
                {
                    "game_profile_id": "game-a",
                    "capture_kind": "full_map",
                    "capture_id": "game-a-full-map",
                    "workflow_stage_id": "full-map",
                    "experiment_id": "persisted-workbench",
                    "target_runs": 1,
                    "capture_duration_s": 20,
                    "window_title": "Popular Game A",
                }
            )
            state.close()

            restored = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=profiles,
                desktop_api=ArbitraryDesktop(),
            )
            try:
                self.assertEqual(
                    restored.descriptor()["armed"]["experiment_id"],
                    "persisted-workbench",
                )
                restored.disarm()
            finally:
                restored.close()
            reopened = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=profiles,
                desktop_api=ArbitraryDesktop(),
            )
            try:
                self.assertIsNone(reopened.descriptor()["armed"])
            finally:
                reopened.close()

    def test_restart_recovers_latest_session_when_legacy_state_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_root = root / "profiles"
            write_catalog(profile_root)
            path = root / "sessions" / "recovered-experiment" / "run_01"
            writer = SessionWriter(
                path,
                [DescribedSource()],
                [],
                video_encoding="mjpeg",
                session_context={
                    "experiment_id": "recovered-experiment",
                    "game_profile_id": "game-a",
                    "route_profile_id": None,
                    "route_id": "game-a-full-map",
                    "capture_kind": "full_map",
                    "capture_id": "game-a-full-map",
                    "workflow_stage_id": "full-map",
                    "workflow_stage": None,
                    "run_index": 1,
                    "frame_adapter": "windows_window",
                    "input_adapter": "windows_raw_keyboard_mouse",
                },
            )
            writer.write_frame(
                FramePacket(
                    "main",
                    np.zeros((32, 48, 3), dtype=np.uint8),
                    writer.origin_ns,
                    writer.origin_ns,
                )
            )
            writer.close()

            restored = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(profile_root),
                desktop_api=ArbitraryDesktop(),
            )
            try:
                descriptor = restored.descriptor()
                self.assertEqual(
                    descriptor["armed"]["experiment_id"], "recovered-experiment"
                )
                self.assertEqual(descriptor["runs"][0]["frames"], 1)
                self.assertTrue(
                    (root / "artifacts" / "workbench_state.json").is_file()
                )
            finally:
                restored.close()

    def test_queued_take_starts_on_raw_input_while_game_is_foreground(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_root = root / "profiles"
            write_catalog(profile_root)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(profile_root),
                desktop_api=ArbitraryDesktop(),
                raw_input_api=SettlingRawInputApi(),
            )
            try:
                state.INPUT_SETTLE_DELAY_S = 0.1
                state.arm(
                    {
                        "game_profile_id": "game-a",
                        "capture_kind": "full_map",
                        "capture_id": "game-a-full-map",
                        "workflow_stage_id": "full-map",
                        "experiment_id": "no-focus-gate-test",
                        "target_runs": 1,
                        "capture_duration_s": 5,
                        "window_title": "Popular Game A",
                        "input_source": {
                            "adapter": "windows_raw_keyboard_mouse",
                        },
                    }
                )
                state._armed["capture_duration_s"] = 0.5
                state.queue_next_take()
                deadline = time.time() + 5
                while state.descriptor()["active_run"] is not None:
                    self.assertLess(time.time(), deadline)
                    time.sleep(0.02)

                path = root / "sessions" / "no-focus-gate-test" / "run_01"
                self.assertTrue(path.exists(), state.descriptor()["last_error"])
                reader = SessionReader(path)
                diagnostics = reader.manifest["input_sources"][0][
                    "raw_input_diagnostics"
                ]
                self.assertEqual(diagnostics["packets_received"], 3)
                self.assertEqual(diagnostics["packets_accepted"], 3)
                self.assertEqual(diagnostics["packets_rejected_foreground"], 0)
                self.assertEqual(
                    diagnostics["foreground_authority"],
                    "active_capture_lifecycle",
                )
                self.assertTrue(reader.manifest["recording_start"]["started"])
                self.assertEqual(
                    reader.manifest["recording_start"]["input_settle_delay_s"],
                    0.1,
                )
                self.assertEqual(len(reader.inputs), 2)
                self.assertEqual(reader.inputs[0]["kind"], "pc_raw_keyboard")
                self.assertEqual(reader.inputs[0]["session_time_ns"], 0)
                self.assertTrue(reader.frames_by_stream["main"])
            finally:
                state.close()

    def test_genshin_profile_exposes_guided_poc_workflow(self):
        game = ProfileCatalog().game("genshin-impact-pc")
        self.assertEqual(game["status"], "first_poc_game")
        self.assertEqual(
            game["default_input_source"]["adapter"],
            "windows_raw_keyboard_mouse",
        )
        self.assertEqual(
            [stage["capture_kind"] for stage in game["poc_workflow"]],
            [
                "game_profile",
                "scene_yaw_calibration",
                "minimap_calibration",
                "minimap_calibration",
                "minimap_calibration",
                "full_map",
                "game_behavior",
                "route",
            ],
        )
        labels = [stage.get("segment_label") for stage in game["poc_workflow"]]
        self.assertEqual(
            labels[:5],
            [
                "ordinary_cruise",
                "scene_rotation_360",
                "rotation_only",
                "movement_only",
                "forward_no_turn",
            ],
        )
        forward = game["poc_workflow"][4]
        self.assertEqual(forward["capture_duration_s"], 10)
        self.assertEqual(forward["start_trigger"], "settled_timer")
        self.assertEqual(forward["input_requirement"], "optional")
        self.assertEqual(
            forward["segment_semantics"]["movement_direction"],
            "cursor_heading",
        )
        instructions = " ".join(forward["instructions"]).casefold()
        self.assertIn("mini-map shift", instructions)
        self.assertIn("do not turn", instructions)
        self.assertTrue(game["profile_editor"]["controls"])

        teleport = game["poc_workflow"][6]
        self.assertEqual(teleport["capture_kind"], "game_behavior")
        self.assertEqual(teleport["segment_label"], "teleportation")
        self.assertEqual(teleport["input_requirement"], "required")
        self.assertEqual(
            teleport["segment_semantics"]["required_learned_coordinates"],
            ["teleport_target_global_xy", "destination_global_xy"],
        )

    def test_optional_labeled_segment_can_arm_without_input_capture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(),
                desktop_api=ArbitraryDesktop(),
            )
            try:
                armed = state.arm(
                    {
                        "game_profile_id": "genshin-impact-pc",
                        "capture_kind": "minimap_calibration",
                        "capture_id": "genshin-minimap-forward-no-turn",
                        "workflow_stage_id": "minimap-forward-no-turn",
                        "experiment_id": "forward-no-input-test",
                        "window_title": "Popular Game A",
                        "input_source": {"adapter": "none"},
                    }
                )["armed"]
                self.assertEqual(armed["segment_label"], "forward_no_turn")
                self.assertEqual(armed["start_trigger"], "settled_timer")
                self.assertEqual(armed["input_requirement"], "optional")
            finally:
                state.close()

        health = input_capture_health(
            {
                "context": {
                    "input_adapter": "windows_raw_keyboard_mouse",
                    "input_requirement": "optional",
                    "capture_kind": "minimap_calibration",
                },
                "input_counts": {},
            },
            [],
        )
        self.assertFalse(health["required"])
        self.assertEqual(health["requirement"], "optional")
        self.assertTrue(health["healthy"])

    def test_hud_reports_waiting_countdown_and_completion_without_disk_scan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_root = root / "profiles"
            write_catalog(profile_root)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(profile_root),
                desktop_api=ArbitraryDesktop(),
            )
            try:
                state.arm(
                    {
                        "game_profile_id": "game-a",
                        "capture_kind": "full_map",
                        "capture_id": "game-a-full-map",
                        "workflow_stage_id": "full-map",
                        "experiment_id": "hud-test",
                        "target_runs": 1,
                        "capture_duration_s": 20,
                        "window_title": "Popular Game A",
                    }
                )
                self.assertFalse(state.hud_descriptor()["visible"])
                state._active = {
                    "run_index": 1,
                    "phase": "arming_sources",
                    "recording_deadline_host_time_ns": None,
                }
                waiting = state.hud_descriptor()
                self.assertEqual(waiting["state"], "arming_sources")
                self.assertIn("1/1", waiting["title"])
                state._active.update(
                    phase="settling_queue_input",
                    input_eligible_host_time_ns=time.perf_counter_ns()
                    + 1_000_000_000,
                )
                settling = state.hud_descriptor()
                self.assertEqual(settling["status"], "SWITCH TO GAME")
                state._active.update(phase="waiting_for_first_input")
                self.assertEqual(
                    state.hud_descriptor()["status"], "PLAY TO START"
                )
                state._active.update(
                    phase="recording_uninterrupted_take",
                    recording_deadline_host_time_ns=time.perf_counter_ns()
                    + 5_000_000_000,
                )
                recording = state.hud_descriptor()
                self.assertEqual(recording["state"], "recording")
                self.assertGreaterEqual(recording["remaining_s"], 4)
                self.assertIn("REC", recording["status"])
                state._active = None
                state._hud_notice = {"state": "complete", "run_index": 1}
                complete = state.hud_descriptor()
                self.assertEqual(complete["status"], "CAPTURE COMPLETE")
                state.set_hud_runtime(True, capture_exclusion=True)
                self.assertTrue(
                    state.descriptor()["hud_runtime"]["capture_exclusion"]
                )
                toggles = []
                state.configure_hud_control(toggles.append)
                hidden = state.set_hud_enabled(False)
                self.assertEqual(toggles, [False])
                self.assertFalse(hidden["hud_runtime"]["enabled"])
                shown = state.set_hud_enabled(True)
                self.assertEqual(toggles, [False, True])
                self.assertTrue(shown["hud_runtime"]["enabled"])
            finally:
                state._active = None
                state.close()

    def test_profile_catalog_keeps_game_and_route_data_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "profiles"
            write_catalog(root)
            catalog = ProfileCatalog(root)
            self.assertEqual(catalog.game("game-a")["display_name"], "Popular Game A")
            self.assertEqual(
                catalog.route("route-a-profile")["game_profile_id"], "game-a"
            )
            self.assertNotIn("route_id", catalog.game("game-a"))

    def test_source_factory_exposes_faithful_gamepad_and_replaceable_frames(self):
        desktop = ArbitraryDesktop()
        raw_api = FakeRawInputApi()
        factory = SourceFactory(
            desktop_api=desktop,
            xinput_api=FakeXInputApi(),
            raw_input_api=raw_api,
        )
        frame = factory.frame(
            {
                "adapter": "windows_window",
                "window_title": "Popular Game A",
                "exact_title": True,
            }
        )
        keyboard = factory.input(
            {
                "adapter": "windows_keyboard_mouse",
                "window_title": "Popular Game A",
                "exact_title": True,
            }
        )
        gamepad = factory.input(
            {
                "adapter": "windows_xinput",
                "window_title": "Popular Game A",
                "exact_title": True,
            }
        )
        raw_input = factory.input(
            {
                "adapter": "windows_raw_keyboard_mouse",
                "window_title": "Popular Game A",
                "exact_title": True,
            }
        )
        self.assertIsInstance(frame, WindowsWindowFrameSource)
        self.assertIsInstance(keyboard, WindowsKeyboardMouseSource)
        self.assertIsInstance(gamepad, WindowsXInputSource)
        self.assertIsInstance(raw_input, WindowsRawKeyboardMouseSource)
        self.assertIs(raw_input.raw_input_api, raw_api)
        self.assertIsNone(factory.input({"adapter": "none"}))
        self.assertEqual(
            {item["adapter"] for item in factory.descriptor()["frame_adapters"]},
            {
                "windows_window",
                "android_scrcpy",
                "hik_mvs",
                "hik_rig_calibrated",
                "uvc",
                "adb_screenshot",
            },
        )
        xinput = next(
            item
            for item in factory.descriptor()["input_adapters"]
            if item["adapter"] == "windows_xinput"
        )
        self.assertEqual(xinput["status"], "recommended_pc_mvp")

    def test_hik_source_factory_exposes_native_and_rig_calibrated_sources(self):
        factory = SourceFactory()
        native = object()
        calibrated = object()
        with patch(
            "acquisition.workbench.NativeHikFrameSource", return_value=native
        ) as native_type, patch(
            "acquisition.workbench.CalibratedHikFrameSource",
            return_value=calibrated,
        ) as calibrated_type:
            self.assertIs(
                factory.frame(
                    {
                        "adapter": "hik_mvs",
                        "camera_id": "HIK123",
                        "fps": 25,
                    }
                ),
                native,
            )
            self.assertIs(
                factory.frame(
                    {
                        "adapter": "hik_rig_calibrated",
                        "calibration": "rig/hik_camera_calibration.json",
                    }
                ),
                calibrated,
            )
        self.assertEqual(native_type.call_args[0][0], "HIK123")
        self.assertEqual(native_type.call_args[1]["stream_id"], "main")
        self.assertEqual(
            calibrated_type.call_args[0][0],
            Path("rig/hik_camera_calibration.json"),
        )
        rig_descriptor = next(
            item
            for item in factory.descriptor()["frame_adapters"]
            if item["adapter"] == "hik_rig_calibrated"
        )
        self.assertEqual(rig_descriptor["label"], "Calibrated rig (ADB + HIK)")

    def test_rig_recording_uses_dual_bundle_boundary(self):
        calls = []
        expected = object()

        def build(calibration, **options):
            calls.append((calibration, options))
            return expected

        factory = SourceFactory(rig_bundle_builder=build)
        with patch.object(factory, "_adb", return_value=Path("adb.exe")):
            actual = factory.recording_bundle(
                {
                    "adapter": "hik_rig_calibrated",
                    "calibration": "rig/hik_camera_calibration.json",
                    "max_fps": 55,
                },
                {"adapter": "adb_getevent", "source_id": "touch"},
            )
        self.assertIs(actual, expected)
        self.assertEqual(calls[0][0], Path("rig/hik_camera_calibration.json"))
        self.assertEqual(calls[0][1]["adb"], Path("adb.exe"))
        self.assertEqual(calls[0][1]["input_adapter"], "adb_getevent")
        self.assertEqual(calls[0][1]["input_source_id"], "touch")
        self.assertEqual(calls[0][1]["max_fps"], 55.0)

    def test_rig_live_source_uses_oriented_bundle_primary(self):
        class Frame:
            def __init__(self, stream_id):
                self.stream_id = stream_id
                self.stopped = False

            def stop(self):
                self.stopped = True

        android = Frame("android_phone")
        hik = Frame("hik_phone")
        bundle = SimpleNamespace(
            frame_sources=[android, hik],
            primary_stream_id="hik_phone",
        )
        factory = SourceFactory(rig_bundle_builder=lambda *_args, **_kwargs: bundle)
        with patch.object(factory, "_adb", return_value=Path("adb.exe")):
            frame, input_source = factory.capture_sources(
                {
                    "adapter": "hik_rig_calibrated",
                    "calibration": "rig/hik_camera_calibration.json",
                },
                {"adapter": "none"},
            )
        self.assertIs(frame, hik)
        self.assertIsNone(input_source)
        self.assertTrue(android.stopped)
        self.assertFalse(hik.stopped)

    def test_hik_device_discovery_does_not_open_camera(self):
        closed = []

        class Adapter:
            def devices(self, probe=False):
                self.assert_probe = probe
                return (
                    SimpleNamespace(
                        device_id="HIK123",
                        label="USB3 MV-CS016 HIK123",
                        metadata={"model": "MV-CS016"},
                    ),
                )

            def close(self):
                closed.append(True)

        adapter = Adapter()
        factory = SourceFactory(hik_adapter_factory=lambda: adapter)
        devices = factory.hik_devices()
        self.assertTrue(adapter.assert_probe)
        self.assertEqual(devices[0]["camera_id"], "HIK123")
        self.assertEqual([True], closed)

    def test_parses_available_and_unavailable_adb_devices(self):
        devices = parse_adb_devices(
            "List of devices attached\n"
            "ABC123 device product:r8q model:SM_G781B device:r8q transport_id:1\n"
            "OFFLINE offline transport_id:2\n"
        )
        self.assertEqual(devices[0]["serial"], "ABC123")
        self.assertEqual(devices[0]["model"], "SM_G781B")
        self.assertTrue(devices[0]["available"])
        self.assertFalse(devices[1]["available"])

    def test_android_capture_pair_shares_one_device_clock(self):
        factory = SourceFactory()
        clock = object()
        hub = type(
            "FakeHub",
            (),
            {"register": lambda self, _stream_id: object()},
        )()
        input_source = object()
        factory._adb = lambda _config: Path("adb")
        with patch("acquisition.workbench.AdbClockMapper", return_value=clock), patch(
            "acquisition.workbench.find_scrcpy_server", return_value=Path("server")
        ), patch("acquisition.workbench.ScrcpyCaptureHub", return_value=hub) as hub_type, patch(
            "acquisition.workbench.AdbGetEventSource", return_value=input_source
        ) as input_type:
            frame, captured_input = factory.capture_sources(
                {"adapter": "android_scrcpy", "serial": "ANDROID123"},
                {"adapter": "adb_getevent"},
            )
        self.assertEqual(frame.stream_id, "main")
        self.assertIs(captured_input, input_source)
        self.assertIs(hub_type.call_args[1]["clock"], clock)
        self.assertIs(input_type.call_args[1]["clock"], clock)

    def test_simple_android_session_uses_scrcpy_and_getevent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(),
                desktop_api=ArbitraryDesktop(),
            )
            try:
                state.queue_next_take = state.descriptor
                descriptor = state.start_session(
                    {
                        "game_profile_id": "genshin-impact-android",
                        "capture_adapter": "android_scrcpy",
                        "serial": "ANDROID123",
                        "input_adapter": "adb_getevent",
                        "capture_duration_s": 20,
                    }
                )
                self.assertEqual(
                    descriptor["armed"]["frame_source"]["adapter"],
                    "android_scrcpy",
                )
                self.assertEqual(
                    descriptor["armed"]["frame_source"]["serial"], "ANDROID123"
                )
                self.assertEqual(
                    descriptor["armed"]["input_source"]["adapter"],
                    "adb_getevent",
                )
                self.assertEqual(descriptor["armed"]["start_trigger"], "first_input")
            finally:
                state.close()

    @staticmethod
    def _write_usable_rig_calibration(root: Path) -> Path:
        directory = root / "hik-calibration-test"
        directory.mkdir(parents=True)
        (directory / "rectification_maps.npz").write_bytes(b"maps")
        (directory / "valid_screen_mask.png").write_bytes(b"mask")
        path = directory / "hik_camera_calibration.json"
        path.write_text(
            json.dumps(
                {
                    "camera": {
                        "device_id": "HIK123",
                        "metadata": {"model": "MV-CS016"},
                    },
                    "phone": {"serial": "ANDROID123", "model": "Phone 1"},
                    "normalization": {"dense_map_file": "rectification_maps.npz"},
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_capture_inventory_lists_phone_hik_camera_and_saved_rig(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = self._write_usable_rig_calibration(root / "artifacts")
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(),
                desktop_api=ArbitraryDesktop(),
            )
            state.android_devices = lambda: {
                "devices": [
                    {
                        "serial": "ANDROID123",
                        "model": "Phone 1",
                        "available": True,
                    }
                ],
                "error": None,
            }
            state.sources.hik_devices = lambda: [
                {
                    "camera_id": "HIK123",
                    "label": "USB3 MV-CS016 HIK123",
                    "metadata": {},
                    "available": True,
                }
            ]
            try:
                inventory = state.capture_source_inventory()
            finally:
                state.close()
            self.assertEqual(
                inventory["android_devices"][0]["serial"], "ANDROID123"
            )
            self.assertEqual(inventory["hik_cameras"][0]["camera_id"], "HIK123")
            rig = inventory["rig_calibrations"][0]
            self.assertEqual(rig["path"], str(calibration.resolve()))
            self.assertTrue(rig["usable"])
            self.assertEqual(rig["phone_serial"], "ANDROID123")

    def test_simple_hik_rig_session_uses_calibrated_adapter_and_rig_phone(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = self._write_usable_rig_calibration(root / "artifacts")
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(),
                desktop_api=ArbitraryDesktop(),
            )
            try:
                state.queue_next_take = state.descriptor
                descriptor = state.start_session(
                    {
                        "game_profile_id": "genshin-impact-android",
                        "capture_adapter": "hik_rig_calibrated",
                        "rig_calibration": str(calibration),
                        "input_adapter": "adb_getevent",
                        "capture_duration_s": 20,
                    }
                )
                frame = descriptor["armed"]["frame_source"]
                captured_input = descriptor["armed"]["input_source"]
                self.assertEqual(frame["adapter"], "hik_rig_calibrated")
                self.assertEqual(frame["camera_id"], "HIK123")
                self.assertEqual(frame["calibration"], str(calibration.resolve()))
                self.assertEqual(captured_input["adapter"], "adb_getevent")
                self.assertEqual(captured_input["serial"], "ANDROID123")
            finally:
                state.close()

    def test_simple_native_hik_session_selects_camera_without_rig(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(),
                desktop_api=ArbitraryDesktop(),
            )
            try:
                state.queue_next_take = state.descriptor
                descriptor = state.start_session(
                    {
                        "game_profile_id": "genshin-impact-android",
                        "capture_adapter": "hik_mvs",
                        "camera_id": "HIK123",
                        "input_adapter": "none",
                        "capture_duration_s": 20,
                    }
                )
                frame = descriptor["armed"]["frame_source"]
                self.assertEqual(frame["adapter"], "hik_mvs")
                self.assertEqual(frame["camera_id"], "HIK123")
                self.assertNotIn("calibration", frame)
            finally:
                state.close()

    def test_android_straight_forward_uses_fixed_motion_events_and_releases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(),
                desktop_api=ArbitraryDesktop(),
            )
            stop = threading.Event()
            session = root / "sessions" / "android" / "run_01"
            session.mkdir(parents=True)
            state._armed = {
                "frame_source": {"adapter": "android_scrcpy", "serial": "ANDROID123"}
            }
            state._active = {
                "phase": "waiting_for_first_input",
                "stop": stop,
                "path": session,
                "android_control_events": [],
            }
            state.sources._adb = lambda _config: Path("adb")
            state.descriptor = lambda: {}
            calls = []

            def capture(command, **_kwargs):
                calls.append(command[-3:])
                if command[-3] == "MOVE":
                    stop.set()

            try:
                with self.assertRaisesRegex(ValueError, "non-zero movement vector"):
                    state.run_android_straight_forward(
                        {
                            "start_x": 420,
                            "start_y": 820,
                            "end_x": 420,
                            "end_y": 820,
                            "duration_s": 8,
                        }
                    )
                with patch("acquisition.workbench.subprocess.check_call", side_effect=capture):
                    state.run_android_straight_forward(
                        {
                            "start_x": 420,
                            "start_y": 820,
                            "end_x": 420,
                            "end_y": 640,
                            "duration_s": 8,
                        }
                    )
                    deadline = time.time() + 2
                    while (state._android_control or {}).get("status") == "running":
                        self.assertLess(time.time(), deadline)
                        time.sleep(0.01)
                self.assertEqual(
                    calls,
                    [["DOWN", "420", "820"], ["MOVE", "420", "640"], ["UP", "420", "640"]],
                )
                self.assertEqual(state._android_control["status"], "complete")
                self.assertEqual(len(state._active["android_control_events"]), 1)
                self.assertFalse((session / "android_control.json").exists())
            finally:
                state._active = None
                state.close()

    def test_raw_input_decoder_preserves_mouse_and_keyboard_details(self):
        mouse = _RawInput()
        mouse.header.dwType = 0
        mouse.header.hDevice = 41
        mouse.data.mouse.usFlags = 0
        mouse.data.mouse.data.usButtonFlags = 0x0001 | 0x0400
        mouse.data.mouse.data.usButtonData = ctypes.c_ushort(-120).value
        mouse.data.mouse.lLastX = 23
        mouse.data.mouse.lLastY = -11
        mouse_record = decode_raw_input(mouse, 555)
        self.assertEqual(mouse_record["kind"], "pc_raw_mouse")
        self.assertEqual(mouse_record["host_time_ns"], 555)
        self.assertEqual(mouse_record["payload"]["movement_mode"], "relative")
        self.assertEqual(mouse_record["payload"]["delta_x"], 23)
        self.assertEqual(mouse_record["payload"]["delta_y"], -11)
        self.assertEqual(
            mouse_record["payload"]["button_transitions"], ["left_down"]
        )
        self.assertEqual(mouse_record["payload"]["wheel_delta"], -120)

        keyboard = _RawInput()
        keyboard.header.dwType = 1
        keyboard.header.hDevice = 42
        keyboard.data.keyboard.MakeCode = 17
        keyboard.data.keyboard.Flags = 0x0002
        keyboard.data.keyboard.VKey = 87
        keyboard.data.keyboard.Message = 0x0100
        key_record = decode_raw_input(keyboard, 556)
        self.assertEqual(key_record["kind"], "pc_raw_keyboard")
        self.assertEqual(key_record["payload"]["key_name"], "W")
        self.assertEqual(key_record["payload"]["scan_code"], 17)
        self.assertTrue(key_record["payload"]["pressed"])
        self.assertTrue(key_record["payload"]["extended_e0"])

    def test_raw_input_source_emits_relative_mouse_and_keyboard_transitions(self):
        events = []
        source = WindowsRawKeyboardMouseSource(
            "Popular Game A",
            exact_title=True,
            desktop_api=ArbitraryDesktop(),
            raw_input_api=FakeRawInputApi(),
        )
        source.start(events.append)
        deadline = time.time() + 1
        while len(events) < 2 and time.time() < deadline:
            time.sleep(0.005)
        source.stop()
        self.assertEqual(
            [event.kind for event in events],
            ["pc_raw_mouse", "pc_raw_keyboard"],
        )
        self.assertEqual(events[0].payload["delta_x"], 14)
        self.assertEqual(events[0].payload["delta_y"], -7)
        self.assertEqual(events[0].payload["button_transitions"], ["left_down"])
        self.assertEqual(events[1].payload["scan_code"], 17)
        self.assertTrue(events[1].payload["pressed"])
        self.assertTrue(events[1].payload["foreground"])
        diagnostics = source.describe()["raw_input_diagnostics"]
        self.assertEqual(diagnostics["packets_received"], 2)
        self.assertEqual(diagnostics["packets_accepted"], 2)
        self.assertEqual(diagnostics["packets_rejected_foreground"], 0)

    def test_raw_input_source_uses_active_capture_lifetime_not_focus(self):
        events = []
        source = WindowsRawKeyboardMouseSource(
            "Popular Game A",
            exact_title=True,
            desktop_api=NeverForegroundDesktop(),
            raw_input_api=FakeRawInputApi(),
        )
        source.disable_foreground_filter()
        source.start(events.append)
        deadline = time.time() + 1
        while len(events) < 2 and time.time() < deadline:
            time.sleep(0.005)
        source.stop()
        self.assertEqual(len(events), 2)
        diagnostics = source.describe()["raw_input_diagnostics"]
        self.assertEqual(
            diagnostics["foreground_authority"], "active_capture_lifecycle"
        )
        self.assertEqual(diagnostics["packets_accepted"], 2)
        self.assertEqual(diagnostics["packets_rejected_foreground"], 0)
        self.assertTrue(all(not event.payload["foreground"] for event in events))

    def test_xinput_source_emits_complete_controller_state(self):
        events = []
        source = WindowsXInputSource(
            "Popular Game A",
            poll_hz=1000,
            exact_title=True,
            desktop_api=ArbitraryDesktop(),
            xinput_api=FakeXInputApi(),
        )
        source.start(events.append)
        deadline = time.time() + 1
        while len(events) < 2 and time.time() < deadline:
            time.sleep(0.005)
        source.stop()
        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(events[0].kind, "pc_xinput_state")
        self.assertTrue(events[0].payload["foreground"])
        self.assertEqual(events[0].payload["state"]["buttons"], ["a"])
        self.assertEqual(events[0].payload["state"]["left_stick_raw"], [12000, -5000])

    def test_automatic_take_bounds_use_first_control_without_in_game_marker(self):
        frames = [
            {"host_capture_time_ns": 100},
            {"host_capture_time_ns": 200},
            {"host_capture_time_ns": 300},
            {"host_capture_time_ns": 400},
        ]
        inputs = [
            {
                "kind": "pc_xinput_state",
                "host_time_ns": 120,
                "payload": {
                    "foreground": True,
                    "state": {
                        "buttons": [],
                        "left_trigger": 0,
                        "right_trigger": 0,
                        "left_stick": [0, 0],
                        "right_stick": [0, 0],
                    },
                },
            },
            {
                "kind": "pc_xinput_state",
                "host_time_ns": 270,
                "payload": {
                    "foreground": True,
                    "state": {
                        "buttons": [],
                        "left_trigger": 0,
                        "right_trigger": 0,
                        "left_stick": [0.6, 0],
                        "right_stick": [0, 0.2],
                    },
                },
            },
        ]
        self.assertEqual(automatic_take_bounds(frames, inputs), (2, 3))

    def test_automatic_take_bounds_ignore_idle_raw_mouse_packets(self):
        frames = [
            {"host_capture_time_ns": 100},
            {"host_capture_time_ns": 200},
            {"host_capture_time_ns": 300},
        ]
        inputs = [
            {
                "kind": "pc_raw_mouse",
                "host_time_ns": 110,
                "payload": {
                    "foreground": True,
                    "delta_x": 0,
                    "delta_y": 0,
                    "button_transitions": [],
                    "wheel_delta": 0,
                    "horizontal_wheel_delta": 0,
                },
            },
            {
                "kind": "pc_raw_mouse",
                "host_time_ns": 260,
                "payload": {
                    "foreground": True,
                    "delta_x": -5,
                    "delta_y": 3,
                    "button_transitions": [],
                    "wheel_delta": 0,
                    "horizontal_wheel_delta": 0,
                },
            },
        ]
        self.assertEqual(automatic_take_bounds(frames, inputs), (2, 2))
        self.assertEqual(
            automatic_take_bounds(frames, [], fallback_host_time_ns=260),
            (2, 2),
        )

    def test_workbench_arms_arbitrary_game_without_gameplay_hotkeys(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_root = root / "profiles"
            write_catalog(profile_root)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(profile_root),
                desktop_api=ArbitraryDesktop(),
                xinput_api=FakeXInputApi(),
            )
            try:
                descriptor = state.arm(
                    {
                        "game_profile_id": "game-a",
                        "route_profile_id": "route-a-profile",
                        "experiment_id": "arbitrary-game-poc",
                        "window_title": "Popular Game A",
                    }
                )
                self.assertEqual(
                    descriptor["armed"]["frame_source"]["window_title"],
                    "Popular Game A",
                )
                self.assertEqual(
                    descriptor["armed"]["input_source"]["adapter"],
                    "windows_xinput",
                )
                self.assertEqual(descriptor["armed"]["capture_duration_s"], 20)
                self.assertEqual(
                    descriptor["capture_policy"]["in_game_controls"], "none"
                )

                server = WorkbenchHttpServer(("127.0.0.1", 0), make_handler(state))
                state.configure_server_endpoint(
                    "127.0.0.1", server.server_address[1]
                )
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    base = "http://127.0.0.1:{}".format(server.server_address[1])
                    html = urllib.request.urlopen(base + "/").read().decode("utf-8")
                    api = json.loads(urllib.request.urlopen(base + "/api/state").read())
                    instance_api = json.loads(
                        urllib.request.urlopen(base + "/api/instance").read()
                    )
                    hud_api = json.loads(
                        urllib.request.urlopen(base + "/api/hud").read()
                    )
                    self.assertIn(
                        "Record first, then label and organize each session afterward.",
                        html,
                    )
                    self.assertIn("Start recording", html)
                    self.assertIn("Input capture", html)
                    self.assertIn("Sessions", html)
                    self.assertIn("Label", html)
                    self.assertIn("Delete", html)
                    self.assertIn("Show overlay", html)
                    self.assertIn("three-second settling countdown", html)
                    self.assertIn("function labelMenuActive()", html)
                    self.assertNotIn("F9", html)
                    self.assertNotIn("Combat Master", html)
                    self.assertEqual(api["armed"]["game_profile_id"], "game-a")
                    self.assertEqual(
                        instance_api["service"], "aria-trace-workbench"
                    )
                    self.assertEqual(instance_api, api["instance"])
                    self.assertEqual(instance_api["port"], server.server_address[1])
                    discovered = discover_workbench_instance(
                        "127.0.0.1", server.server_address[1]
                    )
                    self.assertEqual(
                        discovered["instance_id"], instance_api["instance_id"]
                    )
                    message = occupied_port_message(
                        "127.0.0.1", server.server_address[1], discovered
                    )
                    self.assertIn("PID {}".format(instance_api["process_id"]), message)
                    self.assertIn("did not replace or stop it", message)
                    self.assertFalse(hud_api["visible"])
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)
            finally:
                state.close()

    def test_state_api_replaces_nonfinite_tracker_metrics_with_json_null(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(),
                desktop_api=ArbitraryDesktop(),
            )
            state._live_tracker = {
                "status": "stopped",
                "latest": {
                    "global_fix": {
                        "decision": "rejected-quality:too-few-ratio-matches",
                        "reprojection_p95_localization_px": float("inf"),
                        "feature_correlation_center_agreement_px": float("-inf"),
                        "synthetic_nonfinite_probe": float("nan"),
                    }
                },
            }
            server = WorkbenchHttpServer(("127.0.0.1", 0), make_handler(state))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = "http://127.0.0.1:{}/api/state".format(
                    server.server_address[1]
                )
                body = urllib.request.urlopen(url).read()

                def reject_nonfinite(token):
                    raise ValueError("Non-finite JSON token: {}".format(token))

                parsed = json.loads(body, parse_constant=reject_nonfinite)
                global_fix = parsed["live_tracker"]["latest"]["global_fix"]
                self.assertIsNone(
                    global_fix["reprojection_p95_localization_px"]
                )
                self.assertIsNone(
                    global_fix["feature_correlation_center_agreement_px"]
                )
                self.assertIsNone(global_fix["synthetic_nonfinite_probe"])
                self.assertNotIn(b"Infinity", body)
                self.assertNotIn(b"NaN", body)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
                state.close()

    def test_post_take_confirmation_is_required_before_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_root = root / "profiles"
            write_catalog(profile_root)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(profile_root),
                desktop_api=ArbitraryDesktop(),
            )
            try:
                state.arm(
                    {
                        "game_profile_id": "game-a",
                        "route_profile_id": "route-a-profile",
                        "experiment_id": "confirm-test",
                        "window_title": "Popular Game A",
                    }
                )
                path = state._run_path(1)
                writer = SessionWriter(
                    path,
                    [DescribedSource()],
                    [],
                    video_encoding="mjpeg",
                )
                for index in range(3):
                    timestamp = writer.origin_ns + index * 33_000_000
                    writer.write_frame(
                        FramePacket(
                            "main",
                            np.full((32, 32, 3), index, dtype=np.uint8),
                            timestamp,
                            timestamp,
                        )
                    )
                writer.close()
                state._finalize_take(path, "short-route-a", failed=False)
                self.assertEqual(
                    state._slot(1)["status"], "captured_needs_confirmation"
                )
                state.confirm_take(1)
                self.assertEqual(state._slot(1)["status"], "ready")
            finally:
                state.close()

    def test_empty_configured_input_stream_cannot_remain_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_root = root / "profiles"
            write_catalog(profile_root)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(profile_root),
                desktop_api=ArbitraryDesktop(),
            )
            try:
                state.arm(
                    {
                        "game_profile_id": "game-a",
                        "capture_kind": "full_map",
                        "capture_id": "game-a-full-map",
                        "workflow_stage_id": "full-map",
                        "experiment_id": "empty-input-test",
                        "target_runs": 1,
                        "capture_duration_s": 20,
                        "window_title": "Popular Game A",
                        "input_source": {
                            "adapter": "windows_raw_keyboard_mouse",
                        },
                    }
                )
                path = state._run_path(1)
                writer = SessionWriter(
                    path,
                    [DescribedSource()],
                    [],
                    video_encoding="mjpeg",
                    session_context={
                        "game_profile_id": "game-a",
                        "capture_kind": "full_map",
                        "capture_id": "game-a-full-map",
                        "workflow_stage_id": "full-map",
                        "experiment_id": "empty-input-test",
                        "run_index": 1,
                        "input_adapter": "windows_raw_keyboard_mouse",
                    },
                )
                for index in range(3):
                    timestamp = writer.origin_ns + index * 33_000_000
                    writer.write_frame(
                        FramePacket(
                            "main",
                            np.full((32, 32, 3), index, dtype=np.uint8),
                            timestamp,
                            timestamp,
                        )
                    )
                writer.close()
                state._finalize_take(
                    path,
                    "game-a-full-map",
                    failed=False,
                    capture_kind="full_map",
                    capture_id="game-a-full-map",
                )
                store = AnnotationStore(path)
                store.add("capture_start", 0, "main", 0)
                store.add("capture_complete", 66_000_000, "main", 2)

                slot = state._slot(1)
                self.assertEqual(slot["status"], "needs_rerecord")
                self.assertFalse(slot["input_capture"]["healthy"])
                self.assertEqual(slot["control_input_events"], 0)
                with self.assertRaisesRegex(RuntimeError, "not waiting"):
                    state.confirm_take(1)

                evidence = state._refresh_poc_evidence_index("game-a")
                self.assertEqual(evidence["stages"][0]["status"], "needs_capture")
                indexed = evidence["stages"][0]["sessions"][0]
                self.assertEqual(indexed["status"], "failed")
                self.assertFalse(indexed["input_capture"]["healthy"])
            finally:
                state.close()

    def test_generic_evidence_capture_uses_non_route_markers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_root = root / "profiles"
            write_catalog(profile_root)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(profile_root),
                desktop_api=ArbitraryDesktop(),
            )
            try:
                state.arm(
                    {
                        "game_profile_id": "game-a",
                        "capture_kind": "full_map",
                        "capture_id": "game-a-full-map",
                        "workflow_stage_id": "full-map",
                        "experiment_id": "full-map-test",
                        "target_runs": 1,
                        "capture_duration_s": 20,
                        "window_title": "Popular Game A",
                    }
                )
                path = state._run_path(1)
                writer = SessionWriter(
                    path,
                    [DescribedSource()],
                    [],
                    video_encoding="mjpeg",
                    session_context={
                        "game_profile_id": "game-a",
                        "capture_kind": "full_map",
                        "capture_id": "game-a-full-map",
                        "workflow_stage_id": "full-map",
                        "experiment_id": "full-map-test",
                        "run_index": 1,
                    },
                )
                for index in range(3):
                    timestamp = writer.origin_ns + index * 33_000_000
                    writer.write_frame(
                        FramePacket(
                            "main",
                            np.full((32, 32, 3), index, dtype=np.uint8),
                            timestamp,
                            timestamp,
                        )
                    )
                writer.close()
                state._finalize_take(
                    path,
                    "game-a-full-map",
                    failed=False,
                    capture_kind="full_map",
                    capture_id="game-a-full-map",
                )
                self.assertEqual(
                    state._slot(1)["status"], "captured_needs_confirmation"
                )
                state.confirm_take(1)
                self.assertEqual(state._slot(1)["status"], "ready")
                annotations = AnnotationStore(path).list()
                kinds = [item["kind"] for item in annotations]
                self.assertIn("capture_start", kinds)
                self.assertIn("capture_complete", kinds)
                self.assertNotIn("route_start", kinds)
                descriptor = state.descriptor()
                self.assertEqual(descriptor["schema_version"], "1.2")
                self.assertEqual(descriptor["compile_state"], "not_applicable")
                evidence = descriptor["poc_evidence_indexes"]["game-a"]
                self.assertEqual(evidence["stages"][0]["status"], "ready")
                self.assertEqual(evidence["stages"][0]["ready_captures"], 1)
                self.assertEqual(
                    evidence["stages"][0]["sessions"][0]["capture_kind"],
                    "full_map",
                )
                self.assertTrue(
                    (
                        root
                        / "artifacts"
                        / "poc_evidence"
                        / "game-a"
                        / "evidence_index.json"
                    ).is_file()
                )
                with self.assertRaisesRegex(RuntimeError, "Only route captures"):
                    state.compile_and_evaluate()
            finally:
                state.close()

    def test_profile_draft_is_persisted_without_rewriting_source_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(),
                desktop_api=ArbitraryDesktop(),
            )
            try:
                descriptor = state.save_profile_draft(
                    {
                        "game_profile_id": "genshin-impact-pc",
                        "controls": {
                            "move_forward": {
                                "binding": "W",
                                "activation": "hold",
                                "status": "human_confirmed",
                            },
                            "dash": {
                                "binding": "Right Mouse",
                                "activation": "single_click_or_hold",
                                "status": "human_confirmed",
                            },
                        },
                        "behavior_notes": "Dash requires later timing measurement.",
                        "map_viewer_notes": "Confirm region switching during map capture.",
                    }
                )
                draft = descriptor["game_profile_drafts"]["genshin-impact-pc"]
                self.assertEqual(draft["controls"]["move_forward"]["binding"], "W")
                self.assertEqual(draft["controls"]["dash"]["binding"], "Right Mouse")
                self.assertTrue(
                    (
                        root
                        / "artifacts"
                        / "game_profiles"
                        / "genshin-impact-pc"
                        / "draft.json"
                    ).is_file()
                )
                self.assertEqual(
                    ProfileCatalog()
                    .game("genshin-impact-pc")["control_profile"]["status"],
                    "human_confirmation_required",
                )
                armed = state.arm(
                    {
                        "game_profile_id": "genshin-impact-pc",
                        "capture_kind": "game_profile",
                        "capture_id": "genshin-control-cruise",
                        "workflow_stage_id": "control-cruise",
                        "experiment_id": "genshin-profile-test",
                        "target_runs": 1,
                        "capture_duration_s": 20,
                        "window_title": "Popular Game A",
                    }
                )["armed"]
                self.assertEqual(armed["workflow_stage"]["stage_id"], "control-cruise")
                self.assertEqual(armed["segment_label"], "ordinary_cruise")
                self.assertEqual(armed["start_trigger"], "settled_timer")
                self.assertEqual(
                    armed["game_profile_draft"]["controls"]["dash"]["binding"],
                    "Right Mouse",
                )
            finally:
                state.close()

    def test_session_manager_lists_labels_appends_and_trashes_sessions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(),
                desktop_api=ArbitraryDesktop(),
            )
            try:
                state.arm(
                    {
                        "game_profile_id": "genshin-impact-pc",
                        "experiment_id": "managed-sessions",
                        "capture_kind": "game_profile",
                        "capture_id": "unlabeled-session",
                        "target_runs": 1,
                        "capture_duration_s": 10,
                        "window_title": "Popular Game A",
                        "frame_source": {"adapter": "windows_window", "fps": 30},
                        "input_source": {"adapter": "none"},
                        "start_trigger": "settled_timer",
                        "input_requirement": "none",
                    }
                )
                path = state._run_path(1)
                writer = SessionWriter(
                    path,
                    [DescribedSource()],
                    [],
                    video_encoding="mjpeg",
                    session_context={
                        "experiment_id": "managed-sessions",
                        "game_profile_id": "genshin-impact-pc",
                        "capture_kind": "game_profile",
                        "capture_id": "unlabeled-session",
                        "run_index": 1,
                        "input_adapter": "none",
                        "input_requirement": "none",
                    },
                )
                for index in range(3):
                    timestamp = writer.origin_ns + index * 33_000_000
                    writer.write_frame(
                        FramePacket(
                            "main",
                            np.full((32, 32, 3), index, dtype=np.uint8),
                            timestamp,
                            timestamp,
                        )
                    )
                writer.close()
                state._finalize_take(
                    path,
                    "unlabeled-session",
                    failed=False,
                    capture_kind="game_profile",
                    capture_id="unlabeled-session",
                )

                sessions = state.descriptor()["sessions"]
                self.assertEqual(len(sessions), 1)
                self.assertEqual(sessions[0]["session_key"], "managed-sessions/run_01")
                self.assertEqual(sessions[0]["status"], "recorded")
                self.assertEqual(sessions[0]["label"], "")

                labeled = state.label_session(
                    "managed-sessions/run_01", "rotation_only"
                )["sessions"][0]
                self.assertEqual(labeled["label"], "rotation_only")
                self.assertEqual(labeled["status"], "ready")
                metadata = json.loads(
                    (path / "session_metadata.json").read_text(encoding="utf-8")
                )
                self.assertEqual(
                    metadata["workflow_stage_id"], "minimap-rotation-only"
                )
                self.assertEqual(state._next_run_index(), 2)
                candidates = state.descriptor()["analysis_candidates"][
                    "genshin-impact-pc"
                ]["rotation_only"]
                self.assertEqual(candidates[0]["session_key"], "managed-sessions/run_01")
                self.assertTrue(candidates[0]["recommended"])

                scene_labeled = state.label_session(
                    "managed-sessions/run_01", "scene_rotation_360"
                )["sessions"][0]
                self.assertEqual(scene_labeled["label"], "scene_rotation_360")
                metadata = json.loads(
                    (path / "session_metadata.json").read_text(encoding="utf-8")
                )
                self.assertEqual(metadata["capture_kind"], "scene_yaw_calibration")
                self.assertEqual(metadata["workflow_stage_id"], "scene-rotation-360")
                self.assertEqual(metadata["capture_id"], "genshin-scene-rotation-360")
                scene_candidates = state.descriptor()["analysis_candidates"][
                    "genshin-impact-pc"
                ]["scene_rotation_360"]
                self.assertEqual(
                    scene_candidates[0]["session_key"], "managed-sessions/run_01"
                )

                teleport_labeled = state.label_session(
                    "managed-sessions/run_01", "teleportation"
                )["sessions"][0]
                self.assertEqual(teleport_labeled["label"], "teleportation")
                teleport_candidates = state.descriptor()["analysis_candidates"][
                    "genshin-impact-pc"
                ]["teleportation"]
                self.assertEqual(
                    teleport_candidates[0]["session_key"], "managed-sessions/run_01"
                )

                opened = []
                state._folder_opener = opened.append
                state.open_session_folder("managed-sessions/run_01")
                self.assertEqual(opened, [str(path.resolve())])
                with self.assertRaisesRegex(ValueError, "Invalid session"):
                    state.open_session_folder("../outside/run_01")

                deleted = state.delete_session("managed-sessions/run_01")
                self.assertEqual(deleted["sessions"], [])
                self.assertFalse(path.exists())
                self.assertEqual(
                    len(list((root / "sessions" / ".trash").iterdir())), 1
                )
                with self.assertRaisesRegex(ValueError, "Invalid session"):
                    state.delete_session("../outside")
            finally:
                state.close()

    def test_calibration_and_pose_results_keep_exact_session_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(),
                desktop_api=ArbitraryDesktop(),
            )

            def make_session(run_index, label):
                path = (
                    root
                    / "sessions"
                    / "analysis-sources"
                    / "run_{:02d}".format(run_index)
                )
                writer = SessionWriter(
                    path,
                    [DescribedSource()],
                    [],
                    video_encoding="mjpeg",
                    session_context={
                        "experiment_id": "analysis-sources",
                        "game_profile_id": "genshin-impact-pc",
                        "capture_kind": "game_profile",
                        "capture_id": label,
                        "run_index": run_index,
                        "input_adapter": "none",
                        "input_requirement": "none",
                    },
                )
                timestamp = writer.origin_ns
                writer.write_frame(
                    FramePacket(
                        "main",
                        np.full((32, 32, 3), run_index, dtype=np.uint8),
                        timestamp,
                        timestamp,
                    )
                )
                writer.close()
                (path / "session_metadata.json").write_text(
                    json.dumps({"label": label, "status": "ready"}),
                    encoding="utf-8",
                )
                return path

            rotation = make_session(1, "rotation_only")
            movement = make_session(2, "movement_only")
            forward = make_session(3, "forward_no_turn")
            scene = make_session(4, "scene_rotation_360")
            ordinary = make_session(5, "ordinary_cruise")

            def fake_calibration(
                _rotation,
                _movement,
                output,
                _config,
                progress=None,
                ordinary_session_path=None,
            ):
                self.assertEqual(ordinary_session_path, ordinary)
                output.mkdir(parents=True, exist_ok=True)
                (output / "cursor_pose_overlays.png").write_bytes(b"pose")
                return {
                    "schema_version": "1.0",
                    "generated_utc": "2026-08-27T12:00:00+00:00",
                    "status": "review_required",
                    "evidence": [
                        {
                            "name": "cursor_pose_overlays.png",
                            "title": "Cursor pose overlays",
                            "category": "pose",
                        }
                    ],
                }

            try:
                with patch(
                    "acquisition.workbench.calibrate_segment_sessions",
                    side_effect=fake_calibration,
                ), patch("acquisition.workbench.verify_forward_session") as verify:
                    descriptor = state.run_minimap_calibration(
                        {
                            "game_profile_id": "genshin-impact-pc",
                            "rotation_session_relative_path": "analysis-sources/run_01",
                            "movement_session_relative_path": "analysis-sources/run_02",
                            "ordinary_session_relative_path": "analysis-sources/run_05",
                        }
                    )
                verify.assert_not_called()
                calibration = descriptor["minimap_calibrations"][
                    "genshin-impact-pc"
                ][0]
                self.assertEqual(
                    calibration["source_sessions"]["rotation_only"]["session_key"],
                    "analysis-sources/run_01",
                )
                self.assertEqual(
                    calibration["source_sessions"]["movement_only"]["session_key"],
                    "analysis-sources/run_02",
                )
                self.assertEqual(
                    calibration["source_sessions"]["ordinary_cruise"]["session_key"],
                    "analysis-sources/run_05",
                )

                def fake_verification(
                    _forward, _calibration, output, progress=None
                ):
                    (output / "forward_pose_shift.png").write_bytes(b"shift")
                    return {
                        "status": "review_required",
                        "evidence": [
                            {
                                "name": "forward_pose_shift.png",
                                "title": "Cursor pose and map-shift relationship",
                                "category": "pose_verification",
                            }
                        ],
                    }

                with patch(
                    "acquisition.workbench.verify_forward_session",
                    side_effect=fake_verification,
                ):
                    descriptor = state.run_pose_verification(
                        {
                            "game_profile_id": "genshin-impact-pc",
                            "calibration_id": calibration["calibration_id"],
                            "rotation_session_relative_path": "analysis-sources/run_01",
                            "movement_session_relative_path": "analysis-sources/run_02",
                            "forward_session_relative_path": "analysis-sources/run_03",
                        }
                    )
                updated = descriptor["minimap_calibrations"][
                    "genshin-impact-pc"
                ][0]
                self.assertEqual(
                    updated["forward_verification"]["source_session_key"],
                    "analysis-sources/run_03",
                )
                self.assertEqual(
                    state.minimap_calibration_image(
                        "genshin-impact-pc",
                        calibration["calibration_id"],
                        "forward_pose_shift.png",
                    ),
                    b"shift",
                )

                def fake_scene_yaw(_session, output, config=None, progress=None):
                    output.mkdir(parents=True, exist_ok=True)
                    (output / "scene_yaw_curve.png").write_bytes(b"yaw")
                    return {
                        "schema_version": "1.0",
                        "generated_utc": "2026-08-27T12:10:00+00:00",
                        "status": "review_required",
                        "closure_error_deg": 1.2,
                        "evidence": [
                            {
                                "name": "scene_yaw_curve.png",
                                "title": "Scene yaw curve",
                                "category": "yaw",
                            }
                        ],
                    }

                with patch(
                    "acquisition.workbench.calibrate_scene_yaw_session",
                    side_effect=fake_scene_yaw,
                ):
                    descriptor = state.run_scene_yaw_calibration(
                        {
                            "game_profile_id": "genshin-impact-pc",
                            "session_relative_path": "analysis-sources/run_04",
                        }
                    )
                scene_result = descriptor["scene_yaw_calibrations"][
                    "genshin-impact-pc"
                ][0]
                self.assertEqual(
                    scene_result["source_session_key"], "analysis-sources/run_04"
                )
                self.assertEqual(
                    state.scene_yaw_image(
                        "genshin-impact-pc",
                        scene_result["calibration_id"],
                        "scene_yaw_curve.png",
                    ),
                    b"yaw",
                )
            finally:
                state.close()

    def test_minimap_calibration_keeps_ordinary_motion_optional_and_accepts_route(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(),
                desktop_api=ArbitraryDesktop(),
            )

            def make_session(run_index, label):
                path = (
                    root
                    / "sessions"
                    / "optional-motion"
                    / "run_{:02d}".format(run_index)
                )
                writer = SessionWriter(
                    path,
                    [DescribedSource()],
                    [],
                    video_encoding="mjpeg",
                    session_context={
                        "experiment_id": "optional-motion",
                        "game_profile_id": "genshin-impact-pc",
                        "capture_kind": "game_profile",
                        "capture_id": label,
                        "run_index": run_index,
                        "input_adapter": "none",
                        "input_requirement": "none",
                    },
                )
                timestamp = writer.origin_ns
                writer.write_frame(
                    FramePacket(
                        "main",
                        np.full((32, 32, 3), run_index, dtype=np.uint8),
                        timestamp,
                        timestamp,
                    )
                )
                writer.close()
                (path / "session_metadata.json").write_text(
                    json.dumps({"label": label}), encoding="utf-8"
                )
                return path

            rotation = make_session(1, "rotation_only")
            movement = make_session(2, "movement_only")
            route = make_session(6, "route")
            calls = []

            def fake_calibration(
                _rotation,
                _movement,
                output,
                _config,
                progress=None,
                ordinary_session_path=None,
            ):
                calls.append((_rotation, _movement, ordinary_session_path))
                output.mkdir(parents=True, exist_ok=True)
                return {
                    "schema_version": "1.0",
                    "generated_utc": "2026-08-29T12:00:00+00:00",
                    "status": "review_required",
                    "evidence": [],
                }

            try:
                request = {
                    "game_profile_id": "genshin-impact-pc",
                    "rotation_session_relative_path": "optional-motion/run_01",
                    "movement_session_relative_path": "optional-motion/run_02",
                }
                with patch(
                    "acquisition.workbench.calibrate_segment_sessions",
                    side_effect=fake_calibration,
                ):
                    descriptor = state.run_minimap_calibration(request)
                    without_motion = descriptor["minimap_calibrations"][
                        "genshin-impact-pc"
                    ][0]
                    descriptor = state.run_minimap_calibration(
                        dict(
                            request,
                            ordinary_session_relative_path="optional-motion/run_06",
                        )
                    )

                with_route = descriptor["minimap_calibrations"][
                    "genshin-impact-pc"
                ][0]
                self.assertEqual(
                    calls,
                    [
                        (rotation, movement, None),
                        (rotation, movement, route),
                    ],
                )
                self.assertNotIn(
                    "ordinary_cruise", without_motion["source_sessions"]
                )
                self.assertEqual(
                    with_route["source_sessions"]["ordinary_cruise"],
                    {
                        "session_key": "optional-motion/run_06",
                        "session_id": json.loads(
                            (route / "manifest.json").read_text(encoding="utf-8")
                        )["session_id"],
                        "recorded_label": "route",
                    },
                )
                self.assertEqual(
                    without_motion["calibration_id"], with_route["calibration_id"]
                )
                self.assertTrue(with_route["calibration_id"].startswith("segments-"))
            finally:
                state.close()

    def test_teleport_analysis_uses_selected_session_and_spatial_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(),
                desktop_api=ArbitraryDesktop(),
            )
            session = root / "sessions" / "teleport-sources" / "run_08"
            writer = SessionWriter(
                session,
                [DescribedSource()],
                [],
                video_encoding="mjpeg",
                session_context={
                    "experiment_id": "teleport-sources",
                    "game_profile_id": "genshin-impact-pc",
                    "capture_kind": "game_behavior",
                    "capture_id": "teleportation",
                    "run_index": 8,
                    "input_adapter": "none",
                    "input_requirement": "none",
                },
            )
            timestamp = writer.origin_ns
            writer.write_frame(
                FramePacket(
                    "main",
                    np.zeros((32, 32, 3), dtype=np.uint8),
                    timestamp,
                    timestamp,
                )
            )
            writer.close()
            (session / "session_metadata.json").write_text(
                json.dumps({"label": "teleportation"}), encoding="utf-8"
            )
            calibration_id = "calibration-a"
            calibration_root = (
                root
                / "artifacts"
                / "minimap_calibrations"
                / "genshin-impact-pc"
                / calibration_id
            )
            calibration_root.mkdir(parents=True)
            (calibration_root / "calibration.json").write_text(
                json.dumps({"calibration_id": calibration_id}), encoding="utf-8"
            )
            stitch_id = "stitch-a"
            stitch_root = (
                root
                / "artifacts"
                / "map_stitches"
                / "genshin-impact-pc"
                / stitch_id
            )
            stitch_root.mkdir(parents=True)
            (stitch_root / "map_stitch.json").write_text(
                json.dumps(
                    {
                        "stitch_id": stitch_id,
                        "source_minimap_calibration_id": calibration_id,
                        "localization": {"status": "ready"},
                    }
                ),
                encoding="utf-8",
            )
            calls = []

            def fake_analysis(
                source,
                output,
                *,
                game_profile_id,
                minimap_config,
                minimap_calibration,
                map_stitch,
                map_stitch_root,
                progress=None,
            ):
                calls.append(
                    (
                        source,
                        game_profile_id,
                        minimap_calibration["calibration_id"],
                        map_stitch["stitch_id"],
                        map_stitch_root,
                    )
                )
                output.mkdir(parents=True, exist_ok=True)
                (output / "teleport_phase_timeline.png").write_bytes(b"teleport")
                return {
                    "schema_version": "2.0",
                    "status": "review_required",
                    "coordinate_space_id": "map-stitch:stitch-a:original-map-px",
                    "teleport_target_global_xy": [10.0, 20.0],
                    "destination_global_xy": [11.0, 21.0],
                    "target_to_destination_offset_xy": [1.0, 1.0],
                    "phases": [],
                    "arrival_model": {},
                    "quality": {"generalization_status": "single_observed_episode"},
                    "provenance": {"generated_utc": "2026-08-29T12:00:00+00:00"},
                    "evidence_files": [
                        {
                            "name": "teleport_phase_timeline.png",
                            "title": "Teleport phases",
                            "category": "phases",
                        }
                    ],
                }

            try:
                with patch(
                    "acquisition.workbench.analyze_teleport_session",
                    side_effect=fake_analysis,
                ):
                    descriptor = state.run_teleport_analysis(
                        {
                            "game_profile_id": "genshin-impact-pc",
                            "session_relative_path": "teleport-sources/run_08",
                            "minimap_calibration_id": calibration_id,
                            "map_stitch_id": stitch_id,
                        }
                    )
                behavior = descriptor["teleport_behaviors"]["genshin-impact-pc"][0]
                self.assertEqual(behavior["source_session_key"], "teleport-sources/run_08")
                self.assertEqual(behavior["minimap_calibration_id"], calibration_id)
                self.assertEqual(behavior["map_stitch_id"], stitch_id)
                self.assertEqual(
                    calls,
                    [
                        (
                            session,
                            "genshin-impact-pc",
                            calibration_id,
                            stitch_id,
                            stitch_root,
                        )
                    ],
                )
                self.assertEqual(
                    state.teleport_behavior_image(
                        "genshin-impact-pc",
                        behavior["behavior_id"],
                        "teleport_phase_timeline.png",
                    ),
                    b"teleport",
                )
                (stitch_root / "map_stitch.json").write_text(
                    json.dumps(
                        {
                            "stitch_id": stitch_id,
                            "source_minimap_calibration_id": calibration_id,
                            "localization": {
                                "status": "review_required",
                                "quality": {
                                    "gradient_correlation_score": 0.309,
                                    "gradient_correlation_margin": 0.199,
                                    "reprojection_p95_original_map_px": 4.53,
                                },
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                with patch(
                    "acquisition.workbench.analyze_teleport_session"
                ) as blocked_analysis:
                    with self.assertRaisesRegex(
                        ValueError,
                        "localization is review required.*correlation 0.309.*"
                        "before teleport analysis",
                    ):
                        state.run_teleport_analysis(
                            {
                                "game_profile_id": "genshin-impact-pc",
                                "session_relative_path": "teleport-sources/run_08",
                                "minimap_calibration_id": calibration_id,
                                "map_stitch_id": stitch_id,
                            }
                        )
                blocked_analysis.assert_not_called()
            finally:
                state.close()

    def test_analysis_job_returns_immediately_and_reports_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(),
                desktop_api=ArbitraryDesktop(),
            )
            started = threading.Event()
            release = threading.Event()

            def blocked(_value, progress):
                progress("Synthetic long-running phase")
                started.set()
                release.wait(2)

            try:
                before = time.perf_counter()
                descriptor = state._queue_analysis("test", {}, blocked)
                self.assertLess(time.perf_counter() - before, 1.0)
                self.assertIn(descriptor["analysis_jobs"]["test"]["status"], {"queued", "running"})
                self.assertTrue(started.wait(1))
                self.assertEqual(
                    state.descriptor()["analysis_jobs"]["test"]["status"],
                    "running",
                )
                self.assertEqual(
                    state.descriptor()["analysis_jobs"]["test"]["message"],
                    "Synthetic long-running phase",
                )
                release.set()
                deadline = time.time() + 2
                while time.time() < deadline:
                    if state.descriptor()["analysis_jobs"]["test"]["status"] == "complete":
                        break
                    time.sleep(0.01)
                self.assertEqual(
                    state.descriptor()["analysis_jobs"]["test"]["status"],
                    "complete",
                )
            finally:
                release.set()
                state.close()

    def test_map_stitch_uses_ready_full_map_session_and_serves_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(),
                desktop_api=ArbitraryDesktop(),
            )
            try:
                path = root / "sessions" / "map-captures" / "run_01"
                writer = SessionWriter(
                    path,
                    [DescribedSource()],
                    [],
                    video_encoding="mjpeg",
                    session_context={
                        "experiment_id": "map-captures",
                        "game_profile_id": "genshin-impact-pc",
                        "capture_kind": "full_map",
                        "capture_id": "genshin-full-map",
                        "run_index": 1,
                        "input_adapter": "none",
                        "input_requirement": "none",
                    },
                )
                for index in range(2):
                    timestamp = writer.origin_ns + index * 33_000_000
                    writer.write_frame(
                        FramePacket(
                            "main",
                            np.full((32, 32, 3), index, dtype=np.uint8),
                            timestamp,
                            timestamp,
                        )
                    )
                writer.close()
                (path / "session_metadata.json").write_text(
                    json.dumps({"label": "full_map", "status": "ready"}),
                    encoding="utf-8",
                )
                calibration_id = "fixture-minimap-calibration"
                calibration_root = (
                    root
                    / "artifacts"
                    / "minimap_calibrations"
                    / "genshin-impact-pc"
                    / calibration_id
                )
                calibration_root.mkdir(parents=True)
                cv2.imwrite(
                    str(calibration_root / "forward_start.png"),
                    np.full((80, 120, 3), 127, dtype=np.uint8),
                )
                (calibration_root / "calibration.json").write_text(
                    json.dumps(
                        {
                            "outer_boundary": {
                                "center_x": 60,
                                "center_y": 40,
                                "radius": 30,
                            },
                            "forward_verification": {
                                "evidence": [{"name": "forward_start.png"}]
                            },
                        }
                    ),
                    encoding="utf-8",
                )

                def fake_stitch(
                    _session_path,
                    output,
                    progress=None,
                    localization_reference=None,
                ):
                    self.assertEqual(
                        localization_reference["calibration_id"], calibration_id
                    )
                    self.assertEqual(len(localization_reference["candidates"]), 1)
                    output.mkdir(parents=True, exist_ok=True)
                    (output / "mosaic.png").write_bytes(b"png-evidence")
                    return {
                        "schema_version": "1.0",
                        "generated_utc": "2026-08-27T12:00:00+00:00",
                        "status": "review_required",
                        "evidence": [
                            {
                                "name": "mosaic.png",
                                "title": "Observed full-map mosaic",
                                "category": "map",
                            }
                        ],
                        "registrations": [{"accepted": True}],
                        "provenance": {"source_frame_records": [{"frame_index": 0}]},
                    }

                with patch(
                    "acquisition.workbench.stitch_map_session",
                    side_effect=fake_stitch,
                ):
                    descriptor = state.run_map_stitch(
                        {
                            "game_profile_id": "genshin-impact-pc",
                            "session_relative_path": "map-captures/run_01",
                            "minimap_calibration_id": calibration_id,
                        }
                    )
                stitch = descriptor["map_stitches"]["genshin-impact-pc"][0]
                self.assertEqual(stitch["status"], "review_required")
                self.assertEqual(
                    stitch["source_session_key"], "map-captures/run_01"
                )
                self.assertEqual(
                    stitch["source_minimap_calibration_id"], calibration_id
                )
                self.assertNotIn("registrations", stitch)
                self.assertNotIn("source_frame_records", stitch["provenance"])
                self.assertEqual(
                    state.map_stitch_image(
                        "genshin-impact-pc", stitch["stitch_id"], "mosaic.png"
                    ),
                    b"png-evidence",
                )
                with self.assertRaisesRegex(ValueError, "Invalid map-stitch"):
                    state.map_stitch_image(
                        "genshin-impact-pc", stitch["stitch_id"], "../mosaic.png"
                    )
            finally:
                state.close()

    def test_builds_map_atlas_and_compiles_route_tracking_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(),
                desktop_api=ArbitraryDesktop(),
            )
            try:
                stitch_root = (
                    root / "artifacts" / "map_stitches" / "genshin-impact-pc"
                )
                for stitch_id in ("world-stitch", "town-stitch"):
                    path = stitch_root / stitch_id
                    path.mkdir(parents=True)
                    (path / "map_stitch.json").write_text(
                        json.dumps({"stitch_id": stitch_id}), encoding="utf-8"
                    )

                def fake_atlas(layers, output, canonical_mode_id, atlas_id):
                    output.mkdir(parents=True)
                    (output / "canonical_mosaic.png").write_bytes(b"atlas-image")
                    (output / "canonical_coverage.png").write_bytes(b"coverage")
                    result = {
                        "schema_version": "1.0",
                        "atlas_id": atlas_id,
                        "generated_utc": "2026-08-29T12:00:00+00:00",
                        "status": "ready",
                        "coordinate_space_id": "map-atlas:{}:canonical-map-px".format(
                            atlas_id
                        ),
                        "canonical_mode_id": canonical_mode_id,
                        "canonical_mosaic_file": "canonical_mosaic.png",
                        "canonical_coverage_file": "canonical_coverage.png",
                        "layers": [],
                    }
                    (output / "map_atlas.json").write_text(
                        json.dumps(result), encoding="utf-8"
                    )
                    return result

                with patch(
                    "acquisition.workbench.build_map_atlas", side_effect=fake_atlas
                ):
                    descriptor = state.run_map_atlas(
                        {
                            "game_profile_id": "genshin-impact-pc",
                            "atlas_id": "atlas-a",
                            "world_stitch_id": "world-stitch",
                            "town_stitch_id": "town-stitch",
                        }
                    )
                atlas = descriptor["map_atlases"]["genshin-impact-pc"][0]
                self.assertEqual(atlas["atlas_id"], "atlas-a")
                self.assertEqual(
                    state.map_atlas_image(
                        "genshin-impact-pc", "atlas-a", "canonical_mosaic.png"
                    ),
                    b"atlas-image",
                )

                calibration_root = (
                    root
                    / "artifacts"
                    / "minimap_calibrations"
                    / "genshin-impact-pc"
                    / "cal-a"
                )
                calibration_root.mkdir(parents=True)
                (calibration_root / "calibration.json").write_text(
                    json.dumps({"outer_boundary": {}}), encoding="utf-8"
                )
                route_session = root / "sessions" / "routes" / "run_01"
                route_session.mkdir(parents=True)

                def fake_compile(_session, output, **kwargs):
                    output.mkdir(parents=True)
                    result = {
                        "schema_version": "1.0",
                        "generated_utc": "2026-08-29T12:01:00+00:00",
                        "route_id": kwargs["route_id"],
                        "atlas_id": "atlas-a",
                        "coordinate_space_id": "map-atlas:atlas-a:canonical-map-px",
                        "state_count": 12,
                    }
                    (output / "manifest.json").write_text(
                        json.dumps(result), encoding="utf-8"
                    )
                    return result

                with patch.object(
                    state, "_session_path", return_value=route_session
                ), patch.object(
                    state,
                    "_describe_session",
                    return_value={
                        "session_id": "route-session-a",
                        "game_profile_id": "genshin-impact-pc",
                        "label": "route",
                        "route_id": "route-a",
                    },
                ), patch(
                    "acquisition.workbench.compile_route_session",
                    side_effect=fake_compile,
                ):
                    descriptor = state.run_route_tracking_compile(
                        {
                            "game_profile_id": "genshin-impact-pc",
                            "session_relative_path": "routes/run_01",
                            "map_atlas_id": "atlas-a",
                            "minimap_calibration_id": "cal-a",
                        }
                    )
                package = descriptor["route_tracking_packages"][
                    "genshin-impact-pc"
                ][0]
                self.assertEqual(package["route_id"], "route-a")
                self.assertEqual(package["source_session_key"], "routes/run_01")
                self.assertEqual(package["minimap_calibration_id"], "cal-a")
            finally:
                state.close()

    def test_map_atlas_uses_transition_endpoints_for_independent_layer_scales(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(),
                desktop_api=ArbitraryDesktop(),
            )
            try:
                stitch_root = (
                    root / "artifacts" / "map_stitches" / "genshin-impact-pc"
                )
                for stitch_id in ("world-stitch",):
                    path = stitch_root / stitch_id
                    path.mkdir(parents=True)
                    (path / "map_stitch.json").write_text(
                        json.dumps({"stitch_id": stitch_id}), encoding="utf-8"
                    )
                calibration_root = (
                    root
                    / "artifacts"
                    / "minimap_calibrations"
                    / "genshin-impact-pc"
                    / "cal-a"
                )
                calibration_root.mkdir(parents=True)
                (calibration_root / "calibration.json").write_text(
                    json.dumps(
                        {
                            "outer_boundary": {
                                "center_xy": [64.0, 64.0],
                                "radius": 60.0,
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                transition_session = root / "sessions" / "transitions" / "run_01"
                transition_session.mkdir(parents=True)
                source = {
                    "image": np.full((64, 64, 3), 40, np.uint8),
                    "mask": np.full((64, 64), 255, np.uint8),
                    "source_frame_index": 2,
                    "laplacian_variance": 18.0,
                }
                target = {
                    "image": np.full((64, 64, 3), 80, np.uint8),
                    "mask": np.full((64, 64), 255, np.uint8),
                    "source_frame_index": 58,
                    "laplacian_variance": 22.0,
                }

                def fake_atlas(layers, output, canonical_mode_id, atlas_id):
                    self.assertIs(layers[0]["minimap_reference"], source["image"])
                    self.assertIs(layers[1]["minimap_reference"], target["image"])
                    self.assertEqual(
                        layers[0]["stitch_root"], layers[1]["stitch_root"]
                    )
                    output.mkdir(parents=True)
                    result = {
                        "schema_version": "1.0",
                        "atlas_id": atlas_id,
                        "status": "ready",
                        "canonical_mode_id": canonical_mode_id,
                        "canonical_mosaic_file": "canonical_mosaic.png",
                        "canonical_coverage_file": "canonical_coverage.png",
                        "layers": [
                            {
                                "mode_id": "world",
                                "map_pixels_per_minimap_pixel": 2.6,
                            },
                            {
                                "mode_id": "town",
                                "map_pixels_per_minimap_pixel": 0.9,
                            },
                        ],
                    }
                    (output / "canonical_mosaic.png").write_bytes(b"image")
                    (output / "canonical_coverage.png").write_bytes(b"mask")
                    return result

                with patch.object(
                    state, "_session_path", return_value=transition_session
                ), patch.object(
                    state,
                    "_describe_session",
                    return_value={
                        "game_profile_id": "genshin-impact-pc",
                        "label": "minimap_transition",
                    },
                ), patch(
                    "acquisition.workbench.transition_endpoint_references",
                    return_value={"source": source, "target": target},
                ), patch(
                    "acquisition.workbench.build_map_atlas", side_effect=fake_atlas
                ), patch(
                    "acquisition.workbench.analyze_transition_session",
                    return_value={
                        "quality": {"confidence": 0.9},
                        "evidence_file": "transition_scale_timeline.png",
                    },
                ):
                    descriptor = state.run_map_atlas(
                        {
                            "game_profile_id": "genshin-impact-pc",
                            "atlas_id": "atlas-transition",
                            "map_stitch_id": "world-stitch",
                            "transition_session_relative_path": "transitions/run_01",
                            "minimap_calibration_id": "cal-a",
                        }
                    )

                atlas = descriptor["map_atlases"]["genshin-impact-pc"][0]
                self.assertEqual(
                    atlas["transition_reference"]["source_session_key"],
                    "transitions/run_01",
                )
                self.assertEqual(
                    atlas["transition_reference"]["source_frame_index"], 2
                )
                self.assertEqual(
                    atlas["transition_reference"]["target_frame_index"], 58
                )
                self.assertEqual(atlas["source_map_stitch_id"], "world-stitch")
            finally:
                state.close()

    def test_helpers_reject_empty_ids_and_map_nearest_frame(self):
        self.assertEqual(safe_id(" Route trial 01 "), "Route-trial-01")
        with self.assertRaises(ValueError):
            safe_id(" / ")
        frames = [
            {"host_capture_time_ns": 100},
            {"host_capture_time_ns": 300},
            {"host_capture_time_ns": 900},
        ]
        self.assertEqual(nearest_frame_index(frames, 260), 1)


if __name__ == "__main__":
    unittest.main()
