import ctypes
import json
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from acquisition.annotations import AnnotationStore
from acquisition.models import FramePacket, InputPacket
from acquisition.profiles import ProfileCatalog
from acquisition.session import SessionReader, SessionWriter, input_capture_health
from acquisition.workbench import (
    AcquisitionWorkbench,
    SourceFactory,
    automatic_take_bounds,
    make_handler,
    nearest_frame_index,
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
                        "virtual_key": 87,
                        "key_name": "W",
                        "scan_code": 17,
                        "pressed": True,
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

    def test_queued_take_accepts_raw_input_without_any_focus_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile_root = root / "profiles"
            write_catalog(profile_root)
            state = AcquisitionWorkbench(
                root / "sessions",
                root / "artifacts",
                profiles=ProfileCatalog(profile_root),
                desktop_api=NeverForegroundDesktop(),
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
                reader = SessionReader(path)
                diagnostics = reader.manifest["input_sources"][0][
                    "raw_input_diagnostics"
                ]
                self.assertEqual(diagnostics["packets_received"], 2)
                self.assertEqual(diagnostics["packets_accepted"], 2)
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
                self.assertEqual(len(reader.inputs), 1)
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
                "minimap_calibration",
                "minimap_calibration",
                "minimap_calibration",
                "full_map",
                "route",
            ],
        )
        labels = [stage.get("segment_label") for stage in game["poc_workflow"]]
        self.assertEqual(
            labels[:4],
            [
                "ordinary_cruise",
                "rotation_only",
                "movement_only",
                "forward_no_turn",
            ],
        )
        forward = game["poc_workflow"][3]
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
            {"windows_window", "uvc", "adb_screenshot"},
        )
        xinput = next(
            item
            for item in factory.descriptor()["input_adapters"]
            if item["adapter"] == "windows_xinput"
        )
        self.assertEqual(xinput["status"], "recommended_pc_mvp")

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

                server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    base = "http://127.0.0.1:{}".format(server.server_address[1])
                    html = urllib.request.urlopen(base + "/").read().decode("utf-8")
                    api = json.loads(urllib.request.urlopen(base + "/api/state").read())
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
                    self.assertIn("three-second settling countdown", html)
                    self.assertNotIn("F9", html)
                    self.assertNotIn("Combat Master", html)
                    self.assertEqual(api["armed"]["game_profile_id"], "game-a")
                    self.assertFalse(hud_api["visible"])
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)
            finally:
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
