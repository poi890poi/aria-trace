import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import yaml

from aria_trace.workflows import minimap_capture as capture
from aria_trace.adapters.filesystem.session import SessionReader
from aria_trace.domain.packets import FramePacket, InputPacket
from acquisition.capture_game_minimap_zigzag import (
    _game_booster_lock_showing,
    _hik_fallback_allowed,
    _keyguard_showing,
    _launch_or_defer_game,
    _session_game_label,
    _wake_phone_for_preparation,
)


class FakePhone:
    def __init__(self, keyguard=True):
        self.keyguard = keyguard
        self.commands = []

    def ensure_display_on(self, timeout_seconds):
        self.commands.append(("ensure_display_on", timeout_seconds))
        return {"state": "ON", "probes": 2}

    def shell(self, *arguments):
        self.commands.append(arguments)
        if arguments == ("dumpsys", "window", "policy"):
            return "isKeyguardShowing={}".format(
                "true" if self.keyguard else "false"
            )
        if arguments[:2] == ("input", "swipe"):
            self.keyguard = False
        return ""


class GamePhonePreparationTests(unittest.TestCase):
    def test_hik_fallback_is_limited_to_absence_or_ownership_failures(self):
        self.assertTrue(
            _hik_fallback_allowed(RuntimeError("open camera failed with MVS status 0x80000203"))
        )
        self.assertTrue(
            _hik_fallback_allowed(RuntimeError("Configured HIK camera was not found"))
        )
        self.assertFalse(
            _hik_fallback_allowed(RuntimeError("Rig calibration is for another camera"))
        )
        self.assertFalse(
            _hik_fallback_allowed(RuntimeError("Saved HIK bundle is incomplete"))
        )

    def test_capture_cli_allows_android_only_and_require_hik_is_explicit(self):
        arguments = capture.parser().parse_args([])
        self.assertIsNone(arguments.game_id)
        self.assertIsNone(arguments.diagnostic_rig_calibration_override)
        self.assertFalse(arguments.require_hik)
        with self.assertRaises(SystemExit):
            capture.parser().parse_args(["--rig-calibration", "legacy.json"])

    def test_adb_screenshot_mode_is_explicit_and_has_settle_delay(self):
        arguments = capture.parser().parse_args(
            [
                "--android-capture",
                "adb-screenshot",
                "--screenshot-settle-seconds",
                "0.6",
            ]
        )
        self.assertEqual("adb-screenshot", arguments.android_capture)
        self.assertEqual(0.6, arguments.screenshot_settle_seconds)

    def test_adb_screenshot_uses_exec_out_png_and_midpoint_timestamp(self):
        image = np.full((10, 12, 3), 40, np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        with patch.object(
            capture.subprocess,
            "check_output",
            return_value=encoded.tobytes(),
        ) as command, patch.object(
            capture.time, "perf_counter_ns", side_effect=[100, 300]
        ):
            packet, png = capture._capture_adb_screenshot_packet(
                Path("adb.exe"), "phone-1", 7
            )
        self.assertEqual(encoded.tobytes(), png)
        self.assertEqual(200, packet.host_capture_time_ns)
        self.assertEqual(300, packet.host_receive_time_ns)
        self.assertEqual(100, packet.metadata["timestamp_uncertainty_ns"])
        self.assertEqual(7, packet.metadata["stroke_index"])
        command.assert_called_once()
        self.assertEqual(
            ["adb.exe", "-s", "phone-1", "exec-out", "screencap", "-p"],
            command.call_args[0][0],
        )

    def test_unidentified_game_uses_current_foreground_app_without_launcher(self):
        arguments = capture.parser().parse_args([])
        with patch.object(capture, "launch_android_game") as launcher:
            result = _launch_or_defer_game(FakePhone(False), arguments)
        launcher.assert_not_called()
        self.assertEqual("manual_current_game", result["status"])
        self.assertIsNone(result["game_id"])
        self.assertEqual("unidentified-game", _session_game_label(None, None))

    def test_explicit_android_package_does_not_require_game_id(self):
        arguments = capture.parser().parse_args(
            ["--android-package", "org.example.game"]
        )
        with patch.object(
            capture,
            "launch_android_game",
            return_value={"status": "launched", "package": "org.example.game"},
        ) as launcher:
            result = _launch_or_defer_game(FakePhone(False), arguments)
        launcher.assert_called_once_with(
            unittest.mock.ANY,
            "org.example.game",
            explicit_package="org.example.game",
        )
        self.assertEqual("launched", result["status"])
        self.assertIsNone(result["game_id"])
        self.assertEqual(
            "org.example.game", _session_game_label(None, "org.example.game")
        )

    def test_visible_game_booster_lock_is_distinct_from_android_keyguard(self):
        phone = FakePhone(False)
        phone.shell = Mock(
            return_value=(
                "Window #9 Window{x u0 GameBooster Lock Screen}:\n"
                "  mHasSurface=true isReadyForDisplay()=true\n"
                "  isOnScreen=true\n"
                "  isVisible=true\n"
                "Window #10 Window{y u0 game}:\n"
            )
        )
        self.assertTrue(_game_booster_lock_showing(phone))

    def test_keyguard_state_comes_from_android_policy(self):
        self.assertTrue(_keyguard_showing(FakePhone(True)))
        self.assertFalse(_keyguard_showing(FakePhone(False)))

    def test_wakeup_uses_no_toggling_power_event_or_physical_control(self):
        phone = FakePhone(True)
        result = _wake_phone_for_preparation(
            phone, [2400, 1080], sleeper=lambda _seconds: None
        )
        flattened = [" ".join(map(str, command)) for command in phone.commands]
        self.assertFalse(result["physical_power_button_used"])
        self.assertIn("KEYCODE_WAKEUP", result["actions"])
        self.assertIn("keyguard_upward_swipe", result["actions"])
        self.assertFalse(result["keyguard_after"])
        self.assertFalse(any("KEYCODE_POWER" in command for command in flattened))
        self.assertFalse(any("KEYCODE_SLEEP" in command for command in flattened))

    def test_unlocked_phone_is_not_swiped_or_touched(self):
        phone = FakePhone(False)
        result = _wake_phone_for_preparation(
            phone, [2400, 1080], sleeper=lambda _seconds: None
        )
        self.assertNotIn("keyguard_upward_swipe", result["actions"])
        self.assertFalse(
            any(command[:2] == ("input", "swipe") for command in phone.commands)
        )

    def test_control_geometry_is_reprobed_after_game_launch(self):
        plan = Mock()
        plan.strokes.return_value = [
            {"start_xy": [1, 1], "end_xy": [2, 2]}
        ] * 20
        camera = SimpleNamespace(device_id="camera-1", label="camera")

        class PreparationCheckpoint(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory:
            calibration = Path(directory) / "hik_camera_calibration.json"
            calibration.write_text(
                json.dumps({"camera": {"device_id": "camera-1"}}),
                encoding="utf-8",
            )
            with patch.object(capture, "HikMvsCameraAdapter"), patch.object(
                capture, "_select_camera", return_value=camera
            ), patch.object(
                capture, "resolve_adb_executable", return_value=Path("adb.exe")
            ), patch.object(
                capture, "_select_phone", return_value="phone-1"
            ), patch.object(
                capture, "find_scrcpy_server", return_value=Path("scrcpy-server")
            ), patch.object(
                capture,
                "_phone_surface",
                side_effect=[
                    {"logical_size_px": [1080, 2400]},
                    {"logical_size_px": [2400, 1080]},
                ],
            ) as surface, patch.object(
                capture, "AdbPhoneSession"
            ), patch.object(
                capture, "_wake_phone_for_preparation", return_value={"keyguard_after": False}
            ), patch.object(
                capture,
                "launch_android_game",
                return_value={"package": "game", "status": "launched"},
            ), patch.object(
                capture.time, "sleep"
            ), patch.object(
                capture, "ZigzagTouchPlan", return_value=plan
            ) as constructor, patch(
                "builtins.input", side_effect=PreparationCheckpoint
            ):
                with self.assertRaises(PreparationCheckpoint):
                    capture.main(
                        [
                            "--diagnostic-rig-calibration-override",
                            str(calibration),
                            "--game-id",
                            "game-1",
                        ]
                    )

        self.assertEqual(2, surface.call_count)
        constructor.assert_called_once_with(
            start_xy=[1320, 540],
            end_x=1077,
            vertical_amplitude_px=486,
            move_count=12,
            step_seconds=0.35,
            settle_seconds=1.5,
            reset_seconds=0.10,
        )

    def test_adb_screenshot_entry_path_does_not_resolve_scrcpy(self):
        class PreparationCheckpoint(Exception):
            pass

        surface = {
            "logical_size_px": [2400, 1080],
            "natural_size_px": [1080, 2400],
            "quarter_turns_clockwise_from_natural": 1,
        }
        with patch.object(capture, "HikMvsCameraAdapter"), patch.object(
            capture, "_select_camera", return_value=None
        ), patch.object(
            capture, "resolve_adb_executable", return_value=Path("adb.exe")
        ), patch.object(
            capture, "_select_phone", return_value="phone-1"
        ), patch.object(
            capture, "find_scrcpy_server"
        ) as find_server, patch.object(
            capture, "_phone_surface", return_value=surface
        ), patch.object(
            capture, "AdbPhoneSession"
        ), patch.object(
            capture,
            "_wake_phone_for_preparation",
            return_value={"keyguard_after": False},
        ), patch.object(
            capture,
            "_launch_or_defer_game",
            return_value={"package": None, "status": "manual_current_game"},
        ), patch.object(
            capture.time, "sleep"
        ), patch.object(
            capture,
            "_dismiss_game_booster_lock",
            return_value={"detected": False, "dismissed": False, "attempts": 0},
        ) as dismiss, patch(
            "builtins.input", side_effect=PreparationCheckpoint
        ):
            with self.assertRaises(PreparationCheckpoint):
                capture.main(["--android-capture", "adb-screenshot"])

        find_server.assert_not_called()
        dismiss.assert_called_once()

    def test_control_only_skips_hik_capture_and_calibration(self):
        plan = Mock()
        directions = [
            "up", "up", "down", "down", "down", "down",
            "up", "up", "up", "up", "down", "down",
        ]
        plan.strokes.return_value = [
            {"direction": direction} for direction in directions
        ]
        plan.sampled_strokes.return_value = [{}] * 12
        plan.duration_seconds = 10.0
        control = Mock()
        control.wait_completed.return_value = True
        control.error = None
        control.completed = True
        control.events_issued = 120
        control.expected_event_count = 120

        with patch.object(capture, "HikMvsCameraAdapter") as hik, patch.object(
            capture, "resolve_adb_executable", return_value=Path("adb.exe")
        ), patch.object(
            capture, "_select_phone", return_value="phone-1"
        ), patch.object(
            capture, "find_scrcpy_server", return_value=Path("scrcpy-server")
        ), patch.object(
            capture,
            "_phone_surface",
            side_effect=[
                {"logical_size_px": [1080, 2400]},
                {"logical_size_px": [2400, 1080]},
            ],
        ), patch.object(
            capture, "AdbPhoneSession"
        ), patch.object(
            capture, "_wake_phone_for_preparation", return_value={"keyguard_after": False}
        ), patch.object(
            capture,
            "launch_android_game",
            return_value={"package": "game", "status": "launched"},
        ), patch.object(
            capture.time, "sleep"
        ), patch.object(
            capture, "ZigzagTouchPlan", return_value=plan
        ), patch.object(
            capture, "ScrcpyTouchController"
        ), patch.object(
            capture, "AndroidZigzagInputSource", return_value=control
        ), patch.object(
            capture,
            "_dismiss_game_booster_lock",
            return_value={"detected": False, "dismissed": False, "attempts": 0},
        ) as dismiss:
            self.assertEqual(
                capture.main(
                    ["--control-only", "--yes", "--game-id", "game-1"]
                ),
                0,
            )

        hik.assert_not_called()
        control.start.assert_called_once()
        control.stop.assert_called_once()
        self.assertEqual(dismiss.call_count, 2)

    def test_settled_adb_screenshots_write_a_canonical_minimap_session(self):
        plan = capture.ZigzagTouchPlan(
            start_xy=[24, 12],
            end_x=8,
            vertical_amplitude_px=8,
            move_count=4,
            step_seconds=0.03,
            settle_seconds=0.0,
            reset_seconds=0.0,
            move_sample_hz=10.0,
        )

        class FakeControl:
            source_id = "android_zigzag_control"

            def __init__(self, _adb, _serial, _plan, controller=None):
                self.completed = False
                self.error = None
                self.events_issued = 0
                self.expected_event_count = 8
                self._done = False
                self.controller = controller

            def start(self, emit):
                for index, stroke in enumerate(plan.strokes()):
                    for action, point in (
                        ("DOWN", stroke["start_xy"]),
                        ("UP", stroke["end_xy"]),
                    ):
                        self.events_issued += 1
                        emit(
                            InputPacket(
                                self.source_id,
                                "zigzag_touch",
                                time.perf_counter_ns(),
                                {
                                    "action": action,
                                    "point_xy": point,
                                    "point_index": index,
                                },
                            )
                        )
                self.completed = True
                self._done = True

            def wait_completed(self, _timeout):
                return self._done

            def stop(self):
                pass

            def describe(self):
                return {
                    "type": "FakeControl",
                    "source_id": self.source_id,
                    "events_issued": self.events_issued,
                }

        image = np.full((24, 32, 3), 80, np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)

        def screenshot(_adb, _serial, stroke_index):
            now = time.perf_counter_ns()
            return (
                FramePacket(
                    "android_phone",
                    image.copy(),
                    now,
                    now,
                    metadata={"stroke_index": stroke_index},
                ),
                encoded.tobytes(),
            )

        with tempfile.TemporaryDirectory() as directory, patch.object(
            capture, "AndroidZigzagInputSource", FakeControl
        ), patch.object(
            capture, "_capture_adb_screenshot_packet", side_effect=screenshot
        ), patch.object(capture, "ScrcpyTouchController") as scrcpy_control:
            pending = Path(directory) / "session.pending"
            manifest, control, hik = capture._record_adb_screenshot_zigzag(
                pending,
                adb=Path("adb.exe"),
                serial="phone-1",
                plan=plan,
                screenshot_settle_seconds=0.0,
                surface={
                    "quarter_turns_clockwise_from_natural": 1,
                    "logical_size_px": [32, 24],
                    "natural_size_px": [24, 32],
                },
                selected_camera=None,
                hik_adapter=None,
                rig_calibration=None,
                calibration_revision=None,
                calibration_selection=None,
                require_hik=False,
                hik_fallback_reason="camera absent",
                game_id="test-game",
                preparation={},
                game_launch={"status": "manual_current_game"},
                ffmpeg=None,
                video_encoding="mjpeg",
            )
            reader = SessionReader(pending)
            spaces = yaml.safe_load(
                (pending / "coordinate_spaces.yaml").read_text(encoding="utf-8")
            )
            self.assertIsNone(hik)
            self.assertTrue(control.completed)
            self.assertEqual("complete", manifest["status"])
            self.assertEqual(4, len(reader.frames_by_stream["android_phone"]))
            first_space = reader.frames_by_stream["android_phone"][0][
                "metadata"
            ]["image_space"]
            self.assertEqual(
                "android_phone_natural_display_pixels",
                first_space["canonical_space_id"],
            )
            self.assertEqual([24, 32], first_space["canonical_size_px"])
            self.assertEqual(8, len(reader.inputs))
            self.assertEqual(
                "settled_swipe_endpoint_screenshots",
                reader.manifest["context"]["capture_schedule"],
            )
            self.assertFalse(
                reader.manifest["context"]["android_capture"]["scrcpy_used"]
            )
            self.assertEqual(
                "compatible_android_phone_session",
                reader.manifest["context"]["calibration_compatibility"][
                    "minimap"
                ],
            )
            self.assertEqual(
                "requires_hik_pairs_not_present",
                reader.manifest["context"]["calibration_compatibility"][
                    "game_color"
                ],
            )
            self.assertIn(
                "midpoint of the ADB screencap request/receive interval",
                spaces["streams"]["android_phone"]["timestamp_authority"],
            )
            self.assertEqual(
                4,
                len(list((pending / "screenshots" / "android_phone").glob("*.png"))),
            )
            scrcpy_control.assert_not_called()

    def test_settled_adb_screenshots_keep_optional_hik_pairs_for_color(self):
        plan = capture.ZigzagTouchPlan(
            start_xy=[24, 12],
            end_x=8,
            vertical_amplitude_px=8,
            move_count=4,
            step_seconds=0.03,
            settle_seconds=0.0,
            reset_seconds=0.0,
            move_sample_hz=10.0,
        )

        class FakeControl:
            source_id = "android_zigzag_control"

            def __init__(self, *_args, **_kwargs):
                self.completed = False
                self.error = None
                self.events_issued = 0
                self.expected_event_count = 8

            def start(self, emit):
                for index, stroke in enumerate(plan.strokes()):
                    for action, point in (
                        ("DOWN", stroke["start_xy"]),
                        ("UP", stroke["end_xy"]),
                    ):
                        self.events_issued += 1
                        emit(
                            InputPacket(
                                self.source_id,
                                "zigzag_touch",
                                time.perf_counter_ns(),
                                {
                                    "action": action,
                                    "point_xy": point,
                                    "point_index": index,
                                },
                            )
                        )
                self.completed = True

            def wait_completed(self, _timeout):
                return self.completed

            def stop(self):
                pass

            def describe(self):
                return {"type": "FakeControl", "source_id": self.source_id}

        image = np.full((24, 32, 3), 90, np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)

        def screenshot(_adb, _serial, stroke_index):
            now = time.perf_counter_ns()
            return (
                FramePacket(
                    "android_phone",
                    image.copy(),
                    now,
                    now,
                    metadata={"stroke_index": stroke_index},
                ),
                encoded.tobytes(),
            )

        class FakeHik:
            stream_id = "hik_phone"

            def __init__(self, *_args, **_kwargs):
                self.turns = 0
                self.read_count = 0

            def start(self):
                pass

            def stop(self):
                pass

            def read(self):
                self.read_count += 1
                now = time.perf_counter_ns()
                return FramePacket(
                    self.stream_id,
                    image.copy(),
                    now,
                    now,
                    metadata={
                        "video_encoding_padding_right_bottom_px": [0, 0]
                    },
                )

            def alignment_evidence_image(self, packet):
                return packet.image

            def set_output_orientation(self, turns, _evidence=None):
                self.turns = turns

            def describe(self):
                return {
                    "type": "FakeHik",
                    "stream_id": self.stream_id,
                    "output_quarter_turns_clockwise_from_calibration_display": self.turns,
                }

        class FakeEvidence:
            def __init__(self, *_args, **_kwargs):
                pass

            def start(self, *_args):
                pass

            def process(self, *_args):
                pass

            def close(self, **_kwargs):
                pass

            def describe(self):
                return {"type": "FakeEvidence"}

        orientation = {
            "status": "selected",
            "selection_basis": "test_images",
            "selected_confidence": 0.9,
            "confidence_margin": 0.2,
            "selected_adb_surface_quarter_turns_clockwise_from_phone_natural": 1,
            "selected_camera_adapter_image_quarter_turns_clockwise_from_calibration_display": 0,
        }

        def write_spaces(path, *_args):
            target = Path(path) / "coordinate_spaces.yaml"
            target.write_text(
                yaml.safe_dump(
                    {
                        "streams": {"android_phone": {}, "hik_phone": {}},
                        "conversions": {
                            "adb_to_hik_phone_video_3x3": np.eye(3).tolist()
                        },
                    }
                ),
                encoding="utf-8",
            )
            return target

        with tempfile.TemporaryDirectory() as directory, patch.object(
            capture, "AndroidZigzagInputSource", FakeControl
        ), patch.object(
            capture, "_capture_adb_screenshot_packet", side_effect=screenshot
        ), patch.object(
            capture, "RectifiedHikCamera"
        ), patch.object(
            capture, "CalibratedHikFrameSource", FakeHik
        ), patch.object(
            capture,
            "match_game_camera_orientation",
            return_value=(orientation, {}),
        ), patch.object(
            capture, "GameCrossSourceEvidenceRecorder", FakeEvidence
        ), patch.object(
            capture, "write_dual_source_space_yaml", side_effect=write_spaces
        ):
            pending = Path(directory) / "session.pending"
            calibration = Path(directory) / "hik_camera_calibration.json"
            calibration.write_text("{}", encoding="utf-8")
            manifest, _control, hik = capture._record_adb_screenshot_zigzag(
                pending,
                adb=Path("adb.exe"),
                serial="phone-1",
                plan=plan,
                screenshot_settle_seconds=0.0,
                surface={
                    "quarter_turns_clockwise_from_natural": 1,
                    "logical_size_px": [32, 24],
                    "natural_size_px": [24, 32],
                },
                selected_camera=SimpleNamespace(device_id="camera-1"),
                hik_adapter=Mock(),
                rig_calibration=calibration,
                calibration_revision="rig-1",
                calibration_selection="active_profile_registry",
                require_hik=True,
                hik_fallback_reason=None,
                game_id="test-game",
                preparation={},
                game_launch={"status": "manual_current_game"},
                ffmpeg=None,
                video_encoding="mjpeg",
            )
            reader = SessionReader(pending)
            self.assertIsNotNone(hik)
            self.assertEqual(4, len(reader.frames_by_stream["android_phone"]))
            self.assertEqual(4, len(reader.frames_by_stream["hik_phone"]))
            self.assertEqual(
                "compatible_synchronized_adb_hik_pairs",
                manifest["context"]["calibration_compatibility"]["game_color"],
            )


if __name__ == "__main__":
    unittest.main()
