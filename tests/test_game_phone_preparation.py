import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from acquisition import capture_game_minimap_zigzag as capture
from acquisition.capture_game_minimap_zigzag import (
    _keyguard_showing,
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
                capture.main([])

        self.assertEqual(2, surface.call_count)
        constructor.assert_called_once_with(
            start_xy=[1872, 540],
            end_x=1386,
            vertical_amplitude_px=486,
            move_count=20,
            step_seconds=0.35,
            settle_seconds=1.5,
            reset_seconds=0.10,
        )


if __name__ == "__main__":
    unittest.main()
