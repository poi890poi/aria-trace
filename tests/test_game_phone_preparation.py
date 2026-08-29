import unittest

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


if __name__ == "__main__":
    unittest.main()
