import unittest

from acquisition.android_game_launcher import (
    launch_android_game,
    resolve_game_package,
)


class FakePhone:
    def __init__(self, packages, foreground=None, resolvable=True):
        self.packages = list(packages)
        self.foreground = foreground
        self.resolvable = resolvable
        self.commands = []

    def shell(self, *arguments):
        self.commands.append(arguments)
        if arguments == ("pm", "list", "packages"):
            return "\n".join("package:" + item for item in self.packages)
        if arguments == ("dumpsys", "activity", "activities"):
            return (
                "mResumedActivity: ActivityRecord{{abc u0 "
                + self.foreground
                + " t1}}"
                if self.foreground
                else ""
            )
        if arguments[:3] == ("cmd", "package", "resolve-activity"):
            if self.resolvable:
                package = arguments[-1]
                return package + "/com.miHoYo.GetMobileInfo.MainActivity"
            return "No activity found"
        if arguments[:2] == ("am", "start"):
            component = arguments[-1]
            self.foreground = component
            return "Status: ok"
        if arguments[:1] == ("monkey",):
            package = arguments[arguments.index("-p") + 1]
            self.foreground = package + "/.MainActivity"
            return "Events injected: 1"
        return ""


class AndroidGameLauncherTests(unittest.TestCase):
    def test_genshin_resolves_the_installed_distribution(self):
        self.assertEqual(
            "com.miHoYo.GenshinImpact",
            resolve_game_package(
                "genshin-impact", ["com.miHoYo.GenshinImpact"]
            ),
        )
        self.assertEqual(
            "com.miHoYo.Yuanshen",
            resolve_game_package("genshin-impact", ["com.miHoYo.Yuanshen"]),
        )

    def test_explicit_package_must_be_installed(self):
        with self.assertRaisesRegex(RuntimeError, "not installed"):
            resolve_game_package("custom", [], "example.game")

    def test_unknown_game_remains_manual_without_guessing(self):
        phone = FakePhone(["example.game"])
        result = launch_android_game(phone, "unknown-game")
        self.assertEqual("manual_unknown_game", result["status"])
        self.assertFalse(any(command[:2] == ("am", "start") for command in phone.commands))

    def test_already_foreground_game_is_not_restarted(self):
        package = "com.miHoYo.GenshinImpact"
        phone = FakePhone([package], package + "/.MainActivity")
        result = launch_android_game(phone, "genshin-impact")
        self.assertEqual("already_foreground", result["status"])
        self.assertFalse(any(command[:2] == ("am", "start") for command in phone.commands))

    def test_resolved_launcher_is_started_and_verified(self):
        package = "com.miHoYo.GenshinImpact"
        phone = FakePhone([package])
        result = launch_android_game(
            phone, "genshin-impact", sleeper=lambda _seconds: None
        )
        self.assertEqual("launched", result["status"])
        self.assertEqual("resolved_launcher_activity", result["method"])
        self.assertTrue(result["foreground_after"].startswith(package + "/"))
        self.assertFalse(result["game_input_injected"])
        self.assertFalse(result["calibration_controls_changed"])

    def test_launcher_resolution_falls_back_to_package_launch(self):
        package = "com.miHoYo.GenshinImpact"
        phone = FakePhone([package], resolvable=False)
        result = launch_android_game(
            phone, "genshin-impact", sleeper=lambda _seconds: None
        )
        self.assertEqual("package_launcher_fallback", result["method"])
        self.assertTrue(any(command[:1] == ("monkey",) for command in phone.commands))


if __name__ == "__main__":
    unittest.main()
