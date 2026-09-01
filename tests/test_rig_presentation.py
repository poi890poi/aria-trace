import io
import os
import re
import unittest
from unittest import mock

from aria_trace.apps.rig_presentation import (
    console_print,
    message_role,
    styled_console_text,
)
from aria_trace.workflows.hik_rig_calibration import (
    HikCalibrationOptions,
    HikRigCalibrationSession,
    PANEL_TEXT_REFRESH_SECONDS,
)


ANSI = re.compile(r"\x1b\[[0-9;]*m")


class _Terminal(io.StringIO):
    def isatty(self):
        return True


class RigConsolePresentationTests(unittest.TestCase):
    def test_styling_never_changes_message_characters(self):
        messages = (
            "Warning: camera result is outside the preferred range.",
            "Calibrating refresh-quantized exposure and gain...",
            "Complete calibration succeeded.",
            "Save blocked because the rig moved.",
        )
        for message in messages:
            rendered = styled_console_text(message, enabled=True)
            plain = ANSI.sub("", rendered)
            self.assertTrue(plain.endswith(message))
            self.assertEqual(plain.count(message), 1)

    def test_redirected_output_remains_plain_and_machine_safe(self):
        output = io.StringIO()
        console_print("Warning: unchanged content", file=output)
        self.assertEqual(output.getvalue(), "Warning: unchanged content\n")

    def test_no_color_disables_interactive_styling(self):
        output = _Terminal()
        with mock.patch.dict(os.environ, {"NO_COLOR": "1"}):
            console_print("Calibrating exposure...", file=output)
        self.assertEqual(output.getvalue(), "Calibrating exposure...\n")

    def test_message_roles_distinguish_operator_meaning(self):
        self.assertEqual(message_role("Warning: low contrast"), "warning")
        self.assertEqual(
            message_role("Rig reuse was skipped: no active profile."), "warning"
        )
        self.assertEqual(
            message_role(
                "Calibration bundle saved. Status remains warning until review."
            ),
            "warning",
        )
        self.assertEqual(message_role("Calibrating exposure..."), "stage")
        self.assertEqual(message_role("Calibration complete."), "success")
        self.assertEqual(message_role("Save blocked because the rig moved."), "error")
        self.assertIsNone(message_role('{"status": "complete"}'))

    def test_interactive_status_uses_text_cue_in_addition_to_color(self):
        rendered = styled_console_text(
            "Warning: low contrast", role="warning", enabled=True
        )
        self.assertEqual(
            ANSI.sub("", rendered), "[WARN] Warning: low contrast"
        )


class RigGuiPresentationTests(unittest.TestCase):
    def test_dynamic_telemetry_is_held_while_stage_updates_can_invalidate_it(self):
        session = HikRigCalibrationSession.__new__(HikRigCalibrationSession)
        session._panel_text_cache = {}
        first = session._readable_panel_lines("preview", ["Frame: 1"], now=10.0)
        held = session._readable_panel_lines(
            "preview",
            ["Frame: 2"],
            now=10.0 + PANEL_TEXT_REFRESH_SECONDS / 2.0,
        )
        refreshed = session._readable_panel_lines(
            "preview",
            ["Frame: 3"],
            now=10.0 + PANEL_TEXT_REFRESH_SECONDS,
        )
        self.assertEqual(first, ["Frame: 1"])
        self.assertEqual(held, ["Frame: 1"])
        self.assertEqual(refreshed, ["Frame: 3"])

        session._preview_stage = "old"
        session._preview_settings = {}
        session._set_preview_stage("new")
        self.assertNotIn("preview", session._panel_text_cache)


if __name__ == "__main__":
    unittest.main()
