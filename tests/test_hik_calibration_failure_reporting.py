import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np

from rig_runtime.workflows.hik_rig_calibration import (
    HikCalibrationOptions,
    HikRigCalibrationSession,
)


class HikCalibrationFailureReportingTests(unittest.TestCase):
    def session(self, directory, messages):
        target = mock.Mock(last_target=None, last_screenshot=None)
        return HikRigCalibrationSession(
            HikCalibrationOptions("fake", "phone", Path(directory) / "rig", headless=True),
            camera=mock.Mock(), phone=mock.Mock(), target=target, progress=messages.append,
        )

    def fail_geometry(self, session, error):
        with ExitStack() as stack:
            for name in ("open", "close", "calibrate_lens_distortion", "wait_for_positioning_confirmation"):
                stack.enter_context(mock.patch.object(session, name, return_value=True))
            stack.enter_context(mock.patch.object(session, "calibrate_geometry", side_effect=error))
            with self.assertRaises(type(error)) as raised:
                session.run()
            self.assertIs(raised.exception, error)
            session.close.assert_called_once_with()

    def test_reason_precedes_saved_evidence_and_matches_failure_json(self):
        with tempfile.TemporaryDirectory() as directory:
            messages = []
            session = self.session(directory, messages)
            session.target.last_target = np.zeros((20, 20, 3), np.uint8)
            error = RuntimeError("Only 3 ChArUco corners detected; need at least 6")
            self.fail_geometry(session, error)
            failures = [item for item in messages if str(error) in item]
            self.assertEqual(len(failures), 1)
            self.assertIn("geometry", failures[0])
            self.assertIn("RuntimeError", failures[0])
            saved = next(item for item in messages if "Failure evidence saved for review:" in item)
            self.assertLess(messages.index(failures[0]), messages.index(saved))
            evidence = Path(saved.split("review: ", 1)[1])
            report = json.loads((evidence / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(str(error), report["error"])
            self.assertEqual("geometry", report["failed_stage"])
            self.assertFalse(session.options.output_directory.exists())

    def test_no_images_still_reports_the_original_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            messages = []
            session = self.session(directory, messages)
            error = TimeoutError("Camera frame timed out after 10 seconds")
            self.fail_geometry(session, error)
            self.assertTrue(any(str(error) in item and "geometry" in item for item in messages))
            self.assertFalse(any("Failure evidence saved" in item for item in messages))

    def test_evidence_permission_error_does_not_replace_calibration_error(self):
        with tempfile.TemporaryDirectory() as directory:
            messages = []
            session = self.session(directory, messages)
            error = RuntimeError("Visible phone region could not be located")
            with mock.patch.object(session, "_write_failure_evidence", side_effect=PermissionError("Access is denied")):
                self.fail_geometry(session, error)
            self.assertIn(str(error), messages[0])
            self.assertTrue(any("warning" in item.lower() and "Access is denied" in item for item in messages))
            self.assertFalse(any("Failure evidence saved" in item for item in messages))

    def test_empty_exception_message_still_names_the_error_type(self):
        with tempfile.TemporaryDirectory() as directory:
            messages = []
            session = self.session(directory, messages)
            self.fail_geometry(session, TimeoutError())
            self.assertTrue(any("TimeoutError" in item and "geometry" in item for item in messages))


if __name__ == "__main__":
    unittest.main()
