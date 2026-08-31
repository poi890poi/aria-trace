import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    import qrcode  # noqa: F401
    from PySide6.QtWidgets import QApplication

    from aria_trace.apps.rig_calibrator.application import RigCalibrationWindow

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


@unittest.skipUnless(GUI_AVAILABLE, "rig-calibrator GUI dependencies are unavailable")
class RigCalibratorGuiStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.window = RigCalibrationWindow(
            camera=mock.Mock(),
            adb=mock.Mock(),
            phone_target=mock.Mock(),
            output_root=Path(self.temporary.name),
        )

    def tearDown(self):
        self.window.deleteLater()
        self.temporary.cleanup()

    def test_actions_follow_target_camera_and_geometry_state(self):
        self.assertFalse(self.window.fit_geometry_button.isEnabled())
        self.assertFalse(self.window.quality_button.isEnabled())
        self.window._target_running = True
        self.window._camera_active = True
        self.window.latest_sample = object()
        self.window._update_action_state()
        self.assertTrue(self.window.fit_geometry_button.isEnabled())
        self.assertFalse(self.window.screen_width.isEnabled())
        self.assertFalse(self.window.camera_width.isEnabled())

        self.window.analysis = object()
        self.window._update_action_state()
        self.assertTrue(self.window.quality_button.isEnabled())
        self.assertTrue(self.window.latency_button.isEnabled())
        self.assertTrue(self.window.save_button.isEnabled())

    def test_geometry_input_change_invalidates_dependent_results(self):
        self.window.analysis = object()
        self.window.image_quality = {"result": "old"}
        self.window.timing = {"result": "old"}
        self.window.phone_diagonal.setValue(self.window.phone_diagonal.value() + 0.1)
        self.assertIsNone(self.window.analysis)
        self.assertIsNone(self.window.image_quality)
        self.assertIsNone(self.window.timing)


if __name__ == "__main__":
    unittest.main()
