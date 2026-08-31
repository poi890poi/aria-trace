import queue
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from androidcam import driver


class AndroidCameraTests(unittest.TestCase):
    def test_camera_server_command_selects_camera_source(self):
        clock = Mock()
        with patch(
            "aria_trace.adapters.android.capture.find_ffmpeg",
            return_value=Path("ffmpeg.exe"),
        ):
            hub = driver.ScrcpyCameraHub(
                adb=Path("adb.exe"),
                scrcpy_server=Path("scrcpy-server"),
                serial="phone",
                ffmpeg=Path("ffmpeg.exe"),
                clock=clock,
                camera_id="1",
                camera_width_px=1280,
                camera_height_px=720,
                camera_fps=30,
            )
        hub._scid = 0x1234
        command = hub._server_command("/data/local/tmp/server.jar")
        self.assertIn("video_source=camera", command)
        self.assertIn("camera_id=1", command)
        self.assertIn("camera_size=1280x720", command)
        self.assertIn("camera_fps=30", command)
        self.assertIn("capture_orientation=@", command)
        self.assertNotIn("power_on=false", command)
        self.assertNotIn("camera_facing=front", command)

    def test_driver_returns_frame_metadata_and_closes_hub(self):
        frame_queue = queue.Queue()
        image = np.zeros((720, 1280, 3), np.uint8)
        frame_queue.put(("frame", 7, image, 10, 20, 30))
        hub = Mock()
        hub.register.return_value = frame_queue
        hub.describe.return_value = {"video_source": "camera"}
        hub.take_drops.return_value = 0
        with patch.object(driver, "find_adb", return_value=Path("adb.exe")):
            with patch.object(
                driver, "find_server", return_value=Path("scrcpy-server")
            ):
                with patch.object(driver, "select_serial", return_value="phone"):
                    with patch.object(driver, "ScrcpyCameraHub", return_value=hub):
                        camera = driver.AndroidCamera("phone").open()
                        sample = camera.read_sample()
                        self.assertEqual((720, 1280, 3), sample.image.shape)
                        self.assertEqual(7, sample.sequence)
                        self.assertEqual(
                            "android_camera_scrcpy_bgr_pixels",
                            sample.metadata["image_space"]["space_id"],
                        )
                        camera.close()
        hub.start.assert_called_once_with()
        hub.stop.assert_called_once_with()

    def test_multiple_devices_require_explicit_source(self):
        output = (
            "List of devices attached\n"
            "phone device product:p\n"
            "tablet device product:t\n"
        )
        with patch.object(driver.subprocess, "check_output", return_value=output):
            with self.assertRaisesRegex(RuntimeError, "Multiple Android devices"):
                driver.select_serial(Path("adb.exe"), None)


if __name__ == "__main__":
    unittest.main()
