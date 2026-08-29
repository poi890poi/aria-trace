import unittest

import numpy as np

from acquisition.hik_capture import NativeHikFrameSource
from acquisition.rig_calibration.contracts import FrameSample


class FakeNativeAdapter:
    def __init__(self):
        self.opened = False
        self.closed = False
        self.roi = None

    def open(self, configuration):
        self.opened = True
        self.configuration = configuration
        return {
            "device_id": configuration.device_id,
            "width_px": 100,
            "height_px": 80,
            "fps": 30.0,
        }

    def controls(self):
        return {
            "genicam": {
                "SensorWidth": {"value": 101},
                "SensorHeight": {"value": 81},
            }
        }

    def set_roi(self, roi):
        self.roi = list(roi)
        return list(roi)

    def read(self):
        return FrameSample(
            np.zeros((81, 101, 3), np.uint8),
            100,
            receive_time_ns=110,
            metadata={"frame_number": 7, "device_timestamp": 55},
        )

    def close(self):
        self.closed = True


class NativeHikFrameSourceTests(unittest.TestCase):
    def test_native_source_forces_full_sensor_and_does_not_require_rig(self):
        adapter = FakeNativeAdapter()
        source = NativeHikFrameSource("camera-1", adapter=adapter)
        source.start()
        packet = source.read()
        self.assertEqual([0, 0, 101, 81], adapter.roi)
        self.assertEqual("hik_full", packet.stream_id)
        self.assertEqual((82, 102, 3), packet.image.shape)
        self.assertEqual(
            "native_hik_sensor_bgr_pixels", packet.metadata["coordinate_space"]
        )
        self.assertEqual([1, 1], packet.metadata["video_encoding_padding_right_bottom_px"])
        self.assertNotIn("rig_calibration", packet.metadata)
        source.stop()
        self.assertTrue(adapter.closed)


if __name__ == "__main__":
    unittest.main()
