import unittest

import numpy as np

from aria_trace.adapters.rig.devices import CameraAdapter
from aria_trace.services.calibration.rig.contracts import FrameSample
from aria_trace.workflows.rig_evidence_review import capture_full_camera_sample


class FakeHikAdapter(CameraAdapter):
    adapter_id = "fake_hik"

    def __init__(self):
        self.opened = False
        self.closed = False
        self.roi = None
        self.manual = None
        self.white_balance = None
        self.black_level = None
        self.read_count = 0

    def open(self, configuration):
        self.opened = True
        self.configuration = configuration
        return {}

    def reset_full_sensor_roi(self):
        self.roi = [0, 0, 80, 60]
        return list(self.roi)

    def set_manual_imaging(self, exposure_us, gain):
        self.manual = [exposure_us, gain]

    def set_white_balance(self, red, green, blue):
        self.white_balance = [red, green, blue]

    def set_black_level(self, value):
        self.black_level = value

    def read(self):
        self.read_count += 1
        image = np.full((60, 80, 3), self.read_count, np.uint8)
        return FrameSample(
            image,
            self.read_count,
            receive_time_ns=self.read_count,
            metadata={
                "image_space": {
                    "space_id": "hik_camera_acquisition_pixels",
                    "stored_size_px": [80, 60],
                    "parent_space_id": "hik_full_sensor_camera_pixels",
                    "roi_in_parent_xywh": [0, 0, 80, 60],
                    "local_to_parent_3x3": (
                        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                    ),
                    "orientation": "hik_camera_native",
                    "color_order": "BGR",
                }
            },
        )

    def close(self):
        self.closed = True
        self.opened = False


class RigEvidenceReviewTests(unittest.TestCase):
    def test_full_camera_capture_resets_roi_and_uses_saved_imaging(self):
        adapter = FakeHikAdapter()
        sample = capture_full_camera_sample(
            adapter,
            {
                "camera": {
                    "device_id": "camera-1",
                    "adapter_id": "fake_hik",
                    "full_sensor_mode": {
                        "width_px": 80,
                        "height_px": 60,
                        "fps": 30,
                    },
                },
                "imaging": {
                    "black_level": 2,
                    "exposure_us": 8000,
                    "gain": 3,
                    "white_balance": {
                        "ratio_red": 1200,
                        "ratio_green": 1000,
                        "ratio_blue": 1400,
                    },
                },
            },
            settle_frames=3,
        )
        self.assertEqual([0, 0, 80, 60], adapter.roi)
        self.assertEqual([8000.0, 3.0], adapter.manual)
        self.assertEqual([1200, 1000, 1400], adapter.white_balance)
        self.assertEqual(2, adapter.black_level)
        self.assertEqual(3, adapter.read_count)
        self.assertEqual(3, int(sample.image[0, 0, 0]))
        self.assertTrue(adapter.closed)


if __name__ == "__main__":
    unittest.main()
