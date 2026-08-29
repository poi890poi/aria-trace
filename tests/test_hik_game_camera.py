import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from acquisition.rig_calibration.contracts import FrameSample
from acquisition.rig_calibration.hik.game_camera import (
    ProfiledHikGameCamera,
    _source_crop_to_canonical_phone,
)


class FakeAdapter:
    def __init__(self):
        self.open_config = None
        self.roi = None
        self.read_count = 0
        self.closed = False

    def open(self, config):
        self.open_config = config
        return self

    def set_black_level(self, value):
        self.black_level = value

    def set_manual_imaging(self, exposure, gain):
        self.imaging = (exposure, gain)

    def set_white_balance(self, red, green, blue):
        self.white_balance = (red, green, blue)

    def align_roi(self, roi):
        return list(roi)

    def set_roi(self, roi):
        self.roi = list(roi)
        return list(roi)

    def read(self):
        self.read_count += 1
        x, y, width, height = self.roi
        xx = np.arange(x, x + width, dtype=np.uint8)[None, :]
        yy = np.arange(y, y + height, dtype=np.uint8)[:, None]
        image = np.empty((height, width, 3), dtype=np.uint8)
        image[:, :, 0] = xx
        image[:, :, 1] = yy
        image[:, :, 2] = 7
        return FrameSample(
            image=image,
            time_ns=1234,
            receive_time_ns=1250,
            metadata={"frame_number": 42},
        )

    def close(self):
        self.closed = True


def rig_document():
    return {
        "camera": {
            "device_id": "camera-1",
            "full_sensor_mode": {"width_px": 100, "height_px": 80, "fps": 30.0},
            "controls": {
                "genicam": {
                    "SensorWidth": {"value": 100},
                    "SensorHeight": {"value": 80},
                }
            },
        },
        "phone": {"natural_screen_size_px": [100, 80]},
        "imaging": {
            "black_level": 0,
            "exposure_us": 8000.0,
            "gain": 1.5,
            "white_balance": {
                "ratio_red": 1001,
                "ratio_green": 1002,
                "ratio_blue": 1003,
            },
        },
        "geometry": {
            "screen_to_full_sensor_camera_3x3": np.eye(3).tolist(),
            "full_sensor_camera_to_screen_3x3": np.eye(3).tolist(),
        },
        "normalization": {"output_size_px": [100, 80]},
    }


class HikGameCameraTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.rig_path = root / "hik_camera_calibration.json"
        self.minimap_path = root / "minimap_calibration.json"
        self.rig_path.write_text(json.dumps(rig_document()), encoding="utf-8")
        self.minimap_path.write_text(
            json.dumps({"canonical_phone_crop_xywh": [10, 20, 30, 20]}),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def camera(self, mode, rectify, adapter):
        return ProfiledHikGameCamera(
            self.rig_path,
            self.minimap_path,
            mode=mode,
            rectify_minimap=rectify,
            minimap_margin_px=0,
            adapter=adapter,
        )

    def test_minimap_mode_uses_hardware_roi_without_rectification(self):
        adapter = FakeAdapter()
        camera = self.camera("minimap", False, adapter).open()
        frame_set = camera.read_streams()
        self.assertEqual([10, 20, 30, 20], adapter.roi)
        self.assertEqual((20, 30, 3), frame_set.streams["minimap"].shape)
        self.assertEqual(1, adapter.read_count)
        self.assertFalse(frame_set.metadata["rectified_minimap"])

    def test_dual_mode_derives_both_streams_from_one_acquisition(self):
        adapter = FakeAdapter()
        camera = self.camera("dual", True, adapter).open()
        frame_set = camera.read_streams()
        self.assertEqual([0, 0, 100, 80], adapter.roi)
        self.assertEqual((80, 100, 3), frame_set.streams["full"].shape)
        self.assertEqual((20, 30, 3), frame_set.streams["minimap"].shape)
        self.assertEqual(10, int(frame_set.streams["minimap"][0, 0, 0]))
        self.assertEqual(20, int(frame_set.streams["minimap"][0, 0, 1]))
        self.assertEqual(1, adapter.read_count)
        self.assertEqual(42, frame_set.frame_number)
        self.assertEqual(1234, frame_set.time_ns)
        self.assertTrue(frame_set.metadata["one_acquisition_for_all_streams"])

    def test_full_mode_returns_only_full_sensor_stream(self):
        adapter = FakeAdapter()
        camera = self.camera("full", False, adapter).open()
        frame_set = camera.read_streams()
        self.assertEqual(["full"], list(frame_set.streams))
        self.assertEqual([0, 0, 100, 80], adapter.roi)

    def test_android_logical_crop_can_be_resolved_without_image_matching(self):
        crop = _source_crop_to_canonical_phone(
            {
                "image_source": "android_scrcpy",
                "crop_xywh": [5, 10, 20, 15],
                "source_space": {"origin_in_canonical_phone_xy": [0, 0]},
                "phone_surface_orientation": {
                    "quarter_turns_clockwise_from_natural": 1,
                    "natural_size_px": [80, 100],
                },
            }
        )
        self.assertEqual([10, 75, 15, 20], crop)

    def test_ambiguous_legacy_crop_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Ambiguous mini-map crop"):
            _source_crop_to_canonical_phone({"crop_xywh": [0, 0, 20, 20]})


if __name__ == "__main__":
    unittest.main()
