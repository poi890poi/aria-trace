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
        self.bayer_conversion_calls = []

    def open(self, config):
        self.open_config = config
        return self

    def set_black_level(self, value):
        self.black_level = value

    def set_manual_imaging(self, exposure, gain):
        self.imaging = (exposure, gain)

    def set_white_balance(self, red, green, blue):
        self.white_balance = (red, green, blue)

    def set_bayer_conversion(self, gamma, ccm):
        self.bayer_conversion_calls.append((gamma, ccm))

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
            "hardware_roi_xywh": [0, 0, 100, 80],
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
        "normalization": {
            "output_size_px": [100, 80],
            "origin_screen_xy": [0, 0],
            "screen_units_per_output_pixel_xy": [1, 1],
            "full_sensor_camera_to_output_3x3": np.eye(3).tolist(),
        },
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
        self.assertTrue(frame_set.metadata["full_output_normalized_by_base_rig"])

    def test_full_mode_returns_only_normalized_visible_phone_stream(self):
        adapter = FakeAdapter()
        camera = self.camera("full", False, adapter).open()
        frame_set = camera.read_streams()
        self.assertEqual(["full"], list(frame_set.streams))
        self.assertEqual([0, 0, 100, 80], adapter.roi)

    def test_bayer_color_match_is_set_once_at_open_not_per_frame(self):
        document = {
            "canonical_phone_crop_xywh": [10, 20, 30, 20],
            "hik_bayer_conversion": {
                "status": "selected",
                "gamma": 0.75,
                "ccm_rgb_3x3": [[1.1, 0, 0], [0, 1, 0], [0, 0, 0.9]],
            },
        }
        self.minimap_path.write_text(json.dumps(document), encoding="utf-8")
        adapter = FakeAdapter()
        camera = self.camera("minimap", False, adapter).open()
        camera.read_streams()
        camera.read_streams()
        self.assertEqual(1, len(adapter.bayer_conversion_calls))
        self.assertEqual(0.75, adapter.bayer_conversion_calls[0][0])

    def test_full_mode_uses_base_rig_hardware_roi_and_output_mapping(self):
        document = rig_document()
        document["camera"]["hardware_roi_xywh"] = [10, 5, 80, 60]
        document["normalization"].update(
            {
                "output_size_px": [80, 60],
                "origin_screen_xy": [10, 5],
                "full_sensor_camera_to_output_3x3": [
                    [1, 0, -10], [0, 1, -5], [0, 0, 1]
                ],
            }
        )
        self.rig_path.write_text(json.dumps(document), encoding="utf-8")
        adapter = FakeAdapter()
        frame_set = self.camera("full", False, adapter).open().read_streams()
        self.assertEqual([10, 5, 80, 60], adapter.roi)
        self.assertEqual((60, 80, 3), frame_set.streams["full"].shape)
        self.assertEqual(10, int(frame_set.streams["full"][0, 0, 0]))
        self.assertEqual(5, int(frame_set.streams["full"][0, 0, 1]))

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

    def test_current_profile_pointer_is_rejected(self):
        revision = self.minimap_path.parent / "revisions" / "r1"
        revision.mkdir(parents=True)
        calibration = revision / "minimap_calibration.json"
        calibration.write_text(
            json.dumps({"canonical_phone_crop_xywh": [10, 20, 30, 20]}),
            encoding="utf-8",
        )
        pointer = self.minimap_path.parent / "current.json"
        pointer.write_text(
            json.dumps({"current_revision": str(revision)}), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "obsolete"):
            ProfiledHikGameCamera(
                self.rig_path,
                pointer,
                mode="minimap",
                rectify_minimap=False,
                minimap_margin_px=0,
                adapter=FakeAdapter(),
            )


if __name__ == "__main__":
    unittest.main()
