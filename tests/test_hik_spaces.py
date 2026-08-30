import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from acquisition.dual_source_spaces import write_dual_source_space_yaml
from acquisition.rig_calibration.hik.camera import HikCamera
from acquisition.rig_calibration.hik.driver import RectifiedHikCamera
from acquisition.rig_calibration.hik.spaces import RigCalibratedSpaceConverter


def calibration_document():
    return {
        "camera": {
            "device_id": "camera-1",
            "full_sensor_mode": {"width_px": 100, "height_px": 80, "fps": 30},
            "hardware_roi_xywh": [0, 0, 100, 80],
        },
        "phone": {
            "serial": "phone-1",
            "natural_screen_size_px": [100, 200],
            "screen_size_px": [200, 100],
            "orientation_quarter_turns": 1,
        },
        "imaging": {
            "exposure_us": 8000,
            "gain": 1,
            "black_level": 0,
            "white_balance": {
                "ratio_red": 1000,
                "ratio_green": 1000,
                "ratio_blue": 1000,
            },
        },
        "normalization": {
            "output_size_px": [30, 40],
            "origin_screen_xy": [10, 20],
            "full_sensor_camera_to_output_3x3": np.eye(3).tolist(),
        },
    }


class HikSpaceConversionTests(unittest.TestCase):
    def test_forward_inverse_and_bounds_match_rotated_adb_pixels(self):
        converter = RigCalibratedSpaceConverter(calibration_document(), 1)
        mapped = converter.camera_adapter_to_adb_points([[0, 0], [29, 39]])
        np.testing.assert_allclose(mapped, [[10, 20], [39, 59]])
        np.testing.assert_allclose(
            converter.adb_to_camera_adapter_points(mapped),
            [[0, 0], [29, 39]],
            atol=1.0e-9,
        )
        self.assertEqual([10, 20, 30, 40], converter.camera_adapter_bounds_in_adb_xywh())
        self.assertEqual((200, 100), converter.adb_size_px)
        self.assertEqual(
            0,
            converter.output_image_quarter_turns_clockwise_from_calibration_display,
        )

    def test_public_camera_adapters_expose_same_conversion_without_opening(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hik_camera_calibration.json"
            path.write_text(json.dumps(calibration_document()), encoding="utf-8")
            rectified = RectifiedHikCamera(path)
            facade = HikCamera(config={"calibration": str(path), "color_order": "BGR"})
            expected = np.asarray([[10, 20], [39, 59]], dtype=np.float64)
            for camera in (rectified, facade):
                mapped = camera.camera_adapter_to_adb_points([[0, 0], [29, 39]], 1)
                np.testing.assert_allclose(mapped, expected)
                np.testing.assert_allclose(
                    camera.adb_to_camera_adapter_points(mapped, 1),
                    [[0, 0], [29, 39]],
                )

    def test_session_rotation_is_relative_to_calibration_display_space(self):
        converter = RigCalibratedSpaceConverter(calibration_document(), 3)
        self.assertEqual(
            2,
            converter.output_image_quarter_turns_clockwise_from_calibration_display,
        )
        self.assertEqual((30, 40), converter.output_image_size_px)
        self.assertEqual(
            [160, 40, 30, 40],
            converter.camera_adapter_bounds_in_adb_xywh(),
        )
        mapped = converter.camera_adapter_to_adb_points([[0, 0], [29, 39]])
        np.testing.assert_allclose(mapped, [[189, 79], [160, 40]])

    def test_session_yaml_is_commented_and_describes_saved_video_spaces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "hik_camera_calibration.json"
            calibration.write_text(json.dumps(calibration_document()), encoding="utf-8")
            session = root / "session"
            session.mkdir()
            frames = [
                {"stream_id": "android_phone", "width": 200, "height": 100},
                {"stream_id": "hik_phone", "width": 30, "height": 40},
            ]
            (session / "frames.jsonl").write_text(
                "".join(json.dumps(frame) + "\n" for frame in frames),
                encoding="utf-8",
            )
            (session / "video_android_phone.mkv").write_bytes(b"test")
            (session / "video_hik_phone.mkv").write_bytes(b"test")
            path = write_dual_source_space_yaml(
                session,
                calibration,
                {
                    "quarter_turns_clockwise_from_natural": 1,
                    "natural_size_px": [100, 200],
                },
                {
                    "videos": {
                        "android_phone": "video_android_phone.mkv",
                        "hik_phone": "video_hik_phone.mkv",
                    }
                },
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn("# Named image rasters", text)
            result = yaml.safe_load(text)
            self.assertEqual(
                [10, 20, 30, 40],
                result["conversions"]["hik_phone_video_bounds_in_adb_xywh"],
            )
            self.assertEqual(
                [0, 0],
                result["streams"]["hik_phone"]["encoder_padding_right_bottom_px"],
            )
            self.assertEqual(
                "video_hik_phone.mkv", result["streams"]["hik_phone"]["video"]
            )
            self.assertEqual(
                {"video_android_phone.mkv", "video_hik_phone.mkv"},
                {row["file"] for row in result["media"]},
            )


if __name__ == "__main__":
    unittest.main()
