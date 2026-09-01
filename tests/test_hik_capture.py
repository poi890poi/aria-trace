import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from acquisition.hik_capture import CalibratedHikFrameSource, NativeHikFrameSource
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


class FakeRectifiedReader:
    def __init__(self):
        self.opened = False
        self.released = False

    def open(self):
        self.opened = True

    def read_sample(self):
        image = np.zeros((3, 5, 3), np.uint8)
        image[0, 0] = [1, 2, 3]
        return FrameSample(
            image,
            100,
            receive_time_ns=110,
            metadata={"device_timestamp": 55},
        )

    def release(self):
        self.released = True

    def rectify_for_evidence(self, image):
        return cv2.resize(image, (5, 3), interpolation=cv2.INTER_NEAREST)


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


class CalibratedHikFrameSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.calibration = Path(self.temporary.name) / "rig.json"
        self.calibration.write_text(
            json.dumps(
                {
                    "phone": {
                        "natural_screen_size_px": [10, 20],
                        "screen_size_px": [10, 20],
                        "orientation_quarter_turns": 0,
                    },
                    "normalization": {
                        "output_size_px": [5, 3],
                        "origin_screen_xy": [2, 4],
                        "screen_units_per_output_pixel_xy": [1, 1],
                    },
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_rotates_rectified_phone_view_to_android_logical_orientation(self):
        reader = FakeRectifiedReader()
        source = CalibratedHikFrameSource(
            self.calibration,
            reader=reader,
            output_quarter_turns_clockwise=1,
        )
        source.start()
        packet = source.read()
        self.assertTrue(reader.opened)
        self.assertEqual("hik_phone", packet.stream_id)
        self.assertEqual(
            "hik_session_aligned_visible_phone_pixels",
            packet.metadata["coordinate_space"],
        )
        self.assertEqual(
            "hik_rig_rectified_visible_phone_pixels",
            packet.metadata["source_coordinate_space"],
        )
        self.assertEqual((6, 4, 3), packet.image.shape)
        self.assertEqual([3, 5], packet.metadata["calibrated_source_size_px"])
        self.assertEqual(
            1,
            packet.metadata[
                "output_quarter_turns_clockwise_from_calibration_display"
            ],
        )
        self.assertEqual([1, 1], packet.metadata["video_encoding_padding_right_bottom_px"])
        np.testing.assert_array_equal(packet.image[0, 2], [1, 2, 3])
        self.assertEqual(
            "android_phone_natural_display_pixels",
            packet.metadata["image_space"]["canonical_space_id"],
        )
        self.assertEqual(
            [10, 20], packet.metadata["image_space"]["canonical_size_px"]
        )
        np.testing.assert_allclose(
            packet.metadata["image_space"]["local_to_canonical_3x3"],
            [[0, 1, 2], [-1, 0, 6], [0, 0, 1]],
        )
        source.stop()
        self.assertTrue(reader.released)

    def test_orientation_can_be_updated_from_evidence_before_recording(self):
        reader = FakeRectifiedReader()
        source = CalibratedHikFrameSource(
            self.calibration,
            reader=reader,
            output_quarter_turns_clockwise=0,
        )
        source.start()
        source.set_output_orientation(
            2,
            {
                "selection_basis": "first_game_adb_and_hik_image_evidence_only",
                "selected_confidence": 0.91,
            },
        )
        packet = source.read()
        np.testing.assert_array_equal(packet.image[2, 4], [1, 2, 3])
        self.assertEqual(
            2,
            packet.metadata[
                "output_quarter_turns_clockwise_from_calibration_display"
            ],
        )
        self.assertEqual(
            "first_game_adb_and_hik_image_evidence_only",
            packet.metadata["output_orientation_evidence"]["selection_basis"],
        )

    def test_unrectified_stream_rectifies_only_explicit_evidence_image(self):
        reader = FakeRectifiedReader()
        source = CalibratedHikFrameSource(
            self.calibration,
            reader=reader,
            rectify=False,
            output_quarter_turns_clockwise=1,
        )
        source.start()
        packet = source.read()
        evidence = source.alignment_evidence_image(packet)
        self.assertEqual((3, 5, 3), evidence.shape)
        np.testing.assert_array_equal(evidence[0, 0], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
