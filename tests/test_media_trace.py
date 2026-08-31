import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from acquisition.dual_source_spaces import build_dual_source_media_registry
from acquisition.media_trace import raster_record, validate_media_registry
from acquisition.rig_calibration.hik.media_trace import (
    build_hik_calibration_media_registry,
)
from acquisition.rig_calibration.contracts import FrameSample
from aria_trace.evidence.rig_spatial import (
    SYNTHETIC_BACKGROUND_BGR,
    expanded_rig_camera_review,
    validate_rig_image_space,
)


class MediaTraceTests(unittest.TestCase):
    @staticmethod
    def _roi_sample(image, roi, parent_size):
        return FrameSample(
            image,
            1,
            metadata={
                "image_space": {
                    "space_id": "hik_camera_acquisition_pixels",
                    "stored_size_px": [image.shape[1], image.shape[0]],
                    "parent_space_id": "hik_full_sensor_camera_pixels",
                    "parent_size_px": list(parent_size),
                    "roi_in_parent_xywh": list(roi),
                    "local_to_parent_3x3": [
                        [1.0, 0.0, float(roi[0])],
                        [0.0, 1.0, float(roi[1])],
                        [0.0, 0.0, 1.0],
                    ],
                    "orientation": "hik_camera_native",
                    "color_order": "BGR",
                }
            },
        )

    def test_rig_space_rejects_missing_or_inconsistent_producer_metadata(self):
        image = np.zeros((20, 30, 3), np.uint8)
        with self.assertRaisesRegex(ValueError, "did not supply image_space"):
            validate_rig_image_space(image, None)
        sample = self._roi_sample(image, [7, 9, 30, 20], [80, 60])
        bad = dict(sample.metadata["image_space"])
        bad["roi_in_parent_xywh"] = [7, 9, 29, 20]
        with self.assertRaisesRegex(ValueError, "inconsistent ROI"):
            validate_rig_image_space(image, bad)

    def test_expanded_review_keeps_full_sensor_and_outlying_phone_projection(self):
        image = np.full((20, 30, 3), 40, np.uint8)
        sample = self._roi_sample(image, [10, 8, 30, 20], [80, 60])
        phone_quad = [[-12, -8], [86, -5], [90, 67], [-9, 70]]
        review = expanded_rig_camera_review(
            sample,
            full_sensor_size_px=[80, 60],
            phone_display_size_px=[100, 200],
            phone_display_to_full_sensor_3x3=None,
            phone_display_quadrilateral_full_sensor_xy=phone_quad,
            title="test",
        )
        self.assertEqual(
            phone_quad,
            review.geometry["phone_display_quadrilateral_full_sensor_xy"],
        )
        self.assertEqual(
            "explicit_raw_sensor_quadrilateral",
            review.geometry["phone_projection_model"],
        )
        synthetic = np.all(
            review.image == np.asarray(SYNTHETIC_BACKGROUND_BGR, np.uint8), axis=2
        )
        self.assertGreater(int(np.count_nonzero(synthetic)), 0)
        self.assertGreater(review.image.shape[1], 80)

    def test_registry_requires_every_media_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imwrite(str(root / "frame.png"), np.zeros((4, 6, 3), np.uint8))
            record = raster_record(
                "frame.png",
                media_type="image",
                stored_size_px=[6, 4],
                space_id="test_output_pixels",
                operation="crop_without_resampling",
                source_space_id="test_source_pixels",
                source_region={"kind": "crop", "xywh": [2, 3, 6, 4]},
            )
            validate_media_registry(root, [record])
            cv2.imwrite(str(root / "unregistered.png"), np.zeros((2, 2), np.uint8))
            with self.assertRaisesRegex(RuntimeError, "missing=.*unregistered.png"):
                validate_media_registry(root, [record])

    def test_hik_bundle_distinguishes_hardware_roi_from_rectified_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv2.imwrite(
                str(root / "last_camera_frame.png"),
                np.zeros((80, 90, 3), np.uint8),
            )
            cv2.imwrite(
                str(root / "valid_screen_mask.png"),
                np.full((40, 30), 255, np.uint8),
            )
            config = {
                "camera": {
                    "full_sensor_mode": {"width_px": 100, "height_px": 90},
                    "hardware_roi_xywh": [5, 6, 90, 80],
                },
                "normalization": {"output_size_px": [30, 40]},
                "results": {"cross_source_check": {}},
            }
            sample = FrameSample(
                np.zeros((80, 90, 3), np.uint8),
                1,
                metadata={
                    "image_space": {
                        "space_id": "hik_camera_adapter_roi_image_pixels",
                        "stored_size_px": [90, 80],
                        "parent_space_id": "hik_full_sensor_camera_pixels",
                        "parent_size_px": [100, 90],
                        "roi_in_parent_xywh": [5, 6, 90, 80],
                        "local_to_parent_3x3": [
                            [1.0, 0.0, 5.0],
                            [0.0, 1.0, 6.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "orientation": "hik_camera_native",
                        "color_order": "BGR",
                    }
                },
            )
            records = build_hik_calibration_media_registry(
                root, config, last_camera_sample=sample
            )
            by_file = {row["file"]: row for row in records}
            camera = by_file["last_camera_frame.png"]
            self.assertEqual(
                "hik_camera_adapter_roi_image_pixels", camera["space"]["id"]
            )
            self.assertEqual(
                [5, 6, 90, 80],
                camera["provenance"]["source_region"]["xywh"],
            )
            self.assertEqual(
                "hik_rig_rectified_visible_phone_pixels",
                by_file["valid_screen_mask.png"]["space"]["id"],
            )

    def test_dual_source_registry_covers_videos_and_evidence_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "video_android_phone.mkv").write_bytes(b"test")
            (root / "video_hik_phone.mkv").write_bytes(b"test")
            evidence = root / "cross_source_check"
            evidence.mkdir()
            cv2.imwrite(
                str(evidence / "adb_visible_crop.png"),
                np.zeros((30, 40, 3), np.uint8),
            )
            (evidence / "summary.json").write_text(
                '{"logical_adb_crop_xywh":[10,20,40,30]}', encoding="utf-8"
            )
            rig = root / "rig.json"
            rig.write_text(
                '{"camera":{"hardware_roi_xywh":[4,5,90,80]}}',
                encoding="utf-8",
            )
            document = {
                "rig": {"calibration": str(rig)},
                "spaces": {},
                "orientation_selection": {
                    "effective_quarter_turns_clockwise_from_phone_natural": 1,
                    "source": "first_game_adb_and_hik_image_evidence",
                },
                "conversions": {
                    "camera_adapter_image_quarter_turns_clockwise_to_hik_phone_video": 1,
                    "hik_phone_video_bounds_in_adb_xywh": [10, 20, 40, 30],
                },
                "streams": {
                    "android_phone": {
                        "video": "video_android_phone.mkv",
                        "stored_size_px": [100, 60],
                        "timestamp_authority": "adb clock",
                    },
                    "hik_phone": {
                        "video": "video_hik_phone.mkv",
                        "stored_size_px": [40, 30],
                        "content_size_px": [40, 30],
                        "timestamp_authority": "hik clock",
                    },
                },
            }
            records = build_dual_source_media_registry(root, document)
            by_file = {row["file"]: row for row in records}
            self.assertEqual(3, len(records))
            self.assertEqual(
                [10, 20, 40, 30],
                by_file["cross_source_check/adb_visible_crop.png"]
                ["provenance"]["source_region"]["xywh"],
            )


if __name__ == "__main__":
    unittest.main()
