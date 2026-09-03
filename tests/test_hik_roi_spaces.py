import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from acquisition.commented_yaml import (
    HIK_CONFIG_COMMENTS,
    HIK_CONFIG_HEADER,
    write_commented_yaml,
)
from acquisition.rig_calibration.hik.algorithms import (
    camera_adapter_roi_to_output_homography,
    hik_image_space_conversions,
    validate_hik_coordinate_contract,
)
from acquisition.rig_calibration.hik.driver import HikMvsCameraAdapter
from acquisition.rig_calibration.hik.driver import RectifiedHikCamera
from acquisition.rig_calibration.contracts import FrameSample


class PersistentRoiBackend:
    def __init__(self):
        self.values = {
            "OffsetX": 4,
            "OffsetY": 4,
            "Width": 1300,
            "Height": 1040,
            "SensorWidth": 1440,
            "SensorHeight": 1080,
        }
        self.stops = 0
        self.starts = 0

    def stop(self):
        self.stops += 1

    def start(self):
        self.starts += 1

    def set_int(self, node, value):
        self.values[node] = int(value)

    def get_int(self, node):
        return self.values[node]

    def int_range(self, node):
        if node == "Width":
            return {
                "minimum": 32,
                "maximum": 1440 - self.values["OffsetX"],
                "increment": 4,
            }
        if node == "Height":
            return {
                "minimum": 8,
                "maximum": 1080 - self.values["OffsetY"],
                "increment": 4,
            }
        if node == "OffsetX":
            return {"minimum": 0, "maximum": 1408, "increment": 4}
        if node == "OffsetY":
            return {"minimum": 0, "maximum": 1072, "increment": 4}
        raise AssertionError(node)


class StrictPersistentRoiBackend(PersistentRoiBackend):
    """Model GenICam nodes whose legal dimension depends on current offset."""

    def set_int(self, node, value):
        value = int(value)
        if node == "Width" and value + self.values["OffsetX"] > 1440:
            raise RuntimeError("Width exceeds residual sensor extent")
        if node == "Height" and value + self.values["OffsetY"] > 1080:
            raise RuntimeError("Height exceeds residual sensor extent")
        super().set_int(node, value)


class ResidualRangeOnlyBackend(StrictPersistentRoiBackend):
    """Older plugins may expose residual ranges but no SensorWidth nodes."""

    def get_int(self, node):
        if node in ("SensorWidth", "SensorHeight"):
            raise RuntimeError("node unavailable")
        return super().get_int(node)


class RuntimeAdapter:
    def __init__(self, image):
        self.image = image
        self.roi = None

    def open(self, _configuration):
        return {}

    def set_black_level(self, _value):
        pass

    def set_manual_imaging(self, _exposure, _gain):
        pass

    def set_white_balance(self, _red, _green, _blue):
        pass

    def set_roi(self, roi):
        self.roi = list(roi)
        return list(roi)

    def read(self):
        return FrameSample(self.image.copy(), 1, receive_time_ns=1)

    def close(self):
        pass


class HikRoiSpaceTests(unittest.TestCase):
    def test_roi_alignment_uses_absolute_sensor_extent_after_persistent_crop(self):
        backend = PersistentRoiBackend()
        backend.values.update(
            OffsetX=136, OffsetY=132, Width=1304, Height=948
        )
        adapter = HikMvsCameraAdapter(backend=backend)
        adapter._opened = True

        self.assertEqual(
            [136, 132, 1304, 948],
            adapter.align_roi([136, 132, 1304, 948]),
        )
        self.assertEqual(
            [136, 132, 1304, 948],
            adapter.set_roi([136, 132, 1304, 948]),
        )

    def test_roi_can_move_between_offsets_without_offset_dependent_shrinkage(self):
        backend = StrictPersistentRoiBackend()
        backend.values.update(
            OffsetX=136, OffsetY=132, Width=1304, Height=948
        )
        adapter = HikMvsCameraAdapter(backend=backend)
        adapter._opened = True

        requested = (
            [40, 20, 1360, 1040],
            [200, 160, 1200, 880],
            [0, 0, 1440, 1080],
            [136, 132, 1304, 948],
        )
        for roi in requested:
            with self.subTest(roi=roi):
                self.assertEqual(roi, adapter.set_roi(roi))
                self.assertEqual(roi, [
                    backend.values["OffsetX"],
                    backend.values["OffsetY"],
                    backend.values["Width"],
                    backend.values["Height"],
                ])
        self.assertEqual(len(requested), backend.stops)
        self.assertEqual(len(requested), backend.starts)

    def test_roi_alignment_recovers_absolute_extent_from_residual_ranges(self):
        backend = ResidualRangeOnlyBackend()
        backend.values.update(
            OffsetX=136, OffsetY=132, Width=1304, Height=948
        )
        adapter = HikMvsCameraAdapter(backend=backend)
        adapter._opened = True

        self.assertEqual(
            [136, 132, 1304, 948],
            adapter.set_roi([136, 132, 1304, 948]),
        )

    def test_calibration_reset_clears_persistent_roi_to_full_sensor(self):
        backend = PersistentRoiBackend()
        adapter = HikMvsCameraAdapter(backend=backend)
        adapter._opened = True

        effective = adapter.reset_full_sensor_roi()

        self.assertEqual([0, 0, 1440, 1080], effective)
        self.assertEqual(1, backend.stops)
        self.assertEqual(1, backend.starts)

    def test_both_image_spaces_include_roi_origin_in_their_matrices(self):
        full_to_screen = np.asarray(
            [[2.0, 0.0, 5.0], [0.0, 3.0, 7.0], [0.0, 0.0, 1.0]]
        )
        spaces = hik_image_space_conversions(
            full_to_screen,
            np.linalg.inv(full_to_screen),
            [[1, 0, -10], [0, 1, -20], [0, 0, 1]],
            [1440, 1080],
            [10, 20, 100, 80],
            [100, 80],
        )

        roi_to_screen = np.asarray(
            spaces["conversions"][
                "camera_adapter_roi_image_to_phone_display_3x3"
            ]
        )
        roi_to_output = np.asarray(
            spaces["conversions"][
                "camera_adapter_roi_image_to_calibrated_output_3x3"
            ]
        )
        np.testing.assert_allclose(roi_to_screen.dot([0, 0, 1])[:2], [25, 67])
        np.testing.assert_allclose(roi_to_output.dot([0, 0, 1])[:2], [0, 0])
        self.assertEqual(
            "rig_calibration_only",
            spaces["spaces"]["full_sensor_image"]["owner"],
        )
        self.assertEqual(
            "production_camera_adapter_only",
            spaces["spaces"]["camera_adapter_roi_image"]["owner"],
        )

    def test_version_two_contract_names_and_validates_every_runtime_space(self):
        spaces = hik_image_space_conversions(
            np.eye(3),
            np.eye(3),
            np.eye(3),
            [100, 80],
            [10, 20, 40, 30],
            [100, 80],
            calibration_display_size_px=[80, 100],
            phone_natural_size_px=[100, 80],
            calibration_display_quarter_turns=1,
        )
        calibration = {
            "camera": {
                "full_sensor_mode": {"width_px": 100, "height_px": 80},
                "hardware_roi_xywh": [10, 20, 40, 30],
            },
            "phone": {
                "screen_size_px": [80, 100],
                "natural_screen_size_px": [100, 80],
            },
            "normalization": {"output_size_px": [100, 80]},
            "coordinate_spaces": spaces,
        }
        result = validate_hik_coordinate_contract(calibration, [10, 20, 40, 30])
        self.assertEqual(result["status"], "validated")
        self.assertEqual(
            spaces["runtime_chain"],
            [
                "hik_full_sensor_bgr_pixels",
                "hik_camera_adapter_hardware_roi_bgr_pixels",
                "hik_rig_rectified_visible_phone_pixels",
                "android_calibration_logical_display_pixels",
                "android_phone_natural_display_pixels",
            ],
        )

    def test_version_two_contract_rejects_effective_roi_drift(self):
        spaces = hik_image_space_conversions(
            np.eye(3), np.eye(3), np.eye(3),
            [100, 80], [10, 20, 40, 30], [100, 80],
            calibration_display_size_px=[100, 80],
            phone_natural_size_px=[100, 80],
        )
        calibration = {
            "camera": {
                "full_sensor_mode": {"width_px": 100, "height_px": 80},
                "hardware_roi_xywh": [10, 20, 40, 30],
            },
            "phone": {
                "screen_size_px": [100, 80],
                "natural_screen_size_px": [100, 80],
            },
            "normalization": {"output_size_px": [100, 80]},
            "coordinate_spaces": spaces,
        }
        with self.assertRaisesRegex(ValueError, "Effective HIK ROI"):
            validate_hik_coordinate_contract(calibration, [12, 20, 40, 30])

    def test_projective_roundtrip_is_compared_up_to_homogeneous_scale(self):
        full_to_screen = np.asarray(
            [
                [1.8, 0.07, 13.0],
                [-0.04, 2.1, 9.0],
                [0.0008, -0.0003, 1.0],
            ],
            dtype=np.float64,
        )
        spaces = hik_image_space_conversions(
            full_to_screen,
            np.linalg.inv(full_to_screen),
            np.eye(3),
            [100, 80],
            [10, 20, 40, 30],
            [100, 80],
            calibration_display_size_px=[100, 80],
            phone_natural_size_px=[100, 80],
        )
        calibration = {
            "camera": {
                "full_sensor_mode": {"width_px": 100, "height_px": 80},
                "hardware_roi_xywh": [10, 20, 40, 30],
            },
            "phone": {
                "screen_size_px": [100, 80],
                "natural_screen_size_px": [100, 80],
            },
            "normalization": {"output_size_px": [100, 80]},
            "coordinate_spaces": spaces,
        }
        result = validate_hik_coordinate_contract(
            calibration, [10, 20, 40, 30]
        )
        self.assertLess(result["maximum_matrix_roundtrip_error"], 1.0e-10)

    def test_distortion_contract_uses_one_precomputed_runtime_remap(self):
        lens = {
            "source": "measured",
            "model": "opencv_radtan",
            "camera_matrix_3x3": [[100, 0, 4], [0, 100, 4], [0, 0, 1]],
            "distortion_coefficients": [-0.1, 0.01, 0, 0, 0],
        }
        spaces = hik_image_space_conversions(
            np.eye(3), np.eye(3), np.eye(3),
            [8, 8], [2, 3, 4, 4], [4, 4],
            calibration_display_size_px=[8, 8],
            phone_natural_size_px=[8, 8],
            lens_model=lens,
        )
        config = {
            "camera": {
                "device_id": "fake",
                "full_sensor_mode": {"width_px": 8, "height_px": 8, "fps": 30},
                "hardware_roi_xywh": [2, 3, 4, 4],
            },
            "phone": {
                "screen_size_px": [8, 8],
                "natural_screen_size_px": [8, 8],
            },
            "imaging": {
                "black_level": 0,
                "exposure_us": 1000,
                "gain": 0,
                "white_balance": {
                    "ratio_red": 1000,
                    "ratio_green": 1000,
                    "ratio_blue": 1000,
                },
            },
            "normalization": {
                "full_sensor_camera_to_output_3x3": np.eye(3).tolist(),
                "output_size_px": [4, 4],
                "dense_map_file": "rectification_maps.npz",
                "lens_correction_in_dense_map": True,
            },
            "coordinate_spaces": spaces,
            "optics": {"lens_model": lens},
        }
        self.assertEqual(
            validate_hik_coordinate_contract(config)["schema_version"], 3
        )
        image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape((4, 4, 3))
        adapter = RuntimeAdapter(image)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "hik_camera_calibration.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            xx, yy = np.meshgrid(np.arange(2, 6), np.arange(3, 7))
            np.savez_compressed(
                str(root / "rectification_maps.npz"),
                map_x=xx.astype(np.float32),
                map_y=yy.astype(np.float32),
            )
            real_remap = __import__("cv2").remap
            with mock.patch(
                "rig_runtime.adapters.hik.driver.cv2.remap", wraps=real_remap
            ) as remap:
                camera = RectifiedHikCamera(path, adapter=adapter, rectify=True).open()
                sample = camera.read_sample()
                camera.release()
        self.assertEqual(remap.call_count, 1)
        np.testing.assert_array_equal(sample.image, image)
        self.assertEqual(sample.metadata["image_space"]["runtime_resampling_passes"], 1)
        self.assertTrue(sample.metadata["image_space"]["lens_distortion_corrected"])

    def test_runtime_uses_saved_roi_matrix_and_legacy_mismatch_fallback(self):
        calibration = {
            "normalization": {
                "full_sensor_camera_to_output_3x3": np.eye(3).tolist()
            },
            "coordinate_spaces": {
                "spaces": {
                    "camera_adapter_roi_image": {
                        "roi_in_full_sensor_xywh": [10, 20, 100, 80]
                    }
                },
                "conversions": {
                    "camera_adapter_roi_image_to_calibrated_output_3x3": [
                        [1, 0, 111],
                        [0, 1, 222],
                        [0, 0, 1],
                    ]
                },
            },
        }
        np.testing.assert_allclose(
            camera_adapter_roi_to_output_homography(
                calibration, [10, 20, 100, 80]
            ),
            [[1, 0, 111], [0, 1, 222], [0, 0, 1]],
        )
        np.testing.assert_allclose(
            camera_adapter_roi_to_output_homography(
                calibration, [12, 24, 100, 80]
            ),
            [[1, 0, 12], [0, 1, 24], [0, 0, 1]],
        )

    def test_legacy_origin_and_screen_matrix_fallback_remains_supported(self):
        calibration = {
            "normalization": {"origin_screen_xy": [10, 20]},
            "geometry": {
                "full_sensor_camera_to_screen_3x3": np.eye(3).tolist()
            },
        }
        np.testing.assert_allclose(
            camera_adapter_roi_to_output_homography(
                calibration, [10, 20, 100, 80]
            ),
            np.eye(3),
        )

    def test_rectified_camera_consumes_saved_roi_local_conversion(self):
        image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape((4, 4, 3))
        adapter = RuntimeAdapter(image)
        config = {
            "camera": {
                "device_id": "fake",
                "full_sensor_mode": {"width_px": 8, "height_px": 8, "fps": 30},
                "hardware_roi_xywh": [2, 3, 4, 4],
            },
            "imaging": {
                "black_level": 0,
                "exposure_us": 1000,
                "gain": 0,
                "white_balance": {
                    "ratio_red": 1000,
                    "ratio_green": 1000,
                    "ratio_blue": 1000,
                },
            },
            "normalization": {
                # Deliberately incompatible fallback: the saved ROI-local
                # conversion below must win for the exact saved ROI.
                "full_sensor_camera_to_output_3x3": np.eye(3).tolist(),
                "output_size_px": [4, 4],
            },
            "coordinate_spaces": {
                "spaces": {
                    "camera_adapter_roi_image": {
                        "roi_in_full_sensor_xywh": [2, 3, 4, 4]
                    }
                },
                "conversions": {
                    "camera_adapter_roi_image_to_calibrated_output_3x3": (
                        np.eye(3).tolist()
                    )
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hik_camera_calibration.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            camera = RectifiedHikCamera(path, adapter=adapter).open()
            ok, output = camera.read()
            camera.release()
        self.assertTrue(ok)
        self.assertEqual([2, 3, 4, 4], adapter.roi)
        np.testing.assert_array_equal(image, output)

    def test_yaml_comments_explain_full_sensor_and_adapter_roi_ownership(self):
        value = {
            "schema_version": 1,
            "camera": {},
            "coordinate_spaces": {"spaces": {}, "conversions": {}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hik_camera_calibration.yaml"
            write_commented_yaml(
                path,
                value,
                header=HIK_CONFIG_HEADER,
                section_comments=HIK_CONFIG_COMMENTS,
            )
            text = path.read_text(encoding="utf-8")
        self.assertIn("# Two HIK acquisition spaces are intentional", text)
        self.assertIn("# camera_adapter_roi_image is the reduced production", text)


if __name__ == "__main__":
    unittest.main()
