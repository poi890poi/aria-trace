import ctypes
import types
import unittest

import cv2
import numpy as np

from acquisition.hik_bayer_color_match import (
    apply_mvs_bayer_model,
    optimize_mvs_bayer_conversion,
    synchronized_frame_pairs,
)
from acquisition.rig_calibration.hik.driver import MvsPythonBackend


class HikBayerColorMatchTests(unittest.TestCase):
    def test_backend_sets_vendor_gamma_and_ccm_once(self):
        class Ccm(ctypes.Structure):
            _fields_ = [
                ("bCCMEnable", ctypes.c_bool),
                ("nCCMat", ctypes.c_int * 9),
            ]

        class Camera:
            def __init__(self):
                self.gamma = None
                self.ccm = None

            def MV_CC_SetGammaValue(self, pixel_type, gamma):
                self.gamma = (pixel_type, gamma)
                return 0

            def MV_CC_SetBayerCCMParam(self, value):
                self.ccm = (bool(value.bCCMEnable), list(value.nCCMat))
                return 0

        backend = object.__new__(MvsPythonBackend)
        backend.sdk = types.SimpleNamespace(MV_CC_CCM_PARAM=Ccm)
        backend.camera = Camera()
        backend.get_enum = lambda _node: 0x01080009
        result = backend.set_bayer_conversion(
            0.75, [[1.1, 0, 0], [0, 1, 0], [0, 0, 0.9]]
        )
        self.assertEqual(0x01080009, backend.camera.gamma[0])
        self.assertEqual(1126, backend.camera.ccm[1][0])
        self.assertEqual(922, backend.camera.ccm[1][8])
        self.assertEqual(0, result["additional_frame_passes"])

    def test_mvs_model_applies_ccm_before_gamma(self):
        image = np.asarray([[[29, 15, 21]]], np.uint8)
        adjusted = apply_mvs_bayer_model(
            image,
            0.5,
            [[2.0, 0, 0], [0, 1.0, 0], [0, 0, 0.5]],
        )
        np.testing.assert_allclose(adjusted[0, 0], [61, 62, 103], atol=1)

    def test_synchronized_pairs_are_unique(self):
        pairs = synchronized_frame_pairs(
            np.asarray([0, 10, 20, 30], np.int64),
            np.asarray([1, 2, 11, 21, 31], np.int64),
            maximum_pairs=4,
        )
        self.assertEqual(4, len(pairs))
        self.assertEqual(4, len({item[0] for item in pairs}))

    def test_optimizer_reduces_validation_error(self):
        height, width = 96, 128
        frames = []
        targets = []
        known_gamma = 0.65
        known_ccm = [[1.15, -0.05, 0], [-0.03, 1.08, 0], [0, -0.04, 0.92]]
        for frame_index in range(8):
            image = np.zeros((height, width, 3), np.uint8)
            for y in range(0, height, 16):
                for x in range(0, width, 16):
                    color = (
                        20 + (x * 3 + frame_index * 7) % 180,
                        20 + (y * 2 + frame_index * 11) % 180,
                        20 + (x + y + frame_index * 13) % 180,
                    )
                    image[y : y + 16, x : x + 16] = color
            frames.append(image)
            targets.append(apply_mvs_bayer_model(image, known_gamma, known_ccm))
        frames = np.stack(frames)
        targets = np.stack(targets)
        mask = np.zeros((height, width), np.uint8)
        cv2.circle(mask, (width // 2, height // 2), 42, 255, -1)
        times = np.arange(8, dtype=np.int64) * 10_000_000
        result, evidence = optimize_mvs_bayer_conversion(
            targets,
            times,
            frames,
            times,
            np.eye(3),
            mask,
            maximum_pairs=8,
            maximum_pixels_per_pair=1200,
        )
        self.assertEqual("selected", result["status"])
        self.assertGreater(result["fit"]["relative_rgb_mae_improvement"], 0.5)
        self.assertLess(
            result["fit"]["selected_validation"]["rgb_mae_dn"],
            result["fit"]["baseline_validation"]["rgb_mae_dn"],
        )
        self.assertIn("bayer_color_match_review.png", evidence)


if __name__ == "__main__":
    unittest.main()
