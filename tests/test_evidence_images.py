import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from aria_trace.apps.workbench.api import make_handler
from rig_runtime.evidence.images import write_evidence_image
from rig_runtime.workflows.profile_management import parser
from rig_runtime.adapters.filesystem.profile_retention import RetentionPolicy


class EvidenceImageTests(unittest.TestCase):
    def test_review_is_jpeg_and_machine_mask_remains_exact_png(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = np.zeros((128, 128, 3), np.uint8)
            cv2.circle(image, (64, 64), 15, (0, 255, 0), 1)
            cv2.putText(image, "Pivot", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            review = root / "circle.jpg"
            self.assertTrue(write_evidence_image(review, image))
            self.assertTrue(review.read_bytes().startswith(b"\xff\xd8"))
            self.assertEqual(image.shape, cv2.imread(str(review)).shape)
            mask = np.zeros((128, 128), np.uint8)
            cv2.circle(mask, (64, 64), 40, 255, -1)
            machine = root / "valid_mask.png"
            self.assertTrue(write_evidence_image(machine, mask))
            self.assertTrue(machine.read_bytes().startswith(b"\x89PNG"))
            np.testing.assert_array_equal(mask, cv2.imread(str(machine), cv2.IMREAD_UNCHANGED))

    def test_evidence_routes_serve_new_jpeg_and_legacy_png_types(self):
        for endpoint, method, identifier in (
            ("minimap-calibration", "minimap_calibration_image", "calibration_id"),
            ("scene-yaw", "scene_yaw_image", "calibration_id"),
            ("map-atlas", "map_atlas_image", "atlas_id"),
        ):
            for suffix, expected in (("jpg", "image/jpeg"), ("png", "image/png")):
                with self.subTest(endpoint=endpoint, suffix=suffix):
                    state = mock.Mock()
                    state.session_root = Path("unused-session-root")
                    getattr(state, method).return_value = b"review"
                    handler = object.__new__(make_handler(state))
                    handler.path = "/api/{}/image?game_id=g&{}=c&name=review.{}".format(endpoint, identifier, suffix)
                    handler._send = mock.Mock()
                    handler.do_GET()
                    handler._send.assert_called_once_with(200, expected, b"review")

    def test_manual_and_automatic_evidence_budgets_are_64_mb(self):
        self.assertEqual(64, parser().parse_args(["purge"]).evidence_max_mb)
        self.assertEqual(64, RetentionPolicy().evidence_max_mb)


if __name__ == "__main__":
    unittest.main()
