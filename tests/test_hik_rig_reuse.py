import json
import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from acquisition.rig_calibration.hik.driver import RectifiedHikCamera
from acquisition.rig_calibration.hik.reuse_precheck import (
    compare_calibration_snapshots,
    discover_previous_calibration,
)


def write_calibration(path: Path, camera_id="CAM-1", phone_serial="PHONE-1") -> Path:
    path.mkdir(parents=True)
    config = {
        "camera": {
            "device_id": camera_id,
            "full_sensor_mode": {"width_px": 64, "height_px": 48, "fps": 30.0},
            "hardware_roi_xywh": [0, 0, 64, 48],
        },
        "phone": {"serial": phone_serial},
        "imaging": {
            "exposure_us": 1000.0,
            "gain": 2.0,
            "white_balance": {
                "ratio_red": 1000,
                "ratio_green": 1000,
                "ratio_blue": 1000,
            },
        },
        "normalization": {
            "full_sensor_camera_to_output_3x3": np.eye(3).tolist(),
            "output_size_px": [64, 48],
        },
    }
    config_path = path / "hik_camera_calibration.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


class HikRigReuseTests(unittest.TestCase):
    def test_adapter_reports_complete_saved_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_calibration(Path(directory) / "rig")
            self.assertTrue(RectifiedHikCamera(path, rectify=False).is_calibrated())
            config = json.loads(path.read_text(encoding="utf-8"))
            del config["normalization"]["full_sensor_camera_to_output_3x3"]
            path.write_text(json.dumps(config), encoding="utf-8")
            self.assertFalse(RectifiedHikCamera(path, rectify=False).is_calibrated())

    def test_snapshot_check_accepts_repeat_noise_and_rejects_one_pixel_motion(self):
        rng = np.random.RandomState(4)
        reference = np.zeros((120, 160, 3), np.uint8)
        for y in range(0, 120, 20):
            for x in range(0, 160, 20):
                if (x // 20 + y // 20) % 2 == 0:
                    reference[y : y + 20, x : x + 20] = 240
        noise = rng.normal(0.0, 1.0, reference.shape)
        repeated = np.clip(reference.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        moved = np.roll(reference, 1, axis=1)
        self.assertTrue(compare_calibration_snapshots(reference, repeated)["matches"])
        self.assertFalse(compare_calibration_snapshots(reference, moved)["matches"])

    def test_discovery_selects_newest_direct_matching_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = write_calibration(root / "hik-calibration-old")
            time.sleep(0.02)
            newest = write_calibration(root / "hik-calibration-new")
            write_calibration(
                root / "hik-calibration-other", camera_id="CAM-2", phone_serial="PHONE-2"
            )
            self.assertEqual(
                newest.resolve(),
                discover_previous_calibration(
                    root, camera_id="CAM-1", phone_serial="PHONE-1"
                ),
            )
            self.assertNotEqual(old.resolve(), newest.resolve())


if __name__ == "__main__":
    unittest.main()
