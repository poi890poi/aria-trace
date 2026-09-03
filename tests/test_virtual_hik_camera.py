import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from aria_trace.adapters.rig.devices import CameraConfiguration, create_camera_adapter
from virtualhikcam.driver import VirtualHikCameraAdapter


class FakeAndroidCamera:
    instances = []

    def __init__(self, serial, **kwargs):
        self.serial = serial
        self.kwargs = dict(kwargs)
        self.adb = None
        self.closed = False
        self.width = int(kwargs["width_px"])
        self.height = int(kwargs["height_px"])
        self.effective_configuration = {
            "effective": {"width_px": self.width, "height_px": self.height}
        }
        FakeAndroidCamera.instances.append(self)

    def open(self):
        return self

    def read_sample(self):
        y, x = np.indices((self.height, self.width))
        image = np.stack(
            [x % 251, y % 251, (x + y) % 251], axis=2
        ).astype(np.uint8)
        return SimpleNamespace(
            image=image,
            capture_time_ns=100,
            receive_time_ns=200,
            metadata={"source": "fake_android_camera"},
        )

    def close(self):
        self.closed = True


def configuration(device_id, width=1280, height=720, fps=30.0):
    return CameraConfiguration(device_id, width, height, fps, "virtual_hik")


class VirtualHikCameraTests(unittest.TestCase):
    def setUp(self):
        FakeAndroidCamera.instances = []
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.device = (
            "virtual-hik://phone/camera/1?"
            "width=1280&height=720&fps=30&zoom=1&bit_rate=12000000"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def adapter(self):
        return VirtualHikCameraAdapter(
            state_root=self.root,
            source_factory=FakeAndroidCamera,
            sleep_source_on_close=False,
        )

    def test_nonzero_roi_persists_across_close_and_reopen(self):
        first = self.adapter()
        metadata = first.open(configuration(self.device))
        self.assertEqual("virtual_hik_android_camera", metadata["adapter_id"])
        self.assertEqual([137, 91, 803, 517], first.set_roi([137, 91, 803, 517]))
        sample = first.read()
        self.assertEqual((517, 803), sample.image.shape[:2])
        self.assertEqual(
            [137, 91, 803, 517],
            sample.metadata["image_space"]["roi_in_parent_xywh"],
        )
        self.assertEqual(
            137.0,
            sample.metadata["image_space"]["local_to_parent_3x3"][0][2],
        )
        first.close()

        second = self.adapter()
        second.open(configuration(self.device))
        self.assertEqual([137, 91, 803, 517], second.active_roi)
        self.assertEqual((517, 803), second.read().image.shape[:2])
        self.assertEqual([0, 0, 1280, 720], second.reset_full_sensor_roi())
        second.close()

    def test_camera_modes_have_independent_persistent_state(self):
        first = self.adapter()
        first.open(configuration(self.device))
        first.set_roi([10, 20, 500, 400])
        first_path = first.metadata["state_file"]
        first.close()

        other_device = (
            "virtual-hik://phone/camera/1?"
            "width=640&height=480&fps=30&zoom=2&bit_rate=12000000"
        )
        second = self.adapter()
        second.open(configuration(other_device, 640, 480))
        self.assertEqual([0, 0, 640, 480], second.active_roi)
        self.assertNotEqual(first_path, second.metadata["state_file"])
        self.assertEqual(2.0, second.metadata["zoom"])
        second.close()

    def test_second_open_is_rejected_until_first_releases(self):
        first = self.adapter()
        first.open(configuration(self.device))
        second = self.adapter()
        with self.assertRaisesRegex(RuntimeError, "already open"):
            second.open(configuration(self.device))
        first.close()
        second.open(configuration(self.device))
        second.close()

    def test_imaging_contract_state_persists_without_physical_claim(self):
        first = self.adapter()
        first.open(configuration(self.device))
        first.set_manual_imaging(12000.0, 4.5)
        first.set_white_balance(1300, 1024, 1700)
        self.assertFalse(first.controls()["exposure_us"]["available"])
        first.close()

        second = self.adapter()
        second.open(configuration(self.device))
        self.assertEqual({"exposure_us": 12000.0, "gain": 4.5}, second.imaging_state())
        self.assertEqual(
            {"ratio_red": 1300, "ratio_green": 1024, "ratio_blue": 1700},
            second.white_balance_state(),
        )
        self.assertEqual("none", second.metadata["physical_hik_claims"])
        second.close()

    def test_real_adapter_factory_loads_virtual_driver(self):
        adapter = create_camera_adapter(
            "virtualhikcam.driver:create_camera_adapter"
        )
        self.assertIsInstance(adapter, VirtualHikCameraAdapter)

    def test_explicit_displacement_is_applied_before_roi_and_declared(self):
        adapter = self.adapter()
        adapter.open(configuration(self.device))
        adapter.set_simulated_displacement(7.0, 5.0, 0.0)
        self.assertEqual(7.0, adapter.metadata["simulated_displacement"]["x_px"])
        adapter.set_roi([3, 2, 300, 200])
        sample = adapter.read()

        # ROI-local (9, 10) is full-sensor (12, 12), which samples canonical
        # (5, 7) after a +7 X, +5 Y content displacement.
        expected = np.asarray([5 % 251, 7 % 251, 12 % 251], dtype=np.uint8)
        np.testing.assert_array_equal(expected, sample.image[10, 9])
        self.assertEqual(
            [3, 2, 300, 200], sample.metadata["image_space"]["roi_in_parent_xywh"]
        )
        self.assertEqual(
            7.0,
            sample.metadata["image_space"]["content_from_canonical_parent_3x3"][0][2],
        )
        self.assertEqual(
            5.0,
            sample.metadata["image_space"]["content_from_canonical_parent_3x3"][1][2],
        )
        self.assertTrue(sample.metadata["simulated_displacement_active"])
        adapter.close()

    def test_seeded_random_displacement_persists_and_reset_preserves_other_state(self):
        first = self.adapter()
        first.open(configuration(self.device))
        first.set_roi([30, 40, 500, 400])
        first.set_manual_imaging(12000.0, 4.5)
        displaced = dict(first.randomize_simulated_displacement(seed=1729))
        self.assertEqual("random", displaced["mode"])
        self.assertEqual(1729, displaced["seed"])
        first.close()

        second = self.adapter()
        second.open(configuration(self.device))
        self.assertEqual(displaced, dict(second.simulated_displacement()))
        self.assertEqual([30, 40, 500, 400], second.active_roi)
        self.assertEqual({"exposure_us": 12000.0, "gain": 4.5}, second.imaging_state())
        reset = dict(second.reset_simulated_displacement())
        self.assertEqual(
            {
                "x_px": 0.0,
                "y_px": 0.0,
                "rotation_deg_clockwise": 0.0,
                "mode": "canonical",
                "seed": None,
            },
            reset,
        )
        self.assertEqual([30, 40, 500, 400], second.active_roi)
        self.assertEqual({"exposure_us": 12000.0, "gain": 4.5}, second.imaging_state())
        second.close()

        third = self.adapter()
        third.open(configuration(self.device))
        self.assertEqual(reset, dict(third.simulated_displacement()))
        third.close()

    def test_displacement_map_is_reused_until_pose_changes(self):
        adapter = self.adapter()
        adapter.open(configuration(self.device))
        opened_generation = adapter._displacement_map_generation
        opened_map = adapter._displacement_map_x
        adapter.read()
        adapter.read()
        self.assertEqual(opened_generation, adapter._displacement_map_generation)
        self.assertIs(opened_map, adapter._displacement_map_x)

        adapter.set_simulated_displacement(4.0, -3.0, 0.5)
        changed_generation = adapter._displacement_map_generation
        changed_map = adapter._displacement_map_x
        self.assertEqual(opened_generation + 1, changed_generation)
        self.assertIsNot(opened_map, changed_map)
        first = adapter.read()
        second = adapter.read()
        self.assertEqual(changed_generation, adapter._displacement_map_generation)
        self.assertIs(changed_map, adapter._displacement_map_x)
        self.assertEqual(
            changed_generation, first.metadata["displacement_map_generation"]
        )
        self.assertEqual(
            changed_generation, second.metadata["displacement_map_generation"]
        )
        adapter.close()

    def test_schema_one_state_migrates_to_canonical_displacement(self):
        first = self.adapter()
        first.open(configuration(self.device))
        first.set_roi([10, 20, 500, 400])
        state_path = Path(first.metadata["state_file"])
        first.close()
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["schema_version"] = 1
        state.pop("simulated_displacement", None)
        state_path.write_text(json.dumps(state), encoding="utf-8")

        second = self.adapter()
        second.open(configuration(self.device))
        self.assertEqual([10, 20, 500, 400], second.active_roi)
        self.assertEqual("canonical", second.simulated_displacement()["mode"])
        migrated = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(2, migrated["schema_version"])
        self.assertIn("simulated_displacement", migrated)
        second.close()


if __name__ == "__main__":
    unittest.main()
