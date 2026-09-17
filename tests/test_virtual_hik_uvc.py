import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from rig_runtime.adapters.rig.devices import CameraConfiguration, create_camera_adapter
from virtualhikcam.driver import VirtualHikCameraAdapter, _camera_spec
from virtualhikcam import control, instrument
from virtualhikcam.uvc import device_uri


class Capture:
    def __init__(self, opened=True, readable=True):
        self.opened = opened
        self.readable = readable
        self.released = False
        self.reads = 0
        self.settings = {}
        y, x = np.indices((240, 320))
        self.image = np.stack([x % 251, y % 251, (x+y) % 251], axis=2).astype(np.uint8)

    def isOpened(self):
        return self.opened

    def set(self, key, value):
        self.settings[key] = value
        return True

    def get(self, key):
        return {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 29.97,
        }.get(key, 0)

    def read(self):
        self.reads += 1
        return (True, self.image.copy()) if self.readable else (False, None)

    def release(self):
        self.opened = False
        self.released = True


def configuration(uri=None):
    return CameraConfiguration(uri or device_uri(0,1280,720,30),1280,720,30,'virtual_hik')


class VirtualHikUvcTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def adapter(self):
        return VirtualHikCameraAdapter(state_root=self.root)

    def test_real_opencv_bridge_reports_actual_geometry_and_preserves_timing(self):
        cap = Capture()
        with patch.object(cv2,'VideoCapture',return_value=cap) as factory:
            adapter = self.adapter()
            with patch.object(adapter,'source_factory',side_effect=AssertionError('Android opened')):
                try:
                    metadata = adapter.open(configuration())
                    self.assertEqual('virtual_hik_uvc_camera',metadata['adapter_id'])
                    self.assertEqual([320,240],adapter.full_size)
                    self.assertEqual(640,metadata['source_configuration']['reported']['width_px'])
                    self.assertEqual(1280,cap.settings[cv2.CAP_PROP_FRAME_WIDTH])
                    factory.assert_called_once_with(0,cv2.CAP_DSHOW)
                    sample = adapter.read()
                    self.assertEqual(1,cap.reads)  # Opening's first sample is retained.
                    self.assertEqual('host_receive_no_sensor_timestamp',sample.metadata['timestamp_basis'])
                    self.assertEqual(sample.time_ns,sample.receive_time_ns)
                    self.assertGreater(sample.receive_time_ns,0)
                    self.assertEqual('as_delivered_by_uvc_backend',sample.metadata['image_space']['orientation'])
                    np.testing.assert_array_equal(cap.image,sample.image)
                finally:
                    adapter.close()
        self.assertTrue(cap.released)

    def test_displacement_precedes_roi_and_state_survives_reopen(self):
        with patch.object(cv2,'VideoCapture',side_effect=lambda *args:Capture()):
            first = self.adapter()
            try:
                first.open(configuration())
                first.set_roi([3,2,200,150])
                first.set_simulated_displacement(7,5,0)
                first.set_manual_imaging(12000,4)
                sample = first.read()
                np.testing.assert_array_equal(sample.image[10,9],[5,7,12])
                self.assertEqual([3,2,200,150],sample.metadata['image_space']['roi_in_parent_xywh'])
            finally:
                first.close()
            second = self.adapter()
            try:
                second.open(configuration())
                self.assertEqual([3,2,200,150],second.active_roi)
                self.assertEqual(7,second.simulated_displacement()['x_px'])
                self.assertEqual(12000,second.imaging_state()['exposure_us'])
                self.assertEqual([0,0,320,240],second.reset_full_sensor_roi())
            finally:
                second.close()

    def test_mode_state_is_separate_but_capture_ownership_is_shared(self):
        with patch.object(cv2,'VideoCapture',side_effect=lambda *args:Capture()) as factory:
            first = self.adapter()
            second = self.adapter()
            other = configuration(device_uri(0,640,480,29.97,2,'msmf'))
            try:
                first.open(configuration())
                first.set_roi([10,20,100,80])
                saved = first.metadata['state_file']
                with self.assertRaisesRegex(RuntimeError,'already open'):
                    second.open(other)
                self.assertEqual(1,factory.call_count)
                first.close()
                second.open(other)
                self.assertNotEqual(saved,second.metadata['state_file'])
                self.assertEqual([0,0,320,240],second.active_roi)
                zoomed = second.read().image
                self.assertGreater(int(zoomed[0,0,0]),0)
            finally:
                first.close()
                second.close()

    def test_failed_open_and_first_read_release_camera_and_lease(self):
        for bad in [Capture(opened=False),Capture(readable=False)]:
            with patch.object(cv2,'VideoCapture',side_effect=[bad,Capture()]):
                first = self.adapter()
                second = self.adapter()
                try:
                    with self.assertRaises(RuntimeError):
                        first.open(configuration())
                    self.assertTrue(bad.released)
                    second.open(configuration())
                finally:
                    first.close()
                    second.close()

    def test_size_change_and_disconnect_fail_without_changing_saved_roi(self):
        cap = Capture()
        with patch.object(cv2,'VideoCapture',return_value=cap):
            adapter = self.adapter()
            try:
                adapter.open(configuration())
                adapter.read()
                adapter.set_roi([1,2,100,80])
                cap.image = np.zeros((100,200,3),np.uint8)
                with self.assertRaisesRegex(RuntimeError,'full raster changed'):
                    adapter.read()
                self.assertEqual([1,2,100,80],adapter.active_roi)
                cap.readable = False
                with self.assertRaisesRegex(RuntimeError,'no frame'):
                    adapter.read()
            finally:
                adapter.close()
        self.assertTrue(cap.released)

    def test_identity_normalizes_numbers_and_rejects_invalid_source_options(self):
        a = _camera_spec('virtual-hik+uvc://00?fps=30.0',configuration())
        b = _camera_spec('virtual-hik+uvc://0?fps=30',configuration())
        self.assertEqual(a['canonical_device_id'],b['canonical_device_id'])
        self.assertNotEqual(a['canonical_device_id'],_camera_spec('virtual-hik://0/camera/1',configuration())['canonical_device_id'])
        for uri in ['virtual-hik+uvc://-1','virtual-hik+uvc://webcam','virtual-hik+uvc://0/camera/1',
                    'virtual-hik+uvc://0?fps=nan','virtual-hik+uvc://0?zoom=inf','virtual-hik+uvc://0?backend=wrong',
                    'virtual-hik+uvc://0?width=0','virtual-hik+uvc://0?bit_rate=5']:
            with self.subTest(uri=uri),self.assertRaises(ValueError):
                _camera_spec(uri,configuration())

    def test_control_command_benchmark_opens_selected_uvc_and_always_closes(self):
        cap = Capture()
        out = io.StringIO()
        with patch.object(cv2,'VideoCapture',return_value=cap),redirect_stdout(out):
            result = control.main(['benchmark','--uvc-device','0','--benchmark-frames','5','--state-root',str(self.root)])
        self.assertEqual(0,result)
        self.assertTrue(cap.released)
        self.assertTrue(json.loads(out.getvalue())['camera_model_id'].startswith('virtual-hik+uvc://0?'))

    def test_instrument_accepts_uvc_and_only_sleeps_android_target(self):
        args = instrument._parser().parse_args(['--uvc-device','0','--phone-serial','tablet'])
        self.assertEqual(0,args.uvc_device)
        args.camera_id = device_uri(0,1280,720,30)
        with patch.object(instrument.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='')) as run:
            results = instrument._sleep_devices(args)
        self.assertEqual(['tablet'],[r['serial'] for r in results])
        self.assertEqual('tablet',run.call_args[0][0][2])

    def test_factory_construction_does_not_open_any_camera(self):
        with patch.object(cv2,'VideoCapture') as capture:
            adapter = create_camera_adapter('virtualhikcam.driver:create_camera_adapter')
            self.assertEqual((),adapter.devices())
            adapter.close()
        capture.assert_not_called()


if __name__ == '__main__':
    unittest.main()
