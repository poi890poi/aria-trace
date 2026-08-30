import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acquisition.rig_dual_capture import (
    build_calibrated_rig_recording_bundle,
    single_source_recording_bundle,
)


class FakeSource:
    def __init__(self, stream_id):
        self.stream_id = stream_id
        self.stopped = False

    def stop(self):
        self.stopped = True


class RigDualCaptureTests(unittest.TestCase):
    def test_single_source_bundle_preserves_legacy_stream(self):
        frame = FakeSource("main")
        input_source = object()
        bundle = single_source_recording_bundle(frame, input_source)
        self.assertEqual(bundle.frame_sources, [frame])
        self.assertEqual(bundle.input_sources, [input_source])
        self.assertEqual(bundle.primary_stream_id, "main")
        self.assertEqual(bundle.required_stream_ids, ["main"])

    def test_calibrated_rig_bundle_owns_both_synced_sources(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            calibration = root / "hik_camera_calibration.json"
            calibration.write_text(
                json.dumps(
                    {
                        "camera": {"device_id": "HIK123"},
                        "phone": {"serial": "PHONE456"},
                    }
                ),
                encoding="utf-8",
            )
            android = FakeSource("android_phone")
            hik = FakeSource("hik_phone")
            clock = object()
            hub = object()
            input_source = object()
            processor = object()
            orientation = {
                "selected_adb_surface_quarter_turns_clockwise_from_phone_natural": 1,
                "selected_camera_adapter_image_quarter_turns_clockwise_from_calibration_display": 0,
                "status": "selected",
                "selection_basis": "image_evidence",
                "selected_confidence": 0.9,
                "confidence_margin": 0.2,
            }
            surface = {
                "quarter_turns_clockwise_from_natural": 3,
                "degrees_clockwise_from_natural": 270,
                "logical_size_px": [2400, 1080],
                "natural_size_px": [1080, 2400],
                "source": "test",
            }
            with patch(
                "aria_trace.adapters.rig.dual_capture.AdbClockMapper", return_value=clock
            ) as clock_type, patch(
                "aria_trace.adapters.rig.dual_capture.find_scrcpy_server",
                return_value=Path("scrcpy-server"),
            ), patch(
                "aria_trace.adapters.rig.dual_capture.ScrcpyCaptureHub", return_value=hub
            ) as hub_type, patch(
                "aria_trace.adapters.rig.dual_capture.AndroidRoiFrameSource",
                return_value=android,
            ) as android_type, patch(
                "aria_trace.adapters.rig.dual_capture.CalibratedHikFrameSource",
                return_value=hik,
            ) as hik_type, patch(
                "aria_trace.adapters.rig.dual_capture._phone_surface", return_value=surface
            ), patch(
                "aria_trace.adapters.rig.dual_capture.orient_hik_source_from_first_adb_frame",
                return_value=(orientation, {"first.png": object()}),
            ) as orient, patch(
                "aria_trace.adapters.rig.dual_capture.GameCrossSourceEvidenceRecorder",
                return_value=processor,
            ) as processor_type, patch(
                "aria_trace.adapters.rig.dual_capture.AdbGetEventSource",
                return_value=input_source,
            ) as input_type:
                bundle = build_calibrated_rig_recording_bundle(
                    calibration,
                    adb=Path("adb.exe"),
                    scrcpy_server=Path("scrcpy-server"),
                    input_adapter="adb_getevent",
                )

            clock_type.assert_called_once_with(Path("adb.exe"), "PHONE456")
            self.assertIs(hub_type.call_args[1]["clock"], clock)
            self.assertEqual(android_type.call_args[0][1].stream_id, "android_phone")
            self.assertEqual(hik_type.call_args[0][1], "hik_phone")
            self.assertEqual(
                orient.call_args[1]["android_reported_quarter_turns"], 3
            )
            self.assertIs(input_type.call_args[1]["clock"], clock)
            self.assertEqual(bundle.frame_sources, [android, hik])
            self.assertEqual(bundle.input_sources, [input_source])
            self.assertEqual(bundle.frame_processors, [processor])
            self.assertEqual(bundle.primary_stream_id, "hik_phone")
            self.assertEqual(
                bundle.required_stream_ids, ["android_phone", "hik_phone"]
            )
            self.assertEqual(
                bundle.session_context["phone_surface_orientation"][
                    "quarter_turns_clockwise_from_natural"
                ],
                1,
            )
            processor_type.assert_called_once()

    def test_finalize_writes_space_authority_and_manifest_reference(self):
        with tempfile.TemporaryDirectory() as value:
            root = Path(value)
            calibration = root / "hik_camera_calibration.json"
            calibration.write_text(
                json.dumps(
                    {
                        "camera": {"device_id": "HIK123"},
                        "phone": {"serial": "PHONE456"},
                    }
                ),
                encoding="utf-8",
            )
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            android = FakeSource("android_phone")
            hik = FakeSource("hik_phone")
            with patch(
                "aria_trace.adapters.rig.dual_capture.AdbClockMapper", return_value=object()
            ), patch(
                "aria_trace.adapters.rig.dual_capture.find_scrcpy_server",
                return_value=Path("scrcpy-server"),
            ), patch(
                "aria_trace.adapters.rig.dual_capture.ScrcpyCaptureHub", return_value=object()
            ), patch(
                "aria_trace.adapters.rig.dual_capture.AndroidRoiFrameSource",
                return_value=android,
            ), patch(
                "aria_trace.adapters.rig.dual_capture.CalibratedHikFrameSource",
                return_value=hik,
            ), patch(
                "aria_trace.adapters.rig.dual_capture._phone_surface",
                return_value={
                    "quarter_turns_clockwise_from_natural": 0,
                    "degrees_clockwise_from_natural": 0,
                    "logical_size_px": [100, 200],
                    "natural_size_px": [100, 200],
                    "source": "test",
                },
            ), patch(
                "aria_trace.adapters.rig.dual_capture.orient_hik_source_from_first_adb_frame",
                return_value=(
                    {
                        "selected_adb_surface_quarter_turns_clockwise_from_phone_natural": 0,
                        "selected_camera_adapter_image_quarter_turns_clockwise_from_calibration_display": 0,
                        "status": "selected",
                        "selection_basis": "image_evidence",
                        "selected_confidence": 1.0,
                        "confidence_margin": 1.0,
                    },
                    {},
                ),
            ), patch(
                "aria_trace.adapters.rig.dual_capture.GameCrossSourceEvidenceRecorder",
                return_value=object(),
            ):
                bundle = build_calibrated_rig_recording_bundle(
                    calibration,
                    adb=Path("adb.exe"),
                    scrcpy_server=Path("scrcpy-server"),
                )
            manifest = {
                "status": "complete",
                "frame_counts": {"android_phone": 4, "hik_phone": 3},
            }
            with patch(
                "aria_trace.adapters.rig.dual_capture.write_dual_source_space_yaml"
            ) as write_spaces:
                bundle.finalize(root, manifest)
            write_spaces.assert_called_once()
            self.assertEqual(manifest["coordinate_spaces"], "coordinate_spaces.yaml")
            saved = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["coordinate_spaces"], "coordinate_spaces.yaml")
            with self.assertRaisesRegex(RuntimeError, "hik_phone"):
                bundle.finalize(
                    root,
                    {
                        "status": "complete",
                        "frame_counts": {"android_phone": 4},
                    },
                )

if __name__ == "__main__":
    unittest.main()
