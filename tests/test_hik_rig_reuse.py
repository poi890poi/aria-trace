import contextlib
import io
import json
import subprocess
import sys
import tempfile
import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from acquisition.rig_calibration.hik.driver import RectifiedHikCamera
from acquisition.rig_calibration.hik.reuse_precheck import (
    compare_charuco_alignment,
    discover_active_profile_calibration,
    format_reuse_precheck_failure,
    run_reuse_precheck,
    main as reuse_precheck_main,
    parser as reuse_precheck_parser,
)
from acquisition.profile_registry import ProfileContext, ProfileRegistry
from aria_trace.apps.hik_rig_calibration import (
    _write_standalone_adapter,
    main as rig_calibration_main,
)


def write_calibration(path: Path, camera_id="CAM-1", phone_serial="PHONE-1") -> Path:
    path.mkdir(parents=True)
    config = {
        "camera": {
            "device_id": camera_id,
            "full_sensor_mode": {"width_px": 64, "height_px": 48, "fps": 30.0},
            "hardware_roi_xywh": [0, 0, 64, 48],
        },
        "phone": {
            "serial": phone_serial,
            "screen_size_px": [64, 48],
            "orientation_quarter_turns": 0,
        },
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
        "geometry": {
            "charuco_layout": {
                "squares_x": 4,
                "squares_y": 3,
                "margin_px": [0, 0],
            },
            "screen_to_full_sensor_camera_3x3": np.eye(3).tolist(),
        },
    }
    config_path = path / "hik_camera_calibration.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


class HikRigReuseTests(unittest.TestCase):
    def test_precheck_resets_full_sensor_and_uses_only_charuco_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = write_calibration(root / "rig")
            output = root / "precheck"
            adapter = Mock()
            adapter.reset_full_sensor_roi.return_value = [0, 0, 64, 48]
            adapter.read.return_value = Mock(
                image=np.zeros((48, 64, 3), np.uint8),
                metadata={
                    "image_space": {
                        "space_id": "hik_camera_acquisition_pixels",
                        "stored_size_px": [64, 48],
                    }
                },
            )
            target = Mock()
            phone = Mock()
            points = np.asarray(
                [[10, 10], [30, 10], [10, 30], [30, 30]], np.float64
            )
            with patch(
                "aria_trace.workflows.rig_reuse_precheck.RectifiedHikCamera"
            ) as validator_type, patch(
                "aria_trace.workflows.rig_reuse_precheck.HikMvsCameraAdapter",
                return_value=adapter,
            ), patch(
                "aria_trace.workflows.rig_reuse_precheck.resolve_adb_executable",
                return_value=Path("adb"),
            ), patch(
                "aria_trace.workflows.rig_reuse_precheck.AdbPhoneSession",
                return_value=phone,
            ), patch(
                "aria_trace.workflows.rig_reuse_precheck.AdbDisplayTarget",
                return_value=target,
            ), patch(
                "aria_trace.workflows.rig_reuse_precheck.detect_charuco_correspondences",
                return_value={
                    "camera_points_xy": points,
                    "screen_points_xy": points,
                    "corner_count": 4,
                },
            ):
                validator_type.return_value.is_calibrated.return_value = True
                result = run_reuse_precheck(
                    calibration,
                    output,
                    sample_frames=3,
                    maximum_displacement_px=16.0,
                )

            self.assertTrue(result["reusable"])
            adapter.reset_full_sensor_roi.assert_called_once_with()
            self.assertEqual([0, 0, 64, 48], result["effective_full_sensor_roi_xywh"])
            self.assertFalse((root / "rig" / "last_camera_frame.png").exists())
            self.assertTrue((output / "fresh_full_sensor_frame.png").is_file())
            self.assertEqual(
                "hik_camera_acquisition_pixels",
                result["image_space"]["space_id"],
            )

    def test_repeatability_failure_message_names_geometry_limit_and_outcome(self):
        message = format_reuse_precheck_failure(
            {
                "status": "rig_moved",
                "comparison": {
                    "p95_displacement_px": 18.0,
                    "maximum_allowed_displacement_px": 16.0,
                },
            }
        )
        self.assertIn("detected ChArUco board displacement", message)
        self.assertIn("18.000 full-sensor px; allowed <= 16.000 px [FAIL]", message)
        self.assertIn("Lighting and pixel brightness are not part", message)

    def test_adapter_export_accepts_saved_bundle_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            calibration = bundle / "hik_camera_calibration.json"
            calibration.write_text("{}", encoding="utf-8")
            with patch(
                "aria_trace.adapters.filesystem.profile_registry.context_from_rig_calibration",
                return_value=Mock(),
            ), patch(
                "aria_trace.workflows.adapter_export.export_resolved_adapter",
                return_value={"output": str(bundle / "hikcam_adapter.py")},
            ) as export:
                _write_standalone_adapter(
                    bundle, bundle / "hikcam_adapter.py", root / "profiles"
                )
            self.assertEqual(
                bundle / "hikcam_adapter.py", export.call_args[0][0]
            )

    def test_rig_calibration_uses_configured_profile_root_for_default_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "shared-profiles"
            configured = {
                "devices": {"camera_id": "CAM-1", "phone_id": "PHONE-1"},
                "rig_calibration": {"repeatability_policy": "relaxed"},
            }
            from aria_trace.adapters.filesystem.system_configuration import (
                save_system_configuration,
            )

            save_system_configuration(configured, root)
            session = Mock()
            expected = root / "calibrations" / "hik-calibration-test"
            session.run.return_value = expected
            device = Mock(device_id="CAM-1", label="camera")
            adapter = Mock()
            adapter.devices.return_value = [device]
            with patch.dict(os.environ, {"ARIA_PROFILE_ROOT": str(root)}), patch(
                "aria_trace.apps.hik_rig_calibration.time.strftime",
                return_value="test",
            ), patch(
                "aria_trace.apps.hik_rig_calibration.HikMvsCameraAdapter",
                return_value=adapter,
            ), patch(
                "aria_trace.apps.hik_rig_calibration.connected_adb_devices",
                return_value=["PHONE-1"],
            ), patch(
                "aria_trace.apps.hik_rig_calibration.resolve_adb_executable",
                return_value=Path("adb"),
            ), patch(
                "aria_trace.apps.hik_rig_calibration.HikRigCalibrationSession",
                return_value=session,
            ) as session_type, patch(
                "aria_trace.workflows.profile_management.publish_rig_calibration",
                return_value={"revision_id": "rig-1", "publication": "new_revision"},
            ), patch(
                "aria_trace.apps.hik_rig_calibration._write_standalone_adapter"
            ) as exporter:
                self.assertEqual(0, rig_calibration_main(["--headless", "--save"]))
            options = session_type.call_args[0][0]
            self.assertEqual(expected, options.output_directory)
            self.assertEqual("relaxed", options.repeatability_policy)
            self.assertEqual(12.0, options.save_max_displacement_px)
            exporter.assert_called_once_with(
                expected, expected / "hikcam_adapter.py", root.resolve()
            )

    def test_compatibility_module_runs_the_canonical_cli(self):
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "acquisition.rig_calibration.hik.reuse_precheck",
                "--help",
            ],
            cwd=str(Path(__file__).resolve().parents[1]),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        self.assertEqual(0, completed.returncode, completed.stdout)
        self.assertIn("--profile-root", completed.stdout)

    def test_rig_calibration_reuse_option_skips_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = write_calibration(root / "saved")
            device = Mock(device_id="CAM-1", label="camera")
            adapter = Mock()
            adapter.devices.return_value = [device]
            with patch(
                "aria_trace.apps.hik_rig_calibration.HikMvsCameraAdapter",
                return_value=adapter,
            ), patch(
                "aria_trace.apps.hik_rig_calibration.connected_adb_devices",
                return_value=["PHONE-1"],
            ), patch(
                "aria_trace.apps.hik_rig_calibration.resolve_adb_executable",
                return_value=Path("adb"),
            ), patch(
                "aria_trace.workflows.rig_reuse_precheck.run_active_reuse_precheck",
                return_value={
                    "status": "reusable",
                    "reusable": True,
                    "camera_adapter_is_calibrated": True,
                    "calibration": str(calibration),
                    "comparison": {"p95_displacement_px": 1.0},
                },
            ), patch(
                "aria_trace.apps.hik_rig_calibration._write_standalone_adapter"
            ) as exporter, patch(
                "aria_trace.apps.hik_rig_calibration.HikRigCalibrationSession"
            ) as session:
                result = rig_calibration_main(
                    [
                        "--reuse-if-unchanged",
                        "--headless",
                        "--save",
                        "--output",
                        str(root / "output"),
                    ]
                )
            self.assertEqual(0, result)
            session.assert_not_called()
            exporter.assert_called_once()
            self.assertTrue((root / "output" / "reused_calibration.json").is_file())

    def test_failed_reuse_explains_metrics_before_full_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            device = Mock(device_id="CAM-1", label="camera")
            adapter = Mock()
            adapter.devices.return_value = [device]
            session = Mock()
            session.run.return_value = None
            precheck = {
                "status": "rig_moved",
                "reusable": False,
                "camera_adapter_is_calibrated": True,
                "comparison": {
                    "p95_displacement_px": 18.0,
                    "maximum_allowed_displacement_px": 16.0,
                },
            }
            output = io.StringIO()
            with patch(
                "aria_trace.apps.hik_rig_calibration.HikMvsCameraAdapter",
                return_value=adapter,
            ), patch(
                "aria_trace.apps.hik_rig_calibration.connected_adb_devices",
                return_value=["PHONE-1"],
            ), patch(
                "aria_trace.apps.hik_rig_calibration.resolve_adb_executable",
                return_value=Path("adb"),
            ), patch(
                "aria_trace.workflows.rig_reuse_precheck.run_active_reuse_precheck",
                return_value=precheck,
            ), patch(
                "aria_trace.apps.hik_rig_calibration.HikRigCalibrationSession",
                return_value=session,
            ) as session_type, contextlib.redirect_stdout(output):
                self.assertEqual(
                    0,
                    rig_calibration_main(
                        [
                            "--reuse-if-unchanged",
                            "--headless",
                            "--output",
                            str(root / "output"),
                        ]
                    ),
                )
            text = output.getvalue()
            self.assertIn(
                "18.000 full-sensor px; allowed <= 16.000 px [FAIL]", text
            )
            self.assertIn("Full rig calibration will start now.", text)
            self.assertIn("Repeatability evidence:", text)
            session_type.assert_called_once()
            session.run.assert_called_once()

    def test_legacy_calibration_option_is_obsolete(self):
        with self.assertRaises(SystemExit):
            reuse_precheck_parser().parse_args(
                ["--calibration", "legacy.json", "--output", "out"]
            )
        with self.assertRaises(SystemExit):
            reuse_precheck_parser().parse_args(
                ["--artifacts-root", "artifacts", "--output", "out"]
            )

    def test_active_registry_profile_is_resolved_without_artifact_scanning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = write_calibration(root / "bundle")
            registry = ProfileRegistry(root / "profiles")
            context = ProfileContext(
                camera_id="CAM-1",
                phone_id="PHONE-1",
                panel_display={
                    "natural_panel_px": [64, 48],
                    "logical_frame_px": [64, 48],
                    "refresh_hz": 60,
                },
            )
            registry.publish(
                "rig",
                context,
                {"profile_kind": "rig"},
                runtime_files={"hik_camera_calibration": calibration},
                review_state="accepted",
                activate=True,
            )
            resolved = discover_active_profile_calibration(
                root / "profiles", camera_id="CAM-1", phone_serial="PHONE-1"
            )
            self.assertTrue(resolved.is_file())
            self.assertNotEqual(calibration.resolve(), resolved)

    def test_adapter_reports_complete_saved_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = write_calibration(Path(directory) / "rig")
            self.assertTrue(RectifiedHikCamera(path, rectify=False).is_calibrated())
            config = json.loads(path.read_text(encoding="utf-8"))
            del config["normalization"]["full_sensor_camera_to_output_3x3"]
            path.write_text(json.dumps(config), encoding="utf-8")
            self.assertFalse(RectifiedHikCamera(path, rectify=False).is_calibrated())

    def test_charuco_alignment_ignores_pixels_and_detects_geometric_motion(self):
        screen = np.asarray([[10, 10], [30, 10], [10, 30], [30, 30]], np.float64)
        saved = np.asarray([[110, 60], [130, 60], [110, 80], [130, 80]], np.float64)
        unchanged = {
            "screen_points_xy": screen,
            "camera_points_xy": saved + np.asarray([2.0, -1.0]),
        }
        moved = {
            "screen_points_xy": screen,
            "camera_points_xy": saved + np.asarray([20.0, 0.0]),
        }
        matrix = [[1, 0, 100], [0, 1, 50], [0, 0, 1]]
        self.assertTrue(
            compare_charuco_alignment(
                [unchanged, unchanged, unchanged],
                matrix,
                maximum_displacement_px=16.0,
            )["matches"]
        )
        result = compare_charuco_alignment(
            [moved, moved, moved], matrix, maximum_displacement_px=16.0
        )
        self.assertFalse(result["matches"])
        self.assertTrue(result["lighting_invariant"])

    def test_legacy_artifact_directory_is_not_used_for_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_calibration(root / "artifacts" / "hik-calibration-newest")
            output = root / "precheck"
            self.assertEqual(
                0,
                reuse_precheck_main(
                    [
                        "--profile-root",
                        str(root / "profiles"),
                        "--output",
                        str(output),
                    ]
                ),
            )
            result = json.loads((output / "precheck.json").read_text(encoding="utf-8"))
            self.assertEqual("no_previous_calibration", result["status"])
            self.assertNotIn("calibration", result)

if __name__ == "__main__":
    unittest.main()
