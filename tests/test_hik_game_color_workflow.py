import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from rig_runtime.adapters.filesystem.profile_registry import (
    ProfileContext,
    ProfileRegistry,
    ProfileResolutionError,
    context_from_rig_calibration,
)
from rig_runtime.adapters.filesystem.session import SessionReader, SessionWriter
from rig_runtime.domain.packets import FramePacket
from rig_runtime.domain.spatial import bind_geometry, raster_space
from rig_runtime.workflows.hik_game_color_calibration import (
    _check_color_spatial_alignment,
    _decode_session_records,
    calibrate_game_color_session,
    main,
)
from rig_runtime.workflows.profile_management import publish_rig_calibration


class HikGameColorWorkflowTests(unittest.TestCase):
    def test_color_spatial_check_stops_consistent_xy_displacement(self):
        image = np.zeros((120, 160, 3), np.uint8)
        image[10:55, 20:75] = 255
        image[65:105, 90:145] = 180
        cv2.line(image, (5, 115), (155, 5), (90, 90, 90), 3)
        shifted = np.zeros_like(image)
        shifted[:, 14:] = image[:, :-14]
        adb_frames = np.stack([image] * 4)
        hik_frames = np.stack([shifted] * 4)
        mask = np.full(image.shape[:2], 255, np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "alignment"
            with self.assertRaisesRegex(ValueError, "consistently displaced"):
                _check_color_spatial_alignment(
                    adb_frames,
                    hik_frames,
                    np.eye(3, dtype=np.float64),
                    mask,
                    output,
                )
            summary = json.loads(
                (output / "spatial_alignment_check.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("displaced", summary["status"])
            self.assertAlmostEqual(
                14.0,
                summary["aggregate"]["hik_offset_xy_px_from_adb"][0],
                delta=1.0,
            )
            self.assertTrue(
                (output / "spatial_alignment_residual_translation_overlay.jpg").is_file()
            )

    def test_color_decoder_reads_adb_image_series_without_video(self):
        class Source:
            stream_id = "android_phone"

            def describe(self):
                return {
                    "stream_id": self.stream_id,
                    "preferred_frame_storage": "image_series",
                    "preferred_image_format": "png",
                }

        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "session"
            writer = SessionWriter(session, [Source()], [])
            for index in range(3):
                writer.write_frame(
                    FramePacket(
                        "android_phone",
                        np.full((8, 16, 3), 40 + index, np.uint8),
                        writer.origin_ns + index,
                        writer.origin_ns + index,
                    )
                )
            writer.close()
            reader = SessionReader(session)
            records = reader.frames_by_stream["android_phone"]
            decoded = _decode_session_records(reader, "android_phone", records)
            self.assertEqual((3, 8, 16, 3), decoded.shape)
            self.assertFalse(reader.video_path("android_phone").exists())

    def test_cli_derives_evidence_output_from_profile_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = root / "profiles"
            result = {
                "profile_revision": "color-1",
                "hik_bayer_conversion": {
                    "fit": {
                        "baseline_validation": {"rgb_mae_dn": 10.0},
                        "selected_validation": {"rgb_mae_dn": 4.0},
                    }
                },
            }
            with patch(
                "rig_runtime.workflows.hik_game_color_calibration.calibrate_game_color_session",
                return_value=result,
            ) as calibrate:
                self.assertEqual(
                    0,
                    main(
                        [
                            str(root / "session"),
                            "--profile-root",
                            str(profiles),
                            "--game-id",
                            "game-1",
                        ]
                    ),
                )
            positional, keyword = calibrate.call_args
            output = Path(positional[1])
            self.assertEqual(
                profiles / "calibrations" / "game-color", output.parent
            )
            self.assertTrue(output.name.startswith("game-1-"))
            self.assertTrue(keyword["activate"])

    def test_session_color_calibration_publishes_exact_rig_dependency(self):
        self._exercise_color_calibration()

    def test_displacement_does_not_invalidate_recorded_color_pairs(self):
        self._exercise_color_calibration(displace=True)

    def test_unavailable_color_returns_fallback_without_replacing_profiles(self):
        for error in (ValueError("insufficient stable color samples"),
                      RuntimeError("cannot decode optional HIK color video"),
                      ProfileResolutionError("optional sampling profile unavailable")):
            for prior in (False, True):
                with self.subTest(error=str(error), prior=prior):
                    self._exercise_color_calibration(displace=True, fit_error=error, prior_color=prior)

    def test_missing_hik_color_frames_reuse_previous_fit_after_displacement(self):
        self._exercise_color_calibration(displace=True, prior_color=True, missing_hik=True)

    def test_unhelpful_fit_does_not_replace_previous_working_color(self):
        self._exercise_color_calibration(displace=True, prior_color=True, identity_preferred=True)

    def test_cli_reports_optional_fallback_without_requesting_fit_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main([str(root / "missing-session"), str(root / "output"),
                             "--profile-root", str(root / "profiles"), "--game-id", "game-1"])
            self.assertEqual(0, code)
            self.assertIn("Optional color: uncorrected", output.getvalue())
            self.assertNotIn("Validation RGB MAE", output.getvalue())
            self.assertFalse(json.loads((root / "output/game_color_calibration.json").read_text())["fresh_color_measurement"])

    def _exercise_color_calibration(self, *, displace=False, fit_error=None, prior_color=False,
                                    missing_hik=False, identity_preferred=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rig = root / "rig" / "hik_camera_calibration.json"
            rig.parent.mkdir()
            rig_document = {
                "camera": {
                    "adapter_id": "hik_mvs",
                    "device_id": "CAM-1",
                    "full_sensor_mode": {"width_px": 8, "height_px": 8, "fps": 30},
                    "hardware_roi_xywh": [0, 0, 8, 8],
                },
                "phone": {
                    "serial": "PHONE-1",
                    "model": "phone",
                    "natural_screen_size_px": [8, 16],
                    "screen_size_px": [8, 16],
                    "refresh_hz": 120,
                },
                "imaging": {
                    "exposure_us": 1000,
                    "gain": 1,
                    "white_balance": {
                        "ratio_red": 1024,
                        "ratio_green": 1024,
                        "ratio_blue": 1024,
                    },
                },
                "geometry": {
                    "screen_to_full_sensor_camera_3x3": np.eye(3).tolist(),
                    "full_sensor_camera_to_screen_3x3": np.eye(3).tolist(),
                },
                "normalization": {
                    "output_size_px": [8, 8],
                    "full_sensor_camera_to_output_3x3": [
                        [1, 0, 0], [0, 1, 0], [0, 0, 1]
                    ],
                },
            }
            rig.write_text(json.dumps(rig_document), encoding="utf-8")
            registry = ProfileRegistry(root / "profiles")
            rig_profile = publish_rig_calibration(rig, registry=registry)
            phone_game_context = ProfileContext(
                game_id="game-1",
                package="game.package",
                camera_id="CAM-1",
                phone_id="PHONE-1",
                phone_model="phone",
                panel_display={
                    "natural_panel_px": [8, 16],
                    "logical_frame_px": [8, 16],
                    "refresh_hz": 120,
                },
                game_display={
                    "natural_panel_px": [8, 16],
                    "logical_frame_px": [16, 8],
                    "game_viewport_xywh": [0, 0, 16, 8],
                    "rotation_quarter_turns": 1,
                    "ui_layout_id": "default",
                },
            )
            phone_game = registry.publish(
                "phone_game",
                phone_game_context,
                {
                    "profile_kind": "phone_game",
                    "canonical_phone_crop_xywh": [0, 0, 8, 8],
                    "outer_boundary": bind_geometry(
                        {"center_x": 4.0, "center_y": 4.0, "radius": 3.0},
                        "circle",
                        raster_space("android_phone_natural_display_pixels", [8, 16]),
                    ),
                },
                review_state="accepted",
                activate=True,
            )

            session = root / "session"
            session.mkdir()
            manifest = {
                "schema_version": "1.0",
                "status": "complete",
                "context": {
                    "game_id": "game-1",
                    "game_launch": {"game_id": "game-1", "package": "game.package"},
                    "hik_capture": {"rig_calibration": str(rig)},
                    "phone_surface_orientation": {
                        "quarter_turns_clockwise_from_natural": 1,
                        "logical_size_px": [16, 8],
                        "natural_size_px": [8, 16],
                    },
                },
                "videos": {
                    "android_phone": "android.mkv",
                    "hik_phone": "hik.mkv",
                },
            }
            (session / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            rows = []
            for index in range(4):
                rows.append({
                    "stream_id": "android_phone", "frame_index": index,
                    "host_capture_time_ns": 1000 + index * 100,
                    "session_time_ns": index * 100, "width": 16, "height": 8,
                })
                rows.append({
                    "stream_id": "hik_phone", "frame_index": index,
                    "host_capture_time_ns": 1010 + index * 100,
                    "session_time_ns": 10 + index * 100, "width": 8, "height": 8,
                })
            (session / "frames.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            (session / "inputs.jsonl").write_text("", encoding="utf-8")
            (session / "coordinate_spaces.yaml").write_text(
                yaml.safe_dump({
                    "conversions": {"adb_to_hik_phone_video_3x3": np.eye(3).tolist()},
                    "streams": {"hik_phone": {"content_size_px": [8, 8]}},
                }),
                encoding="utf-8",
            )
            android_frames = np.full((4, 8, 16, 3), 100, np.uint8)
            hik_frames = np.full((4, 8, 8, 3), 100, np.uint8)
            conversion = {
                "status": "selected",
                "gamma": 0.8,
                "ccm_rgb_3x3": np.eye(3).tolist(),
                "fit": {
                    "baseline_validation": {"rgb_mae_dn": 10.0},
                    "selected_validation": {"rgb_mae_dn": 4.0},
                },
            }
            previous = None
            if prior_color:
                previous = registry.publish("rig_game_color", phone_game_context,
                                            {"hik_bayer_conversion": conversion},
                                            dependencies={"rig": rig_profile["revision_id"]}, activate=True)
            if displace:
                moved_path = root / "moved_rig.json"
                moved_document = json.loads(json.dumps(rig_document))
                moved_document["normalization"]["full_sensor_camera_to_output_3x3"][0][2] = 1
                moved_path.write_text(json.dumps(moved_document))
                registry.publish("rig", context_from_rig_calibration(moved_document), {"generation": 2},
                                 runtime_files={"hik_camera_calibration": moved_path}, activate=True)
            before = registry.active_revision_ids()
            if missing_hik:
                (session / "frames.jsonl").write_text("".join(
                    json.dumps(row) + "\n" for row in rows if row["stream_id"] != "hik_phone"))
            with patch(
                "rig_runtime.workflows.hik_game_color_calibration._decode_indices",
                side_effect=[android_frames, hik_frames],
            ), patch(
                "rig_runtime.workflows.hik_game_color_calibration.optimize_mvs_bayer_conversion",
                return_value=(conversion, {"review.png": hik_frames[0]}),
            ) as optimize:
                if fit_error is not None:
                    optimize.side_effect = fit_error
                if identity_preferred:
                    optimize.return_value = ({**conversion, "status": "identity_preferred"}, {})
                result = calibrate_game_color_session(
                    session,
                    root / "output",
                    profile_root=root / "profiles",
                )
            if fit_error is not None or missing_hik or identity_preferred:
                self.assertEqual("optional_fallback", result["status"])
                self.assertTrue(result["non_gating"])
                self.assertFalse(result["fresh_color_measurement"])
                if fit_error is not None:
                    self.assertIn(str(fit_error), result["reason"])
                if missing_hik:
                    self.assertIn("no hik_phone frames", result["reason"])
                    optimize.assert_not_called()
                self.assertEqual(before, registry.active_revision_ids())
                self.assertEqual(previous["revision_id"] if previous else None, result["profile_revision"])
                self.assertEqual("reuse_existing" if previous else "uncorrected", result["fallback"])
                if previous:
                    self.assertEqual(conversion, result["hik_bayer_conversion"])
                    self.assertEqual(rig_profile["revision_id"], registry.revision(previous["revision_id"])["dependencies"]["rig"])
                persisted = json.loads((root / "output/game_color_calibration.json").read_text())
                self.assertEqual(result, persisted)
                return
            profile = registry.revision(result["profile_revision"])
            portable_color = registry.revision(
                result["portable_phone_game_color_revision"]
            )
            self.assertEqual(
                rig_profile["revision_id"], profile["dependencies"]["rig"]
            )
            self.assertEqual(
                portable_color["revision_id"],
                profile["dependencies"]["phone_game_color"],
            )
            self.assertEqual(
                "phone_game_color", portable_color["identity"]["kind"]
            )
            self.assertEqual(
                phone_game["revision_id"],
                result["sampling_geometry"]["phone_game_revision"],
            )
            passed_mask = optimize.call_args[0][5]
            self.assertGreater(np.count_nonzero(passed_mask), 0)
            self.assertLess(np.count_nonzero(passed_mask), passed_mask.size)
            self.assertTrue(
                registry.runtime_file(
                    portable_color, "adb_game_color_reference"
                ).is_file()
            )
            self.assertTrue((root / "output" / "hikcam_adapter.py").is_file())
            self.assertTrue(
                registry.runtime_file(
                    portable_color, "adb_game_color_reference_mask"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
