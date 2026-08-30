import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from acquisition.minimap_calibration import (
    _validate_segments,
    android_minimap_discovery_config,
    calibrate_minimap_boundary_frames,
    calibrate_minimap_frames,
    calibrate_segment_sessions,
    discover_android_minimap_seed,
)
from acquisition.models import FramePacket
from acquisition.session import SessionWriter


def synthetic_frames(frame_count=96):
    height, width = 180, 220
    boundary_center = np.array([112.0, 83.0])
    pivot = np.array([111.5, 82.5])
    radius = 69
    cursor_color = tuple(int(value) for value in cv2.cvtColor(np.uint8([[[93, 230, 240]]]), cv2.COLOR_HSV2BGR)[0, 0])
    yy, xx = np.ogrid[:height, :width]
    circle = (xx - boundary_center[0]) ** 2 + (yy - boundary_center[1]) ** 2 <= radius ** 2
    rotation, movement = [], []
    base_polygon = np.array([[-7.0, -6.0], [-7.0, 6.0], [9.0, 0.0]])
    for index in range(frame_count):
        angle = 2.0 * np.pi * index / frame_count
        image = np.full((height, width, 3), 22, np.uint8)
        phase = (np.sin((xx - boundary_center[0]) * 0.12 * np.cos(angle) + (yy - boundary_center[1]) * 0.12 * np.sin(angle)) + 1.0) * 48.0
        for channel, offset in enumerate((35, 50, 65)):
            layer = np.clip(offset + phase, 0, 255).astype(np.uint8)
            image[:, :, channel][circle] = layer[circle]
        cv2.circle(image, tuple(boundary_center.astype(int)), radius, (220, 225, 230), 2, cv2.LINE_AA)
        polygon = np.round(base_polygon + pivot).astype(np.int32)
        cv2.fillConvexPoly(image, polygon, cursor_color, cv2.LINE_AA)
        rotation.append(image)
        moved = image.copy()
        cv2.fillConvexPoly(moved, polygon, (70, 70, 70), cv2.LINE_AA)
        transform = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated_polygon = np.round(base_polygon @ transform.T + pivot).astype(np.int32)
        cv2.fillConvexPoly(moved, rotated_polygon, cursor_color, cv2.LINE_AA)
        movement.append(moved)
    return np.stack(rotation), np.stack(movement), boundary_center, pivot, radius


def synthetic_android_discovery_frames(frame_count=48):
    height, width = 360, 440
    center = np.array([112.0, 83.0])
    radius = 69
    yy, xx = np.ogrid[:height, :width]
    inside = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius ** 2
    values = []
    for index in range(frame_count):
        image = np.full((height, width, 3), 20, np.uint8)
        phase = (
            np.sin(xx * 0.11 + index * 0.23)
            + np.cos(yy * 0.13 - index * 0.17)
            + 2.0
        ) * 38.0
        for channel, offset in enumerate((28, 42, 56)):
            layer = np.clip(offset + phase, 0, 255).astype(np.uint8)
            image[:, :, channel][inside] = layer[inside]
        cv2.circle(
            image,
            tuple(center.astype(int)),
            radius,
            (225, 225, 225),
            2,
            cv2.LINE_AA,
        )
        values.append(image)
    return np.stack(values), center, radius


class MinimapCalibrationTests(unittest.TestCase):
    def test_boundary_only_entry_point_uses_verified_evidence_contract(self):
        rotation, _, boundary_center, _, radius = synthetic_frames()
        with tempfile.TemporaryDirectory() as temporary:
            result = calibrate_minimap_boundary_frames(
                rotation,
                Path(temporary),
                config={
                    "expected_center_xy": boundary_center.tolist(),
                    "center_search_radius_px": 14.0,
                    "radius_range_px": [62.0, 75.0],
                },
            )
            boundary = result["outer_boundary"]
            self.assertLess(
                np.linalg.norm(
                    np.array([boundary["center_x"], boundary["center_y"]])
                    - boundary_center
                ),
                2.0,
            )
            self.assertLess(abs(boundary["radius"] - radius), 2.5)
            declared = {item["name"] for item in result["evidence"]}
            self.assertEqual(
                declared,
                {
                    "minimap_stacked_difference_heatmap.png",
                    "boundary_temporal_heatmap.png",
                    "boundary_radial_heatmap.png",
                    "boundary_points_binary.png",
                    "boundary_fitted_circle.png",
                    "boundary_evidence_overlay.png",
                    "boundary_confidence.png",
                },
            )
            self.assertFalse((Path(temporary) / "model.npz").exists())
            for name in declared:
                self.assertGreater((Path(temporary) / name).stat().st_size, 0)

    def test_default_boundary_entry_discovers_android_minimap_without_precise_prior(self):
        rotation, expected_center, expected_radius = (
            synthetic_android_discovery_frames()
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = calibrate_minimap_boundary_frames(
                rotation, Path(temporary)
            )
            boundary = result["outer_boundary"]
            self.assertLess(
                np.linalg.norm(
                    np.asarray([boundary["center_x"], boundary["center_y"]])
                    - expected_center
                ),
                3.0,
            )
            self.assertLess(abs(boundary["radius"] - expected_radius), 3.0)
            self.assertIsNone(result["config"]["expected_center_xy"])
            self.assertEqual(
                [0.0, 0.0, 0.35, 0.35],
                result["config"]["discovery"][
                    "center_region_xyxy_fraction"
                ],
            )
            self.assertEqual(
                [0.07, 0.22],
                result["config"]["discovery"]["radius_fraction_range"],
            )
            self.assertEqual(
                0.85,
                result["config"]["discovery"][
                    "minimum_circle_visible_fraction"
                ],
            )
            declared = {item["name"] for item in result["evidence"]}
            self.assertIn("boundary_discovery_candidates.png", declared)

    def test_android_discovery_bounds_are_configurable_and_validated(self):
        configured = android_minimap_discovery_config(
            {
                "center_region_xyxy_fraction": [0.0, 0.0, 0.4, 0.3],
                "radius_fraction_range": [0.08, 0.2],
                "minimum_circle_visible_fraction": 0.9,
            }
        )
        self.assertEqual([0.0, 0.0, 0.4, 0.3], configured["center_region_xyxy_fraction"])
        with self.assertRaisesRegex(ValueError, "center region"):
            android_minimap_discovery_config(
                {"center_region_xyxy_fraction": [0, 0, 1.1, 0.4]}
            )

    def test_android_discovery_prefers_stable_disc_over_stronger_hud_ring(self):
        height, width = 480, 640
        true_center = np.array([80.0, 75.0])
        false_center = np.array([180.0, 120.0])
        true_radius, false_radius = 50.0, 45.0
        yy, xx = np.ogrid[:height, :width]
        true_disc = (
            (xx - true_center[0]) ** 2 + (yy - true_center[1]) ** 2
            <= true_radius ** 2
        )
        false_disc = (
            (xx - false_center[0]) ** 2 + (yy - false_center[1]) ** 2
            <= false_radius ** 2
        )
        frames = []
        for index in range(16):
            phase = (
                np.sin(xx * 0.09 + index * 0.45)
                + np.cos(yy * 0.07 - index * 0.31)
            ) * 28.0
            image = np.repeat(
                np.clip(92.0 + phase, 0, 255).astype(np.uint8)[:, :, None],
                3,
                axis=2,
            )
            static_map = np.clip(
                82.0 + np.sin(xx * 0.14) * 18.0 + np.cos(yy * 0.12) * 15.0,
                0,
                255,
            ).astype(np.uint8)
            for channel in range(3):
                image[:, :, channel][true_disc] = static_map[true_disc]
            cv2.circle(
                image,
                tuple(true_center.astype(int)),
                int(true_radius),
                (140, 140, 140),
                2,
                cv2.LINE_AA,
            )
            cv2.circle(
                image,
                tuple(false_center.astype(int)),
                int(false_radius),
                (255, 255, 255),
                3,
                cv2.LINE_AA,
            )
            frames.append(image)
        hough = np.asarray(
            [[[true_center[0], true_center[1], true_radius],
              [false_center[0], false_center[1], false_radius]]],
            dtype=np.float32,
        )
        with patch(
            "acquisition.minimap_calibration.cv2.HoughCircles",
            return_value=hough,
        ):
            result = discover_android_minimap_seed(np.stack(frames))

        selected = result["selected"]
        self.assertLess(
            np.linalg.norm(
                np.asarray([selected["center_x"], selected["center_y"]])
                - true_center
            ),
            1.0,
        )
        self.assertGreater(selected["stable_disc_contrast"], 0.0)
        false_candidate = next(
            candidate
            for candidate in result["candidates"]
            if abs(candidate["center_x"] - false_center[0]) < 1.0
        )
        self.assertGreater(false_candidate["ring_edge"], selected["ring_edge"])
        self.assertGreater(selected["score"], false_candidate["score"])

    def test_recovers_boundary_pivot_shape_and_evidence(self):
        rotation, movement, boundary_center, pivot, radius = synthetic_frames()
        with tempfile.TemporaryDirectory() as temporary:
            result = calibrate_minimap_frames(
                rotation,
                movement,
                Path(temporary),
                provenance={
                    "fixture": "synthetic_rotating_map",
                    "segment_sessions": {
                        "ordinary_cruise": {
                            "session_id": "synthetic-ordinary",
                            "container_fps": 30.0,
                        },
                        "movement_only": {
                            "session_id": "synthetic-movement",
                            "container_fps": 30.0,
                        },
                    },
                },
                ordinary_frames=movement,
            )
            boundary = result["outer_boundary"]
            center = result["rotation_center"]
            self.assertLess(np.linalg.norm(np.array([boundary["center_x"], boundary["center_y"]]) - boundary_center), 2.0)
            self.assertLess(abs(boundary["radius"] - radius), 2.5)
            self.assertLess(np.linalg.norm(np.array([center["x"], center["y"]]) - pivot), 1.0)
            self.assertGreater(center["angular_coverage_10deg_bins"], 0.85)
            self.assertGreater(result["cursor_shape"]["median_frame_template_iou"], 0.80)
            self.assertEqual(
                result["cursor_shape"]["model_type"],
                "symmetry_constrained_rigid_polygon",
            )
            self.assertGreater(result["cursor_shape"]["symmetry_fit_soft_iou"], 0.70)
            self.assertGreater(result["cursor_shape"]["polygon_template_iou"], 0.80)
            self.assertGreaterEqual(result["cursor_shape"]["polygon_vertex_count"], 3)
            self.assertEqual(result["cursor_shape"]["source"], "screen_fixed_hsv_persistence_during_camera_rotation")
            self.assertTrue((Path(temporary) / "calibration.json").is_file())
            self.assertTrue((Path(temporary) / "model.npz").is_file())
            declared = {item["name"] for item in result["evidence"]}
            self.assertIn("minimap_stacked_difference_heatmap.png", declared)
            self.assertIn("boundary_fitted_circle.png", declared)
            self.assertIn("boundary_evidence_overlay.png", declared)
            self.assertIn("cursor_center_orbit.png", declared)
            self.assertIn("cursor_shape_overlay.png", declared)
            self.assertIn("cursor_shape_polar_correlation.png", declared)
            self.assertIn("cursor_pose_gaussian_fits.png", declared)
            self.assertIn("cursor_pose_polar_samples.png", declared)
            pose = result["cursor_pose_validation"]
            pose_benchmark = pose["pose_estimation_benchmark"]
            self.assertEqual(pose_benchmark["sample_count"], len(movement))
            self.assertGreater(pose_benchmark["median_ms"], 0.0)
            self.assertGreaterEqual(
                pose_benchmark["p95_ms"], pose_benchmark["median_ms"]
            )
            self.assertEqual(pose["polar_origin"], "fitted_cursor_rotation_center")
            self.assertGreater(pose["detection_rate"], 0.98)
            self.assertEqual(
                pose["pose_model"], "symmetry_constrained_rigid_polygon"
            )
            self.assertGreater(pose["median_gaussian_fit_r_squared"], 0.90)
            self.assertLess(pose["median_polygon_symmetric_chamfer_px"], 1.0)
            self.assertLess(pose["median_polygon_pixel_agreement_abs_deg"], 5.0)
            self.assertLess(pose["median_gaussian_center_std_deg"], 1.0)
            dynamics = result["cursor_temporal_dynamics"]
            self.assertEqual(
                dynamics["sources"]["ordinary_cruise"]["provenance"][
                    "session_id"
                ],
                "synthetic-ordinary",
            )
            self.assertGreater(
                dynamics["recommended_runtime_envelope"][
                    "calibrated_turn_rate_p99_deg_s"
                ],
                0.0,
            )
            measurements = [
                json.loads(line)
                for line in (Path(temporary) / "cursor_poses.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            expected = np.arange(len(measurements)) * 360.0 / len(measurements)
            measured = np.array(
                [item["relative_rotation_deg"] for item in measurements]
            )
            error = (measured - expected + 180.0) % 360.0 - 180.0
            self.assertLess(float(np.median(np.abs(error))), 5.0)
            for name in declared:
                self.assertGreater((Path(temporary) / name).stat().st_size, 0)

    def test_segment_validation_rejects_missing_or_reversed_ranges(self):
        with self.assertRaisesRegex(ValueError, "rotation_only"):
            _validate_segments({"movement_only": [1, 2]})
        with self.assertRaisesRegex(ValueError, "invalid"):
            _validate_segments({"rotation_only": [3, 2], "movement_only": [4, 5]})

    def test_segment_session_label_is_validated_before_calibration(self):
        class Source:
            stream_id = "main"

            def describe(self):
                return {"type": "test", "stream_id": "main"}

        with tempfile.TemporaryDirectory() as temporary:
            session = Path(temporary) / "wrong-label"
            writer = SessionWriter(
                session,
                [Source()],
                [],
                video_encoding="mjpeg",
                session_context={"segment_label": "movement_only"},
            )
            writer.write_frame(
                FramePacket(
                    "main",
                    np.zeros((180, 220, 3), np.uint8),
                    writer.origin_ns,
                    writer.origin_ns,
                )
            )
            writer.close()
            with self.assertRaisesRegex(ValueError, "Expected a rotation_only"):
                calibrate_segment_sessions(
                    session,
                    session,
                    Path(temporary) / "output",
                )

    def test_route_session_is_accepted_as_ordinary_motion_evidence(self):
        class Source:
            stream_id = "main"

            def describe(self):
                return {"type": "test", "stream_id": "main"}

        def make_session(root, name, label):
            session = root / name
            writer = SessionWriter(
                session,
                [Source()],
                [],
                video_encoding="mjpeg",
                session_context={"segment_label": label},
            )
            writer.write_frame(
                FramePacket(
                    "main",
                    np.zeros((32, 32, 3), np.uint8),
                    writer.origin_ns,
                    writer.origin_ns,
                )
            )
            writer.close()
            manifest_path = session / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["duration_ns"] = 1_000_000_000
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (session / "session_metadata.json").write_text(
                json.dumps({"label": label}), encoding="utf-8"
            )
            return session

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rotation = make_session(root, "rotation", "rotation_only")
            movement = make_session(root, "movement", "movement_only")
            route = make_session(root, "route", "route")
            frames = np.zeros((2, 32, 32, 3), np.uint8)

            with patch(
                "acquisition.minimap_calibration._read_video_segment",
                return_value=(frames, 30.0),
            ), patch(
                "acquisition.minimap_calibration.calibrate_minimap_frames",
                side_effect=lambda *args, **kwargs: kwargs["provenance"],
            ):
                provenance = calibrate_segment_sessions(
                    rotation,
                    movement,
                    root / "output",
                    ordinary_session_path=route,
                )

            self.assertEqual(
                provenance["segment_sessions"]["ordinary_cruise"][
                    "recorded_label"
                ],
                "route",
            )


if __name__ == "__main__":
    unittest.main()
