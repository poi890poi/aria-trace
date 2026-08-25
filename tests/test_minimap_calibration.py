import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from acquisition.minimap_calibration import (
    _validate_segments,
    calibrate_minimap_frames,
    calibrate_segment_sessions,
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


class MinimapCalibrationTests(unittest.TestCase):
    def test_recovers_boundary_pivot_shape_and_evidence(self):
        rotation, movement, boundary_center, pivot, radius = synthetic_frames()
        with tempfile.TemporaryDirectory() as temporary:
            result = calibrate_minimap_frames(rotation, movement, Path(temporary), provenance={"fixture": "synthetic_rotating_map"})
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
            self.assertIn("boundary_evidence_overlay.png", declared)
            self.assertIn("cursor_center_orbit.png", declared)
            self.assertIn("cursor_shape_overlay.png", declared)
            self.assertIn("cursor_shape_polar_correlation.png", declared)
            self.assertIn("cursor_pose_gaussian_fits.png", declared)
            self.assertIn("cursor_pose_polar_samples.png", declared)
            pose = result["cursor_pose_validation"]
            self.assertEqual(pose["polar_origin"], "fitted_cursor_rotation_center")
            self.assertGreater(pose["detection_rate"], 0.98)
            self.assertEqual(
                pose["pose_model"], "symmetry_constrained_rigid_polygon"
            )
            self.assertGreater(pose["median_gaussian_fit_r_squared"], 0.90)
            self.assertLess(pose["median_polygon_symmetric_chamfer_px"], 1.0)
            self.assertLess(pose["median_polygon_pixel_agreement_abs_deg"], 5.0)
            self.assertLess(pose["median_gaussian_center_std_deg"], 1.0)
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


if __name__ == "__main__":
    unittest.main()
