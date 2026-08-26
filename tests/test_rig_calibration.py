import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from acquisition.rig_calibration import (
    CharucoLayout,
    ControlEvent,
    FrameNormalizer,
    MatchResult,
    MatchTrial,
    PhaseCorrelationMatcher,
    SignalObservation,
    build_calibration,
    detect_charuco_correspondences,
    estimate_latency,
    estimate_paired_delay,
    estimate_screen_geometry,
    evaluate_matchability,
    export_spatial_fragment,
    extract_one_to_one_patch,
    generate_band_limited_target,
    generate_charuco_target,
    load_calibration_yaml,
    nearest_neighbor_magnify,
    render_geometry_overlay,
    render_latency_timeline,
    render_matchability_curve,
    validate_spatial_fragment,
    warp_target,
    write_calibration_bundle,
)
from acquisition.rig_calibration.geometry import transform_points


def synthetic_correspondences():
    camera_size = (640, 480)
    screen_size = (320, 640)
    screen_to_camera = np.asarray(
        [
            [0.72, 0.025, 180.0],
            [0.018, 0.55, 45.0],
            [0.00008, 0.00004, 1.0],
        ],
        dtype=np.float64,
    )
    xx, yy = np.meshgrid(np.linspace(20, 300, 6), np.linspace(20, 620, 8))
    screen_points = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    camera_points = transform_points(screen_points, screen_to_camera)
    return camera_points, screen_points, camera_size, screen_size, screen_to_camera


def calibration_bundle():
    camera_points, screen_points, camera_size, screen_size, _ = synthetic_correspondences()
    required = np.asarray([[20, 30], [140, 30], [140, 150], [20, 150]], np.float64)
    value, geometry, mask = build_calibration(
        "synthetic-rig-001",
        camera_points,
        screen_points,
        camera_size,
        screen_size,
        "aria://rig/synthetic/camera/undistorted",
        "aria://device/synthetic/screen/portrait/layout-320x640",
        required_region_screen_xy=required,
        rig={
            "rig_id": "synthetic",
            "camera": {
                "hardware_id": "synthetic-camera",
                "width_px": camera_size[0],
                "height_px": camera_size[1],
                "fps": 30.0,
            },
            "phone": {"orientation": "portrait"},
        },
        required_roi={"kind": "test-region", "polygon_xy": required.tolist()},
        status="accepted",
    )
    return value, geometry, mask


class RigGeometryTests(unittest.TestCase):
    def test_homography_coverage_and_required_region(self):
        camera_points, screen_points, camera_size, screen_size, _ = synthetic_correspondences()
        required = np.asarray([[20, 30], [140, 30], [140, 150], [20, 150]], np.float64)
        result = estimate_screen_geometry(
            camera_points,
            screen_points,
            camera_size,
            screen_size,
            required_region_screen_xy=required,
        )
        projected = transform_points(camera_points, result.matrix_3x3)
        self.assertLess(float(np.max(np.linalg.norm(projected - screen_points, axis=1))), 1.0e-4)
        self.assertGreater(result.metrics["screen_coverage"], 0.999)
        self.assertGreater(result.metrics["required_region_coverage"], 0.999)
        self.assertLess(result.metrics["camera_utilization"], 0.75)
        self.assertGreater(result.confidence, 0.85)
        self.assertEqual(result.warnings, ())

    def test_geometry_overlay_is_inspectable(self):
        value, geometry, _ = calibration_bundle()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        overlay = render_geometry_overlay(image, geometry)
        self.assertEqual(overlay.shape, image.shape)
        self.assertGreater(int(np.max(overlay)), 0)
        self.assertEqual(value["geometry"]["mode"], "homography_only")

    def test_charuco_dependency_is_explicit(self):
        layout = CharucoLayout((320, 640), squares_x=5, squares_y=9)
        if hasattr(cv2, "aruco"):
            target = generate_charuco_target(layout)
            self.assertEqual(target.shape, (640, 320, 3))
            detection = detect_charuco_correspondences(target, layout)
            self.assertGreaterEqual(detection["corner_count"], 12)
            self.assertEqual(
                detection["camera_points_xy"].shape,
                detection["screen_points_xy"].shape,
            )
        else:
            with self.assertRaisesRegex(RuntimeError, "opencv-contrib"):
                generate_charuco_target(layout)


class RigArtifactTests(unittest.TestCase):
    def test_commented_yaml_normalizer_and_spatial_export(self):
        value, _, mask = calibration_bundle()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            yaml_path = write_calibration_bundle(root, value, mask)
            text = yaml_path.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("# AriaTrace camera-to-phone rig calibration."))
            self.assertIn("# Stable downstream contract", text)
            loaded = load_calibration_yaml(yaml_path)
            normalizer = FrameNormalizer(loaded, root)
            sample = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.circle(sample, (250, 150), 12, (20, 220, 240), -1)
            normalized = normalizer.normalize(sample)
            self.assertEqual(normalized.shape[:2], (640, 320))
            camera_points, screen_points, _, _, _ = synthetic_correspondences()
            measured = normalizer.input_to_screen_points(camera_points[:4])
            self.assertLess(float(np.max(np.abs(measured - screen_points[:4]))), 1.0e-4)
            fragment = export_spatial_fragment(loaded)
            validate_spatial_fragment(fragment)
            self.assertEqual(len(fragment["frames"]), 3)
            self.assertEqual(len(fragment["transforms"]), 2)
            self.assertEqual(
                fragment["transforms"][0]["from_frame"],
                loaded["normalization"]["input_frame_id"],
            )

    def test_one_to_one_patch_does_not_interpolate(self):
        image = np.arange(10 * 12, dtype=np.uint8).reshape((10, 12))
        patch, descriptor = extract_one_to_one_patch(image, (5, 4), (5, 3))
        np.testing.assert_array_equal(patch, image[3:6, 3:8])
        self.assertEqual(descriptor["zoom"], "1:1")
        self.assertEqual(descriptor["interpolation"], "none")
        magnified = nearest_neighbor_magnify(patch, 3)
        self.assertEqual(magnified.shape, (9, 15))
        self.assertTrue(np.all(magnified[0:3, 0:3] == patch[0, 0]))


class RigLatencyTests(unittest.TestCase):
    @staticmethod
    def observations(events, latency_ns, source_id):
        result = []
        end = events[-1].time_ns + 240_000_000
        event_index = -1
        for time_ns in range(0, end + 1, 20_000_000):
            while (
                event_index + 1 < len(events)
                and events[event_index + 1].time_ns + latency_ns <= time_ns
            ):
                event_index += 1
            state = events[event_index].state if event_index >= 0 else "B"
            other = "B" if state == "A" else "A"
            result.append(
                SignalObservation(
                    time_ns,
                    {state: 0.96, other: 0.04},
                    source_id=source_id,
                )
            )
        return result

    def test_alternating_latency_and_paired_endpoints(self):
        events = [
            ControlEvent("token-{:02d}".format(index), "A" if index % 2 == 0 else "B", index * 300_000_000)
            for index in range(12)
        ]
        adb = estimate_latency(events, self.observations(events, 40_000_000, "adb"))
        camera = estimate_latency(events, self.observations(events, 80_000_000, "camera"))
        self.assertEqual(adb["accepted_transitions"], len(events))
        self.assertEqual(camera["accepted_transitions"], len(events))
        self.assertAlmostEqual(camera["median_ns"], 80_000_000, delta=1)
        self.assertAlmostEqual(adb["median_ns"], 40_000_000, delta=1)
        self.assertIsNotNone(camera["cross_correlation"]["lag_ns"])
        paired = estimate_paired_delay(adb, camera)
        self.assertEqual(paired["paired_transition_count"], len(events))
        self.assertAlmostEqual(paired["median_ns"], 40_000_000, delta=1)
        timeline = render_latency_timeline(camera)
        self.assertEqual(timeline.shape, (500, 1000, 3))
        self.assertGreater(int(np.max(timeline)), 0)

    def test_different_clocks_require_mapping(self):
        events = [ControlEvent("a", "A", 0, "control"), ControlEvent("b", "B", 300, "control")]
        observations = [SignalObservation(100, {"A": 1.0}, "camera")]
        with self.assertRaisesRegex(ValueError, "explicit clock transform"):
            estimate_latency(events, observations, stable_observations=1)


class RigMatchabilityTests(unittest.TestCase):
    def test_phase_correlation_and_target_generation(self):
        target = generate_band_limited_target((256, 192), 20, 17)
        observed = warp_target(target, (4.0, -3.0))
        result = PhaseCorrelationMatcher(minimum_response=0.01).match(target, observed)
        self.assertLess(np.linalg.norm(np.asarray(result.translation_xy) - np.asarray([4.0, -3.0])), 0.5)

    def test_conservative_resolution_score(self):
        trials = []
        results = []
        for cells in (8, 16, 32):
            for index in range(20):
                image = np.zeros((100, 100, 3), dtype=np.uint8)
                trials.append(
                    MatchTrial(
                        cells,
                        image,
                        image,
                        expected_translation_xy=(2.0, -1.0),
                        expected_rotation_deg=0.5,
                        trial_id="{}-{}".format(cells, index),
                    )
                )
                failed = cells == 32 and index < 2
                results.append(
                    MatchResult(
                        (2.0, -1.0),
                        0.5,
                        confidence=0.9,
                        ambiguous=failed,
                    )
                )

        class QueueMatcher:
            def __init__(self, values):
                self.values = iter(values)

            def match(self, reference, observed):
                return next(self.values)

        score = evaluate_matchability(
            trials, QueueMatcher(results), bootstrap_samples=30
        )
        self.assertEqual(score["metric"], "MR95-20")
        self.assertEqual(score["primary_cells_across_patch"], 16)
        self.assertAlmostEqual(score["smallest_matchable_detail_mm"], 1.25)
        curve = render_matchability_curve(score)
        self.assertEqual(curve.shape, (480, 900, 3))
        self.assertGreater(int(np.max(curve)), 0)


if __name__ == "__main__":
    unittest.main()
