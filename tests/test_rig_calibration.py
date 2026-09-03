import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from acquisition.rig_calibration import (
    CharucoLayout,
    ControlEvent,
    DataMatrixObservation,
    FrameNormalizer,
    SignalObservation,
    aggregate_esfr_measurements,
    aggregate_feature_matching,
    build_calibration,
    detect_charuco_correspondences,
    decode_data_matrix_payloads,
    estimate_latency,
    estimate_paired_delay,
    estimate_screen_geometry,
    encode_data_matrix_modules,
    evaluate_feature_matching,
    export_spatial_fragment,
    extract_one_to_one_patch,
    grade_data_matrix_decode,
    generate_charuco_target,
    generate_feature_target,
    generate_slanted_edge_target,
    load_calibration_yaml,
    measure_slanted_edge_esfr,
    nearest_neighbor_magnify,
    render_esfr_curve,
    render_feature_matching_curve,
    render_feature_matching_overlay,
    render_geometry_overlay,
    render_latency_timeline,
    render_data_matrix_target,
    select_visible_quality_region,
    summarize_data_matrix_decode_sweep,
    validate_spatial_fragment,
    write_calibration_bundle,
)
from acquisition.rig_calibration.geometry import transform_points
from rig_runtime.adapters.android.display import Presentation
from rig_runtime.services.calibration.rig.presentation import (
    matching_paint_acknowledgement,
    presentation_freshness_boundary_ns,
    sample_host_time_ns,
)


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


class PresentationFreshnessTests(unittest.TestCase):
    def test_acknowledgement_keeps_canonical_and_rotated_surface_sizes_separate(self):
        presentation = Presentation("atlas", "image", 100, 7)
        telemetry = {
            "acknowledgements": [
                {
                    "revision": 7,
                    "painted": True,
                    "canonical_target_size_px": [1080, 2400],
                    "logical_target_size_px": [2400, 1080],
                    "canvas_width": 2400,
                    "canvas_height": 1080,
                    "server_receive_time_ns": 140,
                }
            ]
        }
        acknowledgement = matching_paint_acknowledgement(
            telemetry, presentation, [1080, 2400]
        )
        self.assertIsNotNone(acknowledgement)
        self.assertEqual(
            presentation_freshness_boundary_ns(presentation, acknowledgement), 140
        )

    def test_browser_surface_must_match_canonical_target_raster(self):
        presentation = Presentation("atlas", "image", 100, 2)
        telemetry = {
            "acknowledgements": [
                {
                    "revision": 2,
                    "painted": True,
                    "canvas_width": 1079,
                    "canvas_height": 2400,
                    "image_natural_width": 1080,
                    "image_natural_height": 2400,
                    "server_receive_time_ns": 120,
                }
            ]
        }
        self.assertIsNone(
            matching_paint_acknowledgement(
                telemetry, presentation, [1080, 2400]
            )
        )

    def test_device_clock_requires_host_receive_timestamp(self):
        sample = type(
            "Sample",
            (),
            {"time_ns": 10, "clock_id": "hik_device_ticks", "receive_time_ns": None},
        )()
        with self.assertRaisesRegex(ValueError, "host-monotonic"):
            sample_host_time_ns(sample)
        sample.receive_time_ns = 25
        self.assertEqual(sample_host_time_ns(sample), (25, "host_receive_monotonic"))


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

    def test_partial_display_atlas_selects_only_camera_visible_quality_region(self):
        camera_size = (400, 300)
        screen_size = (800, 600)
        camera_to_screen = np.asarray(
            [[1.0, 0.0, 100.0], [0.0, 1.0, 50.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        xx, yy = np.meshgrid(np.linspace(20, 380, 7), np.linspace(20, 280, 6))
        camera_points = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
        screen_points = transform_points(camera_points, camera_to_screen)
        required = np.asarray(
            [[140, 90], [460, 90], [460, 310], [140, 310]], dtype=np.float64
        )
        geometry = estimate_screen_geometry(
            camera_points,
            screen_points,
            camera_size,
            screen_size,
            required_region_screen_xy=required,
        )
        self.assertLess(geometry.metrics["screen_coverage"], 0.30)
        self.assertLess(geometry.metrics["screen_view_iou"], 0.30)
        region = select_visible_quality_region(
            camera_size,
            screen_size,
            geometry.matrix_3x3,
            required_region_screen_xy=required,
            margin_display_px=8,
        )
        x, y, width, height = region["xywh"]
        self.assertEqual(width, height)
        self.assertGreaterEqual(width, 64)
        self.assertGreaterEqual(x, 108)
        self.assertGreaterEqual(y, 58)
        self.assertLessEqual(x + width, 492)
        self.assertLessEqual(y + height, 342)

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
            self.assertTrue(text.startswith("# IRIS camera-to-phone rig calibration."))
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


class RigDataMatrixDecodeTests(unittest.TestCase):
    @staticmethod
    def decoder(image):
        marker = int(np.asarray(image).reshape(-1)[0])
        if marker == 1:
            return ["A7K2"]
        if marker == 2:
            return ["OLD1"]
        return []

    @staticmethod
    def encoder(_payload):
        yy, xx = np.indices((10, 10))
        return np.where((xx + yy) % 2, 0, 255).astype(np.uint8)

    def test_one_image_returns_standard_binary_decode_grade(self):
        passed = grade_data_matrix_decode(
            np.ones((8, 8), dtype=np.uint8), "A7K2", decoder=self.decoder
        )
        failed = grade_data_matrix_decode(
            np.zeros((8, 8), dtype=np.uint8), "A7K2", decoder=self.decoder
        )
        stale = grade_data_matrix_decode(
            np.full((8, 8), 2, dtype=np.uint8), "A7K2", decoder=self.decoder
        )
        self.assertEqual((passed["grade"], passed["grade_letter"]), (4.0, "A"))
        self.assertEqual((failed["grade"], failed["grade_letter"]), (0.0, "F"))
        self.assertFalse(stale["eligible_for_requested_target"])
        self.assertEqual(stale["grade"], 4.0)

    def test_sweep_retains_grades_without_an_invented_pass_rate(self):
        observations = [
            DataMatrixObservation(np.ones((4, 4), np.uint8), "A7K2", 1, "one-a"),
            DataMatrixObservation(np.zeros((4, 4), np.uint8), "A7K2", 1, "one-f"),
            DataMatrixObservation(np.ones((4, 4), np.uint8), "A7K2", 2, "two-a"),
            DataMatrixObservation(np.full((4, 4), 2, np.uint8), "A7K2", 2, "stale"),
        ]
        value = summarize_data_matrix_decode_sweep(observations, decoder=self.decoder)
        self.assertEqual(value["parameter"], "Decode")
        self.assertEqual(value["aggregation"], "none_standard_grades_reported_as_counts")
        self.assertNotIn("score", value)
        self.assertEqual(value["module_width_results"][0]["decode_grade_4_A_count"], 1)
        self.assertEqual(value["module_width_results"][0]["decode_grade_0_F_count"], 1)
        self.assertEqual(
            value["module_width_results"][1]["ineligible_wrong_target_count"], 1
        )

    def test_target_uses_integer_modules_in_one_fixed_patch(self):
        target = render_data_matrix_target(
            (100, 80), (20, 10, 50, 50), "A7K2", 3, encoder=self.encoder
        )
        self.assertEqual(target.symbol_modules_xy, (10, 10))
        self.assertEqual(target.target_rect_screen_xywh, (20, 10, 50, 50))
        left, top, width, height = target.symbol_rect_screen_xywh
        self.assertEqual((width, height), (36, 36))
        raster = target.image[top : top + height, left : left + width, 0]
        for row in range(12):
            for column in range(12):
                cell = raster[row * 3 : (row + 1) * 3, column * 3 : (column + 1) * 3]
                self.assertEqual(int(cell.min()), int(cell.max()))

    def test_encoder_supports_legacy_zxing_writer_signature(self):
        class LegacyBarcode:
            @staticmethod
            def to_image():
                yy, xx = np.indices((12, 12))
                return np.where((xx + yy) % 2, 0, 255).astype(np.uint8)

        class LegacyZxing:
            class BarcodeFormat:
                DataMatrix = object()

            @staticmethod
            def create_barcode(payload, barcode_format):
                if payload != "A7K2":
                    raise AssertionError("unexpected payload")
                if barcode_format is not LegacyZxing.BarcodeFormat.DataMatrix:
                    raise AssertionError("unexpected barcode format")
                return LegacyBarcode()

        with mock.patch(
            "rig_runtime.services.calibration.rig.data_matrix_readability._zxing_module",
            return_value=LegacyZxing,
        ):
            modules = encode_data_matrix_modules("A7K2")
        self.assertEqual(modules.shape, (12, 12))
        self.assertEqual(set(np.unique(modules)), {0, 255})

    def test_decoder_supports_legacy_zxing_without_try_invert(self):
        class Result:
            text = "A7K2"

        class LegacyZxing:
            class BarcodeFormat:
                DataMatrix = object()

            @staticmethod
            def read_barcodes(
                image,
                formats=None,
                try_rotate=True,
                try_downscale=True,
                return_errors=False,
            ):
                if image.shape != (12, 12):
                    raise AssertionError("unexpected image")
                return [Result()]

        with mock.patch(
            "rig_runtime.services.calibration.rig.data_matrix_readability._zxing_module",
            return_value=LegacyZxing,
        ):
            decoded = decode_data_matrix_payloads(np.zeros((12, 12), np.uint8))
        self.assertEqual(decoded, ["A7K2"])

    def test_decoder_falls_back_to_thresholded_image_after_native_misses(self):
        class Result:
            text = "A7K2"

        class Zxing:
            class BarcodeFormat:
                DataMatrix = object()

            class Binarizer:
                GlobalHistogram = object()

            @staticmethod
            def read_barcodes(
                image,
                formats=None,
                try_rotate=True,
                try_downscale=True,
                try_invert=False,
                binarizer=None,
                return_errors=False,
            ):
                is_binary = image.ndim == 2 and set(np.unique(image)) <= {0, 255}
                return [Result()] if is_binary and not try_downscale else []

        gradient = np.tile(np.arange(32, dtype=np.uint8), (32, 1)) * 8
        image = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
        with mock.patch(
            "rig_runtime.services.calibration.rig.data_matrix_readability._zxing_module",
            return_value=Zxing,
        ):
            decoded = decode_data_matrix_payloads(image)
        self.assertEqual(decoded, ["A7K2"])


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


class RigImageQualityTests(unittest.TestCase):
    def test_display_referred_esfr_uses_prewarp_oversampled_camera_pixels(self):
        screen_size = (320, 240)
        rect = (40, 40, 240, 160)
        target = generate_slanted_edge_target(
            screen_size, rect, edge_angle_deg=5.0, channel="luminance"
        )
        observed = cv2.resize(target, (640, 480), interpolation=cv2.INTER_NEAREST)
        observed = cv2.GaussianBlur(observed, (0, 0), 1.2)
        camera_to_screen = np.asarray(
            [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        result, evidence = measure_slanted_edge_esfr(
            observed,
            camera_to_screen,
            rect,
            edge_angle_deg=5.0,
            channel="luminance",
            display_pixel_pitch_mm_xy=(0.0625, 0.0625),
        )
        display = result["display_referred"]
        self.assertEqual(result["measurement_input_space"], "camera_pre_homography_px")
        self.assertEqual(display["spatial_frequency_unit"], "cycles_per_display_pixel")
        self.assertAlmostEqual(
            result["sampling"]["camera_px_per_display_px_along_edge_normal"],
            2.0,
            places=2,
        )
        self.assertIsNotNone(display["mtf50"])
        self.assertAlmostEqual(
            result["display_physical"]["mtf50"], display["mtf50"] / 0.0625
        )
        self.assertEqual(
            result["display_physical"]["spatial_frequency_unit"],
            "line_pairs_per_mm",
        )
        self.assertLessEqual(max(display["frequency"]), 0.5)
        self.assertGreater(evidence.size, 0)
        aggregate = aggregate_esfr_measurements([result, result])
        curve = render_esfr_curve(aggregate)
        self.assertEqual(curve.shape, (560, 1000, 3))
        self.assertGreater(int(np.max(curve)), 22)

    def test_ground_truth_feature_metrics_and_mma(self):
        screen_size = (480, 360)
        rect = (40, 40, 400, 280)
        reference = generate_feature_target(screen_size, rect, seed=91)
        transform = np.eye(3, dtype=np.float64)
        first = evaluate_feature_matching(reference, reference, transform, rect)
        second = evaluate_feature_matching(
            generate_feature_target(screen_size, rect, seed=92),
            generate_feature_target(screen_size, rect, seed=92),
            transform,
            rect,
        )
        self.assertGreater(first["evaluated_match_count"], 20)
        self.assertGreater(first["repeatability_by_threshold_px"][1], 0.90)
        self.assertGreater(first["mma_by_threshold_px"][1], 0.90)
        self.assertEqual(first["downstream_geometry"]["status"], "estimated")
        aggregate = aggregate_feature_matching([first, second])
        self.assertGreater(aggregate["matching_score_by_threshold_px"][3], 0.80)
        curve = render_feature_matching_curve(aggregate)
        self.assertEqual(curve.shape, (560, 1000, 3))
        self.assertGreater(int(np.max(curve)), 22)
        overlay = render_feature_matching_overlay(reference, reference, first)
        self.assertEqual(overlay.ndim, 3)
        self.assertGreater(int(np.max(overlay)), 22)


if __name__ == "__main__":
    unittest.main()
