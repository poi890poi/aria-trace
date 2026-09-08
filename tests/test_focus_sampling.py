import math
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from rig_runtime.adapters.android.phone import PhoneMetrics
from rig_runtime.apps.hik_rig_calibration import parser
from rig_runtime.services.calibration.rig.focus_sampling import (
    edge_sampling_theory, focus_theory_lines, resolve_camera_sampling,
)
from rig_runtime.services.calibration.rig.image_quality import (
    measure_slanted_edge_esfr, slanted_edge_sampling_geometry,
)
from rig_runtime.services.calibration.rig.hik.patterns import focus_pattern
from rig_runtime.workflows.hik_rig_calibration import (
    HikCalibrationOptions, HikRigCalibrationSession,
)


def focus_session(directory, **options):
    session = HikRigCalibrationSession(
        HikCalibrationOptions("camera", "phone", Path(directory), **options),
        camera=mock.Mock(), phone=mock.Mock(), target=mock.Mock(), progress=mock.Mock(),
    )
    session.phone_metrics = PhoneMetrics("phone", "test", "test", "14", [700, 900], 400, 60)
    session.visible_region = {"safe_xywh": [100, 100, 500, 700]}
    session.geometry = SimpleNamespace(matrix_3x3=np.eye(3), inverse_matrix_3x3=np.eye(3), confidence=1.0)
    session._raw_roi_for_screen_region = mock.Mock(side_effect=lambda rect, *_args, **_kw: list(map(int, rect)))
    return session


class FocusSamplingTests(unittest.TestCase):
    def test_axis_and_diagonal_color_limits(self):
        axis = edge_sampling_theory(0, np.eye(2), "diamond-pentile", "bayer")
        self.assertEqual(axis["panel"]["diamond-pentile"], {"green": .5, "red_blue": .5})
        self.assertEqual(axis["camera"]["bayer"], {"green": .5, "red_blue": .25})
        diagonal = edge_sampling_theory(45, np.eye(2), "diamond-pentile", "bayer")
        self.assertAlmostEqual(diagonal["panel"]["diamond-pentile"]["green"], math.sqrt(.5))
        self.assertAlmostEqual(diagonal["panel"]["diamond-pentile"]["red_blue"], math.sqrt(.125))
        self.assertAlmostEqual(diagonal["camera"]["bayer"]["green"], math.sqrt(.125))
        self.assertAlmostEqual(diagonal["camera"]["bayer"]["red_blue"], math.sqrt(.125))

    def test_magnification_and_camera_rotation_change_bottleneck(self):
        # Two camera samples per display pixel: Bayer R/B rises from .25 to .5.
        large = edge_sampling_theory(0, np.eye(2) / 2, "rgb-stripe", "bayer")
        self.assertEqual(large["camera"]["bayer"]["red_blue"], .5)
        self.assertEqual(large["combined"][0]["ideal_luminance_mtf50"], .5)
        rotation = np.array([[1, -1], [1, 1]]) / math.sqrt(2)
        rotated = edge_sampling_theory(0, rotation, "rgb-stripe", "bayer")
        self.assertAlmostEqual(rotated["combined"][0]["ideal_luminance_mtf50"], math.sqrt(.125))

    def test_fresh_angles_against_independent_reciprocal_lattice(self):
        # Brillouin boundary found by enumerating reciprocal-lattice vectors,
        # rather than reusing the production max/sum formulas.
        def boundary(basis, vector):
            reciprocal = np.linalg.inv(basis).T
            candidates = []
            for x in range(-3, 4):
                for y in range(-3, 4):
                    g = reciprocal @ [x, y]
                    dot = abs(np.dot(g, vector))
                    if dot > 1e-10:
                        candidates.append(np.dot(g, g) / (2 * dot))
            return min(candidates)

        rng = np.random.default_rng(20260908)
        for angle in rng.uniform(-179, 179, 24):
            jacobian = np.array([[.81, .23], [-.17, 1.31]])
            n = [-math.sin(math.radians(angle)), math.cos(math.radians(angle))]
            q = jacobian.T @ n
            result = edge_sampling_theory(angle, jacobian, "diamond-pentile", "bayer")
            self.assertAlmostEqual(result["camera"]["bayer"]["green"], boundary([[1, 1], [1, -1]], q))
            self.assertAlmostEqual(result["camera"]["bayer"]["red_blue"], boundary(np.eye(2) * 2, q))
            self.assertAlmostEqual(result["panel"]["diamond-pentile"]["red_blue"], boundary([[1, 1], [1, -1]], n))

    def test_bayer_alias_witness(self):
        # Different diagonal sinusoids alias on green sites once outside its
        # diamond Nyquist domain; this is a property of samples, not our model.
        xy = np.array([(x, y) for x in range(12) for y in range(12) if (x + y) % 2 == 0])
        a = np.cos(2 * np.pi * (xy @ [.3, .3]))
        b = np.cos(2 * np.pi * (xy @ [-.2, -.2]))
        np.testing.assert_allclose(a, b, atol=1e-13)
        limit = edge_sampling_theory(-45, np.eye(2), "rgb-stripe", "bayer")["camera"]["bayer"]["green"]
        self.assertLess(limit, np.linalg.norm([.3, .3]))

    def test_pixel_format_is_evidence_only_for_recognized_bayer(self):
        for value in (0x01080008, 0x01080009, 0x0108000A, 0x0108000B, "BayerRG12Packed", "BayerBG8"):
            self.assertEqual(resolve_camera_sampling("auto", value), ("bayer", "PixelFormat"))
        for value in (None, "BGR8Packed", "RGB8", "Mono8", "QuadBayerRG8", 0, 0x01080001):
            self.assertEqual(resolve_camera_sampling("auto", value)[0], "unknown")
        self.assertEqual(resolve_camera_sampling("full-grid", "BayerRG8"), ("full-grid", "configured"))

    def test_unknown_scenarios_and_invalid_geometry(self):
        theory = edge_sampling_theory(5, np.eye(2))
        self.assertEqual(len(theory["combined"]), 4)
        text = "\n".join(focus_theory_lines([{"sampling_theory": theory}] * 4, "unknown"))
        for label in ("Diamond PenTile?", "RGB stripe?", "Bayer?", "full grid?", "unknown layout", "actual may differ"):
            self.assertIn(label, text)
        for bad in (np.zeros((2, 2)), [[1, 0], [0, float("nan")]], np.eye(3)):
            with self.assertRaises(ValueError):
                edge_sampling_theory(0, bad)
        self.assertIn("unavailable", focus_theory_lines([{}] * 4, "unknown")[0])

    def test_sampling_geometry_handles_crop_translation_and_projective_scale(self):
        transform = np.array([[.7, .1, 11], [-.05, .8, 13], [.0002, -.0001, 1.]])
        base = slanted_edge_sampling_geometry(transform, [40, 50, 100, 100], 15)
        # Moving camera origin by an ROI offset preserves the local frequency.
        shift = np.array([[1, 0, 23], [0, 1, 17], [0, 0, 1]])
        crop = slanted_edge_sampling_geometry(transform @ shift, [40, 50, 100, 100], 15)
        np.testing.assert_allclose(base["jacobian_display_px_per_camera_px"], crop["jacobian_display_px_per_camera_px"], atol=1e-12)

    def test_lens_distortion_is_included_in_local_frequency_mapping(self):
        # Independent affine stand-in isolates the distortion handoff.
        with mock.patch("rig_runtime.services.calibration.rig.distortion.distort_pixel_points", side_effect=lambda points, _: np.asarray(points) * 2), mock.patch("rig_runtime.services.calibration.rig.distortion.undistort_pixel_points", side_effect=lambda points, _: np.asarray(points) / 2):
            sampling = slanted_edge_sampling_geometry(np.eye(3), [0, 0, 100, 100], 0, {"test": True})
        np.testing.assert_allclose(sampling["jacobian_display_px_per_camera_px"], np.eye(2) / 2)

    def test_focus_measurement_preserves_esfr_and_theory_survives_bad_image(self):
        with tempfile.TemporaryDirectory() as directory:
            session = focus_session(directory, focus_panel_layout="diamond-pentile")
            session.camera_controls = {"genicam": {"PixelFormat": {"value": 0x01080009}}}
            frame = cv2.GaussianBlur(focus_pattern((700, 900), session.visible_region["safe_xywh"]), (0, 0), 1.2)
            row = session._focus_measurement(frame)
            for edge in row["edges"]:
                expected, _ = measure_slanted_edge_esfr(frame, np.eye(3), edge["rect_screen_xywh"], edge["angle_deg"])
                self.assertEqual(edge["mtf50_cycles_per_display_pixel"], expected["display_referred"]["mtf50"])
            bad = session._focus_measurement(np.full_like(frame, 128))
            self.assertIsNone(bad["mtf50"])
            self.assertEqual([e["sampling_theory"] for e in row["edges"]], [e["sampling_theory"] for e in bad["edges"]])

    def test_cli_defaults_and_explicit_layouts(self):
        self.assertEqual(parser().parse_args([]).focus_panel_layout, "unknown")
        args = parser().parse_args(["--focus-panel-layout", "diamond-pentile", "--focus-camera-sampling", "bayer"])
        self.assertEqual((args.focus_panel_layout, args.focus_camera_sampling), ("diamond-pentile", "bayer"))

    def test_live_focus_theory_toggle_preserves_details_and_save(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            session = focus_session(directory)
            frame = cv2.GaussianBlur(focus_pattern((700, 900), session.visible_region["safe_xywh"]), (0, 0), 1.2)
            session._read_camera = mock.Mock(return_value=SimpleNamespace(image=frame))
            session._wait_painted = mock.Mock()
            session._close_preview = mock.Mock()
            session._update_focus_panel_axis = mock.Mock(return_value={"reason": "test"})
            session._desktop_work_area = mock.Mock(return_value=(1920, 1080))
            for name in ("namedWindow", "resizeWindow", "imshow", "destroyWindow"):
                stack.enter_context(mock.patch("cv2." + name))
            stack.enter_context(mock.patch("cv2.waitKey", side_effect=[ord("t"), ord("s")]))
            stack.enter_context(mock.patch("rig_runtime.workflows.hik_rig_calibration.detect_focus_pose_frame", side_effect=ValueError("test pose unavailable")))
            rendered = []
            original = session._draw_text_panel

            def capture(*args):
                result = original(*args)
                rendered.append(result)
                return result

            session._draw_text_panel = capture
            self.assertEqual(session.focus_loop(), "save")
            self.assertIn("Theory:", "\n".join(rendered[0]))
            self.assertIn("measured range", "\n".join(rendered[0]))
            self.assertIn("pose unavailable", "\n".join(rendered[1]))
            self.assertNotIn("Theory:", "\n".join(rendered[1]))
            self.assertIn("sampling_theory", session.focus_history[0]["edges"][0])
            self.assertNotIn("sampling_theory", session.focus_history[1]["edges"][0])


if __name__ == "__main__":
    unittest.main()
