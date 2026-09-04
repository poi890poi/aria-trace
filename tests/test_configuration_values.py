from __future__ import annotations

import unittest
from pathlib import Path

from rig_runtime.adapters.filesystem.profile_registry import AdapterRequest
from rig_runtime.adapters.hik.compat import _adapter_request_from_config
from rig_runtime.apps import hik_stream
from rig_runtime.apps import hik_rig_calibration
from rig_runtime.adapters.android.cursor_orbit import CursorOrbitTouchPlan
from rig_runtime.adapters.android.zigzag import ZigzagTouchPlan
from rig_runtime.domain.configuration import (
    ACQUISITION_DEFAULTS,
    ADAPTER_DEFAULTS,
    CURSOR_ORBIT_PLAN_DEFAULTS,
    RIG_CALIBRATION_DEFAULTS,
    ZIGZAG_PLAN_DEFAULTS,
    ResolvedAdapterPlan,
)
from rig_runtime.workflows import profile_management
from rig_runtime.workflows import minimap_capture
from rig_runtime.workflows.hik_rig_calibration import HikCalibrationOptions


class AdapterConfigurationTests(unittest.TestCase):
    def test_every_public_entry_point_uses_the_domain_defaults(self):
        request = AdapterRequest()
        facade = _adapter_request_from_config({})
        demo = hik_stream.parser().parse_args([])
        manager = profile_management.parser().parse_args(["resolve"])

        for actual in (request, facade):
            self.assertEqual(ADAPTER_DEFAULTS.mode, actual.mode)
            self.assertEqual(ADAPTER_DEFAULTS.normalization, actual.normalization)
            self.assertEqual(ADAPTER_DEFAULTS.color_policy, actual.color_policy)
            self.assertEqual(ADAPTER_DEFAULTS.roi_policy, actual.roi_policy)
            self.assertEqual(ADAPTER_DEFAULTS.mask_policy, actual.mask_policy)
            self.assertEqual(
                ADAPTER_DEFAULTS.minimap_margin_px, actual.minimap_margin_px
            )
            self.assertEqual(
                ADAPTER_DEFAULTS.orientation_behavior,
                actual.orientation_behavior,
            )
            self.assertEqual(
                ADAPTER_DEFAULTS.rotate_degrees_clockwise, actual.rotate
            )

        for actual in (demo, manager):
            self.assertEqual(ADAPTER_DEFAULTS.mode, actual.mode)
            self.assertEqual(ADAPTER_DEFAULTS.color_policy, actual.color_policy)
            self.assertEqual(ADAPTER_DEFAULTS.mask_policy, actual.mask_policy)
            self.assertEqual(
                ADAPTER_DEFAULTS.minimap_margin_px, actual.minimap_margin_px
            )
            self.assertEqual(
                ADAPTER_DEFAULTS.orientation_behavior,
                actual.orientation_behavior,
            )
            self.assertEqual(
                ADAPTER_DEFAULTS.rotate_degrees_clockwise, actual.rotate
            )

    def test_resolved_plan_composes_profile_and_manual_rotation_once(self):
        common = {
            "mode": "dual",
            "normalization": "auto",
            "color_order": "RGB",
            "color_policy": "rig_locked",
            "roi_policy": "auto",
            "mask_policy": "none",
            "minimap_margin_px": 6,
            "profile_game_upright_quarter_turns_clockwise": 3,
            "manual_rotate_degrees_clockwise": 90,
            "initialization_surface_quarter_turns_clockwise_from_natural": 1,
            "game_model": {},
        }
        projected = ResolvedAdapterPlan.create(
            orientation_behavior="projection", **common
        )
        unchanged = ResolvedAdapterPlan.create(
            orientation_behavior="as_is", **common
        )

        self.assertEqual(3, projected.profile_game_upright_quarter_turns_clockwise)
        self.assertEqual(0, projected.game_upright_quarter_turns_clockwise)
        self.assertEqual(1, unchanged.game_upright_quarter_turns_clockwise)
        self.assertEqual(
            projected.game_upright_quarter_turns_clockwise,
            projected.as_dict()["game_upright_quarter_turns_clockwise"],
        )


class RigCalibrationConfigurationTests(unittest.TestCase):
    def test_cli_and_options_use_one_rig_calibration_policy(self):
        cli = hik_rig_calibration.parser().parse_args([])
        options = HikCalibrationOptions(
            camera_id="camera",
            phone_serial="phone",
            output_directory=Path("unused"),
        )
        pairs = (
            (cli.camera_width, options.camera_width_px, "camera_width_px"),
            (cli.camera_height, options.camera_height_px, "camera_height_px"),
            (cli.camera_fps, options.camera_fps, "camera_fps"),
            (cli.target_port, options.target_port, "target_port"),
            (cli.target_presenter, options.target_presenter, "target_presenter"),
            (cli.panel_scale, options.panel_scale_mode, "panel_scale_mode"),
            (
                cli.operation_timeout_seconds,
                options.operation_timeout_seconds,
                "operation_timeout_seconds",
            ),
            (
                cli.max_shutter_multiplier,
                options.maximum_shutter_multiplier,
                "maximum_shutter_multiplier",
            ),
            (
                cli.max_exposure_periods,
                options.maximum_exposure_periods,
                "maximum_exposure_periods",
            ),
            (cli.max_auto_gain_db, options.maximum_auto_gain_db, "maximum_auto_gain_db"),
            (cli.geometry_frames, options.geometry_frames, "geometry_frames"),
            (
                cli.visible_screen_margin_px,
                options.visible_screen_margin_px,
                "visible_screen_margin_px",
            ),
            (cli.settle_frames, options.settle_frames, "settle_frames"),
            (
                cli.distortion_correction,
                options.distortion_correction,
                "distortion_correction",
            ),
            (
                cli.distortion_views,
                options.distortion_view_count,
                "distortion_view_count",
            ),
            (cli.final_benchmark, options.final_benchmark_mode, "final_benchmark_mode"),
        )
        for cli_value, option_value, policy_name in pairs:
            self.assertEqual(
                getattr(RIG_CALIBRATION_DEFAULTS, policy_name), cli_value
            )
            self.assertEqual(cli_value, option_value)


class AcquisitionConfigurationTests(unittest.TestCase):
    def test_capture_cli_uses_acquisition_policy(self):
        cli = minimap_capture.parser().parse_args([])
        self.assertEqual(ACQUISITION_DEFAULTS.camera_width_px, cli.camera_width)
        self.assertEqual(ACQUISITION_DEFAULTS.camera_height_px, cli.camera_height)
        self.assertEqual(ACQUISITION_DEFAULTS.camera_fps, cli.camera_fps)
        self.assertEqual(ACQUISITION_DEFAULTS.android_capture, cli.android_capture)
        self.assertEqual(ACQUISITION_DEFAULTS.sample_count, cli.moves)
        self.assertEqual(ACQUISITION_DEFAULTS.capture_mode, cli.capture_mode)
        self.assertEqual(
            ACQUISITION_DEFAULTS.horizontal_swipe_fraction,
            cli.horizontal_swipe_fraction,
        )
        self.assertEqual(
            ACQUISITION_DEFAULTS.vertical_swipe_fraction,
            cli.vertical_swipe_fraction,
        )

    def test_direct_touch_plans_use_their_own_domain_policy(self):
        zigzag = ZigzagTouchPlan(
            start_xy=[100, 100], end_x=50, vertical_amplitude_px=40
        )
        orbit = CursorOrbitTouchPlan(center_xy=[100, 100], radius_px=20)
        self.assertEqual(ZIGZAG_PLAN_DEFAULTS.move_count, zigzag.move_count)
        self.assertEqual(ZIGZAG_PLAN_DEFAULTS.step_seconds, zigzag.step_seconds)
        self.assertEqual(
            ZIGZAG_PLAN_DEFAULTS.endpoint_hold_seconds,
            zigzag.endpoint_hold_seconds,
        )
        self.assertEqual(
            CURSOR_ORBIT_PLAN_DEFAULTS.direction_count, orbit.direction_count
        )
        self.assertEqual(CURSOR_ORBIT_PLAN_DEFAULTS.repeats, orbit.repeats)
        self.assertEqual(
            CURSOR_ORBIT_PLAN_DEFAULTS.reset_seconds, orbit.reset_seconds
        )


if __name__ == "__main__":
    unittest.main()
