from __future__ import annotations

import unittest

from rig_runtime.adapters.filesystem.profile_registry import AdapterRequest
from rig_runtime.adapters.hik.compat import _adapter_request_from_config
from rig_runtime.apps import hik_stream
from rig_runtime.domain.configuration import (
    ADAPTER_DEFAULTS,
    ResolvedAdapterPlan,
)
from rig_runtime.workflows import profile_management


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


if __name__ == "__main__":
    unittest.main()
