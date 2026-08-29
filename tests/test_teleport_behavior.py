import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from acquisition.teleport_analysis import (
    TELEPORT_BEHAVIOR_MODEL,
    _target_change_component,
    parse_teleport_inputs,
)
from acquisition.teleport_behavior import (
    make_teleport_behavior_sample,
    save_teleport_behavior_sample,
)


class TeleportBehaviorTests(unittest.TestCase):
    def test_persists_target_and_destination_in_one_named_space(self):
        sample = make_teleport_behavior_sample(
            game_profile_id="genshin-impact-pc",
            session_id="session-1",
            coordinate_space_id="global-mosaic-v2-original-px",
            teleport_target_global_xy=(300.5, 400.25),
            destination_global_xy=(304.0, 398.0),
            portal_id="statue-1",
            phases=[
                {
                    "state": "loading",
                    "start_s": 5.0,
                    "end_s": 8.0,
                    "guard": "destination_not_ready",
                }
            ],
            behavior_model=TELEPORT_BEHAVIOR_MODEL,
            arrival_model={"sample_count": 4, "radial_p95_px": 0.8},
            evidence={"target_click_frame": 42, "world_ready_frame": 91},
            provenance={"map_stitch_id": "mosaic-v2"},
            quality={"destination_confidence": 0.96},
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = save_teleport_behavior_sample(
                sample, Path(temporary) / "teleport.json"
            )
            value = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(value["behavior_type"], "teleportation")
        self.assertEqual(value["teleport_target_global_xy"], [300.5, 400.25])
        self.assertEqual(value["destination_global_xy"], [304.0, 398.0])
        self.assertEqual(value["coordinate_space_id"], "global-mosaic-v2-original-px")
        self.assertNotIn("origin_global_xy", value)
        self.assertEqual(value["schema_version"], "2.0")
        self.assertEqual(value["phases"][0]["state"], "loading")
        self.assertTrue(
            any(item.get("optional") for item in value["behavior_model"]["transitions"])
        )

    def test_rejects_coordinates_without_reusable_provenance(self):
        with self.assertRaisesRegex(ValueError, "provenance"):
            make_teleport_behavior_sample(
                game_profile_id="genshin-impact-pc",
                session_id="session-1",
                coordinate_space_id="global-map",
                teleport_target_global_xy=(1, 2),
                destination_global_xy=(3, 4),
                evidence={"target_click_frame": 1},
                provenance={},
            )

    def test_input_model_separates_navigation_drags_from_stationary_clicks(self):
        def mouse(time_ms, *, transitions=(), dx=0, dy=0, wheel=0):
            return {
                "kind": "pc_raw_mouse",
                "session_time_ns": time_ms * 1_000_000,
                "payload": {
                    "button_transitions": list(transitions),
                    "delta_x": dx,
                    "delta_y": dy,
                    "wheel_delta": wheel,
                },
            }

        inputs = [
            {
                "kind": "pc_raw_keyboard",
                "session_time_ns": 0,
                "payload": {"key_name": "M", "pressed": True},
            },
            mouse(100, wheel=-120),
            mouse(150, wheel=-120),
            mouse(500, transitions=("left_down",)),
            mouse(520, dx=20, dy=-10),
            mouse(550, transitions=("left_up",)),
            mouse(900, transitions=("left_down",)),
            mouse(980, transitions=("left_up",)),
        ]
        result = parse_teleport_inputs(inputs)
        self.assertEqual(result["map_open_events"][0]["key"], "M")
        self.assertEqual(result["wheel_bursts"][0]["action"], "zoom_out")
        self.assertEqual(result["wheel_bursts"][0]["notches"], 2.0)
        self.assertEqual(len(result["drag_episodes"]), 1)
        self.assertEqual(len(result["stationary_clicks"]), 1)

    def test_visual_change_selects_compact_target_instead_of_side_panel(self):
        before = np.zeros((300, 500, 3), np.uint8)
        after = before.copy()
        after[140:180, 180:220] = 255
        after[:, 390:] = 220
        result = _target_change_component(before, after)
        self.assertAlmostEqual(result["screen_xy"][0], 199.5, delta=3.0)
        self.assertAlmostEqual(result["screen_xy"][1], 159.5, delta=3.0)


if __name__ == "__main__":
    unittest.main()
