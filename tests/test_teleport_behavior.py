import json
import tempfile
import unittest
from pathlib import Path

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
            origin_global_xy=(10, 20),
            teleport_target_global_xy=(300.5, 400.25),
            destination_global_xy=(304.0, 398.0),
            portal_id="statue-1",
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


if __name__ == "__main__":
    unittest.main()
