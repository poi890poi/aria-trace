import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.localization.build_transition_reference import build
from replay.route_tracking import RouteTrackingPackage


class TransitionReferenceTests(unittest.TestCase):
    def test_builds_loadable_calibration_proxy_with_explicit_role(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atlas = root / "atlas"
            session = root / "session"
            output = root / "reference"
            atlas.mkdir()
            session.mkdir()
            (session / "manifest.json").write_text(
                json.dumps({"session_id": "session-9"}), encoding="utf-8"
            )
            (atlas / "map_atlas.json").write_text(
                json.dumps(
                    {
                        "atlas_id": "atlas-1",
                        "coordinate_space_id": "space-1",
                        "transition_model": {
                            "analysis": {
                                "samples": [
                                    {
                                        "frame_index": 10,
                                        "session_time_ns": 100,
                                        "canonical_xy": [1.0, 2.0],
                                        "likelihoods": {"world": 0.9, "town": 0.1},
                                    },
                                    {
                                        "frame_index": 20,
                                        "session_time_ns": 200,
                                        "canonical_xy": [4.0, 6.0],
                                        "likelihoods": {"world": 0.1, "town": 0.9},
                                    },
                                ]
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            manifest = build(atlas, session, output)
            package = RouteTrackingPackage(output)

            self.assertEqual(
                manifest["reference_role"], "offline-transition-calibration-proxy"
            )
            self.assertFalse(manifest["independent_ground_truth"])
            self.assertEqual(package.states[0]["mode_id"], "world")
            self.assertEqual(package.states[1]["mode_id"], "town")
            self.assertEqual(package.states[1]["route_distance_px"], 5.0)


if __name__ == "__main__":
    unittest.main()
