import tempfile
import unittest
from pathlib import Path

import numpy as np

from replay.route_tracking import (
    RouteTrackingPackage,
    compile_route_tracking_package,
)


class RouteTrackingPackageTests(unittest.TestCase):
    @staticmethod
    def _observations(count=10):
        values = []
        for index in range(count):
            descriptor = np.zeros(16, np.float32)
            descriptor[index % len(descriptor)] = 1.0
            values.append(
                {
                    "source_frame_index": 100 + index,
                    "session_time_ns": 1_000_000_000 + index * 200_000_000,
                    "x": index * 10.0,
                    "y": 2.0 * np.sin(index / 3.0),
                    "mode_id": "world" if index < 5 else "town",
                    "localization_score": 0.8,
                    "localization_margin": 0.2,
                    "descriptor": descriptor,
                }
            )
        return values

    def test_compiles_geometry_motion_envelope_and_transition(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "route"
            manifest = compile_route_tracking_package(
                self._observations(),
                output,
                route_id="town-entry",
                atlas_id="atlas-a",
                coordinate_space_id="map-atlas:atlas-a:canonical-map-px",
                max_step_px=20.0,
            )
            package = RouteTrackingPackage(output)

            self.assertEqual(manifest["state_count"], 10)
            self.assertEqual(manifest["transition_count"], 1)
            self.assertGreater(manifest["route_length_px"], 90.0)
            self.assertGreater(manifest["motion_envelope"]["speed_px_s"]["p95"], 40.0)
            self.assertEqual(package.transitions[0]["source_mode_id"], "world")
            self.assertEqual(package.transitions[0]["target_mode_id"], "town")
            self.assertEqual(
                package.transitions[0]["position_semantics"],
                "continuous_no_displacement",
            )

    def test_candidate_search_is_restricted_to_adjacent_route_progress(self):
        observations = self._observations(14)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "route"
            compile_route_tracking_package(
                observations,
                output,
                route_id="route-a",
                atlas_id="atlas-a",
                coordinate_space_id="map-atlas:atlas-a:canonical-map-px",
            )
            package = RouteTrackingPackage(output)
            query = observations[11]["descriptor"]

            unrestricted = package.candidates(query, top_k=1)
            restricted = package.candidates(
                query,
                previous_state_index=4,
                backward_states=1,
                forward_states=2,
                top_k=5,
            )

            self.assertEqual(unrestricted[0]["state_index"], 11)
            self.assertTrue(all(3 <= item["state_index"] <= 6 for item in restricted))

    def test_rejects_discontinuous_route_geometry(self):
        observations = self._observations()
        observations[6]["x"] = 500.0
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "discontinuous"):
                compile_route_tracking_package(
                    observations,
                    Path(temporary) / "route",
                    route_id="bad-route",
                    atlas_id="atlas-a",
                    coordinate_space_id="map-atlas:atlas-a:canonical-map-px",
                    max_step_px=30.0,
                )


if __name__ == "__main__":
    unittest.main()
