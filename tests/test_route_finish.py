import tempfile
import unittest
from pathlib import Path

import numpy as np

from aria_trace.services.localization.route.tracker import RouteFinishGate
from replay.route_tracking import RouteTrackingPackage, compile_route_tracking_package


class RouteFinishGateTests(unittest.TestCase):
    def _package(self, root: Path, positions):
        observations = []
        for index, (x, y) in enumerate(positions):
            descriptor = np.zeros(8, np.float32)
            descriptor[index % len(descriptor)] = 1.0
            observations.append(
                {
                    "source_frame_index": index,
                    "session_time_ns": 1_000_000_000 + index * 100_000_000,
                    "x": x,
                    "y": y,
                    "mode_id": "world",
                    "descriptor": descriptor,
                }
            )
        compile_route_tracking_package(
            observations,
            root,
            route_id="test-route",
            atlas_id="atlas",
            coordinate_space_id="map",
            corridor_radius_px=20,
        )
        return RouteTrackingPackage(root)

    @staticmethod
    def _state(x, *, accepted=True, fresh=True, mode="world"):
        return {
            "pose": {"x": x, "y": 0.0},
            "active_map_mode_id": mode,
            "route_tracking_fresh": fresh,
            "route_tracking": {"measurement_accepted": accepted},
        }

    def test_requires_departure_and_three_visual_confirmations(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = self._package(Path(temporary), [(0, 0), (50, 0), (100, 0)])
            gate = RouteFinishGate(package, radius_px=5, consecutive_measurements=3)
            self.assertFalse(gate.update(self._state(100))["reached"])
            self.assertFalse(gate.update(self._state(20))["reached"])
            self.assertTrue(gate.armed)
            self.assertFalse(gate.update(self._state(98))["reached"])
            self.assertFalse(gate.update(self._state(99))["reached"])
            result = gate.update(self._state(100))
            self.assertTrue(result["reached"])
            self.assertEqual(
                "consecutive-current-frame-map-measurements",
                result["evidence_policy"],
            )

    def test_rejected_and_wrong_layer_measurements_reset_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = self._package(Path(temporary), [(0, 0), (50, 0), (100, 0)])
            gate = RouteFinishGate(package, radius_px=5, consecutive_measurements=2)
            gate.update(self._state(20))
            gate.update(self._state(100))
            self.assertEqual(1, gate.confirmations)
            gate.update(self._state(100, accepted=False))
            self.assertEqual(0, gate.confirmations)
            gate.update(self._state(100, mode="town"))
            self.assertEqual(0, gate.confirmations)

    def test_lap_endpoint_does_not_finish_before_departure(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = self._package(Path(temporary), [(0, 0), (50, 0), (0, 0)])
            gate = RouteFinishGate(package, radius_px=5, consecutive_measurements=2)
            self.assertFalse(gate.update(self._state(0))["armed"])
            self.assertFalse(gate.update(self._state(0))["reached"])
            self.assertTrue(gate.update(self._state(30))["armed"])
            self.assertFalse(gate.update(self._state(1))["reached"])
            self.assertTrue(gate.update(self._state(0))["reached"])


if __name__ == "__main__":
    unittest.main()
