import math
import unittest

from benchmarks.localization.repeated_waypoints import (
    discover_repeated_waypoints,
    evaluate_candidate_repeatability,
)


def _circle_laps(count=5):
    rows = []
    frame = 0
    time_s = 0.0
    for lap in range(count):
        samples = 70 + (lap % 3) * 9
        for index in range(samples):
            phase = 2.0 * math.pi * index / samples
            radius = 60.0 + 1.5 * math.sin(phase * 3.0 + lap)
            rows.append(
                {
                    "source_frame_index": frame,
                    "session_time_ns": int(time_s * 1.0e9),
                    "canonical_xy": [
                        200.0 + radius * math.cos(phase),
                        300.0 + radius * math.sin(phase),
                    ],
                    "route_heading_deg": math.degrees(phase + math.pi / 2.0) % 360.0,
                    "mode_id": "world",
                }
            )
            frame += 1
            time_s += 0.17 + 0.025 * ((index + lap) % 4)
    return rows


class RepeatedWaypointTests(unittest.TestCase):
    def test_finds_variable_speed_laps_without_pause_or_boundary(self):
        result = discover_repeated_waypoints(
            _circle_laps(), minimum_recurrence_s=6.0, minimum_visits=4
        )
        self.assertGreaterEqual(result["group_count"], 20)
        self.assertTrue(all(item["visit_count"] >= 4 for item in result["groups"]))
        member_keys = [
            (member["frame_index"], member["session_time_ns"])
            for group in result["groups"]
            for member in group["members"]
        ]
        self.assertEqual(len(member_keys), len(set(member_keys)))
        self.assertTrue(
            all(
                item["reference_position_spread_px"]["p95"] <= 5.0
                for item in result["groups"]
            )
        )

    def test_opposite_directions_are_distinct_complete_states(self):
        rows = []
        frame = 0
        time_s = 0.0
        for _ in range(4):
            for direction, points, heading in (
                ("forward", range(41), 0.0),
                ("reverse", range(40, -1, -1), 180.0),
            ):
                for point in points:
                    rows.append(
                        {
                            "source_frame_index": frame,
                            "session_time_ns": int(time_s * 1.0e9),
                            "x": point * 2.0,
                            "y": 10.0 + 0.4 * math.sin(frame),
                            "heading_deg": heading,
                            "mode_id": "town",
                            "direction": direction,
                        }
                    )
                    frame += 1
                    time_s += 0.2
        result = discover_repeated_waypoints(
            rows,
            spatial_radius_px=3.0,
            heading_radius_deg=25.0,
            waypoint_spacing_px=8.0,
            minimum_recurrence_s=5.0,
            minimum_visits=3,
        )
        near_middle = [
            item for item in result["groups"] if abs(item["anchor_xy"][0] - 40.0) < 5.0
        ]
        headings = sorted(round(item["anchor_heading_deg"]) for item in near_middle)
        self.assertIn(0, headings)
        self.assertIn(180, headings)

    def test_repeatability_subtracts_real_path_variation_and_constant_bias(self):
        reference = _circle_laps()
        waypoints = discover_repeated_waypoints(
            reference, minimum_recurrence_s=6.0, minimum_visits=4
        )
        candidate = []
        for index, row in enumerate(reference):
            candidate.append(
                {
                    "frame_index": row["source_frame_index"],
                    "session_time_ns": row["session_time_ns"],
                    "x": row["canonical_xy"][0] + 8.0 + 0.15 * math.sin(index),
                    "y": row["canonical_xy"][1] - 6.0 + 0.15 * math.cos(index),
                    "heading_deg": row["route_heading_deg"] + 4.0,
                    "valid": True,
                    "measurement_accepted": True,
                }
            )
        result = evaluate_candidate_repeatability(waypoints, candidate)
        self.assertGreater(result["reference_position_error_px"]["median"], 9.0)
        self.assertLess(result["position_repeatability_residual_px"]["p95"], 0.4)
        self.assertLess(result["heading_repeatability_residual_deg"]["p95"], 0.01)
        self.assertEqual(result["availability_rate"], 1.0)

    def test_candidate_can_match_by_nearest_timestamp_across_sources(self):
        reference = _circle_laps()
        waypoints = discover_repeated_waypoints(
            reference, minimum_recurrence_s=6.0, minimum_visits=4
        )
        candidate = [
            {
                "session_time_ns": row["session_time_ns"] + 5_000_000,
                "x": row["canonical_xy"][0],
                "y": row["canonical_xy"][1],
                "heading_deg": row["route_heading_deg"],
                "valid": True,
            }
            for row in reference
        ]
        result = evaluate_candidate_repeatability(
            waypoints, candidate, maximum_time_delta_ms=10.0
        )
        self.assertEqual(result["availability_rate"], 1.0)
        self.assertAlmostEqual(result["time_alignment_delta_ms"]["median"], 5.0)


if __name__ == "__main__":
    unittest.main()
