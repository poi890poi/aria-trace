import unittest
from types import SimpleNamespace

from benchmarks.cursor_pose.stateful import (
    Measurement,
    apply_fallback_strategy,
    summarize_e2e_rows,
)
from benchmarks.cursor_pose.run_e2e import (
    CausalOutlierGate,
    _forward_only_input_audit,
)


def _measurement(frame, angle, accepted=True):
    return Measurement(
        frame_index=frame,
        session_time_ns=frame * 100_000_000,
        angle_deg=angle,
        confidence=0.9 if accepted else 0.1,
        accepted=accepted,
        rejection_reason=None if accepted else "low_confidence",
        latency_ns=1_000_000,
    )


class CursorPoseE2EBenchmarkTests(unittest.TestCase):
    def test_e2e_reference_requires_forward_only_input_evidence(self):
        reader = SimpleNamespace(
            inputs=[
                {"kind": "pc_raw_keyboard", "payload": {"key_name": "W"}},
                {"kind": "pc_raw_keyboard", "payload": {"key_name": "SPACE"}},
            ]
        )
        self.assertTrue(_forward_only_input_audit(reader)["valid_forward_only_control"])

        reader.inputs.append(
            {"kind": "pc_raw_mouse", "payload": {"delta_x": 2, "delta_y": 0}}
        )
        self.assertFalse(
            _forward_only_input_audit(reader)["valid_forward_only_control"]
        )

    def test_final_gate_rejects_outlier_without_updating_state(self):
        gate = CausalOutlierGate(
            confidence_min=0.45,
            innovation_limit_deg=8.0,
            large_innovation_confidence_min=0.7,
        )
        first = gate.decide(_measurement(0, 10.0))
        second = gate.decide(_measurement(1, 12.0))
        outlier = gate.decide(
            Measurement(2, 200_000_000, 80.0, 0.6, True, None, 1_000_000)
        )
        recovered = gate.decide(_measurement(3, 16.0))

        self.assertTrue(first.accepted)
        self.assertTrue(second.accepted)
        self.assertFalse(outlier.accepted)
        self.assertEqual(
            outlier.rejection_reason,
            "large_innovation_without_high_confidence",
        )
        self.assertTrue(recovered.accepted)
        self.assertEqual(len(gate.accepted_history), 3)

    def test_baseline_gate_disables_temporal_rejection(self):
        gate = CausalOutlierGate(
            confidence_min=0.45,
            innovation_limit_deg=8.0,
            large_innovation_confidence_min=0.7,
            temporal_check_enabled=False,
        )

        self.assertTrue(gate.decide(_measurement(0, 10.0)).accepted)
        candidate = Measurement(
            1, 100_000_000, 100.0, 0.6, True, None, 1_000_000
        )
        self.assertTrue(gate.decide(candidate).accepted)

    def test_reuse_previous_state_never_becomes_fresh(self):
        states = apply_fallback_strategy(
            [_measurement(0, 10.0), _measurement(1, 90.0, accepted=False)],
            "reuse_previous_state",
        )

        self.assertEqual(states[1].angle_deg, 10.0)
        self.assertEqual(states[1].provenance, "held")
        self.assertFalse(states[1].primary_accepted)
        self.assertTrue(states[1].fallback_invoked)
        self.assertTrue(states[1].fallback_produced_output)

    def test_constant_velocity_uses_only_past_accepted_measurements(self):
        states = apply_fallback_strategy(
            [
                _measurement(0, 350.0),
                _measurement(1, 0.0),
                _measurement(2, 270.0, accepted=False),
            ],
            "constant_velocity_last_2_accepted",
        )

        self.assertAlmostEqual(states[2].angle_deg, 10.0, places=6)
        self.assertEqual(states[2].provenance, "predicted")

    def test_rate_names_have_explicit_denominators_and_partition_outputs(self):
        rows = [
            {
                "session": "run",
                "frame_index": 0,
                "session_time_ns": 0,
                "primary_accepted": True,
                "fallback_invoked": False,
                "fallback_produced_output": False,
                "provenance": "fresh_measurement",
                "output_angle_deg": 10.0,
                "reference_angle_deg": 12.0,
                "absolute_error_deg": 2.0,
                "e2e_latency_ns": 1_000_000,
                "fallback_strategy_latency_ns": 10_000,
            },
            {
                "session": "run",
                "frame_index": 1,
                "session_time_ns": 100_000_000,
                "primary_accepted": False,
                "fallback_invoked": True,
                "fallback_produced_output": True,
                "provenance": "held",
                "output_angle_deg": 10.0,
                "reference_angle_deg": 15.0,
                "absolute_error_deg": 5.0,
                "e2e_latency_ns": 1_010_000,
                "fallback_strategy_latency_ns": 10_000,
            },
            {
                "session": "run",
                "frame_index": 2,
                "session_time_ns": 200_000_000,
                "primary_accepted": False,
                "fallback_invoked": True,
                "fallback_produced_output": False,
                "provenance": "unavailable",
                "output_angle_deg": None,
                "reference_angle_deg": 18.0,
                "absolute_error_deg": None,
                "e2e_latency_ns": 1_020_000,
                "fallback_strategy_latency_ns": 10_000,
            },
        ]

        result = summarize_e2e_rows(rows)

        self.assertEqual(
            result["rate_denominator"],
            "all chronological primary measurement attempts",
        )
        self.assertAlmostEqual(result["primary_measurement_accepted_rate"], 1 / 3)
        self.assertAlmostEqual(result["primary_measurement_rejected_rate"], 2 / 3)
        self.assertAlmostEqual(result["fallback_invocation_rate"], 2 / 3)
        self.assertAlmostEqual(result["fallback_output_success_rate"], 1 / 2)
        self.assertAlmostEqual(result["final_output_available_rate"], 2 / 3)
        self.assertAlmostEqual(
            sum(result["output_provenance_rate"].values()), 1.0
        )
        self.assertEqual(result["e2e_absolute_error_deg"]["worst"], 5.0)
        self.assertNotIn("worst", result["fresh_measurement_error_deg"])
        self.assertNotIn("worst", result["fallback_output_error_deg"])

    def test_no_fallback_reports_unavailable_without_fabricating_angle(self):
        states = apply_fallback_strategy(
            [_measurement(0, None, accepted=False)], "no_fallback"
        )

        self.assertIsNone(states[0].angle_deg)
        self.assertEqual(states[0].provenance, "unavailable")


if __name__ == "__main__":
    unittest.main()
