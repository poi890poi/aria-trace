import unittest

from rig_runtime.domain import (
    CheckStatus,
    ComponentError,
    DataEnvelope,
    Decision,
    DiagnosticValue,
    EnvelopeIdentity,
    FailureKind,
    ImageFrame,
    ProducerRef,
    ProvenanceInfo,
    QualityCheck,
    QualityInfo,
    QualityMetric,
    SchemaRef,
    SpaceRef,
    TimePoint,
    TimingInfo,
)
from rig_runtime.ports import CancellationToken, ComponentContext


class _Cancelled(CancellationToken):
    def is_cancelled(self) -> bool:
        return True


class ComponentContractTests(unittest.TestCase):
    def test_envelope_carries_explicit_time_space_provenance_and_quality(self):
        image = object()
        envelope = DataEnvelope(
            schema=SchemaRef("aria.image-frame", "1.0"),
            value=ImageFrame("phone", image, dropped_before=2),
            identity=EnvelopeIdentity("frame-7", "trace-2", ("capture-1",)),
            timing=TimingInfo(
                capture=TimePoint(100, "host-monotonic"),
                receive=TimePoint(125, "host-monotonic"),
            ),
            space=SpaceRef("phone-display", "raster"),
            provenance=ProvenanceInfo(ProducerRef("camera-adapter", "1.0")),
            quality=QualityInfo(
                Decision.ACCEPTED,
                metrics=(QualityMetric("queue_depth", 0, "frames"),),
                checks=(QualityCheck("fresh", CheckStatus.PASS),),
            ),
            diagnostics=(DiagnosticValue("transport", "scrcpy"),),
        )

        self.assertIs(image, envelope.value.image)
        self.assertEqual(25, envelope.timing.capture_to_receive_ns)
        self.assertEqual("phone-display", envelope.space.space_id)
        self.assertEqual("camera-adapter", envelope.provenance.producer.component_id)
        self.assertEqual(Decision.ACCEPTED, envelope.quality.decision)

    def test_latency_is_unknown_across_unrelated_clocks(self):
        timing = TimingInfo(
            capture=TimePoint(100, "device-monotonic"),
            receive=TimePoint(120, "host-monotonic"),
        )
        self.assertIsNone(timing.capture_to_receive_ns)

    def test_duplicate_quality_or_diagnostic_names_are_rejected(self):
        with self.assertRaises(ValueError):
            QualityInfo(
                metrics=(QualityMetric("score", 1), QualityMetric("score", 2))
            )

        with self.assertRaises(ValueError):
            DataEnvelope(
                schema=SchemaRef("test", "1"),
                value="value",
                identity=EnvelopeIdentity("value-1", "trace-1"),
                timing=TimingInfo(),
                space=SpaceRef("unknown"),
                provenance=ProvenanceInfo(ProducerRef("test", "1")),
                diagnostics=(
                    DiagnosticValue("same", 1),
                    DiagnosticValue("same", 2),
                ),
            )

    def test_cancellation_is_a_structured_component_failure(self):
        context = ComponentContext("trace-1", "run-1", {}, _Cancelled())
        with self.assertRaises(ComponentError) as raised:
            context.raise_if_cancelled()
        self.assertEqual(FailureKind.CANCELED, raised.exception.kind)
        self.assertEqual("cancelled", raised.exception.code)

    def test_unknown_time_and_space_must_be_explicit(self):
        with self.assertRaises(ValueError):
            TimePoint(1, "")
        with self.assertRaises(ValueError):
            SpaceRef("")
        self.assertEqual("unknown", TimePoint(1, "unknown").clock_id)
        self.assertEqual("unknown", SpaceRef("unknown").space_id)


if __name__ == "__main__":
    unittest.main()
