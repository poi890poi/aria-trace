import json
import tempfile
import unittest
from pathlib import Path

from aria_trace.adapters.jsonl_trace import JsonlInvocationSink
from aria_trace.domain import (
    ComponentInvocation,
    ConfigurationRef,
    FailureRecord,
    InvocationStatus,
    ProducerRef,
    TimePoint,
)
from aria_trace.evidence import canonical_json_bytes


class InvocationTraceTests(unittest.TestCase):
    def completed(self, invocation_id="invocation-1"):
        return ComponentInvocation(
            invocation_id=invocation_id,
            trace_id="trace-1",
            run_id="run-1",
            producer=ProducerRef(
                "minimap-boundary", "1.0", "radial-circle-fit", "abc123"
            ),
            configuration=ConfigurationRef("default-boundary"),
            status=InvocationStatus.COMPLETED,
            started=TimePoint(100, "host-monotonic"),
            finished=TimePoint(125, "host-monotonic"),
            input_envelope_ids=("frames-1",),
            output_envelope_ids=("boundary-1",),
        )

    def test_terminal_lifecycle_and_clock_order_are_validated(self):
        with self.assertRaises(ValueError):
            ComponentInvocation(
                invocation_id="bad",
                trace_id="trace",
                run_id="run",
                producer=ProducerRef("test", "1"),
                status=InvocationStatus.COMPLETED,
                started=TimePoint(100, "host"),
            )
        with self.assertRaises(ValueError):
            ComponentInvocation(
                invocation_id="bad",
                trace_id="trace",
                run_id="run",
                producer=ProducerRef("test", "1"),
                status=InvocationStatus.FAILED,
                started=TimePoint(100, "host"),
                finished=TimePoint(110, "host"),
            )

    def test_failure_is_structured_and_deterministically_serialized(self):
        value = ComponentInvocation(
            invocation_id="failed-1",
            trace_id="trace-1",
            run_id="run-1",
            producer=ProducerRef("global-localizer", "2.0"),
            status=InvocationStatus.FAILED,
            started=TimePoint(100, "host"),
            finished=TimePoint(150, "host"),
            failure=FailureRecord(
                "no-consensus", "rejected", "No stable consensus", {"count": 2}
            ),
        )
        first = canonical_json_bytes(value)
        second = canonical_json_bytes(value)
        self.assertEqual(first, second)
        decoded = json.loads(first)
        self.assertEqual("no-consensus", decoded["failure"]["code"])
        self.assertEqual("failed", decoded["status"])

    def test_jsonl_sink_appends_complete_independent_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace" / "invocations.jsonl"
            sink = JsonlInvocationSink(path, create_parent=True)
            sink.record(self.completed("invocation-1"))
            sink.record(self.completed("invocation-2"))

            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(
                ["invocation-1", "invocation-2"],
                [row["invocation_id"] for row in rows],
            )
            self.assertEqual("radial-circle-fit", rows[0]["producer"]["algorithm_id"])
            self.assertEqual(["frames-1"], rows[0]["input_envelope_ids"])


if __name__ == "__main__":
    unittest.main()
