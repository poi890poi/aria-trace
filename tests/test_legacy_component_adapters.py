import unittest

import numpy as np

from acquisition.models import FramePacket
from rig_runtime.adapters.legacy_acquisition import (
    HOST_CLOCK_METADATA_KEY,
    SOURCE_CLOCK_METADATA_KEY,
    SPACE_METADATA_KEY,
    LegacyFrameSourceAdapter,
    envelope_to_frame_packet,
    frame_packet_to_envelope,
)
from rig_runtime.domain import ComponentError, ProducerRef, SourceRef, SpaceRef


class _LegacySource:
    def __init__(self, packet):
        self.packet = packet
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def read(self):
        result, self.packet = self.packet, None
        return result

    def stop(self):
        self.stopped = True


class LegacyComponentAdapterTests(unittest.TestCase):
    def packet(self):
        return FramePacket(
            "android_phone",
            np.zeros((3, 4, 3), dtype=np.uint8),
            100,
            125,
            source_time_ns=80,
            metadata={"source_frame_index": 7},
            dropped_before=2,
        )

    def test_packet_round_trip_preserves_payload_and_makes_domains_explicit(self):
        packet = self.packet()
        envelope = frame_packet_to_envelope(
            packet,
            envelope_id="frame-7",
            trace_id="trace-a",
            host_clock_id="host-perf-counter-ns",
            source_clock_id="android-monotonic-ns",
            space=SpaceRef("android-logical-display", "raster"),
            producer=ProducerRef("scrcpy-source", "1.0"),
            sources=(SourceRef("session-1", "recording"),),
        )

        self.assertIs(packet.image, envelope.value.image)
        self.assertEqual("android-logical-display", envelope.space.space_id)
        self.assertEqual("android-monotonic-ns", envelope.timing.source.clock_id)
        self.assertEqual(25, envelope.timing.capture_to_receive_ns)
        self.assertEqual("unknown", envelope.quality.decision.value)

        restored = envelope_to_frame_packet(envelope)
        self.assertIs(packet.image, restored.image)
        self.assertEqual(packet.stream_id, restored.stream_id)
        self.assertEqual(packet.source_time_ns, restored.source_time_ns)
        self.assertEqual(packet.dropped_before, restored.dropped_before)
        self.assertEqual(
            "host-perf-counter-ns", restored.metadata[HOST_CLOCK_METADATA_KEY]
        )
        self.assertEqual(
            "android-monotonic-ns", restored.metadata[SOURCE_CLOCK_METADATA_KEY]
        )
        self.assertEqual(
            "android-logical-display", restored.metadata[SPACE_METADATA_KEY]
        )

    def test_source_timestamp_without_clock_is_rejected(self):
        with self.assertRaises(ComponentError) as raised:
            frame_packet_to_envelope(
                self.packet(),
                envelope_id="frame-7",
                trace_id="trace-a",
                host_clock_id="host-perf-counter-ns",
                source_clock_id=None,
                space=SpaceRef("android-logical-display", "raster"),
                producer=ProducerRef("scrcpy-source", "1.0"),
            )
        self.assertEqual("missing-source-clock", raised.exception.code)

    def test_conflicting_legacy_metadata_is_rejected(self):
        packet = self.packet()
        packet.metadata[SPACE_METADATA_KEY] = "wrong-space"
        with self.assertRaises(ComponentError) as raised:
            frame_packet_to_envelope(
                packet,
                envelope_id="frame-7",
                trace_id="trace-a",
                host_clock_id="host-perf-counter-ns",
                source_clock_id="android-monotonic-ns",
                space=SpaceRef("android-logical-display", "raster"),
                producer=ProducerRef("scrcpy-source", "1.0"),
            )
        self.assertEqual("metadata-conflict", raised.exception.code)

    def test_legacy_source_lifecycle_and_empty_read_are_preserved(self):
        source = _LegacySource(self.packet())
        adapter = LegacyFrameSourceAdapter(
            source,
            trace_id="trace-a",
            host_clock_id="host-perf-counter-ns",
            source_clock_id="android-monotonic-ns",
            space=SpaceRef("android-logical-display", "raster"),
            producer=ProducerRef("scrcpy-source", "1.0"),
            envelope_id=lambda packet: "frame-{}".format(
                packet.metadata["source_frame_index"]
            ),
        )

        adapter.start()
        first = adapter.read()
        second = adapter.read()
        adapter.stop()

        self.assertTrue(source.started)
        self.assertTrue(source.stopped)
        self.assertEqual("frame-7", first.identity.envelope_id)
        self.assertIsNone(second)


if __name__ == "__main__":
    unittest.main()
