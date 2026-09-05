import threading
import time
import unittest

import numpy as np

from acquisition.frame_pump import LatestFramePump
from acquisition.models import FramePacket
from rig_runtime.adapters.legacy_acquisition import LegacyFrameSourceAdapter
from rig_runtime.domain import ProducerRef, SpaceRef


class FastSource:
    def __init__(self):
        self.stopped = threading.Event()
        self.sequence = 0

    def start(self):
        return None

    def read(self):
        if self.stopped.wait(0.001):
            return None
        self.sequence += 1
        return FramePacket(
            "main",
            np.full((2, 2), self.sequence, np.int32),
            self.sequence,
            self.sequence,
        )

    def stop(self):
        self.stopped.set()


class LatestFramePumpTests(unittest.TestCase):
    def test_observer_sees_source_frames_without_consuming_latest_value(self):
        source = FastSource()
        observed = []
        pump = LatestFramePump(
            source,
            observer=lambda packet: observed.append(packet.host_capture_time_ns),
        )
        pump.start()
        try:
            value = pump.read_latest(0.2)
            deadline = time.monotonic() + 0.2
            while (
                value.host_capture_time_ns not in observed
                and time.monotonic() < deadline
            ):
                time.sleep(0.001)
            self.assertIn(value.host_capture_time_ns, observed)
            self.assertIsNone(pump.observer_error)
        finally:
            pump.stop()

    def test_observer_does_not_delay_current_frame_delivery(self):
        source = FastSource()
        observer_started = threading.Event()
        release_observer = threading.Event()

        def block(_packet):
            observer_started.set()
            release_observer.wait(1.0)

        pump = LatestFramePump(source, observer=block)
        pump.start()
        try:
            self.assertTrue(observer_started.wait(0.2))
            self.assertIsNotNone(pump.read_latest(0.05))
        finally:
            release_observer.set()
            pump.stop()

    def test_observer_failure_does_not_stop_capture(self):
        source = FastSource()

        def fail(_packet):
            raise ValueError("observer failed")

        pump = LatestFramePump(source, observer=fail)
        pump.start()
        try:
            value = pump.read_latest(0.2)
            self.assertIsNotNone(value)
            self.assertEqual("ValueError: observer failed", pump.observer_error)
        finally:
            pump.stop()

    def test_slow_consumer_gets_latest_frame_and_drop_count(self):
        source = FastSource()
        pump = LatestFramePump(source)
        pump.start()
        try:
            first = pump.read_latest(0.2)
            deadline = time.monotonic() + 1.0
            while pump.dropped_before_processing == 0 and time.monotonic() < deadline:
                time.sleep(0.001)
            second = pump.read_latest(0.2)
            self.assertGreater(second.host_capture_time_ns, first.host_capture_time_ns)
            self.assertGreater(pump.dropped_before_processing, 0)
        finally:
            pump.stop()

    def test_pump_preserves_universal_envelopes_without_payload_knowledge(self):
        legacy = FastSource()
        source = LegacyFrameSourceAdapter(
            legacy,
            trace_id="trace-pump",
            host_clock_id="host-perf-counter-ns",
            source_clock_id=None,
            space=SpaceRef("test-raster", "raster"),
            producer=ProducerRef("test-source", "1.0"),
            envelope_id=lambda packet: "frame-{}".format(
                packet.host_capture_time_ns
            ),
        )
        pump = LatestFramePump(source)
        pump.start()
        try:
            value = pump.read_latest(0.2)
            self.assertEqual("trace-pump", value.identity.trace_id)
            self.assertEqual("test-raster", value.space.space_id)
            self.assertEqual("main", value.value.stream_id)
        finally:
            pump.stop()


if __name__ == "__main__":
    unittest.main()
