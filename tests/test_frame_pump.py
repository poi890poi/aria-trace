import threading
import time
import unittest

import numpy as np

from acquisition.frame_pump import LatestFramePump
from acquisition.models import FramePacket


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
    def test_slow_consumer_gets_latest_frame_and_drop_count(self):
        source = FastSource()
        pump = LatestFramePump(source)
        pump.start()
        try:
            first = pump.read_latest(0.2)
            time.sleep(0.02)
            second = pump.read_latest(0.2)
            self.assertGreater(second.host_capture_time_ns, first.host_capture_time_ns)
            self.assertGreater(pump.dropped_before_processing, 0)
        finally:
            pump.stop()


if __name__ == "__main__":
    unittest.main()
