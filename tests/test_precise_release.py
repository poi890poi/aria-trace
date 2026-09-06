import threading
import queue
import unittest
from unittest.mock import patch

import numpy as np

from benchmarks.localization.precise_release import wait_until, PreciseReleaseSource


class PreciseReleaseTests(unittest.TestCase):
    def test_wait_preserves_absolute_deadline_and_uses_bounded_sleeps(self):
        now, sleeps = [1000], []
        def sleep(seconds):
            sleeps.append(seconds)
            now[0] += round(seconds*1e9)
        with patch('benchmarks.localization.precise_release.time.perf_counter_ns',side_effect=lambda:now[0]), \
             patch('benchmarks.localization.precise_release.time.sleep',side_effect=sleep):
            self.assertTrue(wait_until(12_001_000,threading.Event()))
        self.assertEqual(12_001_000,now[0])
        self.assertEqual(3,len(sleeps))
        self.assertLessEqual(max(sleeps),.005)

    def test_cancellation_does_not_release_frame(self):
        stop = threading.Event()
        with patch('benchmarks.localization.precise_release.time.perf_counter_ns',return_value=0), \
             patch('benchmarks.localization.precise_release.time.sleep',side_effect=lambda _:stop.set()):
            self.assertFalse(wait_until(1_000_000_000,stop))

    def source(self, decoded_index=0):
        source = PreciseReleaseSource.__new__(PreciseReleaseSource)
        source.frames = [{'frame_index':4,'session_time_ns':10}]
        source.index, source.origin = 0, 1000
        source.stop_event, source.finished = threading.Event(), threading.Event()
        source.decoded_queue = queue.Queue()
        self.pixels = np.arange(12,dtype=np.uint8).reshape(2,2,3)
        source.decoded_queue.put((decoded_index,self.pixels,500,600))
        source.prefetch_error, source.prefetch_setup_ms, source.rows = None, 0., []
        return source

    def test_source_preserves_image_and_scheduled_capture_timestamp(self):
        source = self.source()
        with patch('benchmarks.localization.precise_release.wait_until',return_value=True) as wait, \
             patch('benchmarks.localization.precise_release.time.perf_counter_ns',side_effect=[1020,1040]):
            packet = source.read()
        wait.assert_called_once_with(1010,source.stop_event)
        self.assertIs(self.pixels,packet.image)
        self.assertEqual(1010,packet.host_capture_time_ns)
        self.assertEqual(4,source.rows[0]['frame_index'])
        self.assertEqual(10,source.rows[0]['session_time_ns'])
        self.assertEqual(1,source.index)

    def test_canceled_source_does_not_consume_or_publish_queued_image(self):
        source = self.source()
        with patch('benchmarks.localization.precise_release.wait_until',return_value=False):
            self.assertIsNone(source.read())
        self.assertEqual(0,source.index)
        self.assertEqual(1,source.decoded_queue.qsize())
        self.assertEqual([],source.rows)

    def test_source_rejects_reordered_decode(self):
        source = self.source(decoded_index=1)
        with patch('benchmarks.localization.precise_release.wait_until',return_value=True):
            with self.assertRaisesRegex(RuntimeError,'reordered'):
                source.read()
        self.assertEqual([],source.rows)


if __name__ == '__main__':
    unittest.main()
