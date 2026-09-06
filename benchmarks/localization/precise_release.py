"""Recorded-source wait experiment preserving every original release deadline."""

import queue
import time

from benchmarks.localization.prefetched_replay import PrefetchedSource
from rig_runtime.domain.packets import FramePacket


def wait_until(deadline_ns, stop_event):
    """Never release early; check cancellation between sleeps requested at <=5 ms."""
    while not stop_event.is_set():
        remaining = (deadline_ns-time.perf_counter_ns())/1e9
        if remaining <= 0:
            return True
        time.sleep(min(.005, remaining))
    return False


class PreciseReleaseSource(PrefetchedSource):
    def read(self):
        if self.index >= len(self.frames) or self.stop_event.is_set():
            self.finished.set()
            self.stop_event.wait(.005)
            return None
        frame = self.frames[self.index]
        scheduled = self.origin+frame['session_time_ns']
        if not wait_until(scheduled,self.stop_event):
            return None
        released = time.perf_counter_ns()
        while not self.stop_event.is_set():
            try:
                index,image,decode_start,decode_end = self.decoded_queue.get(timeout=.1)
                break
            except queue.Empty:
                if self.prefetch_error:
                    raise self.prefetch_error
        else:
            return None
        available = time.perf_counter_ns()
        if index != self.index:
            raise RuntimeError('Prefetch reordered input frames')
        self.rows.append({'frame_index':frame['frame_index'],'session_time_ns':frame['session_time_ns'],
            'host_time_ns':scheduled,'decode_ms':(decode_end-decode_start)/1e6,
            'decode_finished_host_time_ns':decode_end,'release_lateness_ms':(released-scheduled)/1e6,
            'prefetch_wait_ms':(available-released)/1e6,'prefetch_setup_ms':self.prefetch_setup_ms,
            'source_pipeline':'bounded-decode-ahead-precise-wait','prefetch_buffer_frames':self.buffer_frames})
        self.index += 1
        return FramePacket('main',image,scheduled,available)
