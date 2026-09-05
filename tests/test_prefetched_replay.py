import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from benchmarks.localization.prefetched_replay import PrefetchedSource


class PrefetchedReplayTests(unittest.TestCase):
    def test_decode_ahead_preserves_order_and_never_releases_a_future_frame(self):
        class Video:
            index=0
            def isOpened(self): return True
            def read(self):
                image=np.full((2,2,3),self.index,dtype=np.uint8)
                self.index+=1
                return True,image
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'frames.jsonl').write_text(''.join(json.dumps({"frame_index":i,"session_time_ns":i*1_000_000})+'\n' for i in range(3)))
            source=PrefetchedSource(root)
            with patch('benchmarks.localization.run_workbench_replay.cv2.VideoCapture',return_value=Video()):
                source.start()
            try:
                for i in range(3):
                    packet=source.read()
                    self.assertEqual(int(packet.image[0,0,0]),i)
                    self.assertEqual(packet.host_capture_time_ns,source.origin+i*1_000_000)
                    self.assertGreaterEqual(time.perf_counter_ns(),packet.host_capture_time_ns)
                self.assertIsNone(source.read())
                self.assertTrue(source.finished.is_set())
            finally:
                source.stop()


if __name__=='__main__': unittest.main()
