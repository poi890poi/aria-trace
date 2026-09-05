"""Bounded decode-ahead experiment; release times and tracker inputs stay causal.

This changes the recorded-source adapter, not physical capture or the tracker.
Startup priming is reported separately and never presented as an engine speedup.
"""
import argparse
import queue
import threading
import time
from pathlib import Path
from unittest.mock import patch

from rig_runtime.domain.packets import FramePacket
from benchmarks.localization import run_workbench_replay as replay


class PrefetchedSource(replay.RecordedSource):
    buffer_frames = 30

    def start(self):
        super().start()
        began=time.perf_counter_ns()
        self.decoded_queue=queue.Queue(maxsize=self.buffer_frames)
        ready=threading.Event()
        def decode():
            try:
                for index, frame in enumerate(self.frames):
                    if self.stop_event.is_set():
                        break
                    started=time.perf_counter_ns()
                    ok,image=self.capture.read()
                    ended=time.perf_counter_ns()
                    if not ok:
                        raise RuntimeError("Unexpected prefetch EOF at frame "+str(index))
                    while not self.stop_event.is_set():
                        try:
                            self.decoded_queue.put((index,image,started,ended),timeout=.1)
                            break
                        except queue.Full:
                            pass
                    if index+1>=min(self.buffer_frames,len(self.frames)):
                        ready.set()
            except Exception as exc:
                self.prefetch_error=exc
            finally:
                ready.set()
        self.prefetch_error=None
        self.decoder=threading.Thread(target=decode,name="recorded-decode-ahead",daemon=True)
        self.decoder.start()
        if not ready.wait(30):
            self.stop()
            raise RuntimeError("Decode prefetch did not initialize")
        if self.prefetch_error:
            raise self.prefetch_error
        self.prefetch_setup_ms=(time.perf_counter_ns()-began)/1e6
        self.origin=time.perf_counter_ns()+100_000_000

    def read(self):
        if self.index>=len(self.frames) or self.stop_event.is_set():
            self.finished.set()
            self.stop_event.wait(.005)
            return None
        frame=self.frames[self.index]
        scheduled=self.origin+frame["session_time_ns"]
        if self.stop_event.wait(max(0,(scheduled-time.perf_counter_ns())/1e9)):
            return None
        released=time.perf_counter_ns()
        while not self.stop_event.is_set():
            try:
                index,image,decode_start,decode_end=self.decoded_queue.get(timeout=.1)
                break
            except queue.Empty:
                if self.prefetch_error:
                    raise self.prefetch_error
        else:
            return None
        available=time.perf_counter_ns()
        if index!=self.index:
            raise RuntimeError("Prefetch reordered input frames")
        self.rows.append({"frame_index":frame["frame_index"],"session_time_ns":frame["session_time_ns"],
            "host_time_ns":scheduled,"decode_ms":(decode_end-decode_start)/1e6,
            "decode_finished_host_time_ns":decode_end,"release_lateness_ms":(released-scheduled)/1e6,
            "prefetch_wait_ms":(available-released)/1e6,"prefetch_setup_ms":self.prefetch_setup_ms,
            "source_pipeline":"bounded-decode-ahead","prefetch_buffer_frames":self.buffer_frames})
        self.index+=1
        return FramePacket("main",image,scheduled,available)

    def stop(self):
        super().stop()
        decoder=getattr(self,"decoder",None)
        if decoder is not None:
            decoder.join(timeout=5)
            if decoder.is_alive():
                raise RuntimeError("Prefetch decoder did not stop")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",nargs="+",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-seconds",type=float)
    p.add_argument("--mode",default="free-roam",choices=["free-roam","route-assisted"])
    p.add_argument("--record-video",action="store_true")
    args=p.parse_args()
    args.atlas="08b6f2d6-820a-4bfd-875a-6a55d1986a4e"
    args.calibration="segments-df624035-833-bd07601f-708"
    args.scene_yaw="01dbaa74-8e00-4763-a215-9ea37e18b1b2"
    args.cache=Path("artifacts/benchmark_cache/atlas_references")
    args.references=Path("artifacts/poc/workbench-rebuilt-atlas-20260905/references/references.json")
    args.references_only=False
    args.loss_error_limit_px=None
    args.experiment={"variant":"decode-ahead", "buffer_frames":30,
        "scope":"recorded-source adapter only; production tracker unchanged"}
    with patch.object(replay,"RecordedSource",PrefetchedSource):
        replay.run(args)


if __name__=="__main__":
    main()
