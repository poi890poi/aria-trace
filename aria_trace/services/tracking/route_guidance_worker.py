"""Own capture/input and a slow scene worker for a human-driven guidance trial."""
import argparse
import json
import os
import threading
import time
from pathlib import Path
import cv2
import numpy as np
from .destination_worker import LearnedFeatures
from .route_context import RouteContext
from .route_guidance import load_guide, LocalCourseGuide, HumanInput


def atomic_json(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')
    try:
        temporary.replace(path)
    except PermissionError:
        # A Windows UI reader may briefly own the old file. The next snapshot
        # retries; its timestamp prevents readers treating the old one as fresh.
        pass


class SceneWorker:
    def __init__(self, model, meta, arrays):
        self.model=model; self.meta=meta; self.memory=RouteContext(meta,arrays)
        self.condition=threading.Condition(); self.pending=None; self.latest=None
        self.error=None; self.stopped=False; self.replaced=0; self.processed=0
        self.thread=threading.Thread(target=self._run,name='route-scene-context',daemon=True)
        self.thread.start()

    def submit(self, image, timestamp):
        with self.condition:
            if self.pending is not None: self.replaced+=1
            self.pending=(image.copy(),timestamp); self.condition.notify_all()

    def snapshot(self):
        with self.condition: return None if self.latest is None else dict(self.latest)

    def _run(self):
        last=-float('inf')
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda:self.stopped or self.pending is not None)
                    if self.stopped:return
                    image,t=self.pending; self.pending=None
                if t-last<.2:continue
                last=t; start=time.perf_counter(); q=self.model.extract(image,self.meta['scene_mask'])
                row=self.memory.step(q,t); row['compute_ms']=(time.perf_counter()-start)*1000
                with self.condition:
                    self.latest=row; self.processed+=1
        except Exception as exc:
            self.error='{}: {}'.format(type(exc).__name__,exc)

    def close(self):
        with self.condition:
            self.stopped=True; self.pending=None; self.condition.notify_all()
        self.thread.join(timeout=4)


class Replay:
    """Paced recorded input; never sends commands to a game."""
    def __init__(self, config, emit):
        folder=Path(config['session']); self.frames=[json.loads(s) for s in (folder/'frames.jsonl').read_text().splitlines()]
        self.inputs=[json.loads(s) for s in (folder/'inputs.jsonl').read_text().splitlines()]
        self.lo=float(config['start_s']); self.hi=float(config['end_s']); self.emit=emit
        self.cap=cv2.VideoCapture(str(folder/'video_main.mkv')); self.index=0; self.ii=0
        # Decode before starting the replay clock; seeking must not manufacture latency.
        while self.index<len(self.frames) and self.frames[self.index]['session_time_ns']/1e9<self.lo:
            ok,_=self.cap.read()
            if not ok:raise RuntimeError('Recorded replay frame missing')
            self.index+=1
        self.origin=time.perf_counter()-self.lo
        # Restore the actual preceding key transitions, not a guessed W state.
        while self.ii<len(self.inputs) and self.inputs[self.ii]['session_time_ns']/1e9<self.lo:
            row=self.inputs[self.ii]
            if row['kind']=='pc_raw_keyboard': self.emit(row['kind'],row['payload'],self.origin+row['session_time_ns']/1e9)
            self.ii+=1

    def read(self):
        while self.index<len(self.frames):
            row=self.frames[self.index]; self.index+=1; ok,image=self.cap.read()
            if not ok:raise RuntimeError('Recorded replay frame missing')
            t=row['session_time_ns']/1e9
            if t<self.lo:continue
            if t>=self.hi:return None
            remaining=self.origin+t-time.perf_counter()
            if remaining>0:time.sleep(remaining)
            while self.ii<len(self.inputs) and self.inputs[self.ii]['session_time_ns']/1e9<=t:
                r=self.inputs[self.ii]; self.emit(r['kind'],r['payload'],self.origin+r['session_time_ns']/1e9); self.ii+=1
            return image,self.origin+t,True
        return None

    def close(self):self.cap.release()


def video_overlay(image, row):
    result=image.copy(); hint=row.get('hint') or {}
    label=hint.get('label') if row.get('available') else row.get('reason','Guidance unavailable')
    cv2.rectangle(result,(10,10),(620,66),(20,25,30),-1)
    cv2.putText(result,str(label),(23,45),cv2.FONT_HERSHEY_SIMPLEX,.8,(230,240,240),2)
    return result


def run(config):
    cv2.setNumThreads(1); output=Path(config['output']); output.mkdir(parents=True,exist_ok=True)
    latest=output/'latest.json'; stop=output/'stop'; scene=None; source=None; inputs=None; replay=None; video=None
    status='stopped'; error=None; count=0; first=None; last_publish=0.; final={}; input_log=None
    atomic_json(latest,dict(status='starting',available=False,hint=None,reason='Loading prepared route'))
    try:
        meta,arrays,maps=load_guide(config['package'],config['game_profile_id'])
        model=LearnedFeatures(config['runtime_root'],'auto')
        warm=np.random.RandomState(913).randint(0,255,(360,640),dtype=np.uint8)
        for _ in range(2):model.extract(warm,meta['scene_mask'])
        scene=SceneWorker(model,meta,arrays); guide=LocalCourseGuide(meta,arrays,maps); human=HumanInput()
        input_log=(output/'inputs.jsonl').open('w',encoding='utf-8')
        with (output/'guidance.jsonl').open('w',encoding='utf-8') as log:
            input_lock=threading.Lock()
            def emit(kind,payload,t):
                human.feed(kind,payload,t)
                with input_lock:
                    if input_log.closed:return  # A late callback during source shutdown.
                    input_log.write(json.dumps(dict(kind=kind,payload=payload,host_time_ns=int(t*1e9)))+'\n');input_log.flush()
            if config.get('record_video'):
                from aria_trace.evidence.route_video import RouteTraceVideoRecorder
                video=RouteTraceVideoRecorder(output/'video',video_overlay,fps=30.)
            if config.get('replay'):
                replay=Replay(config['replay'],emit)
            else:
                from aria_trace.apps.workbench.sources import SourceFactory
                factory=SourceFactory(); source=factory.frame(config['frame_source'])
                inputs=factory.input(dict(adapter='windows_raw_keyboard_mouse',window_title=config['frame_source']['window_title']))
                source.start(); inputs.start(lambda p:emit(p.kind,p.payload,p.host_time_ns/1e9))
            while not stop.exists():
                if scene.error:raise RuntimeError(scene.error)
                if replay:
                    packet=replay.read()
                    if packet is None:break
                    image,t,foreground=packet
                else:
                    packet=source.read()
                    if packet is None:continue
                    image=packet.image; t=packet.host_capture_time_ns/1e9
                    foreground=bool(source.api.is_foreground(source.hwnd))
                started=time.perf_counter(); h,w=image.shape[:2]
                if abs(w/h-1280/720)>.02:
                    raise ValueError('This prepared route expects a 16:9 game view')
                gray=cv2.cvtColor(cv2.resize(image,(1280,720)),cv2.COLOR_BGR2GRAY)
                scene.submit(cv2.resize(gray,(640,360)),t)
                human_state=human.snapshot(t)
                if human_state.get('error'):raise RuntimeError(human_state['error'])
                if not foreground:
                    with human.lock:human.keys.clear()
                    human_state['keys']=[]
                row=guide.step(gray[:180,:220],t,scene.snapshot(),human_state)
                row.update(status='running',capture_ns=int(t*1e9),published_ns=time.perf_counter_ns(),
                           compute_ms=(time.perf_counter()-started)*1000,device=model.device,
                           scene_processed=scene.processed,scene_replaced=scene.replaced,scene_error=scene.error,
                           actual_keys=human_state['keys'],actual_mouse_events=human_state['events'])
                row['capture_to_result_ms']=(row['published_ns']-row['capture_ns'])/1e6
                # Python 3.7 and 3.12 on Windows have different perf-counter origins.
                # Keep capture latency monotonic inside this process; publish an epoch
                # timestamp solely for the Workbench's cross-process freshness check.
                row['published_wall_ns']=time.time_ns()
                if row['capture_to_result_ms']>500:
                    row.update(available=False,hint=None,reason='Capture is too old')
                count+=1; first=t if first is None else first; row['processed_frames']=count; row['update_fps']=count/max(t-first,.001)
                if video:row['video']=video.summary
                log.write(json.dumps(row,allow_nan=False)+'\n'); log.flush(); final=row
                now=time.perf_counter()
                if now-last_publish>=.05:
                    atomic_json(latest,row);last_publish=now
                if video:
                    video.update_state(row);video.submit(image,int(t*1e9))
    except Exception as exc:
        status='failed';error='{}: {}'.format(type(exc).__name__,exc)
        raise
    finally:
        if inputs:inputs.stop()
        if input_log:
            with input_lock:input_log.close()
        if source:source.stop()
        if replay:replay.close()
        atomic_json(latest,dict(final,status='finalizing',available=False,hint=None,reason='Saving guidance evidence'))
        if scene:scene.close()
        video_result=None if video is None else video.close(status='complete' if error is None else 'failed',error=error)
        final.update(status=status,error=error,available=False,hint=None,reason=error or 'Guidance stopped',processed_frames=count,video=video_result)
        atomic_json(latest,final);atomic_json(output/'summary.json',final)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    run(json.loads(Path(parser.parse_args().config).read_text(encoding='utf-8')))
