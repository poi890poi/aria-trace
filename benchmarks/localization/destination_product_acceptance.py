"""Product observer parity and actual bounded worker, driven by recorded frames."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np
from aria_trace.services.tracking.destination import DestinationObserver,load_package
from aria_trace.services.tracking.destination_monitor import DestinationMonitor

ROOT=Path('artifacts/poc/destination-product-20260914')
NATIVE=Path('artifacts/poc/segment-loop-20260913')
PASSAGE=Path('artifacts/poc/passage-loop-20260913')
PACKAGES=Path('artifacts/workbench/destination_targets/genshin-impact-pc')

def read(path):return json.loads(path.read_text())
def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))
def stats(values):return dict(count=len(values),median=float(np.median(values)),p95=float(np.percentile(values,95)),maximum=float(max(values))) if values else None

def parity():
    ROOT.mkdir(parents=True,exist_ok=True);ims=np.load(NATIVE/'native-gray.npy',mmap_mode='r');qq=np.load(PASSAGE/'query.npz');cache={k:qq[k] for k in qq.files};out=[]
    for name,pkg in [('outdoor','bridge-gate-two-demo'),('town','town-gate-two-demo')]:
        rr=read(PASSAGE/'two-demo'/(name+'-lap3-multi_lk.json'));observer=DestinationObserver(*load_package(PACKAGES/pkg));rows=[]
        for ref in rr:
            i=ref['frame'];result=observer.step(ims[i],(cache[str(i)+'p'],cache[str(i)+'d']),ref['time_s']);same=(ref['aim'] is None)==(result['aim'] is None);error=None if ref['aim'] is None or result['aim'] is None else float(np.linalg.norm(np.array(ref['aim'])-result['aim']));rows.append(dict(frame=i,same_availability=same,error=error,same_kind=ref['kind']==result['kind']))
        summary=dict(name=name,frames=len(rows),availability_mismatches=sum(not r['same_availability'] for r in rows),kind_mismatches=sum(not r['same_kind'] for r in rows),coordinate_difference=stats([r['error'] for r in rows if r['error'] is not None]));out.append(summary);dump(ROOT/(name+'-parity-rows.json'),rows);print(summary,flush=True)
    dump(ROOT/'parity.json',out)

def paced():
    ROOT.mkdir(parents=True,exist_ok=True);ims=np.load(NATIVE/'native-gray.npy',mmap_mode='r');data=read(NATIVE/'native-frames.json');summary=[]
    for name,pkg,lo,hi in [('outdoor','bridge-gate-two-demo',192,198),('town','town-gate-two-demo',232,238)]:
        directory=ROOT/('worker-'+name);monitor=DestinationMonitor(PACKAGES/pkg,directory);end=time.monotonic()+65
        while monitor.status=='starting' and time.monotonic()<end:time.sleep(.02)
        if monitor.status!='running':monitor.close();raise RuntimeError(monitor.snapshot())
        samples=[r for r in data if lo<=r['time_s']<hi];began=time.perf_counter();submit=[];snapshots=[]
        try:
            for r in samples:
                target=began+r['time_s']-samples[0]['time_s'];time.sleep(max(0,target-time.perf_counter()));im=cv2.cvtColor(ims[r['frame']],cv2.COLOR_GRAY2BGR)
                # Declared brief blackout: test loss and recovery through actual IPC.
                if lo+2<=r['time_s']<lo+2.6:im[:]=0
                t=time.perf_counter();monitor.submit(im,time.perf_counter_ns());submit.append((time.perf_counter()-t)*1000);snapshots.append(monitor.snapshot())
            time.sleep(.15);(directory/'last-overlay.jpg').write_bytes(monitor.overlay());value=monitor.snapshot()
        finally:monitor.close()
        raw=[json.loads(line) for line in (directory/'observations.jsonl').read_text().splitlines()];value.update(submit_ms=stats(submit),worker_result_ms=stats([r['capture_to_result_ms'] for r in raw]),fresh_at_publication=sum(r['fresh_at_publication'] for r in raw),raw_observations=len(raw),source_interval=[lo,hi],blackout_interval=[lo+2,lo+2.6],worker_exited=monitor.process.poll() is not None,thread_exited=not monitor.thread.is_alive());summary.append(value);dump(directory/'snapshots.json',snapshots);print({k:v for k,v in value.items() if k not in ['points']},flush=True)
    dump(ROOT/'paced.json',summary)

if __name__=='__main__':
    cv2.setNumThreads(1);a=argparse.ArgumentParser();a.add_argument('phase',choices=['parity','paced']);globals()[a.parse_args().phase]()
