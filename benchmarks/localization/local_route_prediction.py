"""Past input compensates observation averaging; future pixels are scorer-only."""
import json
import cv2
import numpy as np
from benchmarks.localization.local_route_control import ROOT,SOURCE,SESSION,read,dump,mapmask,measure

def run():
    images=np.load(ROOT/'minimap.npy',mmap_mode='r');times=np.array([r['time_s'] for r in read(SOURCE/'frames.json')]);mask=mapmask();gain=read(ROOT/'motor-fit.json')['chosen']['gain_deg_per_count'];inputs=[json.loads(s) for s in (SESSION/'inputs.jsonl').read_text().splitlines()];mouse=[r for r in inputs if r['kind']=='pc_raw_mouse'];it=np.array([r['session_time_ns']/1e9 for r in mouse]);dx=np.array([r['payload'].get('delta_x',0) for r in mouse]);out=[]
    for r in read(ROOT/'rows.json'):
        t=r['time_s'];i=r['frame'];past=int(np.argmin(abs(times-t+.5)));dt=t-times[past];motion=r['motion'];v=-np.array(motion['shift'])/dt
        if not motion['accepted'] or np.linalg.norm(v)<2:continue
        base=float(np.degrees(np.arctan2(v[1],v[0])));integral=float(dx[(it>t-dt/2)&(it<=t)].sum());prediction=base+gain*integral
        # Retrospective short symmetric displacement is an approximate current
        # travel-direction reference, not world truth. It is never runtime input.
        a=int(np.argmin(abs(times-t+.15)));b=int(np.argmin(abs(times-t-.15)))
        if times[b]<=t or times[a]>=t:continue
        reference=measure(images[a],images[b],mask);delta=-np.array(reference['shift'])
        if not reference['accepted'] or np.linalg.norm(delta)/(times[b]-times[a])<2:continue
        direction=float(np.degrees(np.arctan2(delta[1],delta[0])));error=lambda x:float((x-direction+180)%360-180)
        out.append(dict(time_s=t,base_deg=base,predicted_deg=prediction,reference_deg=direction,past_mouse_counts=integral,base_error_deg=error(base),predicted_error_deg=error(prediction)))
    dump(ROOT/'current-course-prediction.json',out)
    score={k:dict(samples=len(out),mean_abs=float(np.mean([abs(r[k]) for r in out])),p95_abs=float(np.percentile([abs(r[k]) for r in out],95)),maximum_abs=float(max(abs(r[k]) for r in out))) for k in ['base_error_deg','predicted_error_deg']};dump(ROOT/'current-course-prediction-score.json',score);print(json.dumps(score,indent=2),flush=True)

if __name__=='__main__':cv2.setNumThreads(1);run()
