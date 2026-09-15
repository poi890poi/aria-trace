"""Fit a tiny body-response calibration from teacher input and measured map course."""
import argparse
import json
import cv2
import numpy as np
from benchmarks.localization.local_route_control import ROOT,SOURCE,SESSION,read,dump,mapmask,measure

def run(session='21'):
    from pathlib import Path
    prefix='motor' if session=='21' else 'run20-motor';split=124 if session=='21' else 160
    mapfile=ROOT/'minimap.npy' if session=='21' else Path('artifacts/poc/map-assisted-pair-20260913/minimap.npy');timefile=SOURCE/'frames.json' if session=='21' else Path('artifacts/poc/tiny-controller-20260912/selection.json');folder=SESSION if session=='21' else Path('sessions/workbench/recordings-genshin-impact-pc/run_20')
    images=np.load(mapfile,mmap_mode='r');times=np.array([r['time_s'] for r in read(timefile)]);mask=mapmask();ids=sorted(set(int(np.argmin(abs(times-t))) for t in np.arange(.5,times[-1],.1)));rows=[]
    inputs=[json.loads(s) for s in (folder/'inputs.jsonl').read_text().splitlines()];mouse=[r for r in inputs if r['kind']=='pc_raw_mouse'];it=np.array([r['session_time_ns']/1e9 for r in mouse]);dx=np.array([r['payload'].get('delta_x',0) for r in mouse]);keyboard=[r for r in inputs if r['kind']=='pc_raw_keyboard'];held=set();k=0
    for n,i in enumerate(ids):
        t=times[i];j=int(np.argmin(abs(times-t+.5)));m=measure(images[j],images[i],mask);v=-np.array(m['shift'])/(t-times[j]);speed=float(np.linalg.norm(v));angle=float(np.degrees(np.arctan2(v[1],v[0]))) if m['accepted'] and speed>2 else None
        while k<len(keyboard) and keyboard[k]['session_time_ns']/1e9<=t:
            p=keyboard[k]['payload'];key=p.get('key_name');held.add(key) if p.get('pressed') else held.discard(key);k+=1
        rows.append(dict(frame=i,time_s=float(t),course=angle,speed=speed,forward_only=bool('W' in held and not (held&set('ASD'))),motion=m))
        if n%600==0:print('Body response',n,'/',len(ids),flush=True)
    dump(ROOT/(prefix+'-motion.json'),rows);data=[];rt=np.array([r['time_s'] for r in rows])
    for r in rows:
        previous=rows[int(np.argmin(abs(rt-r['time_s']+.3)))];dt=r['time_s']-previous['time_s']
        if r['course'] is None or previous['course'] is None or dt<.2 or not r['forward_only'] or not previous['forward_only']:continue
        rate=((r['course']-previous['course']+180)%360-180)/dt;u={}
        for lag in [0.,.1,.2,.3,.4,.5,.6,.8,1.]:
            hi=r['time_s']-lag;lo=hi-dt;u[str(lag)]=float(dx[(it>=lo)&(it<hi)].sum()/dt)
        data.append(dict(time_s=r['time_s'],course_rate=rate,input_rates=u))
    fits=[]
    for lag in [0.,.1,.2,.3,.4,.5,.6,.8,1.]:
        samples=[r for r in data if r['time_s']<split and abs(r['input_rates'][str(lag)])>150];x=np.array([r['input_rates'][str(lag)] for r in samples]);y=np.array([r['course_rate'] for r in samples]);gain=float(np.dot(x,y)/np.dot(x,x));error=y-gain*x;fits.append(dict(lag_s=lag,gain_deg_per_count=gain,teacher_samples=len(x),teacher_rmse=float(np.sqrt(np.mean(error**2))),teacher_mae=float(np.mean(abs(error)))))
    chosen=min(fits,key=lambda x:x['teacher_rmse']);lag=str(chosen['lag_s']);test=[r for r in data if r['time_s']>=split and abs(r['input_rates'][lag])>150];error=np.array([r['course_rate']-chosen['gain_deg_per_count']*r['input_rates'][lag] for r in test]);baseline=np.array([r['course_rate'] for r in test]);score=dict(chosen=chosen,candidates=fits,later_turn_samples=len(test),later_mae=float(np.mean(abs(error))),later_p95_abs=float(np.percentile(abs(error),95)),no_input_model_mae=float(np.mean(abs(baseline))),no_input_model_p95=float(np.percentile(abs(baseline),95)))
    dump(ROOT/(prefix+'-fit.json'),score);dump(ROOT/(prefix+'-regression-rows.json'),data);print(json.dumps(score,indent=2),flush=True)

if __name__=='__main__':
    cv2.setNumThreads(1);p=argparse.ArgumentParser();p.add_argument('--session',choices=['20','21'],default='21');run(p.parse_args().session)
