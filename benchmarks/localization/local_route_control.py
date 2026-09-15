"""Physical local minimap relation for an offline course-control screen."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
from benchmarks.localization.route_guidance_trial import SOURCE, SESSION, ROOT as SCENE, read, dump

ROOT=Path('artifacts/poc/local-route-control-20260915')

def mapmask():
    c=read('artifacts/workbench/minimap_calibrations/genshin-impact-pc/segments-df624035-833-bd07601f-708/calibration.json')
    b=c['outer_boundary'];y,x=np.indices((180,220));m=(x-b['center_x'])**2+(y-b['center_y'])**2<(b['radius']-6)**2
    pivot=c['rotation_center'];cx=pivot.get('x',b['center_x']);cy=pivot.get('y',b['center_y']);m&=(x-cx)**2+(y-cy)**2>14**2;m[:58,:75]=False
    return np.uint8(m)*255

def shift(a,b,mask):
    weight=mask.astype(np.float32)/255;aa=a.astype(np.float32);bb=b.astype(np.float32)
    aa=(aa-cv2.GaussianBlur(aa,(0,0),3))*weight;bb=(bb-cv2.GaussianBlur(bb,(0,0),3))*weight
    h,w=a.shape;hh=cv2.getOptimalDFTSize(h);ww=cv2.getOptimalDFTSize(w)
    while hh%2:hh=cv2.getOptimalDFTSize(hh+1)
    while ww%2:ww=cv2.getOptimalDFTSize(ww+1)
    window=cv2.createHanningWindow((w,h),cv2.CV_32F)*weight
    aa=cv2.copyMakeBorder(aa*window,0,hh-h,0,ww-w,cv2.BORDER_CONSTANT)
    bb=cv2.copyMakeBorder(bb*window,0,hh-h,0,ww-w,cv2.BORDER_CONSTANT)
    z,response=cv2.phaseCorrelate(aa,bb)
    return np.array(z),float(response)

def cache():
    ROOT.mkdir(exist_ok=True);rows=read(SOURCE/'frames.json');target=ROOT/'minimap.npy'
    if target.exists():raise RuntimeError('Existing cache preserved')
    images=np.lib.format.open_memmap(target,mode='w+',dtype=np.uint8,shape=(len(rows),180,220));cap=cv2.VideoCapture(str(SESSION/'video_main.mkv'))
    started=time.perf_counter()
    for i in range(len(rows)):
        ok,im=cap.read()
        if not ok:raise RuntimeError('Missing source frame '+str(i))
        images[i]=cv2.cvtColor(im[:180,:220],cv2.COLOR_BGR2GRAY)
        if i%1000==0:print('Native map cache',i,'/',len(rows),flush=True)
    cap.release();images.flush();dump(ROOT/'cache.json',dict(frames=len(rows),source=str(SESSION/'video_main.mkv'),timestamp_source=str(SOURCE/'frames.json'),decode_seconds=time.perf_counter()-started))

def measure(a,b,mask):
    z,response=shift(a,b,mask)
    return dict(shift=z.tolist(),response=response,accepted=bool(response>=.35 and np.linalg.norm(z)<=20))

def run():
    images=np.load(ROOT/'minimap.npy',mmap_mode='r');times=np.array([r['time_s'] for r in read(SOURCE/'frames.json')]);mask=mapmask();bank=read(SCENE/'bank.json');scene=read(SCENE/'vlad-local.json');teacher={};pre=[]
    for v in bank:
        i=v['frame'];future={}
        for h in [1.,3.]:
            j=int(np.argmin(abs(times-v['time_s']-h)))
            if times[j]>=124:continue
            future[str(h)]=dict(measure(images[i],images[j],mask),frame=j,time_s=float(times[j]))
        teacher[v['time_s']]=dict(v,future=future);pre.append(teacher[v['time_s']])
    dump(ROOT/'preparation.json',pre);out=[];last=None;last_t=-999.
    for n,r in enumerate(scene):
        began=time.perf_counter();i=r['frame'];t=r['time_s']
        if r['teacher_time_s'] is not None:last=teacher[r['teacher_time_s']];last_t=t
        past=int(np.argmin(abs(times-t+.5)));motion=measure(images[past],images[i],mask);relation=None;hints={}
        if last is not None and t-last_t<=1:
            relation=measure(images[last['frame']],images[i],mask)
            if relation['accepted'] and motion['accepted']:
                velocity=-np.array(motion['shift'])/max(t-times[past],.001)
                for horizon,goal in last['future'].items():
                    if not goal['accepted']:continue
                    delta=np.array(relation['shift'])-goal['shift'];speed=float(np.linalg.norm(velocity));distance=float(np.linalg.norm(delta));angle=None
                    if speed>=.2 and distance>=.2:angle=float(np.degrees(np.arctan2(velocity[0]*delta[1]-velocity[1]*delta[0],np.dot(velocity,delta))))
                    hints[horizon]=dict(course_difference_deg=angle,target_vector=delta.tolist(),travel_vector=velocity.tolist(),speed=speed,distance=distance)
        out.append(dict(frame=i,time_s=t,teacher_time_s=None if last is None else last['time_s'],scene_age_s=None if last is None else t-last_t,map_relation=relation,motion=motion,hints=hints,elapsed_ms=(time.perf_counter()-began)*1000))
        if n%150==0:print('Local relation',n,'/',len(scene),flush=True)
    dump(ROOT/'rows.json',out)
    print('Relations',sum(r['map_relation'] is not None and r['map_relation']['accepted'] for r in out),'/',len(out),'hints',{h:sum(h in r['hints'] and r['hints'][h]['course_difference_deg'] is not None for r in out) for h in ['1.0','3.0']},flush=True)

def controls():
    images=np.load(ROOT/'minimap.npy',mmap_mode='r');mask=mapmask();out=[]
    for i in np.linspace(0,len(images)-1,20,dtype=int):
        for dx,dy in [(0,0),(.25,0),(0,.25),(1,0),(-1,0),(0,1),(0,-1),(3,-2)]:
            b=cv2.warpAffine(images[i],np.float32([[1,0,dx],[0,1,dy]]),(220,180),borderMode=cv2.BORDER_REFLECT);r=measure(images[i],b,mask);r.update(frame=int(i),truth=[dx,dy],error=float(np.linalg.norm(np.array(r['shift'])-[dx,dy])));out.append(r)
    dump(ROOT/'controls.json',out);print('Map pixel controls P95/max',np.percentile([r['error'] for r in out],[95,100]),flush=True)

def path_progress():
    """Change the target rule: advance along a physical local demo polyline."""
    images=np.load(ROOT/'minimap.npy',mmap_mode='r');times=np.array([r['time_s'] for r in read(SOURCE/'frames.json')]);mask=mapmask();paths={};pre=[]
    for v in read(SCENE/'bank.json'):
        i=v['frame'];t=v['time_s'];nodes={i:np.zeros(2)};failures=[]
        for direction in [-1,1]:
            before=i;position=np.zeros(2)
            for target in np.arange(t+direction*.25,t+direction*(1.5 if direction<0 else 3)+direction*.01,direction*.25):
                if not 0<=target<124:break
                j=int(np.argmin(abs(times-target)));r=measure(images[before],images[j],mask)
                if not r['accepted']:failures.append(float(times[j]));break
                position=position-np.array(r['shift']);nodes[j]=position.copy();before=j
        indices=sorted(nodes);p=np.array([nodes[j] for j in indices]);record=dict(frame=i,time_s=t,points=p.tolist(),times_s=[float(times[j]) for j in indices],failures_s=failures)
        paths[t]=record;pre.append(record)
    dump(ROOT/'local-paths.json',pre);out=[]
    for r in read(ROOT/'rows.json'):
        goal=None;v=paths.get(r['teacher_time_s']);relation=r['map_relation'];motion=r['motion']
        if v and relation and relation['accepted'] and motion['accepted']:
            past=int(np.argmin(abs(times-r['time_s']+.5)));dt=max(r['time_s']-times[past],.001);position=-np.array(relation['shift']);velocity=-np.array(motion['shift'])/dt;speed=float(np.linalg.norm(velocity));path=np.array(v['points']);target=follow_path(path,position,speed*1.)
            if target is not None and speed>=.2:
                delta=target['goal']-position;distance=float(np.linalg.norm(delta))
                if distance>=.2:goal=dict(course_difference_deg=float(np.degrees(np.arctan2(velocity[0]*delta[1]-velocity[1]*delta[0],np.dot(velocity,delta)))),target_vector=delta.tolist(),travel_vector=velocity.tolist(),cross_track_px=target['cross_track'],remaining_px=target['remaining'],position=position.tolist(),goal=target['goal'].tolist(),path=v['points'],speed=speed)
        out.append(dict(frame=r['frame'],time_s=r['time_s'],teacher_time_s=r['teacher_time_s'],hint=goal))
    dump(ROOT/'path-rows.json',out);print('Path-progress hints',sum(r['hint'] is not None for r in out),'/',len(out),flush=True)

def follow_path(path,position,lookahead):
    if len(path)<2:return None
    start=path[:-1];vectors=np.diff(path,axis=0);length=np.linalg.norm(vectors,axis=1);valid=length>1e-6
    if not valid.any():return None
    fractions=np.clip(np.sum((position-start)*vectors,axis=1)/np.maximum(length**2,1e-9),0,1);nearest=start+fractions[:,None]*vectors;distance=np.linalg.norm(nearest-position,axis=1);distance[~valid]=np.inf;k=int(np.argmin(distance));arc=np.r_[0,np.cumsum(length)];s=arc[k]+fractions[k]*length[k];wanted=s+lookahead
    if wanted>arc[-1]:return None
    j=min(len(length)-1,int(np.searchsorted(arc,wanted,side='right')-1));goal=path[j]+(wanted-arc[j])/max(length[j],1e-9)*vectors[j];difference=position-nearest[k];cross=(vectors[k,0]*difference[1]-vectors[k,1]*difference[0])/length[k]
    return dict(goal=goal,cross_track=float(cross),remaining=float(arc[-1]-s))

def corridor():
    """Prepare the observed corridor across demonstrations, not an exact zero line."""
    images=np.load(ROOT/'minimap.npy',mmap_mode='r');times=np.array([r['time_s'] for r in read(SOURCE/'frames.json')]);mask=mapmask();paths={r['time_s']:r for r in read(ROOT/'local-paths.json')};bank=read(SCENE/'bank.json');prepared={};prep=[]
    for v in bank:
        alternatives=[]
        for lap in range(3):
            if lap==v['lap']:alternatives.append(dict(lap=lap,frame=v['frame'],points=paths[v['time_s']]['points'],response=1.));continue
            candidates=[]
            for a in bank:
                if a['lap']!=lap or abs(a['phase']-v['phase'])>.08:continue
                m=measure(images[a['frame']],images[v['frame']],mask)
                if m['accepted']:candidates.append((np.linalg.norm(m['shift']),a,m))
            if candidates:
                _,a,m=min(candidates,key=lambda x:x[0]);points=np.array(paths[a['time_s']]['points'])+m['shift'];alternatives.append(dict(lap=lap,frame=a['frame'],points=points.tolist(),response=m['response']))
        record=dict(frame=v['frame'],time_s=v['time_s'],paths=alternatives);prepared[v['time_s']]=record;prep.append(record)
    dump(ROOT/'corridor-preparation.json',prep);out=[]
    for r in read(ROOT/'rows.json'):
        relation=r['map_relation'];motion=r['motion'];record=prepared.get(r['teacher_time_s']);hints=[];hint=None
        if record and relation and relation['accepted'] and motion['accepted']:
            past=int(np.argmin(abs(times-r['time_s']+.5)));dt=max(r['time_s']-times[past],.001);position=-np.array(relation['shift']);velocity=-np.array(motion['shift'])/dt;speed=float(np.linalg.norm(velocity))
            if speed>=.2:
                for p in record['paths']:
                    target=follow_path(np.array(p['points']),position,speed)
                    if target is None:continue
                    delta=target['goal']-position
                    if np.linalg.norm(delta)<.2:continue
                    angle=float(np.degrees(np.arctan2(velocity[0]*delta[1]-velocity[1]*delta[0],np.dot(velocity,delta))))
                    hints.append(dict(lap=p['lap'],angle=angle,goal=target['goal'].tolist(),cross_track_px=target['cross_track']))
                if hints:
                    lo=min(x['angle'] for x in hints);hi=max(x['angle'] for x in hints)
                    hint=dict(course_window_deg=[lo,hi],course_correction_deg=lo if lo>0 else hi if hi<0 else 0.,demonstrations=len(hints),position=position.tolist(),velocity=velocity.tolist(),targets=hints)
        out.append(dict(frame=r['frame'],time_s=r['time_s'],teacher_time_s=r['teacher_time_s'],hint=hint))
    dump(ROOT/'corridor-rows.json',out);print('Corridor hints',sum(r['hint'] is not None for r in out),'/',len(out),flush=True)

if __name__=='__main__':
    cv2.setNumThreads(1);p=argparse.ArgumentParser();p.add_argument('phase',choices=['cache','run','controls','path_progress','corridor']);globals()[p.parse_args().phase]()
