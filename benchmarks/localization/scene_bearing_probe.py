"""Image-only feature-bearing assay; never issues control or claims yaw truth."""

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from benchmarks.localization.precision_candidates import quadratic_peak
from benchmarks.localization.reference_cache import identity
from benchmarks.localization.run_workbench_replay import distribution, read_rows


def scene_mask(gray):
    h, w = gray.shape
    mask = np.zeros_like(gray)
    mask[int(.16*h):int(.75*h), int(.20*w):int(.82*w)] = 255
    mask[:int(.30*h), :int(.23*w)] = 0
    mask[int(.36*h):, int(.43*w):int(.59*w)] = 0
    return mask


def seed_points(gray, mask_scene=False):
    h,w=gray.shape; mask=np.zeros_like(gray)
    # Upper-central scene: excludes HUD corners, player body, and lower UI.
    mask[int(.22*h):int(.53*h),int(.28*w):int(.72*w)]=255
    if mask_scene:
        mask = cv2.bitwise_and(mask, scene_mask(gray))
    points=cv2.goodFeaturesToTrack(gray,40,.02,12,mask=mask,blockSize=5)
    return points.reshape(-1,2).astype(np.float32) if points is not None else np.empty((0,2),np.float32)


def patch_at(gray,point):
    return cv2.getRectSubPix(gray,(17,17),tuple(float(x) for x in point))


def correlate(template,gray,guess,radius,margin_min=0.):
    h,w=gray.shape; x,y=guess; half=8
    left=max(0,int(round(x))-radius-half);top=max(0,int(round(y))-radius-half)
    right=min(w,int(round(x))+radius+half+1);bottom=min(h,int(round(y))+radius+half+1)
    region=gray[top:bottom,left:right]
    if min(region.shape)<17 or np.std(template)<3:return None
    response=cv2.matchTemplate(region,template,cv2.TM_CCOEFF_NORMED)
    response=np.nan_to_num(response,nan=-1.,posinf=-1.,neginf=-1.)
    _,score,_,location=cv2.minMaxLoc(response)
    if score<.8:return None
    excluded=response.copy();px,py=location
    excluded[max(0,py-3):py+4,max(0,px-3):px+4]=-1
    if score-float(excluded.max())<margin_min:return None
    dx,dy=quadratic_peak(response,location)
    point=np.array([left+px+half+dx,top+py+half+dy],np.float32)
    if np.std(patch_at(gray,point))<3:return None
    return point


class BearingTracker:
    def __init__(self,gray,points,method,reacquire=False,mask_scene=False):
        self.previous=gray.copy();self.points=points.copy();self.alive=np.ones(len(points),bool)
        self.templates=[patch_at(gray,p) for p in points];self.method=method;self.reacquire=reacquire;self.sequence=0
        self.allowed = scene_mask(gray) if mask_scene else None

    def allowed_point(self, point):
        if self.allowed is None:
            return True
        x, y = np.rint(point).astype(int)
        h, w = self.allowed.shape
        return 8 <= x < w-8 and 8 <= y < h-8 and bool(np.all(self.allowed[y-8:y+9, x-8:x+9]))

    def update(self,gray):
        began=time.perf_counter();ids=np.flatnonzero(self.alive);old=self.points.copy();good=np.zeros(len(old),bool)
        if len(ids) and self.method=='lk':
            source=old[ids].reshape(-1,1,2)
            forward,status,_=cv2.calcOpticalFlowPyrLK(self.previous,gray,source,None,winSize=(21,21),maxLevel=3,
                                                   criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
            backward,back_status,_=cv2.calcOpticalFlowPyrLK(gray,self.previous,forward,None,winSize=(21,21),maxLevel=3,
                                                         criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
            valid=(status.ravel()>0)&(back_status.ravel()>0)&(np.linalg.norm(backward.reshape(-1,2)-old[ids],axis=1)<=1.)
            h,w=gray.shape
            for k,idx in enumerate(ids):
                p=forward[k,0]
                if valid[k] and 8<=p[0]<w-8 and 8<=p[1]<h-8 and np.std(patch_at(gray,p))>=3 and self.allowed_point(p):
                    self.points[idx]=p;good[idx]=True
        elif len(ids):
            for idx in ids:
                p=correlate(patch_at(self.previous,old[idx]),gray,old[idx],12)
                back=correlate(patch_at(gray,p),self.previous,old[idx],12) if p is not None else None
                if back is not None and np.linalg.norm(back-old[idx])<=1. and self.allowed_point(p):
                    self.points[idx]=p;good[idx]=True
        reacquired=[]
        if self.reacquire:
            # Bounded work: inspect one quarter of initial IDs per update.
            for idx in np.flatnonzero(~good):
                if idx%4!=self.sequence%4:continue
                p=correlate(self.templates[idx],gray,self.points[idx],45,margin_min=.08)
                if p is not None and self.allowed_point(p):self.points[idx]=p;good[idx]=True;reacquired.append(int(idx))
        self.alive=good;self.previous=gray.copy();self.sequence+=1
        return {"active_ids":np.flatnonzero(good).tolist(),"points":self.points[good].tolist(),"reacquired_ids":reacquired,
                "elapsed_ms":(time.perf_counter()-began)*1000}


def assay(frames,times,method,reacquire=False,truth_shift=None,blackout=None,mask_scene=False):
    points=seed_points(frames[0],mask_scene);tracker=BearingTracker(frames[0],points,method,reacquire,mask_scene);records=[];errors=[]
    lost=None;longest=0.;recovery_after_blackout=None
    for i,gray in enumerate(frames[1:],1):
        row=tracker.update(gray);row.update(time_s=float(times[i]),sample_index=i)
        if len(row['active_ids'])<3 and lost is None:lost=times[i]
        if len(row['active_ids'])>=3 and lost is not None:longest=max(longest,times[i]-lost);lost=None
        if truth_shift is not None and row['active_ids']:
            expected=points[row['active_ids']]+truth_shift[i]
            row['synthetic_error_px']=np.linalg.norm(np.asarray(row['points'])-expected,axis=1).tolist();errors+=row['synthetic_error_px']
        if blackout and times[i]>blackout[1] and recovery_after_blackout is None and len(row['active_ids'])>=3:recovery_after_blackout=times[i]-blackout[1]
        records.append(row)
    if lost is not None:longest=max(longest,times[-1]-lost)
    return {"method":method,"reacquire":reacquire,"mask_scene":mask_scene,"initial_features":len(points),"final_features":len(records[-1]['active_ids']),
            "initial_points":points.tolist(),"availability_at_least_3_features":float(np.mean([len(r['active_ids'])>=3 for r in records])),
            "longest_below_3_features_s":float(longest),"latency_ms":distribution([r['elapsed_ms'] for r in records]),
            "synthetic_error_px":distribution(errors),"blackout_false_fresh_features":sum(len(r['active_ids']) for r in records if blackout and blackout[0]<=r['time_s']<=blackout[1]),
            "recovery_after_blackout_s":float(recovery_after_blackout) if recovery_after_blackout is not None else None,"records":records}


def read_clip(session,start,end,rate=10.):
    records=read_rows(session/'frames.jsonl');chosen=[];next_t=start
    for row in records:
        t=row['session_time_ns']/1e9
        if t>=next_t and t<=end:chosen.append(row);next_t=t+1/rate
    capture=cv2.VideoCapture(str(session/'video_main.mkv'));frames=[];times=[]
    selected={row['frame_index']:row for row in chosen}
    capture.set(cv2.CAP_PROP_POS_FRAMES,chosen[0]['frame_index'])
    for index in range(chosen[0]['frame_index'],chosen[-1]['frame_index']+1):
        ok,image=capture.read()
        if not ok:raise RuntimeError('Unexpected clip EOF')
        if index in selected:
            frames.append(cv2.cvtColor(cv2.resize(image,(640,360)),cv2.COLOR_BGR2GRAY));times.append(selected[index]['session_time_ns']/1e9)
    capture.release();return frames,np.array(times),chosen


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--mask-scene',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'implementation.py').write_text(Path(__file__).read_text(),encoding='utf-8')
    session=Path('sessions/workbench/recordings-genshin-impact-pc/run_20');summary=[]
    for name,start,end in [('return-straight',45.,55.),('outbound-straight',23.,33.),('endpoint-circle',34.,44.),('heldout-return',121.,131.)]:
        frames,times,chosen=read_clip(session,start,end)
        for method,reacquire in [('lk',False),('template',False),('lk',True)]:
            result=assay(frames,times,method,reacquire,mask_scene=args.mask_scene);result.update(clip=name,source_frames=[r['frame_index'] for r in chosen],source_time_s=[start,end])
            prefix=name+'-'+method+('-reacquire' if reacquire else '')
            (args.output/(prefix+'.json')).write_text(json.dumps(result,indent=2));summary.append({k:v for k,v in result.items() if k not in ('records','initial_points','source_frames')})
            first=cv2.cvtColor(frames[0],cv2.COLOR_GRAY2BGR);last=cv2.cvtColor(frames[-1],cv2.COLOR_GRAY2BGR)
            for point in result['initial_points']:cv2.circle(first,tuple(np.rint(point).astype(int)),3,(255,200,0),1)
            for point in result['records'][-1]['points']:cv2.circle(last,tuple(np.rint(point).astype(int)),3,(0,255,70),1)
            cv2.imwrite(str(args.output/(prefix+'.png')),np.hstack([first,last]))
        if name=='return-straight':
            source=frames[0];synthetic=[];shifts=[];synthetic_times=np.arange(121)/10
            for t in synthetic_times:
                shift=[18*np.sin(t*.4),5*np.sin(t*.3)];shifts.append(shift)
                frame=cv2.warpAffine(source,np.float32([[1,0,shift[0]],[0,1,shift[1]]]),(640,360),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
                if 4<=t<=5:frame[:]=0
                synthetic.append(frame)
            for method,reacquire in [('lk',False),('template',False),('lk',True)]:
                result=assay(synthetic,synthetic_times,method,reacquire,np.asarray(shifts),blackout=(4.,5.),mask_scene=args.mask_scene);result['clip']='synthetic-shift-and-occlusion'
                prefix='synthetic-'+method+('-reacquire' if reacquire else '')
                (args.output/(prefix+'.json')).write_text(json.dumps(result,indent=2));summary.append({k:v for k,v in result.items() if k not in ('records','initial_points')})
    output={"role":"image-bearing persistence assay; no yaw truth, route control, or gaze labels","implementation":identity(__file__),
            "source_manifest":identity(session/'manifest.json'),"source_frames":identity(session/'frames.jsonl'),
            "source_video":identity(session/'video_main.mkv'),"frame_width":640,"sampling_hz_requested":10,"results":summary}
    (args.output/'summary.json').write_text(json.dumps(output,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
