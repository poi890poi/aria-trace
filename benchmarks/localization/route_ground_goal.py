"""Small prepared physical-goal screen; no game input or lateral-error labels."""
import argparse
import time
import cv2
import numpy as np
from benchmarks.localization.route_guidance_trial import ROOT, SOURCE, read, dump
from aria_trace.services.tracking.destination import correspondence

MASK = dict(include=[75,190,580,327], exclude=[[273,150,385,327]])
POLY = np.array([[75,327],[220,190],[420,190],[580,327]],np.float32)
FOOT = np.array([[320,275],[320,290],[320,305]],np.float32)

def crop(q):
    p,d=q
    keep=np.array([cv2.pointPolygonTest(POLY,tuple(map(float,x)),False)>=0 for x in p],bool)
    return p[keep],d[keep]

def fit(a,b,goals):
    links=correspondence(a,b)
    if len(links)<10:return None
    p=a[0][links[:,0]];q=b[0][links[:,1]]
    return project(p,q,goals)

def project(p,q,goals):
    if len(p)<10:return None
    cv2.setRNGSeed(914)
    h,mask=cv2.findHomography(p,q,cv2.RANSAC,2.5,maxIters=2000,confidence=.995)
    if h is None or mask is None:return None
    keep=mask.ravel()>0;p=p[keep];q=q[keep]
    if len(p)<10 or keep.mean()<.5 or min(np.ptp(p,axis=0))<30 or min(np.ptp(q,axis=0))<30:return None
    hull=cv2.convexHull(p)
    if any(cv2.pointPolygonTest(hull,tuple(map(float,g)),False)<0 for g in goals):return None
    target=cv2.perspectiveTransform(np.float32(goals)[None],h)[0]
    if not np.isfinite(target).all() or (target<0).any() or (target>np.array([640,360])).any():return None
    return dict(goal=target.tolist(),support=len(p),reference=p.tolist(),points=q.tolist())

def temporal():
    """Change preparation only: short native-frame flow replaces a one-second match."""
    bank=read(ROOT/'bank.json');times=np.array([r['time_s'] for r in read(SOURCE/'frames.json')]);images=np.load(SOURCE/'gray.npy',mmap_mode='r');z=np.load(ROOT/'ground-features.npz');prepared={};log=[]
    area=np.zeros((360,640),np.uint8);cv2.fillConvexPoly(area,POLY.astype(np.int32),255);area[150:327,273:385]=0
    for n,v in enumerate(bank):
        i=v['frame'];j=int(np.argmin(abs(times-v['time_s']-1)))
        if str(i)+'p' not in z or str(j)+'p' not in z or times[j]>=124:continue
        goal=FOOT.copy();last=None;failed=None
        for a in range(j,i,-1):
            p=cv2.goodFeaturesToTrack(images[a],400,.01,6,mask=area)
            if p is None:failed='no corners';break
            q,status,_=cv2.calcOpticalFlowPyrLK(images[a],images[a-1],p,None,winSize=(21,21),maxLevel=3)
            back,status2,_=cv2.calcOpticalFlowPyrLK(images[a-1],images[a],q,None,winSize=(21,21),maxLevel=3)
            keep=(status.ravel()>0)&(status2.ravel()>0)&(np.linalg.norm(p-back,axis=2).ravel()<1.5)
            last=project(p[keep,0],q[keep,0],goal)
            if last is None:failed='unsupported short-step plane';break
            goal=np.float32(last['goal'])
        if failed is None and last:prepared[i]=dict(last,goal=goal.tolist())
        log.append(dict(frame=i,time_s=v['time_s'],future_frame=j,fit=prepared.get(i),failure=failed))
        if n%30==0:print('Temporal preparation',n,'/',len(bank),'accepted',len(prepared),flush=True)
    out=[]
    def q(i):return z[str(i)+'p'],z[str(i)+'d']
    for r in read(ROOT/'hint-prior.json')[::3]:
        i=r['frame'];v=next((v for v in bank if v['time_s']==r['teacher_time_s']),None);result=None
        if v and v['frame'] in prepared:result=fit(q(v['frame']),q(i),prepared[v['frame']]['goal'])
        out.append(dict(frame=i,time_s=r['time_s'],teacher_frame=None if v is None else v['frame'],fit=result))
    dump(ROOT/'ground-temporal-preparation.json',log);dump(ROOT/'ground-temporal-transfer.json',out)
    print('Temporal preparation',len(prepared),'/',len(log),'transfer',sum(r['fit'] is not None for r in out),'/',len(out),flush=True)

def extract():
    from aria_trace.services.tracking.destination_worker import LearnedFeatures
    bank=read(ROOT/'bank.json');times=np.array([r['time_s'] for r in read(SOURCE/'frames.json')]);images=np.load(SOURCE/'gray.npy',mmap_mode='r')
    # Every third prepared exemplar spans approaches, bends, stairs and corners.
    selected=[r for r in bank[::3] if r['time_s']+1<124]
    queries=read(ROOT/'hint-prior.json')[::3]
    ids=set(r['frame'] for r in selected)
    ids.update(int(np.argmin(abs(times-r['time_s']-1))) for r in selected)
    ids.update(r['frame'] for r in queries)
    # Add actual retrieved teacher views to avoid silently reducing query support.
    ids.update(next(v['frame'] for v in bank if v['time_s']==r['teacher_time_s']) for r in queries if r['teacher_time_s'] is not None)
    ids.update(int(np.argmin(abs(times-r['teacher_time_s']-1))) for r in queries if r['teacher_time_s'] is not None and r['teacher_time_s']+1<124)
    model=LearnedFeatures('.tools','auto');arrays={};cost=[]
    for n,i in enumerate(sorted(ids)):
        start=time.perf_counter();p,d=crop(model.extract(images[i],MASK));cost.append(dict(frame=i,ms=(time.perf_counter()-start)*1000));arrays[str(i)+'p']=p;arrays[str(i)+'d']=d
        if n%50==0:print('Ground features',n,'/',len(ids),flush=True)
    np.savez_compressed(ROOT/'ground-features.npz',**arrays);dump(ROOT/'ground-extraction.json',cost)

def run():
    bank=read(ROOT/'bank.json');times=np.array([r['time_s'] for r in read(SOURCE/'frames.json')]);z=np.load(ROOT/'ground-features.npz');prepared={};log=[]
    def q(i):return z[str(i)+'p'],z[str(i)+'d']
    for v in bank:
        i=v['frame'];j=int(np.argmin(abs(times-v['time_s']-1)))
        if str(i)+'p' not in z or str(j)+'p' not in z or times[j]>=124:continue
        result=fit(q(j),q(i),FOOT)
        log.append(dict(frame=i,time_s=v['time_s'],future_frame=j,fit=result))
        if result:prepared[i]=result
    out=[]
    for r in read(ROOT/'hint-prior.json')[::3]:
        i=r['frame'];v=next((v for v in bank if v['time_s']==r['teacher_time_s']),None);result=None
        if v and v['frame'] in prepared:result=fit(q(v['frame']),q(i),prepared[v['frame']]['goal'])
        out.append(dict(frame=i,time_s=r['time_s'],teacher_frame=None if v is None else v['frame'],fit=result))
    dump(ROOT/'ground-preparation.json',log);dump(ROOT/'ground-transfer.json',out)
    print('Ground preparation',len(prepared),'/',len(log),'transfer',sum(r['fit'] is not None for r in out),'/',len(out),flush=True)

if __name__=='__main__':
    cv2.setNumThreads(1);p=argparse.ArgumentParser();p.add_argument('phase',choices=['extract','run','temporal']);globals()[p.parse_args().phase]()
