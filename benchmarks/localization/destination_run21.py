"""Fresh-session survey and frozen prepared-destination holdout evidence."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

SESSION = Path('sessions/workbench/recordings-genshin-impact-pc/run_21')
ROOT = Path('artifacts/poc/destination-run21-20260914')

# First-lap-only annotations identify physical landmark groups, not steering goals.
TARGETS=[dict(name='market_canopy',seed=20,roi=[235,116,403,165],aim=[325,135],span=[17,24]),
         dict(name='stair_top',seed=32,roi=[279,26,337,83],aim=[306,59],span=[30,35]),
         dict(name='corner_lamp',seed=36,roi=[222,78,281,162],aim=[244,120],span=[35,39]),
         dict(name='after_turn_arch',seed=2,roi=[345,49,435,133],aim=[379,91],span=[0,6])]

def transfer():
    import time
    from aria_trace.services.tracking.destination import DEFAULT_MASK, DestinationObserver, flow, local_fit, mask_points
    from aria_trace.services.tracking.destination_worker import LearnedFeatures
    frames=json.loads((ROOT/'frames.json').read_text());times=np.array([r['time_s'] for r in frames]);images=np.load(ROOT/'gray.npy',mmap_mode='r')
    model=LearnedFeatures('.tools','auto');bank={};views={};log=[];cache={};start=time.perf_counter()
    def extract(i):
        if i not in cache:cache[i]=model.extract(images[i],DEFAULT_MASK)
        return cache[i]
    for config in TARGETS:
        name=config['name'];seed=int(np.argmin(abs(times-config['seed'])));p,d=extract(seed);x0,y0,x1,y1=config['roi'];owned=(p[:,0]>=x0)&(p[:,0]<x1)&(p[:,1]>=y0)&(p[:,1]<y1);target=[]
        if owned.sum()<6:
            log.append(dict(target=name,error='fewer than six seed features',support=int(owned.sum())));continue
        def save(i,p,d,own,aim,width):
            key=name+'_'+str(i);bank[key+'p']=p;bank[key+'d']=d;bank[key+'o']=own;target.append(dict(key=key,aim=list(aim),width=width,frame=i,time_s=float(times[i])))
        save(seed,p,d,owned,config['aim'],40.)
        for direction in [-1,1]:
            previous=seed;points=p[owned].copy();aim=config['aim'];width=40.;last_saved=times[seed]
            for i in range(seed+direction,len(times) if direction>0 else -1,direction):
                if not config['span'][0]<=times[i]<=config['span'][1]:break
                q,valid=flow(images[previous],images[i],points);fit=local_fit(points[valid],q[valid],aim)
                if fit is None:log.append(dict(target=name,direction=direction,lost_s=float(times[i])));break
                points=fit['points'];aim=fit['aim'];width*=fit['scale']
                if abs(times[i]-last_saved)>=.75:
                    qp,qd=extract(i);distance=np.linalg.norm(qp[:,None,:]-points[None,:,:],axis=2).min(axis=1);own=distance<8
                    if own.sum()>=6:save(i,qp,qd,own,aim,width);last_saved=times[i]
                previous=i
        views[name]=sorted(target,key=lambda x:x['time_s']);print('Prepared',name,len(target),'views',flush=True)
    np.savez_compressed(ROOT/'first-lap-bank.npz',**bank);(ROOT/'first-lap-views.json').write_text(json.dumps(views,indent=2));(ROOT/'prepare-log.json').write_text(json.dumps(log,indent=2))
    observers={name:DestinationObserver(dict(mask=DEFAULT_MASK,views=value),bank) for name,value in views.items()};rows=[];cost=[]
    for i in range(len(times)):
        if times[i]<41:continue
        needs=any(o.recognition_due(float(times[i])) for o in observers.values());q=None
        if needs:
            began=time.perf_counter();q=extract(i);cost.append(dict(frame=i,ms=(time.perf_counter()-began)*1000))
        for name,o in observers.items():
            r=o.step(images[i],q,float(times[i]));r.update(target=name,frame=i,time_s=float(times[i]));rows.append(r)
        if i%500==0:print('Transfer',i,'/',len(times),flush=True)
    (ROOT/'transfer-rows.json').write_text(json.dumps(rows));(ROOT/'transfer-extraction.json').write_text(json.dumps(cost))
    np.savez_compressed(ROOT/'transfer-features.npz',**{str(i)+s:a for i,q in cache.items() for s,a in zip(['p','d'],q)})
    print('Transfer finished seconds',time.perf_counter()-start,flush=True)

def frozen():
    import time
    from aria_trace.services.tracking.destination import DestinationObserver, load_package
    from aria_trace.services.tracking.destination_worker import LearnedFeatures
    frames=json.loads((ROOT/'frames.json').read_text());times=np.array([r['time_s'] for r in frames]);images=np.load(ROOT/'gray.npy',mmap_mode='r')
    observers={name:DestinationObserver(*load_package(Path('artifacts/workbench/destination_targets/genshin-impact-pc')/name)) for name in ['bridge-gate-two-demo','town-gate-two-demo']}
    model=LearnedFeatures('.tools','auto');features={};rows=[]
    ids=sorted(set(int(np.argmin(abs(times-t))) for t in np.arange(0,times[-1],1)))
    for n,i in enumerate(ids):
        started=time.perf_counter();q=model.extract(images[i],next(iter(observers.values())).meta['mask']);cost=(time.perf_counter()-started)*1000
        features[str(i)+'p'],features[str(i)+'d']=q
        for name,observer in observers.items():
            row=observer.step(images[i],q,float(times[i]));row.update(target=name,frame=i,time_s=float(times[i]),extraction_ms=cost);rows.append(row)
        if n%40==0:print('Frozen holdout',n,'/',len(ids),flush=True)
    np.savez_compressed(ROOT/'frozen-features.npz',**features)
    (ROOT/'frozen-rows.json').write_text(json.dumps(rows))
    result={name:dict(frames=sum(r['target']==name for r in rows),accepted=[dict(frame=r['frame'],time_s=r['time_s'],aim=r['aim'],support=r['support']) for r in rows if r['target']==name and r['aim'] is not None]) for name in observers}
    (ROOT/'frozen-summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

def survey():
    ROOT.mkdir(parents=True, exist_ok=True)
    frames = [json.loads(x) for x in (SESSION/'frames.jsonl').read_text().splitlines()]
    inputs = [json.loads(x) for x in (SESSION/'inputs.jsonl').read_text().splitlines()]
    times = np.array([r['session_time_ns']/1e9 for r in frames])
    cap = cv2.VideoCapture(str(SESSION/'video_main.mkv'))
    gray = np.lib.format.open_memmap(str(ROOT/'gray.npy'), mode='w+', dtype=np.uint8, shape=(len(frames),360,640))
    tiles = []
    wanted = set(int(np.argmin(abs(times-t))) for t in np.arange(0,times[-1],8))
    for i, row in enumerate(frames):
        ok, im = cap.read()
        if not ok: raise RuntimeError('Missing frame '+str(i))
        im = cv2.resize(im,(640,360),interpolation=cv2.INTER_AREA)
        gray[i] = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        if i in wanted:
            tile=np.zeros((202,320,3),np.uint8);tile[22:]=cv2.resize(im,(320,180))
            cv2.putText(tile,'%.2f s | frame %d'%(times[i],i),(5,16),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            tiles.append(tile)
    gray.flush();cap.release()
    for page in range((len(tiles)+14)//15):
        canvas=np.zeros((202*5,320*3,3),np.uint8)
        for j, tile in enumerate(tiles[page*15:(page+1)*15]):canvas[(j//3)*202:(j//3+1)*202,(j%3)*320:(j%3+1)*320]=tile
        cv2.imwrite(str(ROOT/('survey-%d.jpg'%page)),canvas)
    motion=[]
    for t in range(int(times[-1])+1):
        rows=[r['payload'] for r in inputs if r['kind']=='pc_raw_mouse' and t<=r['session_time_ns']/1e9<t+1]
        motion.append(dict(time_s=t,dx=sum(r.get('delta_x',0) for r in rows),abs_dx=sum(abs(r.get('delta_x',0)) for r in rows),dy=sum(r.get('delta_y',0) for r in rows)))
    (ROOT/'frames.json').write_text(json.dumps([dict(frame=i,time_s=float(t)) for i,t in enumerate(times)]))
    (ROOT/'mouse-seconds.json').write_text(json.dumps(motion,indent=2))
    print(json.dumps(dict(frames=len(frames),duration_s=float(times[-1]),interval_ms=dict(median=float(np.median(np.diff(times))*1000),p95=float(np.percentile(np.diff(times),95)*1000)),largest_turn_seconds=sorted(motion,key=lambda x:x['abs_dx'],reverse=True)[:20]),indent=2))

if __name__=='__main__':
    cv2.setNumThreads(1)
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['survey','frozen','transfer']);globals()[parser.parse_args().phase]()
