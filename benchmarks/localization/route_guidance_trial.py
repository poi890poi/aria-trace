"""Route-wide retrieval and demonstrated-action evidence, no input execution."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
from aria_trace.services.tracking.destination import correspondence, DEFAULT_MASK, mask_points

SOURCE=Path('artifacts/poc/destination-run21-20260914')
ROOT=Path('artifacts/poc/route-guidance-20260914')
SESSION=Path('sessions/workbench/recordings-genshin-impact-pc/run_21')
TEACH=[(0,41),(41,82),(82,124)]

def read(path):return json.loads(Path(path).read_text())
def dump(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False))

def signature(image):
    small=cv2.resize(image[25:275,110:550],(44,25),interpolation=cv2.INTER_AREA).astype(np.float32)
    small[13:,16:27]=0
    dx=cv2.Sobel(small,cv2.CV_32F,1,0,ksize=3);dy=cv2.Sobel(small,cv2.CV_32F,0,1,ksize=3)
    out=np.r_[dx.ravel(),dy.ravel()];return out/max(float(np.linalg.norm(out)),1e-8)

def geometry(reference,query):
    links=correspondence(reference,query)
    if len(links)<12:return None
    a=reference[0][links[:,0]];b=query[0][links[:,1]]
    cv2.setRNGSeed(914)
    h,mask=cv2.findHomography(a,b,cv2.RANSAC,3.,maxIters=1000,confidence=.99)
    if h is None or mask is None:return None
    good=mask.ravel()>0;a=a[good];b=b[good]
    if len(a)<12 or good.mean()<.4:return None
    span=np.ptp(b,axis=0)
    cells=set((int(p[0]//110),int(p[1]//85)) for p in b)
    if span[0]<100 or span[1]<45 or len(cells)<3:return None
    return dict(support=len(a),ratio=float(good.mean()),points=b.tolist(),reference=a.tolist())

def data():
    rows=read(SOURCE/'frames.json');times=np.array([r['time_s'] for r in rows]);images=np.load(SOURCE/'gray.npy',mmap_mode='r');features={}
    for file in ['frozen-features.npz','transfer-features.npz']:
        with np.load(SOURCE/file) as z:features.update({k:z[k] for k in z.files})
    ids=np.array(sorted(int(k[:-1]) for k in features if k.endswith('p')))
    return rows,times,images,features,ids

def prepare():
    ROOT.mkdir(parents=True,exist_ok=True);rows,times,images,features,ids=data();inputs=[json.loads(s) for s in (SESSION/'inputs.jsonl').read_text().splitlines()];mouse=[r for r in inputs if r['kind']=='pc_raw_mouse'];bank=[];arrays={}
    for lap,(lo,hi) in enumerate(TEACH):
        available=ids[(times[ids]>=lo)&(times[ids]<hi)]
        chosen=sorted(set(int(available[np.argmin(abs(times[available]-t))]) for t in np.arange(lo,hi,.75)))
        for i in chosen:
            t=times[i];window=[r['payload'].get('delta_x',0) for r in mouse if t<=r['session_time_ns']/1e9<min(t+.6,hi)];rate=sum(window)/max(min(.6,hi-t),.01)
            key=str(i);bank.append(dict(key=key,frame=i,time_s=float(t),lap=lap,phase=float((t-lo)/(hi-lo)),mouse_rate=rate,turn=1 if rate>150 else -1 if rate < -150 else 0))
            for suffix in ['p','d']:arrays[key+suffix]=features[key+suffix]
    np.savez_compressed(ROOT/'bank.npz',**arrays);np.save(ROOT/'signatures.npy',np.array([signature(images[r['frame']]) for r in bank]));dump(ROOT/'bank.json',bank)
    print('Prepared',len(bank),'route exemplars; no neural training',flush=True)

class Memory:
    def __init__(self,bank,arrays,signatures,local=True,centers=None):
        self.bank=bank;self.arrays=arrays;self.signatures=signatures;self.local=local;self.phase=None;self.last_fresh=-999.;self.retry=-999.;self.centers=centers
    def step(self,image,query,t):
        started=time.perf_counter();descriptor=signature(image) if self.centers is None else vlad(query[1],self.centers);similarities=self.signatures@descriptor;order=np.argsort(-similarities);indices=[]
        if self.local and self.phase is not None and t-self.last_fresh<1:
            indices=[int(i) for i in order if -.055<=self.bank[i]['phase']-self.phase<=.13][:8]
        else:indices=[int(i) for i in order[:8]]
        candidates=[]
        for i in indices:
            v=self.bank[i];key=v['key'];fit=geometry((self.arrays[key+'p'],self.arrays[key+'d']),query)
            if fit:fit.update(index=i,**v);candidates.append(fit)
        if not candidates and self.local and self.phase is not None and t>=self.retry:
            self.retry=t+.5
            for i in [int(j) for j in order[:8] if j not in indices]:
                v=self.bank[i];key=v['key'];fit=geometry((self.arrays[key+'p'],self.arrays[key+'d']),query)
                if fit:fit.update(index=i,**v);candidates.append(fit)
        candidates.sort(key=lambda c:-c['support']);chosen=candidates[0] if candidates else None;hint=None;agreement=None
        if chosen:
            self.phase=chosen['phase'];self.last_fresh=t
            nearby=[c for c in candidates if abs(c['phase']-chosen['phase'])<.065];votes={}
            for c in nearby:
                if c['lap'] not in votes:votes[c['lap']]=c
            if len(votes)>=2:
                signs=[c['turn'] for c in votes.values()];agreement=max(signs.count(s) for s in set(signs))/len(signs)
                if len(set(signs))==1:hint=signs[0]
        return dict(time_s=t,phase=None if chosen is None else chosen['phase'],teacher_time_s=None if chosen is None else chosen['time_s'],support=0 if chosen is None else chosen['support'],points=[] if chosen is None else chosen['points'],demo_turn_hint=hint,demo_agreement=agreement,candidates=[{k:c[k] for k in ['time_s','phase','lap','support','turn']} for c in candidates],elapsed_ms=(time.perf_counter()-started)*1000,correction=None)

def run():
    _,times,images,features,ids=data();bank=read(ROOT/'bank.json');z=np.load(ROOT/'bank.npz');arrays={k:z[k] for k in z.files};signatures=np.load(ROOT/'signatures.npy')
    query=ids[times[ids]>=124]
    for mode in ['global','local']:
        memory=Memory(bank,arrays,signatures,mode=='local');rows=[]
        for n,i in enumerate(query):
            r=memory.step(images[i],(features[str(i)+'p'],features[str(i)+'d']),float(times[i]));r['frame']=int(i);rows.append(r)
            if n%150==0:print(mode,n,'/',len(query),flush=True)
        dump(ROOT/(mode+'.json'),rows);print(mode,'recognized',sum(r['phase'] is not None for r in rows),'/',len(rows),'hint',sum(r['demo_turn_hint'] is not None for r in rows),flush=True)

def vlad(descriptors,centers):
    if not len(descriptors):return np.zeros(centers.size,np.float32)
    nearest=np.argmax(descriptors@centers.T,axis=1);out=np.zeros_like(centers)
    for i in range(len(centers)):
        use=descriptors[nearest==i]
        if len(use):out[i]=use.sum(axis=0)-len(use)*centers[i]
    out/=np.maximum(np.linalg.norm(out,axis=1,keepdims=True),1e-8);out=np.sign(out)*np.sqrt(abs(out));out=out.ravel();return out/max(np.linalg.norm(out),1e-8)

def vlad_run():
    _,times,images,features,ids=data();bank=read(ROOT/'bank.json');z=np.load(ROOT/'bank.npz');arrays={k:z[k] for k in z.files};all_desc=np.concatenate([arrays[r['key']+'d'] for r in bank]);rng=np.random.RandomState(914);sample=all_desc[rng.choice(len(all_desc),min(12000,len(all_desc)),replace=False)];cv2.setRNGSeed(914);began=time.perf_counter()
    _,_,centers=cv2.kmeans(np.float32(sample),32,None,(cv2.TERM_CRITERIA_MAX_ITER|cv2.TERM_CRITERIA_EPS,30,.01),1,cv2.KMEANS_PP_CENTERS);centers/=np.maximum(np.linalg.norm(centers,axis=1,keepdims=True),1e-8)
    signatures=np.array([vlad(arrays[r['key']+'d'],centers) for r in bank]);np.save(ROOT/'vlad-centers.npy',centers);np.save(ROOT/'vlad-signatures.npy',signatures);print('VLAD preparation seconds',time.perf_counter()-began,flush=True)
    for mode in ['global','local']:
        memory=Memory(bank,arrays,signatures,mode=='local',centers);rows=[];query=ids[times[ids]>=124]
        for n,i in enumerate(query):
            r=memory.step(images[i],(features[str(i)+'p'],features[str(i)+'d']),float(times[i]));r['frame']=int(i);rows.append(r)
            if n%150==0:print('vlad-'+mode,n,'/',len(query),flush=True)
        dump(ROOT/('vlad-'+mode+'.json'),rows);print('vlad-'+mode,'recognized',sum(r['phase'] is not None for r in rows),'/',len(rows),'hint',sum(r['demo_turn_hint'] is not None for r in rows),flush=True)

def negative():
    native=Path('artifacts/poc/segment-loop-20260913');meta=read(native/'native-frames.json');images=np.load(native/'native-gray.npy',mmap_mode='r');q=np.load('artifacts/poc/passage-loop-20260913/query.npz');bank=read(ROOT/'bank.json');z=np.load(ROOT/'bank.npz');memory=Memory(bank,{k:z[k] for k in z.files},np.load(ROOT/'vlad-signatures.npy'),True,np.load(ROOT/'vlad-centers.npy'));selected=[];last=-999.
    for r in meta:
        if 178<=r['time_s']<208 and r['time_s']-last>=.75 and str(r['frame'])+'p' in q:selected.append(r);last=r['time_s']
    out=[]
    for r in selected:
        i=r['frame'];v=memory.step(images[i],(q[str(i)+'p'],q[str(i)+'d']),r['time_s']);v['frame']=i;out.append(v)
    dump(ROOT/'outdoor-negative.json',out);print('Outdoor control recognized',sum(r['phase'] is not None for r in out),'/',len(out),flush=True)

def hints():
    bank={r['time_s']:r for r in read(ROOT/'bank.json')};rows=read(ROOT/'vlad-local.json');result=[]
    for r in rows:
        candidates=r['candidates'];hint=None;rate=None;peers=[]
        if candidates:
            best=candidates[0];peers=[c for c in candidates if c['support']>=.85*best['support']]
            # A second demonstration is helpful but not mandatory. Comparable
            # support with conflicting actions or distant stages is ambiguous.
            if max(c['phase'] for c in peers)-min(c['phase'] for c in peers)<.13 and len(set(c['turn'] for c in peers))==1:
                hint=best['turn'];rate=float(np.median([bank[c['time_s']]['mouse_rate'] for c in peers]))
        result.append(dict(r,demo_turn_hint=hint,demo_mouse_rate=rate,peer_count=len(peers)))
    dump(ROOT/'hint-prior.json',result);print('Action-prior availability',sum(r['demo_turn_hint'] is not None for r in result),'/',len(result),'directions',{str(k):sum(r['demo_turn_hint']==k for r in result) for k in [-1,0,1]},flush=True)

if __name__=='__main__':
    cv2.setNumThreads(1);p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run','vlad_run','negative','hints']);globals()[p.parse_args().phase]()
