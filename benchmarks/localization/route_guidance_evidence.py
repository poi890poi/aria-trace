"""Compact native-image evidence and raw metrics for the continuous-route assay."""
import hashlib
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np
from benchmarks.localization.route_guidance_trial import ROOT, SOURCE, SESSION, read, dump
from aria_trace.services.tracking.destination import MODEL, MODEL_SHA256

def losses(rows,key):
    episodes=[];start=None
    for r in rows:
        if r[key] is None and start is None:start=r['time_s']
        if r[key] is not None and start is not None:
            episodes.append(dict(start_s=start,end_s=r['time_s'],duration_s=r['time_s']-start,censored=False));start=None
    if start is not None:episodes.append(dict(start_s=start,end_s=rows[-1]['time_s'],duration_s=rows[-1]['time_s']-start,censored=True))
    return episodes

def label(im,s,pos,size=.5,color=(30,30,30)):
    cv2.putText(im,s,pos,cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)

def main():
    frames=read(SOURCE/'frames.json');times=np.array([r['time_s'] for r in frames]);images=np.load(SOURCE/'gray.npy',mmap_mode='r');metrics={}
    for name in ['global','local','vlad-global','vlad-local','hint-prior','outdoor-negative']:
        rows=read(ROOT/(name+'.json'));gaps=losses(rows,'phase');ms=np.array([r['elapsed_ms'] for r in rows]);hint_gaps=losses(rows,'demo_turn_hint')
        metrics[name]=dict(samples=len(rows),fresh=sum(r['phase'] is not None for r in rows),held=0,unavailable=sum(r['phase'] is None for r in rows),losses=gaps,longest_unavailable_s=max((r['duration_s'] for r in gaps),default=0),demonstrated_action_available=sum(r['demo_turn_hint'] is not None for r in rows),action_losses=hint_gaps,elapsed_ms=dict(mean=float(ms.mean()),median=float(np.median(ms)),p95=float(np.percentile(ms,95)),maximum=float(ms.max())))
    dump(ROOT/'summary.json',metrics)
    rows=read(ROOT/'hint-prior.json');mouse=[json.loads(s) for s in (SESSION/'inputs.jsonl').read_text().splitlines() if 'pc_raw_mouse' in s]
    # All six examples declared as mechanism review, not an accuracy sample.
    wanted=[127.91,143.96,159.46,160.44,162.98,203.53]
    sheet=np.full((len(wanted)*300+105,960,3),250,np.uint8)
    label(sheet,'Why route recognition is not yet a steering correction',(16,30),.7)
    label(sheet,'Left: later lap. Right: retrieved teaching view and its next 0.6 s mouse input.',(16,58))
    label(sheet,'The player can need a different action from a similar-looking place. No XY error is scored.',(16,83),.47)
    for n,t in enumerate(wanted):
        r=min(rows,key=lambda r:abs(r['time_s']-t));teacher=int(np.argmin(abs(times-r['teacher_time_s'])));y=105+n*300
        for col,i in enumerate([r['frame'],teacher]):sheet[y+30:y+300,col*480:(col+1)*480]=cv2.resize(cv2.cvtColor(images[i],cv2.COLOR_GRAY2BGR),(480,270))
        observed=sum(r2['payload'].get('delta_x',0) for r2 in mouse if r['time_s']-.3<=r2['session_time_ns']/1e9<r['time_s'])/.3
        label(sheet,'Current %.2f s | preceding input %+.0f/s'%(r['time_s'],observed),(7,y+21),.45)
        label(sheet,'Teach %.2f s | recalled %s/s'%(r['teacher_time_s'],'?' if r['demo_mouse_rate'] is None else '%+.0f'%r['demo_mouse_rate']),(487,y+21),.45)
    cv2.imwrite(str(ROOT/'recognition-versus-advice.jpg'),sheet,[cv2.IMWRITE_JPEG_QUALITY,86])
    canvas=np.full((740,1400,3),250,np.uint8);x0,x1=200,1360;start,end=124,240
    def x(t):return int(x0+(t-start)/(end-start)*(x1-x0))
    label(canvas,'Continuous route memory | later town laps',(20,32),.8)
    label(canvas,'Green = fresh scene recognition. Red = unavailable. No held outputs or position-accuracy claim.',(20,61),.5)
    for n,name in enumerate(['global','local','vlad-global','vlad-local']):
        rr=read(ROOT/(name+'.json'));y=100+n*83
        label(canvas,{'global':'Gradient / global','local':'Gradient / local','vlad-global':'Learned / global','vlad-local':'Learned / local'}[name],(15,y+25),.46)
        for a,b in zip(rr,rr[1:]):cv2.rectangle(canvas,(x(a['time_s']),y),(max(x(a['time_s'])+1,x(b['time_s'])),y+35),(100,165,110) if a['phase'] is not None else (130,130,224),-1)
        label(canvas,'%d/%d; longest gap %.2f s'%(metrics[name]['fresh'],len(rr),metrics[name]['longest_unavailable_s']),(x0,y+57),.43)
    label(canvas,'Retrieved demo action',(15,478),.44)
    colors={-1:(210,135,40),0:(183,183,183),1:(35,165,215),None:(240,240,240)}
    for a,b in zip(rows,rows[1:]):cv2.rectangle(canvas,(x(a['time_s']),454),(max(x(a['time_s'])+1,x(b['time_s'])),493),colors[a['demo_turn_hint']],-1)
    label(canvas,'Blue: left. Yellow: right. Gray: no turn in demo. White: unavailable. Gray does NOT mean aligned.',(x0,521),.44)
    for t in range(124,241,8):label(canvas,str(t)+'s',(x(t)-10,560),.4)
    label(canvas,'Supported conclusion: route-wide acquisition improved. Copied actions still fail the control contract.',(20,613),.56)
    label(canvas,'Example: right-turn prior at 160.44 s; no-turn prior at 162.98 s while recorded turn continues.',(20,644),.48)
    label(canvas,'This is an algorithm-availability chart. It is not a physical-target lifetime chart.',(20,686),.5)
    cv2.imwrite(str(ROOT/'route-memory-availability.png'),canvas)
    for stem in ['ground','ground-temporal']:
        path=ROOT/(stem+'-transfer.json')
        if not path.exists():continue
        rr=[r for r in read(path) if r['fit'] is not None][:6]
        if not rr:continue
        im=np.full((60+len(rr)*300,960,3),250,np.uint8);label(im,stem+': accepted point transfers, physical path accuracy unverified',(15,30),.58)
        prep={r['frame']:r for r in read(ROOT/(stem+'-preparation.json'))}
        for n,r in enumerate(rr):
            y=60+n*300
            for col,i in enumerate([r['teacher_frame'],r['frame']]):
                scene=cv2.cvtColor(images[i],cv2.COLOR_GRAY2BGR);fit=r['fit'] if col else prep[i]['fit']
                for point in (r['fit']['points'] if col else r['fit']['reference']):cv2.circle(scene,tuple(np.rint(point).astype(int)),2,(0,210,255),-1)
                goal=np.float32(fit['goal']);cv2.polylines(scene,[np.rint(goal).astype(np.int32)],False,(255,100,20),3);cv2.drawMarker(scene,tuple(np.rint(goal[1]).astype(int)),(255,100,20),cv2.MARKER_CROSS,18,2)
                im[y+30:y+300,col*480:(col+1)*480]=cv2.resize(scene,(480,270));label(im,('%s %.2f s'%('Teach' if not col else 'Query',times[i])),(col*480+8,y+21),.48)
        cv2.imwrite(str(ROOT/(stem+'-review.jpg')),im,[cv2.IMWRITE_JPEG_QUALITY,88])
    paths=[SESSION/'frames.jsonl',SESSION/'inputs.jsonl',ROOT/'bank.json',ROOT/'bank.npz',ROOT/'vlad-centers.npy',ROOT/'vlad-signatures.npy']
    paths.extend(Path('benchmarks/localization')/n for n in ['route_guidance_trial.py','route_ground_goal.py','route_guidance_evidence.py'])
    dump(ROOT/'provenance.json',dict(base_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),model=MODEL,model_sha256=MODEL_SHA256,teaching_intervals_s=[[0,41],[41,82],[82,124]],query_interval_s=[124,float(times[-1])],fresh_holdout=False,hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},runtime='Python 3.12, OpenCV 5, one OpenCV thread; ALIKED-N16rot / GTX1650 extraction cached separately',evaluation='Recognition availability and sparse image review, no world-position truth or closed-loop trial'))
    print('Evidence and summary written',flush=True)

if __name__=='__main__':main()
