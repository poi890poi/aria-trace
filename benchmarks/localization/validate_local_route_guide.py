"""Frozen production parity and causal replay; no commands are issued."""
import json
import time
from pathlib import Path
import cv2
import numpy as np
from aria_trace.services.tracking.route_context import RouteContext
from aria_trace.services.tracking.route_guidance import load_guide, LocalCourseGuide, HumanInput
from .route_guidance_trial import data, ROOT, SESSION


def main():
    cv2.setNumThreads(1)
    output=Path('artifacts/poc/local-route-control-20260915')
    meta,arrays,maps=load_guide('artifacts/workbench/route_guides/genshin-impact-pc/town-circuit-run21')
    _,times,_,features,ids=data()
    expected=json.loads((ROOT/'vlad-local.json').read_text())
    context=RouteContext(meta,arrays); mismatches=[]; contexts=[]
    for r in expected:
        key=str(r['frame']); q=(features[key+'p'],features[key+'d'])
        v=context.step(q,r['time_s']);contexts.append(v)
        teacher=None if v['view'] is None else meta['views'][v['view']]['time_s']
        if teacher!=r['teacher_time_s'] or v['support']!=r['support']:mismatches.append(r['frame'])
    print('Context parity',len(expected)-len(mismatches),'/',len(expected),flush=True)
    guide=LocalCourseGuide(meta,arrays,maps); human=HumanInput()
    inputs=[json.loads(s) for s in (SESSION/'inputs.jsonl').read_text().splitlines()]
    native=np.load(output/'minimap.npy',mmap_mode='r'); ii=0; ci=0; rows=[]
    for i in ids[times[ids]>=124]:
        t=float(times[i])
        while ii<len(inputs) and inputs[ii]['session_time_ns']/1e9<=t:
            r=inputs[ii];human.feed(r['kind'],r['payload'],r['session_time_ns']/1e9);ii+=1
        started=time.perf_counter(); row=guide.step(native[i],t,contexts[ci],human.snapshot(t));ci+=1
        row['compute_ms']=(time.perf_counter()-started)*1000;row['frame']=int(i);rows.append(row)
    (output/'production-cached-rows.json').write_text(json.dumps(rows,allow_nan=False))
    timing=[r['compute_ms'] for r in rows]; lost=[]; start=None
    for r in rows:
        if not r['available'] and start is None:start=r['capture_s']
        if r['available'] and start is not None:lost.append(r['capture_s']-start);start=None
    if start is not None:lost.append(rows[-1]['capture_s']-start)
    result=dict(context_parity=len(expected)-len(mismatches),context_samples=len(expected),mismatches=mismatches,
                guidance=sum(r['available'] for r in rows),samples=len(rows),longest_lost_s=max(lost or [0]),loss_episodes=len(lost),
                compute_ms=dict(mean=float(np.mean(timing)),p50=float(np.median(timing)),p95=float(np.percentile(timing,95)),maximum=max(timing)))
    (output/'production-cached-summary.json').write_text(json.dumps(result,indent=2));print(result,flush=True)
    assert not mismatches


if __name__=='__main__':main()
