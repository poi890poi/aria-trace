"""Package the completed town-route assay for the opt-in manual guidance trial."""
import hashlib
from pathlib import Path
import numpy as np
from benchmarks.localization.local_route_control import ROOT,SOURCE,SCENE,read,dump,mapmask
from aria_trace.services.tracking.destination import MODEL,MODEL_SHA256,DEFAULT_MASK
from aria_trace.services.tracking.route_guidance import SCHEMA,load_guide

def main():
    target=Path('artifacts/workbench/route_guides/genshin-impact-pc/town-circuit-run21')
    if (target/'guide.json').exists():raise RuntimeError('Prepared guide already exists; preserve it')
    target.mkdir(parents=True,exist_ok=True);bank=read(SCENE/'bank.json');corridor={r['time_s']:r for r in read(ROOT/'corridor-preparation.json')};z=np.load(SCENE/'bank.npz');arrays={k:z[k] for k in z.files};arrays.update(centers=np.load(SCENE/'vlad-centers.npy'),signatures=np.load(SCENE/'vlad-signatures.npy'),mask=mapmask());np.savez_compressed(target/'arrays.npz',**arrays)
    maps=np.load(ROOT/'minimap.npy',mmap_mode='r');np.save(target/'maps.npy',np.array([maps[v['frame']] for v in bank]));views=[dict(key=v['key'],phase=v['phase'],time_s=v['time_s'],frame=v['frame'],lap=v['lap'],paths=corridor[v['time_s']]['paths']) for v in bank]
    meta=dict(schema=SCHEMA,guide_id=target.name,label='Town circuit - session 21',game_profile_id='genshin-impact-pc',model=MODEL,model_sha256=MODEL_SHA256,image_size_wh=[1280,720],scene_mask=DEFAULT_MASK,views=views,motor_gain_deg_per_count=read(ROOT/'motor-fit.json')['chosen']['gain_deg_per_count'],turn_display_scale=1500,source_description='First three Session21 traversals, before 124 s; no route neural training. Human-driven trial.',sha256={n:hashlib.sha256((target/n).read_bytes()).hexdigest() for n in ['arrays.npz','maps.npy']})
    dump(target/'guide.json',meta);load_guide(target,'genshin-impact-pc');print('Prepared and validated',target,flush=True)

if __name__=='__main__':main()
