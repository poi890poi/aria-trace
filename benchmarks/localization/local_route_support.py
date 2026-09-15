"""Check local map relations with separate spatial support before exposing hints."""
import cv2
import numpy as np
from benchmarks.localization.local_route_control import ROOT,SOURCE,SCENE,read,dump,mapmask,measure

def run():
    images=np.load(ROOT/'minimap.npy',mmap_mode='r');mask=mapmask();yy,xx=np.indices(mask.shape);masks=[mask*(xx<110),mask*(xx>=110),mask*(yy>85)];bank={v['time_s']:v for v in read(SCENE/'bank.json')};times=np.array([r['time_s'] for r in read(SOURCE/'frames.json')]);out=[]
    for r in read(ROOT/'rows.json'):
        i=r['frame'];v=bank.get(r['teacher_time_s']);current=[];motion=[]
        if v and r['map_relation'] and r['map_relation']['accepted']:
            for m in masks:
                q=measure(images[v['frame']],images[i],m)
                if q['accepted']:current.append(float(np.linalg.norm(np.array(q['shift'])-r['map_relation']['shift'])))
        if r['motion']['accepted']:
            j=int(np.argmin(abs(times-r['time_s']+.5)))
            for m in masks:
                q=measure(images[j],images[i],m)
                if q['accepted']:motion.append(float(np.linalg.norm(np.array(q['shift'])-r['motion']['shift'])))
        out.append(dict(time_s=r['time_s'],relation_disagreements_px=current,motion_disagreements_px=motion,accepted=bool(len(current)>=2 and len(motion)>=2 and max(current+motion)<=1.)))
    dump(ROOT/'support.json',out);print('Independent subregion support',sum(r['accepted'] for r in out),'/',len(out),'disagreement P95/max',np.percentile([max(r['relation_disagreements_px']+r['motion_disagreements_px']) for r in out if r['relation_disagreements_px'] and r['motion_disagreements_px']],[95,100]),flush=True)

if __name__=='__main__':cv2.setNumThreads(1);run()
