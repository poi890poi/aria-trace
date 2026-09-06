"""Compare native minimaps with current-image atlas matches on identical frames."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from aria_trace.services.tracking.runtime import MinimapExtractor
from benchmarks.localization.run_workbench_replay import read_rows
from rig_runtime.adapters.filesystem.profiles import ProfileCatalog


def build(root, candidate):
    atlas = Path('artifacts/workbench/map_atlases/genshin-impact-pc/08b6f2d6-820a-4bfd-875a-6a55d1986a4e')
    manifest = json.loads((atlas/'map_atlas.json').read_text())
    mosaic = cv2.imread(str(atlas/manifest['canonical_mosaic_file']))
    layers = {x['mode_id']: x for x in manifest['layers']}
    calibration = json.loads(Path('artifacts/workbench/minimap_calibrations/genshin-impact-pc/segments-df624035-833-bd07601f-708/calibration.json').read_text())
    extractor = MinimapExtractor(ProfileCatalog().game('genshin-impact-pc')['minimap_calibration']['crop_xywh'], calibration)
    control = {r['frame_index']: r for r in read_rows(root/'control20/run20/scored_telemetry.jsonl')}
    tested = {r['frame_index']: r for r in read_rows(root/candidate/'run20/scored_telemetry.jsonl')}
    changes = json.loads(Path('artifacts/poc/narrow-lap-20260906/lap-analysis.json').read_text())['reference_transition_brackets']
    selected = []
    for change in changes:
        eligible = [i for i, r in control.items() if i in tested and change['last_old_reference_s']-1 <= r['session_time_ns']/1e9 <= change['first_new_reference_s']+1 and r.get('reference_error_px') is not None]
        selected.append(max(eligible, key=lambda i: control[i]['reference_error_px']))
    target = root/'native-comparison'
    target.mkdir(exist_ok=False)
    capture = cv2.VideoCapture('sessions/workbench/recordings-genshin-impact-pc/run_20/video_main.mkv')
    cache = {}
    for index in selected:
        capture.set(cv2.CAP_PROP_POS_FRAMES,index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError('Missing source frame')
        observation, mask = extractor.extract(frame)
        panels = [observation.copy()]
        labels = ['Native game minimap']
        for label, row in [('Control',control[index]), (candidate,tested[index])]:
            mode = row['active_map_mode_id']
            scale = (row.get('route_tracking',{}).get('precision_scale') or {}).get('estimated_scale') or layers[mode]['map_pixels_per_minimap_pixel']
            if scale not in cache:
                endpoint = next((x for x in layers.values() if abs(x['map_pixels_per_minimap_pixel']-scale)<1e-8),None)
                image = cv2.imread(str(atlas/endpoint['localization_mosaic_file'])) if endpoint else cv2.resize(mosaic,(round(mosaic.shape[1]/scale),round(mosaic.shape[0]/scale)),interpolation=cv2.INTER_AREA)
                cache[scale] = image
            image = cache[scale]
            xy = np.array([row['pose']['x'],row['pose']['y']])*[image.shape[1]/mosaic.shape[1],image.shape[0]/mosaic.shape[0]]-.5
            panels.append(cv2.getRectSubPix(image,(observation.shape[1],observation.shape[0]),tuple(xy)))
            labels.append(f'{label}: scale {scale:.3f}, proxy error {row["reference_error_px"]:.2f}px')
        strips = []
        for panel, label in zip(panels,labels):
            panel[mask==0] = 0
            panel = cv2.resize(panel,(414,414),interpolation=cv2.INTER_NEAREST)
            panel = cv2.copyMakeBorder(panel,55,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
            cv2.putText(panel,label,(7,25),cv2.FONT_HERSHEY_SIMPLEX,.40,(255,255,255),1,cv2.LINE_AA)
            strips.append(panel)
        canvas = np.hstack(strips)
        native = cv2.resize(frame,(1242,699))
        cv2.putText(native,f'Run 20 / frame {index} / source {control[index]["session_time_ns"]/1e9:.3f}s',(20,675),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
        cv2.imwrite(str(target/f'frame-{index:05d}.png'),np.vstack([native,canvas]))
    capture.release()
    (target/'selection.json').write_text(json.dumps({'selection':'Worst control proxy disagreement in each transition window, restricted to identical processed frames',
        'candidate':candidate,'frame_indices':selected,'role':'Visual alignment evidence; same-atlas reference is not independent position truth'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('candidate');args=p.parse_args()
    build(args.root,args.candidate)
