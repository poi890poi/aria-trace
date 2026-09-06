"""Paired native screen of identical hypotheses at different concurrency levels."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import platform
import time

import cv2
import numpy as np

from aria_trace.services.mapping.scale_aware import ScaleAwareLocalizer
from aria_trace.services.tracking.runtime import MinimapExtractor
from benchmarks.localization.scale_cost import ParallelScaleLocalizer
from benchmarks.localization.template_cpu import ATLAS, CALIBRATION
from benchmarks.localization.reference_cache import identity
from benchmarks.localization.run_workbench_replay import read_rows, distribution
from rig_runtime.adapters.filesystem.profiles import ProfileCatalog


def screen(output):
    output.mkdir(parents=True, exist_ok=False)
    prior = Path('artifacts/poc/tracker-implementation-20260906/visual20/run20/scored_telemetry.jsonl')
    peaks = Path('artifacts/poc/precision-round-20260906/native-comparison/selection.json')
    video = Path('sessions/workbench/recordings-genshin-impact-pc/run_20/video_main.mkv')
    atlas = Path('artifacts/workbench/map_atlases/genshin-impact-pc')/ATLAS
    calibration = Path('artifacts/workbench/minimap_calibrations/genshin-impact-pc')/CALIBRATION/'calibration.json'
    rows = read_rows(prior)
    selected = set(np.linspace(1,len(rows)-1,48,dtype=int).tolist())
    for frame_index in json.loads(peaks.read_text())['frame_indices']:
        nearest = min(range(len(rows)), key=lambda i: abs(rows[i]['frame_index']-frame_index))
        selected.update(max(1,min(len(rows)-1,nearest+delta)) for delta in (-8,-4,0,4,8))
    extractor = MinimapExtractor(ProfileCatalog().game('genshin-impact-pc')['minimap_calibration']['crop_xywh'], json.loads(calibration.read_text()))
    capture = cv2.VideoCapture(str(video))
    samples = []
    for index in sorted(selected):
        row, previous = rows[index], rows[index-1]['pose']
        capture.set(cv2.CAP_PROP_POS_FRAMES,row['frame_index'])
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError('Cannot decode selected frame')
        observation, mask = extractor.extract(frame)
        detail = row.get('route_tracking') or {}
        samples.append((observation,mask,detail.get('search_center_canonical_xy',[previous['x'],previous['y']]),
                        detail.get('search_radius_canonical_px',12.), row['active_map_mode_id'],row['frame_index']))
    capture.release()
    cv2.setNumThreads(6)
    baseline = ScaleAwareLocalizer(atlas)
    candidate = ParallelScaleLocalizer.__new__(ParallelScaleLocalizer)
    candidate.__dict__.update(baseline.__dict__)
    candidate._matching_pool = None
    expected = []
    for observation,mask,xy,radius,mode,frame in samples:
        baseline.active_mode_id = mode
        result = baseline.refine_active_near(observation,mask,xy,radius)
        result.pop('elapsed_ms')
        expected.append(result)
    measured, failures = [], []
    try:
        # Rotate block order, so later workers are not always measured warmest.
        for repetition, order in enumerate(((1,2,4,6),(6,4,2,1),(2,6,1,4))):
            for workers in order:
                candidate._matching_pool = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
                try:
                    candidate.refine_active_near(*samples[0][:4])
                    for index,(observation,mask,xy,radius,mode,frame) in enumerate(samples):
                        candidate.active_mode_id = mode
                        wall,cpu = time.perf_counter(),time.process_time()
                        result = candidate.refine_active_near(observation,mask,xy,radius)
                        elapsed,used = (time.perf_counter()-wall)*1000,(time.process_time()-cpu)*1000
                        result.pop('elapsed_ms')
                        equal = result == expected[index]
                        measured.append({'repeat': repetition,'workers':workers,'frame':frame,'wall_ms':elapsed,'cpu_ms':used,'equal':equal})
                        if not equal:
                            failures.append({'workers':workers,'frame':frame,'expected':expected[index],'actual':result})
                finally:
                    if candidate._matching_pool:
                        candidate._matching_pool.shutdown(wait=True)
                        candidate._matching_pool = None
                print('DONE',repetition,workers,flush=True)
    finally:
        baseline.close()
    summaries = []
    for workers in (1,2,4,6):
        chosen = [r for r in measured if r['workers']==workers]
        summaries.append({'workers':workers,'exact':sum(r['equal'] for r in chosen),'count':len(chosen),
                          'wall_ms':distribution([r['wall_ms'] for r in chosen]),'cpu_ms':distribution([r['cpu_ms'] for r in chosen])})
    (output/'measurements.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in measured))
    result = {'role':__doc__,'python':platform.python_version(),'opencv':cv2.__version__,'opencv_threads':cv2.getNumThreads(),
              'samples':len(samples),'inputs':[identity(p) for p in (prior,peaks,video,calibration,atlas/'map_atlas.json')],
              'source':identity(__file__),'candidate_source':identity(Path('benchmarks/localization/scale_cost.py')),
              'setup':baseline.scale_search_setup,'summary':summaries,'failures':failures}
    (output/'screen.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(summaries,indent=2),flush=True)
    if failures:
        raise AssertionError('Native output parity failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    screen(parser.parse_args().output)
