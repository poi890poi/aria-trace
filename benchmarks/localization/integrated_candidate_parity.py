"""Compare reusable scale matching with the frozen experiment on native frames.

This is a transfer check, not independent ground truth. Image selection and
proposal coordinates are frozen from the prior control's six transition peaks.
Neither implementation receives the evaluation reference coordinates.
"""

import argparse
import json
from pathlib import Path

import cv2

from aria_trace.services.mapping.layers import LayeredGlobalLocalizer
from aria_trace.services.mapping.scale_aware import ScaleAwareLocalizer
from aria_trace.services.tracking.runtime import MinimapExtractor
from benchmarks.localization.precision_candidates import installed
from benchmarks.localization.reference_cache import identity
from benchmarks.localization.run_workbench_replay import read_rows
from benchmarks.localization.template_cpu import ATLAS, CALIBRATION
from rig_runtime.adapters.filesystem.profiles import ProfileCatalog


def verify(output):
    root = Path('artifacts/poc/precision-round-20260906')
    selection = root/'native-comparison/selection.json'
    control = root/'control20/run20/scored_telemetry.jsonl'
    video = Path('sessions/workbench/recordings-genshin-impact-pc/run_20/video_main.mkv')
    indices = json.loads(selection.read_text())['frame_indices']
    rows = read_rows(control)
    atlas = Path('artifacts/workbench/map_atlases/genshin-impact-pc')/ATLAS
    calibration_path = Path('artifacts/workbench/minimap_calibrations/genshin-impact-pc')/CALIBRATION/'calibration.json'
    calibration = json.loads(calibration_path.read_text())
    extractor = MinimapExtractor(ProfileCatalog().game('genshin-impact-pc')['minimap_calibration']['crop_xywh'], calibration)
    cv2.setNumThreads(6)
    candidate = ScaleAwareLocalizer(atlas)
    capture = cv2.VideoCapture(str(video))
    results = []
    try:
        with installed(subpixel=True, scale_sweep=True):
            previous = LayeredGlobalLocalizer(atlas)
            try:
                for index, row in enumerate(rows):
                    if row['frame_index'] not in indices:
                        continue
                    capture.set(cv2.CAP_PROP_POS_FRAMES, row['frame_index'])
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError('Cannot read native frame')
                    observation, mask = extractor.extract(frame)
                    detail, prior = row['route_tracking'], rows[index-1]['pose']
                    proposal = detail.get('search_center_canonical_xy', [prior['x'], prior['y']])
                    radius = detail.get('search_radius_canonical_px', 12.)
                    candidate.active_mode_id = previous.active_mode_id = row['active_map_mode_id']
                    before = previous.refine_active_near(observation, mask, proposal, radius)
                    after = candidate.refine_active_near(observation, mask, proposal, radius)
                    before.pop('elapsed_ms'); after.pop('elapsed_ms')
                    if before != after:
                        raise AssertionError('Implementation differs at frame '+str(row['frame_index']))
                    results.append({'frame_index': row['frame_index'], 'proposal': proposal, 'radius': radius, 'result': after})
            finally:
                previous.close()
    finally:
        capture.release()
        candidate.close()
    if len(results) != 6:
        raise AssertionError('Expected the six frozen native comparisons')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'role': __doc__, 'passed': len(results), 'source': identity(__file__),
                                 'inputs': [identity(p) for p in (selection, control, video, calibration_path)],
                                 'results': results}, indent=2))
    print('PASS: six native scale-search outputs match the frozen experiment exactly (excluding runtime).')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    verify(parser.parse_args().output)
