"""Score frozen atlas support on a withheld recording; never feeds runtime XY."""

import argparse
from collections import Counter
import json
from pathlib import Path

import cv2

from benchmarks.localization.reference_cache import identity
from benchmarks.localization.run_workbench_replay import read_rows, validate_reference_inputs


def evaluate(atlas, reference, output):
    validate_reference_inputs(reference)
    marker = json.loads((reference/'cache.json').read_text())
    for item in marker['outputs']:
        if identity(reference/item['name'])['sha256'] != item['sha256']:
            raise RuntimeError('Reference output changed')
    feature_path = atlas/'map_features/observed_modes.json'
    feature = json.loads(feature_path.read_text())
    trained = [r['reference']['sha256'] for r in feature['training_references']]
    if identity(reference/'cache.json')['sha256'] in trained:
        raise ValueError('This recording was included in support training')
    manifest = json.loads((atlas/'map_atlas.json').read_text())
    scales = {layer['mode_id']:layer['map_pixels_per_minimap_pixel'] for layer in manifest['layers']}
    codes = cv2.imread(str(atlas/'map_features/observed_mode_support.png'), cv2.IMREAD_GRAYSCALE)
    counts = Counter(); stable = 0
    for row in read_rows(reference/'route_states.jsonl'):
        mode = row['mode_id']
        if abs(row['map_scale']/scales[mode]-1) > feature['stable_scale_relative_band']:
            continue
        stable += 1
        x,y = (round(v) for v in row['canonical_xy'])
        code = int(codes[y,x]) if 0 <= x < codes.shape[1] and 0 <= y < codes.shape[0] else 0
        counts[(mode,feature['pixel_codes'][str(code)])] += 1
    result = {'role':'post-run lookup at inferred reference XY; best-case support diagnostic, not runtime localization',
              'features':identity(feature_path),'withheld_reference':identity(reference/'cache.json'),
              'support_mask':identity(atlas/'map_features/observed_mode_support.png'),
              'atlas_manifest':identity(atlas/'map_atlas.json'),'implementation':identity(__file__),
              'stable_reference_samples':stable,'counts':[{'reference_mode':k[0],'support':k[1],'samples':v} for k,v in sorted(counts.items())],
              'limitations':['No score for rejected or missing reference samples.',
                             'Sparse support is not a complete town boundary; unknown support must not veto image evidence.',
                             'Directed crossing brackets remain atlas-owned; no route order is used.']}
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('atlas',type=Path);p.add_argument('reference',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();evaluate(args.atlas,args.reference,args.output)
