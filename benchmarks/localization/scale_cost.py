"""Isolated full-scale scheduling experiments; no live default changes."""

import argparse
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np

from aria_trace.services.mapping.scale_aware import ScaleAwareLocalizer
from benchmarks.localization import template_cpu
from benchmarks.localization.reference_cache import identity


class ParallelScaleLocalizer(ScaleAwareLocalizer):
    def __init__(self, atlas_path, workers=4):
        self._matching_pool = None
        super().__init__(atlas_path)
        if workers > 1:
            self._matching_pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='scale-match')

    def close(self):
        if self._matching_pool is not None:
            self._matching_pool.shutdown(wait=True, cancel_futures=True)
            self._matching_pool = None
        super().close()

    def refine_active_near(self, observation, mask, canonical_xy, search_radius_px=18., score_min=.55):
        if canonical_xy is None:
            raise ValueError('Local map refinement requires a canonical proposal')
        began = time.perf_counter()
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        def match(pair):
            scale, layer = pair
            return scale, self._observe_one_mode(layer, gradient, mask, canonical_xy, search_radius_px)
        results = (self._matching_pool.map(match, self._scale_pyramid) if self._matching_pool
                   else map(match, self._scale_pyramid))
        matches = [(scale, result) for scale, result in results if result.get('valid')]
        if not matches:
            return {'valid': False, 'x': None, 'y': None, 'score': 0., 'reason': 'no-covered-scale-hypothesis',
                    'elapsed_ms': (time.perf_counter()-began)*1000}
        scale, best = max(matches, key=lambda item: item[1]['score'])
        accepted = best['score'] >= score_min
        return {
            'valid': accepted,
            'x': canonical_xy[0]+best['best_offset_canonical_xy'][0] if accepted else None,
            'y': canonical_xy[1]+best['best_offset_canonical_xy'][1] if accepted else None,
            'score': best['score'],
            'margin': best['score']-max((m['score'] for s,m in matches if s != scale), default=0.),
            'coverage_fraction': best['coverage_fraction'], 'selected_mode_id': self.active_mode_id,
            'reason': None if accepted else 'correlation-below-threshold',
            'elapsed_ms': (time.perf_counter()-began)*1000,
            'pose_authority': 'current-frame-map-correlation', 'search_center_canonical_xy': list(canonical_xy),
            'search_radius_canonical_px': search_radius_px,
            'precision_scale': {'estimated_scale': scale, 'hypothesis_count': len(matches),
                                'discrete_mode_authority': 'unchanged-visual-controller',
                                'trigger': 'always', 'endpoint_score': None,
                                'scores': [[s,m['score']] for s,m in matches], 'winning_peak': best},
        }


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--scale-workers', type=int, choices=[1,2,4,6], default=4)
    parser.add_argument('--consume-mode-after-xy', action='store_true')
    parser.add_argument('--precise-release', action='store_true')
    selected, remainder = parser.parse_known_args()
    original_run = template_cpu.run
    def run(args):
        args.atlas_localizer_factory = partial(ParallelScaleLocalizer, workers=selected.scale_workers)
        args.experiment['scale_cost'] = {'workers': selected.scale_workers, 'source': identity(__file__),
                                         'operation': 'same-17-hypotheses-in-order',
                                         'consume_mode_after_xy': selected.consume_mode_after_xy,
                                         'precise_release': selected.precise_release}
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/'scale_cost_source.py').write_text(Path(__file__).read_text(), encoding='utf-8')
        if mode_source:
            (args.output/'mode_delivery_update.py').write_text(mode_source, encoding='utf-8')
        if selected.precise_release:
            (args.output/'precise_release_source.py').write_text(Path('benchmarks/localization/precise_release.py').read_text(), encoding='utf-8')
        return original_run(args)
    from benchmarks.localization.mode_delivery import consume_after_xy
    from benchmarks.localization import prefetched_replay
    from benchmarks.localization.precise_release import PreciseReleaseSource
    with consume_after_xy() if selected.consume_mode_after_xy else nullcontext(None) as mode_source:
        source_context = (patch.object(prefetched_replay,'PrefetchedSource',PreciseReleaseSource)
                          if selected.precise_release else nullcontext())
        with source_context:
            with patch.object(template_cpu, 'run', run), patch.object(sys, 'argv', [sys.argv[0], *remainder]):
                template_cpu.main()


if __name__ == '__main__':
    main()
