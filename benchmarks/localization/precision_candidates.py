"""Isolated fractional-XY and intermediate-scale template experiments."""

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

import cv2
import numpy as np

from aria_trace.services.mapping.layers import LayeredGlobalLocalizer
from benchmarks.localization import template_cpu
from benchmarks.localization.reference_cache import identity
from benchmarks.localization.transition_zone_ablation import without_spatial_gates


def quadratic_peak(response, location):
    """Concave three-point fit; edges/flat or convex peaks remain integer."""
    x, y = location
    def offset(a, b, c):
        denominator = float(a) - 2 * float(b) + float(c)
        if denominator >= -1e-8 or not all(math.isfinite(float(v)) for v in (a, b, c)):
            return 0.0
        return float(np.clip(.5 * (float(a) - float(c)) / denominator, -.5, .5))
    dx = offset(*response[y, x-1:x+2]) if 0 < x < response.shape[1]-1 else 0.0
    dy = offset(*response[y-1:y+2, x]) if 0 < y < response.shape[0]-1 else 0.0
    return dx, dy


def fractional_match(localizer, observation_gradient, mask, canonical_xy, search_radius_px):
    height, width = observation_gradient.shape[:2]
    center_x, center_y = localizer._localization_xy(canonical_xy)
    scale_x = math.hypot(localizer.original_to_localization[0, 0], localizer.original_to_localization[1, 0])
    scale_y = math.hypot(localizer.original_to_localization[0, 1], localizer.original_to_localization[1, 1])
    radius = max(4., float(search_radius_px) * (scale_x + scale_y) / 2)
    left = max(0, int(math.floor(center_x-radius-width/2)))
    top = max(0, int(math.floor(center_y-radius-height/2)))
    right = min(localizer.map_gradient.shape[1], int(math.ceil(center_x+radius+width/2)))
    bottom = min(localizer.map_gradient.shape[0], int(math.ceil(center_y+radius+height/2)))
    search = localizer.map_gradient[top:bottom, left:right]
    if search.shape[0] < height or search.shape[1] < width:
        return {"valid": False, "score": 0., "coverage_fraction": 0., "reason": "insufficient-local-map-area"}
    response = cv2.matchTemplate(search, observation_gradient, cv2.TM_CCORR_NORMED, mask=mask)
    response = np.nan_to_num(response, nan=-1., posinf=-1., neginf=-1.)
    _, score, _, location = cv2.minMaxLoc(response)
    match_left, match_top = left+location[0], top+location[1]
    coverage = localizer.coverage[match_top:match_top+height, match_left:match_left+width]
    selected = mask > 0
    fraction = float(np.mean(coverage[selected] > 0) if np.any(selected) else 0.)
    dx, dy = quadratic_peak(response, location)
    center = localizer._original_xy((match_left+width/2+dx, match_top+height/2+dy))
    valid = bool(math.isfinite(score) and score >= 0 and fraction >= .75)
    return {"valid": valid, "score": max(0., float(score)) if valid else 0., "coverage_fraction": fraction,
            "best_offset_canonical_xy": [float(center[0]-canonical_xy[0]), float(center[1]-canonical_xy[1])],
            "search_bounds_localization_xyxy": [left, top, right, bottom],
            "reason": None if valid else "insufficient-observed-coverage",
            "subpixel_offset_localization_xy": [dx, dy]}


class PyramidLayer:
    def __init__(self, mosaic, coverage, requested_scale):
        h, w = mosaic.shape[:2]
        size = (int(round(w/requested_scale)), int(round(h/requested_scale)))
        image = cv2.resize(mosaic, size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.map_gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        self.coverage = cv2.resize(coverage, size, interpolation=cv2.INTER_NEAREST)
        self.scale_xy = np.array([w/size[0], h/size[1]])
        self.original_to_localization = np.diag([1/self.scale_xy[0], 1/self.scale_xy[1], 1.])

    def _localization_xy(self, xy): return np.asarray(xy)/self.scale_xy
    def _original_xy(self, xy): return np.asarray(xy)*self.scale_xy


def build_pyramid(layered, count=17):
    began = time.perf_counter()
    endpoints = sorted(layered.map_scales.items(), key=lambda item: item[1])
    scales = np.geomspace(endpoints[0][1], endpoints[-1][1], count)
    mosaic = cv2.imread(str(layered.atlas_path/layered.manifest['canonical_mosaic_file']))
    coverage = cv2.imread(str(layered.atlas_path/layered.manifest['canonical_coverage_file']), cv2.IMREAD_GRAYSCALE)
    if mosaic is None or coverage is None:
        raise RuntimeError('Scale pyramid input cannot be read')
    layers = [(float(scales[0]), layered.localizers[endpoints[0][0]])]
    layers += [(float(scale), PyramidLayer(mosaic, coverage, scale)) for scale in scales[1:-1]]
    layers.append((float(scales[-1]), layered.localizers[endpoints[-1][0]]))
    return layers, {"scales": [scale for scale, _ in layers], "setup_ms": (time.perf_counter()-began)*1000,
                    "extra_array_bytes": sum(layer.map_gradient.nbytes+layer.coverage.nbytes for _, layer in layers[1:-1])}


@contextmanager
def installed(subpixel=False, scale_sweep=False, selective_scale=False):
    original_init = LayeredGlobalLocalizer.__init__
    original_close = LayeredGlobalLocalizer.close
    original_refine = LayeredGlobalLocalizer.refine_active_near
    original_match = LayeredGlobalLocalizer._observe_one_mode
    setup = []
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if scale_sweep:
            self._precision_pyramid, metadata = build_pyramid(self)
            setup.append(metadata)
    def close(self):
        original_close(self)
        if hasattr(self, '_precision_pyramid'):
            del self._precision_pyramid
    def refine(self, observation, mask, canonical_xy, search_radius_px=18., score_min=.55):
        if not scale_sweep:
            return original_refine(self, observation, mask, canonical_xy, search_radius_px, score_min)
        began = time.perf_counter()
        endpoint = None
        if selective_scale:
            endpoint = original_refine(self, observation, mask, canonical_xy, search_radius_px, score_min)
            if endpoint.get('valid') and endpoint['score'] >= .55:
                endpoint['precision_scale'] = {'estimated_scale': self.map_scales.get(self.active_mode_id),
                                               'hypothesis_count': 1, 'trigger': 'endpoint-score-sufficient',
                                               'trigger_threshold': .55}
                return endpoint
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        matches = []
        for scale, layer in self._precision_pyramid:
            result = self._observe_one_mode(layer, gradient, mask, canonical_xy, search_radius_px)
            if result.get('valid'):
                matches.append((scale, result))
        if not matches:
            return {"valid": False, "x": None, "y": None, "score": 0., "reason": "no-covered-scale-hypothesis", "elapsed_ms": (time.perf_counter()-began)*1000}
        scale, best = max(matches, key=lambda item: item[1]['score'])
        accepted = best['score'] >= score_min
        return {"valid": accepted, "x": canonical_xy[0]+best['best_offset_canonical_xy'][0] if accepted else None,
                "y": canonical_xy[1]+best['best_offset_canonical_xy'][1] if accepted else None,
                "score": best['score'], "margin": best['score']-max((m['score'] for s,m in matches if s != scale), default=0.),
                "coverage_fraction": best['coverage_fraction'], "selected_mode_id": self.active_mode_id,
                "reason": None if accepted else "correlation-below-threshold", "elapsed_ms": (time.perf_counter()-began)*1000,
                "pose_authority": "current-frame-map-correlation", "search_center_canonical_xy": list(canonical_xy),
                "search_radius_canonical_px": search_radius_px,
                "precision_scale": {"estimated_scale": scale, "hypothesis_count": len(matches), "discrete_mode_authority": "unchanged-visual-controller",
                                    "trigger": "weak-endpoint" if selective_scale else "always",
                                    "endpoint_score": endpoint.get('score') if endpoint else None,
                                    "scores": [[s, m['score']] for s,m in matches], "winning_peak": best}}
    with patch.object(LayeredGlobalLocalizer, '__init__', initialize), patch.object(LayeredGlobalLocalizer, 'close', close), patch.object(LayeredGlobalLocalizer, 'refine_active_near', refine), patch.object(LayeredGlobalLocalizer, '_observe_one_mode', staticmethod(fractional_match if subpixel else original_match)):
        yield setup


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--subpixel', action='store_true'); parser.add_argument('--scale-sweep', action='store_true')
    parser.add_argument('--selective-scale', action='store_true')
    candidate, remainder = parser.parse_known_args()
    if candidate.selective_scale and not candidate.scale_sweep:
        parser.error('--selective-scale requires --scale-sweep')
    original_run = template_cpu.run
    with installed(candidate.subpixel, candidate.scale_sweep, candidate.selective_scale) as setup, without_spatial_gates():
        def run(args):
            args.experiment['precision'] = {"subpixel": candidate.subpixel, "scale_sweep": candidate.scale_sweep,
                                           "selective_scale": candidate.selective_scale,
                                           "transition_zone_policy": "visual-only", "setup": setup, "source": identity(__file__)}
            args.output.mkdir(parents=True, exist_ok=True)
            (args.output/'precision_source.py').write_text(Path(__file__).read_text(), encoding='utf-8')
            result = original_run(args)
            (args.output/'precision_setup.json').write_text(json.dumps(setup, indent=2))
            return result
        with patch.object(template_cpu, 'run', run), patch.object(sys, 'argv', [sys.argv[0], *remainder]):
            template_cpu.main()


if __name__ == '__main__':
    main()
