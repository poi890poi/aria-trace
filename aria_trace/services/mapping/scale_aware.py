"""Reusable scale-aware atlas candidate, explicitly selected for benchmarking.

Current image matching owns XY. The existing visual controller still owns the
discrete mode. This preserves the evaluated combination; it does not claim that
the full scale sweep meets control deadlines or fixes published mode/scale lag.
"""

import math
import time

import cv2
import numpy as np

from aria_trace.services.mapping.candidates import VisualTransitionLocalizer
from aria_trace.services.mapping.fractional_template import fractional_match


class _ScaleLayer:
    def __init__(self, mosaic, coverage, scale):
        height, width = mosaic.shape[:2]
        size = (int(round(width/scale)), int(round(height/scale)))
        if min(size) < 1:
            raise ValueError("Atlas scale produces an empty raster")
        image = cv2.resize(mosaic, size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.map_gradient = cv2.magnitude(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
        )
        self.coverage = cv2.resize(coverage, size, interpolation=cv2.INTER_NEAREST)
        self.scale_xy = np.array([width/size[0], height/size[1]])
        self.original_to_localization = np.diag([1/self.scale_xy[0], 1/self.scale_xy[1], 1.])

    def _localization_xy(self, xy):
        return np.asarray(xy)/self.scale_xy

    def _original_xy(self, xy):
        return np.asarray(xy)*self.scale_xy


def _build_pyramid(layered):
    began = time.perf_counter()
    endpoints = sorted(layered.map_scales.items(), key=lambda item: item[1])
    if not endpoints or any(not math.isfinite(s) or s <= 0 for _, s in endpoints):
        raise ValueError("Scale-aware matching requires positive finite atlas scales")
    layers = [(float(endpoints[0][1]), layered.localizers[endpoints[0][0]])]
    extra_bytes = 0
    if endpoints[-1][1] > endpoints[0][1]:
        scales = np.geomspace(endpoints[0][1], endpoints[-1][1], 17)
        mosaic = cv2.imread(str(layered.atlas_path/layered.manifest['canonical_mosaic_file']))
        coverage = cv2.imread(
            str(layered.atlas_path/layered.manifest['canonical_coverage_file']), cv2.IMREAD_GRAYSCALE
        )
        if mosaic is None or coverage is None:
            raise ValueError("Scale pyramid input cannot be read")
        for scale in scales[1:-1]:
            layer = _ScaleLayer(mosaic, coverage, scale)
            layers.append((float(scale), layer))
            extra_bytes += layer.map_gradient.nbytes + layer.coverage.nbytes
        # Reuse the actual endpoint rasters and transforms. Resampling those
        # again would silently change the frozen baseline's coordinate grid.
        layers.append((float(scales[-1]), layered.localizers[endpoints[-1][0]]))
    return layers, {
        "scales": [scale for scale, _ in layers],
        "setup_ms": (time.perf_counter()-began)*1000,
        "extra_array_bytes": extra_bytes,
    }


class ScaleAwareLocalizer(VisualTransitionLocalizer):
    """Fractional XY across a fixed atlas scale pyramid; no route-state inputs."""

    _observe_one_mode = staticmethod(fractional_match)

    def __init__(self, atlas_path):
        super().__init__(atlas_path)
        try:
            self._scale_pyramid, self.scale_search_setup = _build_pyramid(self)
        except Exception:
            self.close()
            raise

    def close(self):
        super().close()
        if hasattr(self, '_scale_pyramid'):
            del self._scale_pyramid

    def refine_active_near(self, observation, mask, canonical_xy, search_radius_px=18., score_min=.55):
        if canonical_xy is None:
            raise ValueError("Local map refinement requires a canonical proposal")
        began = time.perf_counter()
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gradient = cv2.magnitude(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
        )
        matches = []
        for scale, layer in self._scale_pyramid:
            result = self._observe_one_mode(layer, gradient, mask, canonical_xy, search_radius_px)
            if result.get('valid'):
                matches.append((scale, result))
        if not matches:
            return {"valid": False, "x": None, "y": None, "score": 0.,
                    "reason": "no-covered-scale-hypothesis", "elapsed_ms": (time.perf_counter()-began)*1000}
        scale, best = max(matches, key=lambda item: item[1]['score'])
        accepted = best['score'] >= score_min
        return {
            "valid": accepted,
            "x": canonical_xy[0]+best['best_offset_canonical_xy'][0] if accepted else None,
            "y": canonical_xy[1]+best['best_offset_canonical_xy'][1] if accepted else None,
            "score": best['score'],
            "margin": best['score']-max((m['score'] for s, m in matches if s != scale), default=0.),
            "coverage_fraction": best['coverage_fraction'],
            "selected_mode_id": self.active_mode_id,
            "reason": None if accepted else "correlation-below-threshold",
            "elapsed_ms": (time.perf_counter()-began)*1000,
            "pose_authority": "current-frame-map-correlation",
            "search_center_canonical_xy": list(canonical_xy),
            "search_radius_canonical_px": search_radius_px,
            "precision_scale": {
                "estimated_scale": scale, "hypothesis_count": len(matches),
                "discrete_mode_authority": "unchanged-visual-controller",
                "trigger": "always", "endpoint_score": None,
                "scores": [[s, m['score']] for s, m in matches], "winning_peak": best,
            },
        }
