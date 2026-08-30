"""Canonical multi-scale map atlas construction and localization."""

import json
import math
import shutil
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional

import cv2
import numpy as np

from acquisition.live_tracker import GlobalFix, GlobalMapLocalizer


SCHEMA_VERSION = "1.0"


def _read_image(path: Path, flags: int) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise ValueError("Could not decode map-layer image: {}".format(path))
    return image


def _homogeneous(matrix_2x3) -> np.ndarray:
    value = np.eye(3, dtype=np.float64)
    value[:2] = np.asarray(matrix_2x3, dtype=np.float64)
    return value


def estimate_map_layer_alignment(
    canonical_mosaic: np.ndarray,
    layer_mosaic: np.ndarray,
    canonical_coverage: Optional[np.ndarray] = None,
    layer_coverage: Optional[np.ndarray] = None,
) -> dict:
    """Estimate a similarity from one rendered map layer into canonical pixels."""

    canonical_gray = cv2.cvtColor(canonical_mosaic, cv2.COLOR_BGR2GRAY)
    layer_gray = cv2.cvtColor(layer_mosaic, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.004, edgeThreshold=15)
    canonical_points, canonical_descriptors = sift.detectAndCompute(
        canonical_gray, canonical_coverage
    )
    layer_points, layer_descriptors = sift.detectAndCompute(layer_gray, layer_coverage)
    if canonical_descriptors is None or layer_descriptors is None:
        raise RuntimeError("Map-layer alignment found no SIFT descriptors")
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        layer_descriptors, canonical_descriptors, k=2
    )
    matches = [
        first
        for pair in pairs
        if len(pair) == 2
        for first, second in [pair]
        if first.distance < 0.78 * second.distance
    ]
    if len(matches) < 8:
        raise RuntimeError(
            "Map layers need at least 8 ratio-test matches; got {}".format(
                len(matches)
            )
        )
    source = np.float32([layer_points[item.queryIdx].pt for item in matches])
    target = np.float32([canonical_points[item.trainIdx].pt for item in matches])
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
        maxIters=30000,
        confidence=0.999,
    )
    if matrix is None or inlier_mask is None:
        raise RuntimeError("Map-layer matches have no consistent similarity")
    accepted = inlier_mask.ravel().astype(bool)
    inlier_count = int(np.count_nonzero(accepted))
    predicted = cv2.transform(source.reshape((-1, 1, 2)), matrix).reshape((-1, 2))
    errors = np.linalg.norm(predicted - target, axis=1)
    inlier_ratio = float(np.mean(accepted))
    reprojection_p95 = float(np.percentile(errors[accepted], 95))
    if inlier_count < 8 or inlier_ratio < 0.30 or reprojection_p95 > 6.0:
        raise RuntimeError(
            "Map-layer alignment is weak: {} inliers, ratio {:.3f}, p95 {:.2f}px".format(
                inlier_count, inlier_ratio, reprojection_p95
            )
        )
    scale = math.hypot(float(matrix[0, 0]), float(matrix[1, 0]))
    rotation = math.degrees(math.atan2(float(matrix[1, 0]), float(matrix[0, 0])))
    return {
        "layer_original_to_canonical_3x3": _homogeneous(matrix),
        "quality": {
            "ratio_match_count": len(matches),
            "inlier_count": inlier_count,
            "inlier_ratio": inlier_ratio,
            "reprojection_median_px": float(np.median(errors[accepted])),
            "reprojection_p95_px": reprojection_p95,
            "canonical_pixels_per_layer_pixel": float(scale),
            "rotation_deg": float(rotation),
        },
    }


def _load_stitch(stitch_root: Path) -> dict:
    manifest_path = stitch_root / "map_stitch.json"
    if not manifest_path.is_file():
        raise ValueError("Map stitch manifest does not exist: {}".format(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return manifest


def _build_layer_specific_localization(
    mosaic: np.ndarray,
    coverage: np.ndarray,
    reference: np.ndarray,
    reference_mask: np.ndarray,
    output_path: Path,
    mode_id: str,
):
    estimate = estimate_map_layer_alignment(
        mosaic,
        reference,
        coverage,
        reference_mask,
    )
    original_per_minimap = float(
        estimate["quality"]["canonical_pixels_per_layer_pixel"]
    )
    if not 0.25 <= original_per_minimap <= 32.0:
        raise RuntimeError(
            "Layer {} map/mini-map scale {:.3f} is implausible".format(
                mode_id, original_per_minimap
            )
        )
    requested_factor = 1.0 / original_per_minimap
    size = (
        max(64, int(round(mosaic.shape[1] * requested_factor))),
        max(64, int(round(mosaic.shape[0] * requested_factor))),
    )
    interpolation = cv2.INTER_AREA if requested_factor < 1.0 else cv2.INTER_CUBIC
    localization_mosaic = cv2.resize(mosaic, size, interpolation=interpolation)
    localization_coverage = cv2.resize(
        coverage, size, interpolation=cv2.INTER_NEAREST
    )
    factor_x = size[0] / float(mosaic.shape[1])
    factor_y = size[1] / float(mosaic.shape[0])
    localization_to_original = np.asarray(
        [[1.0 / factor_x, 0.0, 0.0], [0.0, 1.0 / factor_y, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    layer_directory = output_path / "layers" / mode_id
    layer_directory.mkdir(parents=True, exist_ok=True)
    localization_file = "layers/{}/localization_mosaic.png".format(mode_id)
    coverage_file = "layers/{}/localization_coverage.png".format(mode_id)
    reference_file = "minimap_reference_{}.png".format(mode_id)
    if not cv2.imwrite(str(output_path / localization_file), localization_mosaic):
        raise RuntimeError("Could not write layer-specific localization mosaic")
    if not cv2.imwrite(str(output_path / coverage_file), localization_coverage):
        raise RuntimeError("Could not write layer-specific localization coverage")
    if not cv2.imwrite(str(output_path / reference_file), reference):
        raise RuntimeError("Could not write layer-specific mini-map reference")
    return {
        "localization_mosaic_file": localization_file,
        "localization_coverage_file": coverage_file,
        "localization_to_original_3x3": localization_to_original,
        "localization_size_wh": list(size),
        "map_pixels_per_minimap_pixel": original_per_minimap,
        "minimap_reference_file": reference_file,
        "minimap_reference_to_original_map_3x3": estimate[
            "layer_original_to_canonical_3x3"
        ].tolist(),
        "minimap_reference_quality": estimate["quality"],
    }


def build_map_atlas(
    layer_specs: Iterable[Mapping],
    output_path: Path,
    *,
    canonical_mode_id: str,
    atlas_id: Optional[str] = None,
) -> dict:
    """Build a portable atlas from independently stitched rendered map scales."""

    specs = [dict(item) for item in layer_specs]
    mode_ids = [str(item.get("mode_id") or "") for item in specs]
    if len(specs) < 2:
        raise ValueError("A multi-scale map atlas needs at least two layers")
    if any(not mode_id for mode_id in mode_ids) or len(set(mode_ids)) != len(mode_ids):
        raise ValueError("Map atlas mode IDs must be non-empty and unique")
    if canonical_mode_id not in mode_ids:
        raise ValueError("Canonical mode must name one of the map layers")
    output_path = Path(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise RuntimeError("Map atlas output directory is not empty")
    output_path.mkdir(parents=True, exist_ok=True)

    loaded = []
    for spec in specs:
        stitch_root = Path(spec["stitch_root"])
        manifest = _load_stitch(stitch_root)
        mosaic = _read_image(stitch_root / "mosaic.png", cv2.IMREAD_COLOR)
        coverage = _read_image(stitch_root / "coverage.png", cv2.IMREAD_GRAYSCALE)
        loaded.append((spec, stitch_root, manifest, mosaic, coverage))
    canonical = next(item for item in loaded if item[0]["mode_id"] == canonical_mode_id)
    canonical_mosaic = canonical[3]
    canonical_coverage = canonical[4]
    atlas_id = str(atlas_id or output_path.name)
    canonical_file = "canonical_mosaic.png"
    canonical_coverage_file = "canonical_coverage.png"
    if not cv2.imwrite(str(output_path / canonical_file), canonical_mosaic):
        raise RuntimeError("Could not write canonical atlas mosaic")
    if not cv2.imwrite(str(output_path / canonical_coverage_file), canonical_coverage):
        raise RuntimeError("Could not write canonical atlas coverage")

    layers = []
    for spec, stitch_root, stitch, mosaic, coverage in loaded:
        mode_id = str(spec["mode_id"])
        shares_canonical_source = stitch_root.resolve() == canonical[1].resolve()
        if mode_id == canonical_mode_id or shares_canonical_source:
            original_to_canonical = np.eye(3, dtype=np.float64)
            alignment_quality = {
                "method": (
                    "canonical_identity"
                    if mode_id == canonical_mode_id
                    else "shared_source_identity"
                ),
                "inlier_count": None,
                "inlier_ratio": 1.0,
                "reprojection_p95_px": 0.0,
                "canonical_pixels_per_layer_pixel": 1.0,
                "rotation_deg": 0.0,
            }
            alignment_file = None
        else:
            estimate = estimate_map_layer_alignment(
                canonical_mosaic,
                mosaic,
                canonical_coverage,
                coverage,
            )
            original_to_canonical = estimate["layer_original_to_canonical_3x3"]
            alignment_quality = dict(estimate["quality"])
            alignment_quality["method"] = "sift_similarity_ransac"
            warped = cv2.warpPerspective(
                mosaic,
                original_to_canonical,
                (canonical_mosaic.shape[1], canonical_mosaic.shape[0]),
            )
            warped_mask = cv2.warpPerspective(
                coverage,
                original_to_canonical,
                (canonical_mosaic.shape[1], canonical_mosaic.shape[0]),
                flags=cv2.INTER_NEAREST,
            )
            alignment_view = canonical_mosaic.copy()
            valid = warped_mask > 0
            alignment_view[valid] = cv2.addWeighted(
                canonical_mosaic[valid], 0.5, warped[valid], 0.5, 0.0
            )
            alignment_file = "alignment_{}.png".format(mode_id)
            if not cv2.imwrite(str(output_path / alignment_file), alignment_view):
                raise RuntimeError("Could not write map-layer alignment evidence")

        reference = spec.get("minimap_reference")
        reference_mask = spec.get("minimap_reference_mask")
        if reference is not None and reference_mask is not None:
            layer_localization = _build_layer_specific_localization(
                mosaic,
                coverage,
                reference,
                reference_mask,
                output_path,
                mode_id,
            )
            localization_file = layer_localization["localization_mosaic_file"]
            localization_coverage_file = layer_localization[
                "localization_coverage_file"
            ]
            localization_to_original = layer_localization[
                "localization_to_original_3x3"
            ]
            localization_source = "transition_endpoint_minimap_reference"
        else:
            localization = stitch.get("localization") or {}
            if localization.get("status") != "ready":
                raise ValueError(
                    "Map stitch {} needs a ready localization raster or a layer-specific mini-map reference"
                    .format(stitch_root.name)
                )
            localization_to_original = np.asarray(
                localization["localization_to_original_map_3x3"], dtype=np.float64
            )
            layer_directory = output_path / "layers" / mode_id
            layer_directory.mkdir(parents=True, exist_ok=True)
            localization_file = "layers/{}/localization_mosaic.png".format(mode_id)
            localization_coverage_file = "layers/{}/localization_coverage.png".format(
                mode_id
            )
            shutil.copy2(
                stitch_root / localization["mosaic_file"],
                output_path / localization_file,
            )
            shutil.copy2(
                stitch_root / localization["coverage_file"],
                output_path / localization_coverage_file,
            )
            layer_localization = {
                "localization_size_wh": localization.get("size_wh"),
                "map_pixels_per_minimap_pixel": localization.get(
                    "map_pixels_per_minimap_pixel"
                ),
                "minimap_reference_file": None,
                "minimap_reference_quality": localization.get("quality"),
            }
            localization_source = "source_stitch_localization"
        localization_to_canonical = original_to_canonical.dot(
            localization_to_original
        )
        layers.append(
            {
                "mode_id": mode_id,
                "display_name": str(spec.get("display_name") or mode_id),
                "source_stitch_id": str(stitch.get("stitch_id") or stitch_root.name),
                "source_minimap_calibration_id": stitch.get(
                    "source_minimap_calibration_id"
                ),
                "localization_mosaic_file": localization_file,
                "localization_coverage_file": localization_coverage_file,
                "localization_to_canonical_3x3": localization_to_canonical.tolist(),
                "localization_source": localization_source,
                "localization_size_wh": layer_localization.get(
                    "localization_size_wh"
                ),
                "map_pixels_per_minimap_pixel": layer_localization.get(
                    "map_pixels_per_minimap_pixel"
                ),
                "minimap_reference_file": layer_localization.get(
                    "minimap_reference_file"
                ),
                "minimap_reference_quality": layer_localization.get(
                    "minimap_reference_quality"
                ),
                "original_map_to_canonical_3x3": original_to_canonical.tolist(),
                "alignment_quality": alignment_quality,
                "alignment_evidence_file": alignment_file,
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "atlas_id": atlas_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ready",
        "coordinate_space_id": "map-atlas:{}:canonical-map-px".format(atlas_id),
        "canonical_mode_id": canonical_mode_id,
        "canonical_mosaic_file": canonical_file,
        "canonical_coverage_file": canonical_coverage_file,
        "canonical_size_wh": [canonical_mosaic.shape[1], canonical_mosaic.shape[0]],
        "layers": layers,
    }
    (output_path / "map_atlas.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


class LayeredGlobalLocalizer:
    """Localize against rendered map scales while returning canonical pixels."""

    supports_bounded_search = True

    def __init__(self, atlas_path: Path) -> None:
        self.atlas_path = Path(atlas_path)
        self.manifest = json.loads(
            (self.atlas_path / "map_atlas.json").read_text(encoding="utf-8")
        )
        self.transition_model = self.manifest.get("transition_model")
        self.localizers = {}
        self.map_scales = {}
        for layer in self.manifest["layers"]:
            mode_id = str(layer["mode_id"])
            self.localizers[mode_id] = GlobalMapLocalizer(
                _read_image(
                    self.atlas_path / layer["localization_mosaic_file"],
                    cv2.IMREAD_COLOR,
                ),
                _read_image(
                    self.atlas_path / layer["localization_coverage_file"],
                    cv2.IMREAD_GRAYSCALE,
                ),
                layer["localization_to_canonical_3x3"],
            )
            scale = layer.get("map_pixels_per_minimap_pixel")
            if scale is not None:
                self.map_scales[mode_id] = float(scale)
        self.active_mode_id = None
        self.last_mode_likelihoods = {}
        self.last_selected_mode_id = None

    def close(self) -> None:
        for localizer in self.localizers.values():
            localizer.close()

    def set_active_mode(self, mode_id: Optional[str]) -> None:
        if mode_id is not None and mode_id not in self.localizers:
            raise ValueError("Unknown map-layer mode: {}".format(mode_id))
        self.active_mode_id = mode_id

    def map_scale_for_mode(self, mode_id: str) -> float:
        """Return the atlas-declared map scale for a representation mode."""

        value = self.map_scales.get(str(mode_id))
        if value is None or not math.isfinite(value) or value <= 0.0:
            raise ValueError("Map layer has no valid declared scale: {}".format(mode_id))
        return value

    @staticmethod
    def _observe_one_mode(
        localizer: GlobalMapLocalizer,
        observation_gradient: np.ndarray,
        mask: np.ndarray,
        canonical_xy,
        search_radius_px: float,
    ) -> dict:
        """Score one normalized layer near a known pose without returning a pose."""

        height, width = observation_gradient.shape[:2]
        center_x, center_y = localizer._localization_xy(canonical_xy)
        scale_x = math.hypot(
            localizer.original_to_localization[0, 0],
            localizer.original_to_localization[1, 0],
        )
        scale_y = math.hypot(
            localizer.original_to_localization[0, 1],
            localizer.original_to_localization[1, 1],
        )
        radius = max(4.0, float(search_radius_px) * (scale_x + scale_y) / 2.0)
        left = max(0, int(math.floor(center_x - radius - width / 2.0)))
        top = max(0, int(math.floor(center_y - radius - height / 2.0)))
        right = min(
            localizer.map_gradient.shape[1],
            int(math.ceil(center_x + radius + width / 2.0)),
        )
        bottom = min(
            localizer.map_gradient.shape[0],
            int(math.ceil(center_y + radius + height / 2.0)),
        )
        search = localizer.map_gradient[top:bottom, left:right]
        if search.shape[0] < height or search.shape[1] < width:
            return {
                "valid": False,
                "score": 0.0,
                "coverage_fraction": 0.0,
                "reason": "insufficient-local-map-area",
            }
        response = cv2.matchTemplate(
            search,
            observation_gradient,
            cv2.TM_CCORR_NORMED,
            mask=mask,
        )
        response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
        _, score, _, location = cv2.minMaxLoc(response)
        match_left = left + int(location[0])
        match_top = top + int(location[1])
        coverage_patch = localizer.coverage[
            match_top : match_top + height,
            match_left : match_left + width,
        ]
        selected = mask > 0
        coverage_fraction = float(
            np.mean(coverage_patch[selected] > 0) if np.any(selected) else 0.0
        )
        match_center_local = (
            match_left + width / 2.0,
            match_top + height / 2.0,
        )
        match_center_canonical = localizer._original_xy(match_center_local)
        offset = (
            float(match_center_canonical[0] - canonical_xy[0]),
            float(match_center_canonical[1] - canonical_xy[1]),
        )
        valid = bool(math.isfinite(score) and score >= 0.0 and coverage_fraction >= 0.75)
        return {
            "valid": valid,
            "score": max(0.0, float(score)) if valid else 0.0,
            "coverage_fraction": coverage_fraction,
            "best_offset_canonical_xy": list(offset),
            "search_bounds_localization_xyxy": [left, top, right, bottom],
            "reason": None if valid else "insufficient-observed-coverage",
        }

    def observe_modes(
        self,
        observation: np.ndarray,
        mask: np.ndarray,
        canonical_xy,
        search_radius_px: float = 40.0,
    ) -> dict:
        """Compare every rendered scale locally while keeping pose read-only.

        This is deliberately separate from global localization.  The returned
        best offsets are diagnostics only; callers may use the scores to change
        representation, but never to correct position or yaw.
        """

        started = time.perf_counter()
        if canonical_xy is None:
            raise ValueError("Mode observation requires an established canonical pose")
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)
        gradient = cv2.magnitude(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
        )
        diagnostics = {
            mode_id: self._observe_one_mode(
                localizer,
                gradient,
                mask,
                canonical_xy,
                search_radius_px,
            )
            for mode_id, localizer in self.localizers.items()
        }
        raw_scores = {
            mode_id: float(value["score"])
            for mode_id, value in diagnostics.items()
            if value["valid"]
        }
        best_score = max(raw_scores.values(), default=0.0)
        likelihoods = {
            mode_id: score / best_score
            for mode_id, score in raw_scores.items()
            if best_score > 0.0
        }
        ordered = sorted(likelihoods.items(), key=lambda item: item[1], reverse=True)
        return {
            "valid": len(likelihoods) >= 2,
            "likelihoods": likelihoods,
            "raw_correlation_scores": raw_scores,
            "score_normalization": "relative_to_best_correlation",
            "selected_mode_id": ordered[0][0] if ordered else None,
            "score_margin": (
                float(ordered[0][1] - ordered[1][1]) if len(ordered) >= 2 else 0.0
            ),
            "canonical_xy_read_only": [float(canonical_xy[0]), float(canonical_xy[1])],
            "search_radius_canonical_px": float(search_radius_px),
            "modes": diagnostics,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "pose_authority": "none",
        }

    def refine_near(
        self,
        observation: np.ndarray,
        mask: np.ndarray,
        canonical_xy,
        search_radius_px: float = 18.0,
        score_min: float = 0.55,
        mode_ids=None,
    ) -> dict:
        """Return a current-frame map pose near a proposed canonical position.

        Unlike ``observe_modes``, this method has explicit XY pose authority.
        The caller supplies only a bounded search proposal; the returned pose is
        the peak measured from the current mini-map against atlas pixels.
        """

        started = time.perf_counter()
        if canonical_xy is None:
            raise ValueError("Local map refinement requires a canonical proposal")
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32)
        gradient = cv2.magnitude(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
        )
        selected_ids = tuple(mode_ids or self.localizers)
        hypotheses = []
        diagnostics = {}
        for mode_id in selected_ids:
            match = self._observe_one_mode(
                self.localizers[mode_id],
                gradient,
                mask,
                canonical_xy,
                search_radius_px,
            )
            diagnostics[str(mode_id)] = match
            if not match.get("valid"):
                continue
            offset = match["best_offset_canonical_xy"]
            hypotheses.append(
                {
                    "x": float(canonical_xy[0]) + float(offset[0]),
                    "y": float(canonical_xy[1]) + float(offset[1]),
                    "score": float(match["score"]),
                    "mode_id": str(mode_id),
                    "coverage_fraction": float(match["coverage_fraction"]),
                }
            )
        hypotheses.sort(key=lambda item: item["score"], reverse=True)
        if not hypotheses:
            return {
                "valid": False,
                "x": None,
                "y": None,
                "score": 0.0,
                "margin": 0.0,
                "selected_mode_id": None,
                "reason": "no-covered-local-map-hypothesis",
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "pose_authority": "current-frame-map-correlation",
                "modes": diagnostics,
            }
        best = hypotheses[0]
        distinct_scores = [
            item["score"]
            for item in hypotheses[1:]
            if math.hypot(item["x"] - best["x"], item["y"] - best["y"])
            >= 8.0
        ]
        second = max(distinct_scores, default=0.0)
        valid = bool(best["score"] >= float(score_min))
        return {
            "valid": valid,
            "x": best["x"] if valid else None,
            "y": best["y"] if valid else None,
            "score": best["score"],
            "margin": best["score"] - second,
            "selected_mode_id": best["mode_id"],
            "coverage_fraction": best["coverage_fraction"],
            "reason": None if valid else "correlation-below-threshold",
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "pose_authority": "current-frame-map-correlation",
            "search_center_canonical_xy": [
                float(canonical_xy[0]),
                float(canonical_xy[1]),
            ],
            "search_radius_canonical_px": float(search_radius_px),
            "modes": diagnostics,
        }

    def localize_all(
        self,
        observation: np.ndarray,
        mask: np.ndarray,
        yaw_prior_deg: Optional[float] = None,
        search_center_xy=None,
        search_radius_px: Optional[float] = None,
        mode_ids=None,
    ):
        selected_ids = tuple(mode_ids or self.localizers)
        fixes = {}
        for mode_id in selected_ids:
            localizer = self.localizers[mode_id]
            fixes[mode_id] = localizer.localize(
                observation,
                mask,
                yaw_prior_deg,
                search_center_xy=search_center_xy,
                search_radius_px=search_radius_px,
            )
        return fixes

    def localize(
        self,
        observation: np.ndarray,
        mask: np.ndarray,
        yaw_prior_deg: Optional[float] = None,
        search_center_xy=None,
        search_radius_px: Optional[float] = None,
    ) -> GlobalFix:
        mode_ids = (
            (self.active_mode_id,) if self.active_mode_id is not None else None
        )
        fixes = self.localize_all(
            observation,
            mask,
            yaw_prior_deg,
            search_center_xy,
            search_radius_px,
            mode_ids,
        )
        best_mode_id, best = max(
            fixes.items(),
            key=lambda item: (
                bool(item[1].valid), float(item[1].score), float(item[1].margin)
            ),
        )
        self.last_mode_likelihoods = {
            mode_id: max(0.0, float(fix.score)) for mode_id, fix in fixes.items()
        }
        self.last_selected_mode_id = best_mode_id
        diagnostics = dict(best.diagnostics or {})
        diagnostics["map_layer"] = {
            "selected_mode_id": best_mode_id,
            "mode_likelihoods": dict(self.last_mode_likelihoods),
        }
        return replace(best, diagnostics=diagnostics)
