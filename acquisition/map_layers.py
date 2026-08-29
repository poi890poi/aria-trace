"""Canonical multi-scale map atlas construction and localization."""

import json
import math
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Optional

import cv2
import numpy as np

from .live_tracker import GlobalFix, GlobalMapLocalizer


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
    localization = manifest.get("localization") or {}
    if localization.get("status") != "ready":
        raise ValueError(
            "Map stitch {} has no ready localization raster".format(stitch_root.name)
        )
    return manifest


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
        if mode_id == canonical_mode_id:
            original_to_canonical = np.eye(3, dtype=np.float64)
            alignment_quality = {
                "method": "canonical_identity",
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

        localization = stitch["localization"]
        localization_to_original = np.asarray(
            localization["localization_to_original_map_3x3"], dtype=np.float64
        )
        localization_to_canonical = original_to_canonical.dot(
            localization_to_original
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
        self.localizers = {}
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
