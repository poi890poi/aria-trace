"""Polar-space cursor pose estimation from a reviewed mini-map calibration."""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from .cursor_shape_model import edge_distance_transform, polygon_edge, render_polygon
from .session import SessionReader


SCHEMA_VERSION = "1.0"


def timing_summary_ms(durations_ns, method: str) -> dict:
    values = np.asarray(durations_ns, dtype=np.float64) / 1.0e6
    if not len(values):
        raise ValueError("At least one timing sample is required")
    return {
        "method": method,
        "clock": "time.perf_counter_ns",
        "sample_count": int(len(values)),
        "total_ms": float(values.sum()),
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def wrap_signed_degrees(value):
    return (np.asarray(value) + 180.0) % 360.0 - 180.0


def circular_difference_degrees(left, right):
    return wrap_signed_degrees(np.asarray(left) - np.asarray(right))


def _confidence_level(value: float) -> str:
    if value >= 0.85:
        return "high"
    if value >= 0.65:
        return "moderate"
    return "low"


def _color_heatmap(values: np.ndarray) -> np.ndarray:
    normalized = cv2.normalize(values, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


class CursorPoseEstimator:
    """Estimate one screen-space cursor angle without temporal assumptions."""

    GAUSSIAN_FIT_METHODS = ("vectorized_grid", "fast_grid", "legacy_grid")

    def __init__(
        self,
        calibration_path: Path,
        gaussian_fit_method: str = "vectorized_grid",
    ) -> None:
        calibration_path = Path(calibration_path)
        if calibration_path.is_dir():
            calibration_path = calibration_path / "calibration.json"
        self.calibration_path = calibration_path.resolve()
        if gaussian_fit_method not in self.GAUSSIAN_FIT_METHODS:
            raise ValueError(
                "Unsupported Gaussian fit method {!r}; expected one of {}".format(
                    gaussian_fit_method, ", ".join(self.GAUSSIAN_FIT_METHODS)
                )
            )
        self.gaussian_fit_method = gaussian_fit_method
        self.root = self.calibration_path.parent
        self.calibration = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        model_path = self.root / self.calibration.get("model_file", "model.npz")
        model = np.load(model_path)
        self.template = model["cursor_binary"].astype(np.float32)
        if "cursor_polygon_relative_xy" not in model:
            raise ValueError("Calibration predates the rigid symmetric polygon model")
        self.polygon = model["cursor_polygon_relative_xy"].astype(np.float64)
        self.symmetry_axis_deg = float(model["cursor_symmetry_axis_deg"][0])
        self.pivot = model["rotation_center"].astype(np.float32)
        self.crop_xywh = tuple(map(int, self.calibration["config"]["crop_xywh"]))
        self.cursor_config = self.calibration["config"]["cursor"]
        self.tip_angle_deg = float(
            self.calibration["cursor_shape"]["farthest_contour_point_angle_screen_deg"]
        )
        centroid = self.calibration["cursor_shape"]["centroid_offset_from_pivot_px"]
        self.canonical_centroid_angle_deg = math.degrees(
            math.atan2(float(centroid["dy"]), float(centroid["dx"]))
        )
        self.patch_size = int(self.template.shape[0])
        self.patch_half = (self.patch_size - 1) / 2.0
        self.theta = np.linspace(0.0, 2.0 * np.pi, 360, endpoint=False)
        self.radii = np.linspace(0.5, 15.0, 36)
        self.x_map = (
            self.patch_half
            + np.cos(self.theta)[:, None] * self.radii[None, :]
        ).astype(np.float32)
        self.y_map = (
            self.patch_half
            + np.sin(self.theta)[:, None] * self.radii[None, :]
        ).astype(np.float32)
        self.template_polar = cv2.remap(
            self.template,
            self.x_map,
            self.y_map,
            cv2.INTER_LINEAR,
        )
        self.template_polar_fft = np.fft.fft(self.template_polar, axis=0)
        self.template_energy = (
            math.sqrt(float(np.sum(self.template_polar ** 2))) + 1.0e-6
        )
        self.polygon_masks = np.stack(
            [
                render_polygon(self.polygon, self.patch_size, angle, supersample=4)
                for angle in range(360)
            ]
        )
        self.polygon_edges = np.stack(
            [polygon_edge(mask) for mask in self.polygon_masks]
        )
        self.polygon_distance_transforms = np.stack(
            [edge_distance_transform(edge) for edge in self.polygon_edges]
        )

    def _crop(self, frame: np.ndarray) -> np.ndarray:
        x, y, width, height = self.crop_xywh
        if frame.shape[:2] == (height, width):
            return frame
        if y + height > frame.shape[0] or x + width > frame.shape[1]:
            raise ValueError("Frame is smaller than the calibrated mini-map crop")
        return frame[y : y + height, x : x + width]

    def _cursor_mask(self, crop: np.ndarray):
        lower = np.asarray(self.cursor_config["hsv_lower"], dtype=np.uint8)
        upper = np.asarray(self.cursor_config["hsv_upper"], dtype=np.uint8)
        binary = cv2.inRange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV), lower, upper)
        height, width = crop.shape[:2]
        yy, xx = np.ogrid[:height, :width]
        search_radius = float(self.cursor_config["search_radius_px"])
        search = (
            (xx - self.pivot[0]) ** 2 + (yy - self.pivot[1]) ** 2
            <= search_radius ** 2
        )
        binary[~search] = 0
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        min_area, max_area = map(int, self.cursor_config["component_area_px"])
        candidates = []
        for index in range(1, count):
            area = int(stats[index, cv2.CC_STAT_AREA])
            distance = float(np.linalg.norm(centroids[index] - self.pivot))
            if min_area <= area <= max_area and distance < search_radius * 0.5:
                candidates.append((distance, index))
        if not candidates:
            return None, None, None
        selected = min(candidates)[1]
        return (
            (labels == selected).astype(np.float32),
            centroids[selected].astype(np.float64),
            int(stats[selected, cv2.CC_STAT_AREA]),
        )

    @staticmethod
    def _gaussian_fit_candidates(correlation: np.ndarray) -> list:
        """Return separated response maxima in descending smoothed height."""
        kernel_x = np.arange(-6, 7, dtype=np.float64)
        kernel = np.exp(-0.5 * (kernel_x / 2.0) ** 2)
        kernel /= kernel.sum()
        padded = np.concatenate([correlation[-6:], correlation, correlation[:6]])
        smoothed = np.convolve(padded, kernel, mode="valid")
        candidates = np.argsort(smoothed)[-12:]
        candidate_centers = []
        for candidate in candidates[np.argsort(smoothed[candidates])[::-1]]:
            if all(
                abs(float(circular_difference_degrees(candidate, prior))) >= 12.0
                for prior in candidate_centers
            ):
                candidate_centers.append(float(candidate))
            if len(candidate_centers) == 4:
                break
        return candidate_centers

    @staticmethod
    def _fit_circular_gaussian_legacy(correlation: np.ndarray) -> dict:
        """Fit a Gaussian lobe on the circular angular-correlation response."""
        angles = np.arange(360, dtype=np.float64)
        candidate_centers = CursorPoseEstimator._gaussian_fit_candidates(correlation)

        best = None
        sigma_grid = np.linspace(2.0, 50.0, 17)
        for seed in candidate_centers:
            for center in np.arange(seed - 8.0, seed + 8.01, 0.5):
                distance = wrap_signed_degrees(angles - center)
                window = np.abs(distance) <= 70.0
                y = correlation[window].astype(np.float64)
                for sigma in sigma_grid:
                    gaussian = np.exp(-0.5 * (distance[window] / sigma) ** 2)
                    design = np.column_stack([np.ones_like(gaussian), gaussian])
                    baseline, amplitude = np.linalg.lstsq(design, y, rcond=None)[0]
                    if amplitude <= 0.0:
                        continue
                    residual = y - (baseline + amplitude * gaussian)
                    score = float(np.sqrt(np.mean(residual ** 2))) / max(
                        float(amplitude), 1.0e-6
                    )
                    if best is None or score < best[0]:
                        best = (score, center, sigma)
        if best is None:
            raise RuntimeError("Circular Gaussian correlation fit failed")

        _, coarse_center, coarse_sigma = best
        refined = None
        for center in np.arange(coarse_center - 0.6, coarse_center + 0.601, 0.05):
            distance = wrap_signed_degrees(angles - center)
            window = np.abs(distance) <= 70.0
            y = correlation[window].astype(np.float64)
            for sigma in np.arange(
                max(1.0, coarse_sigma - 3.0), coarse_sigma + 3.001, 0.1
            ):
                gaussian = np.exp(-0.5 * (distance[window] / sigma) ** 2)
                design = np.column_stack([np.ones_like(gaussian), gaussian])
                baseline, amplitude = np.linalg.lstsq(design, y, rcond=None)[0]
                if amplitude <= 0.0:
                    continue
                fitted = baseline + amplitude * gaussian
                residual = y - fitted
                rss = float(np.sum(residual ** 2))
                score = math.sqrt(rss / len(y)) / max(float(amplitude), 1.0e-6)
                if refined is None or score < refined[0]:
                    refined = (
                        score,
                        float(center % 360.0),
                        float(sigma),
                        float(baseline),
                        float(amplitude),
                        y,
                        fitted,
                        distance[window],
                    )
        if refined is None:
            raise RuntimeError("Circular Gaussian refinement failed")
        score, center, sigma, baseline, amplitude, y, fitted, distance = refined
        residual = y - fitted
        rss = float(np.sum(residual ** 2))
        tss = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - rss / max(tss, 1.0e-12)
        gaussian = np.exp(-0.5 * (distance / sigma) ** 2)
        jacobian = np.column_stack(
            [
                np.ones_like(gaussian),
                gaussian,
                amplitude * gaussian * distance / (sigma ** 2),
                amplitude * gaussian * (distance ** 2) / (sigma ** 3),
            ]
        )
        dof = max(1, len(y) - jacobian.shape[1])
        covariance = (rss / dof) * np.linalg.pinv(jacobian.T @ jacobian)
        center_std = float(math.sqrt(max(float(covariance[2, 2]), 0.0)))
        exclusion = max(20, int(math.ceil(3.0 * sigma)))
        excluded = correlation.copy()
        fitted_index = int(round(center)) % 360
        for offset in range(-exclusion, exclusion + 1):
            excluded[(fitted_index + offset) % 360] = -np.inf
        fitted_peak = float(baseline + amplitude)
        return {
            "center_deg": float(wrap_signed_degrees(center)),
            "sigma_deg": sigma,
            "amplitude": amplitude,
            "baseline": baseline,
            "rmse": float(math.sqrt(rss / len(y))),
            "normalized_rmse": float(score),
            "r_squared": float(r_squared),
            "center_std_deg": center_std,
            "fitted_peak": fitted_peak,
            "second_peak": float(np.max(excluded)),
            "peak_margin": fitted_peak - float(np.max(excluded)),
            "raw_peak_deg": float(wrap_signed_degrees(int(np.argmax(correlation)))),
        }

    @staticmethod
    def _score_gaussian_grid(
        correlation: np.ndarray,
        centers: np.ndarray,
        sigmas: np.ndarray,
        batch_size: int = 512,
    ):
        """Score a Cartesian center/sigma grid with closed-form linear fits."""
        angles = np.arange(360, dtype=np.float64)
        correlation = correlation.astype(np.float64, copy=False)
        pair_centers = np.repeat(np.asarray(centers, dtype=np.float64), len(sigmas))
        pair_sigmas = np.tile(np.asarray(sigmas, dtype=np.float64), len(centers))
        best = None
        for start in range(0, len(pair_centers), batch_size):
            stop = min(start + batch_size, len(pair_centers))
            batch_centers = pair_centers[start:stop]
            batch_sigmas = pair_sigmas[start:stop]
            distance = wrap_signed_degrees(
                angles[None, :] - batch_centers[:, None]
            )
            window = np.abs(distance) <= 70.0
            weights = window.astype(np.float64)
            gaussian = np.exp(
                -0.5 * (distance / batch_sigmas[:, None]) ** 2
            ) * weights
            sample_count = weights.sum(axis=1)
            sum_y = weights @ correlation
            sum_y2 = weights @ (correlation ** 2)
            sum_g = gaussian.sum(axis=1)
            sum_g2 = np.square(gaussian).sum(axis=1)
            sum_gy = gaussian @ correlation
            denominator = sample_count * sum_g2 - np.square(sum_g)
            valid_denominator = np.abs(denominator) > 1.0e-12
            baseline = np.full(len(batch_centers), np.nan, dtype=np.float64)
            amplitude = np.full(len(batch_centers), np.nan, dtype=np.float64)
            baseline[valid_denominator] = (
                sum_y[valid_denominator] * sum_g2[valid_denominator]
                - sum_g[valid_denominator] * sum_gy[valid_denominator]
            ) / denominator[valid_denominator]
            amplitude[valid_denominator] = (
                sample_count[valid_denominator] * sum_gy[valid_denominator]
                - sum_g[valid_denominator] * sum_y[valid_denominator]
            ) / denominator[valid_denominator]
            rss = (
                sum_y2
                - 2.0 * baseline * sum_y
                - 2.0 * amplitude * sum_gy
                + np.square(baseline) * sample_count
                + 2.0 * baseline * amplitude * sum_g
                + np.square(amplitude) * sum_g2
            )
            rss = np.maximum(rss, 0.0)
            scores = np.sqrt(rss / sample_count) / np.maximum(amplitude, 1.0e-6)
            scores[(amplitude <= 0.0) | ~np.isfinite(scores)] = np.inf
            local_index = int(np.argmin(scores))
            candidate = (
                float(scores[local_index]),
                float(batch_centers[local_index]),
                float(batch_sigmas[local_index]),
                float(baseline[local_index]),
                float(amplitude[local_index]),
            )
            if best is None or candidate[0] < best[0]:
                best = candidate
        return best

    @staticmethod
    def _finalize_circular_gaussian_fit(
        correlation: np.ndarray,
        score: float,
        center: float,
        sigma: float,
        baseline: float,
        amplitude: float,
    ) -> dict:
        angles = np.arange(360, dtype=np.float64)
        distance_all = wrap_signed_degrees(angles - center)
        window = np.abs(distance_all) <= 70.0
        distance = distance_all[window]
        y = correlation[window].astype(np.float64)
        gaussian = np.exp(-0.5 * (distance / sigma) ** 2)
        fitted = baseline + amplitude * gaussian
        residual = y - fitted
        rss = float(np.sum(residual ** 2))
        tss = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = 1.0 - rss / max(tss, 1.0e-12)
        jacobian = np.column_stack(
            [
                np.ones_like(gaussian),
                gaussian,
                amplitude * gaussian * distance / (sigma ** 2),
                amplitude * gaussian * (distance ** 2) / (sigma ** 3),
            ]
        )
        dof = max(1, len(y) - jacobian.shape[1])
        covariance = (rss / dof) * np.linalg.pinv(jacobian.T @ jacobian)
        center_std = float(math.sqrt(max(float(covariance[2, 2]), 0.0)))
        exclusion = max(20, int(math.ceil(3.0 * sigma)))
        excluded = correlation.copy()
        fitted_index = int(round(center)) % 360
        for offset in range(-exclusion, exclusion + 1):
            excluded[(fitted_index + offset) % 360] = -np.inf
        fitted_peak = float(baseline + amplitude)
        return {
            "center_deg": float(wrap_signed_degrees(center % 360.0)),
            "sigma_deg": float(sigma),
            "amplitude": float(amplitude),
            "baseline": float(baseline),
            "rmse": float(math.sqrt(rss / len(y))),
            "normalized_rmse": float(score),
            "r_squared": float(r_squared),
            "center_std_deg": center_std,
            "fitted_peak": fitted_peak,
            "second_peak": float(np.max(excluded)),
            "peak_margin": fitted_peak - float(np.max(excluded)),
            "raw_peak_deg": float(wrap_signed_degrees(int(np.argmax(correlation)))),
        }

    @classmethod
    def _fit_circular_gaussian_vectorized(
        cls,
        correlation: np.ndarray,
    ) -> dict:
        candidate_centers = cls._gaussian_fit_candidates(correlation)
        if not candidate_centers:
            raise RuntimeError("Circular Gaussian correlation fit failed")
        center_offsets = np.arange(-8.0, 8.01, 0.5)
        sigma_grid = np.linspace(2.0, 50.0, 17)
        refine_center_offsets = np.arange(-0.6, 0.601, 0.05)
        refine_sigma_offsets = np.arange(-3.0, 3.001, 0.1)
        coarse_centers = (
            np.asarray(candidate_centers)[:, None] + center_offsets[None, :]
        ).reshape(-1)
        coarse = cls._score_gaussian_grid(
            correlation, coarse_centers, sigma_grid
        )
        if coarse is None or not math.isfinite(coarse[0]):
            raise RuntimeError("Circular Gaussian correlation fit failed")
        _, coarse_center, coarse_sigma, _, _ = coarse
        refined_centers = coarse_center + refine_center_offsets
        refined_sigmas = np.maximum(1.0, coarse_sigma + refine_sigma_offsets)
        refined = cls._score_gaussian_grid(
            correlation, refined_centers, refined_sigmas
        )
        if refined is None or not math.isfinite(refined[0]):
            raise RuntimeError("Circular Gaussian refinement failed")
        return cls._finalize_circular_gaussian_fit(correlation, *refined)

    @classmethod
    def _fit_circular_gaussian_fast(cls, correlation: np.ndarray) -> dict:
        """Fit the same model on a reduced grid for latency-sensitive tracking."""
        candidate_centers = cls._gaussian_fit_candidates(correlation)
        if not candidate_centers:
            raise RuntimeError("Circular Gaussian correlation fit failed")
        coarse_centers = (
            np.asarray(candidate_centers)[:, None]
            + np.arange(-8.0, 8.01, 1.0)[None, :]
        ).reshape(-1)
        coarse = cls._score_gaussian_grid(
            correlation,
            coarse_centers,
            np.linspace(2.0, 50.0, 17),
        )
        if coarse is None or not math.isfinite(coarse[0]):
            raise RuntimeError("Circular Gaussian correlation fit failed")
        _, coarse_center, coarse_sigma, _, _ = coarse
        refined = cls._score_gaussian_grid(
            correlation,
            coarse_center + np.arange(-0.6, 0.601, 0.075),
            np.maximum(1.0, coarse_sigma + np.arange(-3.0, 3.001, 0.25)),
        )
        if refined is None or not math.isfinite(refined[0]):
            raise RuntimeError("Circular Gaussian refinement failed")
        return cls._finalize_circular_gaussian_fit(correlation, *refined)

    def _fit_circular_gaussian(self, correlation: np.ndarray) -> dict:
        if self.gaussian_fit_method == "legacy_grid":
            return self._fit_circular_gaussian_legacy(correlation)
        if self.gaussian_fit_method == "fast_grid":
            return self._fit_circular_gaussian_fast(correlation)
        return self._fit_circular_gaussian_vectorized(correlation)

    def _pixel_correlate(self, patch: np.ndarray):
        polar = cv2.remap(
            patch,
            self.x_map,
            self.y_map,
            cv2.INTER_LINEAR,
        )
        correlation = np.fft.ifft(
            np.fft.fft(polar, axis=0) * np.conj(self.template_polar_fft),
            axis=0,
        ).real.sum(axis=1).astype(np.float32)
        energy = math.sqrt(float(np.sum(polar ** 2))) * self.template_energy
        correlation /= energy + 1.0e-6
        gaussian_fit = self._fit_circular_gaussian(correlation)
        return gaussian_fit, correlation, polar

    @staticmethod
    def _symmetric_chamfer_curve(
        observed_edge: np.ndarray,
        observed_distance: np.ndarray,
        polygon_edges: np.ndarray,
        polygon_distance_transforms: np.ndarray,
    ) -> np.ndarray:
        edge_counts = np.maximum(
            polygon_edges.sum(axis=(1, 2), dtype=np.float64), 1.0
        )
        model_to_observed = (
            polygon_edges * observed_distance[None, :, :]
        ).sum(axis=(1, 2), dtype=np.float64) / edge_counts
        observed_to_model = polygon_distance_transforms[:, observed_edge].mean(
            axis=1,
            dtype=np.float64,
        )
        return 0.5 * (model_to_observed + observed_to_model)

    def _polygon_likelihood(self, patch: np.ndarray):
        observed = patch >= 0.5
        observed_edge = polygon_edge(observed)
        if not np.any(observed_edge):
            raise RuntimeError("Observed cursor polygon has no edge")
        observed_distance = edge_distance_transform(observed_edge)
        chamfer = self._symmetric_chamfer_curve(
            observed_edge,
            observed_distance,
            self.polygon_edges,
            self.polygon_distance_transforms,
        )
        scale = max(float(np.percentile(chamfer, 20)), 0.75)
        likelihood = np.exp(-0.5 * (chamfer / scale) ** 2).astype(np.float32)
        gaussian_fit = self._fit_circular_gaussian(likelihood)
        return gaussian_fit, likelihood, chamfer, observed_edge

    def estimate(
        self,
        frame: np.ndarray,
        frame_index: Optional[int] = None,
        session_time_ns: Optional[int] = None,
    ) -> dict:
        crop = self._crop(frame)
        mask, centroid, area = self._cursor_mask(crop)
        common = {
            "schema_version": SCHEMA_VERSION,
            "frame_index": frame_index,
            "session_time_ns": session_time_ns,
            "detected": mask is not None,
            "angle_convention": "screen degrees: 0=right, +clockwise",
            "world_heading_status": "unresolved",
            "polar_origin": "fitted_cursor_rotation_center",
            "polar_origin_x": float(self.pivot[0]),
            "polar_origin_y": float(self.pivot[1]),
        }
        if mask is None:
            common.update(
                {
                    "confidence": 0.0,
                    "confidence_level": "low",
                    "failure": "cursor_component_not_detected",
                }
            )
            return common
        patch = cv2.getRectSubPix(
            mask,
            (self.patch_size, self.patch_size),
            tuple(self.pivot),
        )
        pixel_fit, pixel_correlation, polar = self._pixel_correlate(patch)
        gaussian_fit, correlation, chamfer_curve, observed_edge = (
            self._polygon_likelihood(patch)
        )
        shift = gaussian_fit["center_deg"]
        peak = gaussian_fit["fitted_peak"]
        margin = gaussian_fit["peak_margin"]
        predicted_probability = render_polygon(
            self.polygon, self.patch_size, shift, supersample=4
        )
        predicted = predicted_probability >= 0.5
        shift_radians = math.radians(shift)
        shift_rotation = np.array(
            [
                [math.cos(shift_radians), -math.sin(shift_radians)],
                [math.sin(shift_radians), math.cos(shift_radians)],
            ]
        )
        polygon_points_crop = self.polygon @ shift_rotation.T + self.pivot
        observed = patch >= 0.5
        union = np.logical_or(predicted, observed).sum()
        aligned_iou = (
            float(np.logical_and(predicted, observed).sum() / union)
            if union
            else 0.0
        )
        centroid_vector = centroid - self.pivot
        centroid_angle = math.degrees(
            math.atan2(centroid_vector[1], centroid_vector[0])
        )
        centroid_shift = float(
            wrap_signed_degrees(
                centroid_angle - self.canonical_centroid_angle_deg
            )
        )
        agreement_error = float(
            circular_difference_degrees(shift, centroid_shift)
        )
        pixel_agreement_error = float(
            circular_difference_degrees(shift, pixel_fit["center_deg"])
        )
        fitted_chamfer = float(
            np.interp(shift % 360.0, np.arange(360), chamfer_curve, period=360)
        )
        component_scores = {
            "polygon_likelihood": float(np.clip((peak - 0.55) / 0.45, 0.0, 1.0)),
            "peak_margin": float(np.clip(margin / 0.16, 0.0, 1.0)),
            "gaussian_fit": float(
                np.clip((gaussian_fit["r_squared"] - 0.70) / 0.28, 0.0, 1.0)
            ),
            "angular_precision": float(
                np.clip(1.0 - gaussian_fit["center_std_deg"] / 3.0, 0.0, 1.0)
            ),
            "polygon_iou": float(np.clip(aligned_iou / 0.82, 0.0, 1.0)),
            "edge_distance": float(math.exp(-((fitted_chamfer / 1.5) ** 2))),
            "pixel_polar_agreement": float(
                math.exp(-((pixel_agreement_error / 8.0) ** 2))
            ),
            "centroid_agreement": float(
                math.exp(-((agreement_error / 20.0) ** 2))
            ),
        }
        confidence = float(
            np.prod([max(value, 0.02) for value in component_scores.values()])
            ** (1.0 / len(component_scores))
        )
        common.update(
            {
                "relative_rotation_deg": float(shift),
                "angle_screen_deg": float((self.symmetry_axis_deg + shift) % 360.0),
                "pose_model": "symmetry_constrained_rigid_polygon",
                "symmetry_axis_template_deg": float(self.symmetry_axis_deg),
                "centroid_rotation_deg": centroid_shift,
                "centroid_agreement_error_deg": agreement_error,
                "cursor_centroid_x": float(centroid[0]),
                "cursor_centroid_y": float(centroid[1]),
                "component_area_px": area,
                "angular_likelihood_peak": float(peak),
                "angular_likelihood_margin": float(margin),
                "polygon_symmetric_chamfer_px": fitted_chamfer,
                "pixel_polar_rotation_deg": float(pixel_fit["center_deg"]),
                "polygon_pixel_agreement_error_deg": pixel_agreement_error,
                "polar_correlation_peak": float(pixel_fit["fitted_peak"]),
                "polar_peak_margin": float(pixel_fit["peak_margin"]),
                "gaussian_center_deg": float(gaussian_fit["center_deg"]),
                "gaussian_sigma_deg": float(gaussian_fit["sigma_deg"]),
                "gaussian_center_std_deg": float(gaussian_fit["center_std_deg"]),
                "gaussian_amplitude": float(gaussian_fit["amplitude"]),
                "gaussian_baseline": float(gaussian_fit["baseline"]),
                "gaussian_fit_rmse": float(gaussian_fit["rmse"]),
                "gaussian_fit_normalized_rmse": float(
                    gaussian_fit["normalized_rmse"]
                ),
                "gaussian_fit_r_squared": float(gaussian_fit["r_squared"]),
                "raw_angular_likelihood_peak_deg": float(gaussian_fit["raw_peak_deg"]),
                "raw_correlation_peak_deg": float(pixel_fit["raw_peak_deg"]),
                "template_aligned_iou": aligned_iou,
                "confidence": confidence,
                "confidence_level": _confidence_level(confidence),
                "confidence_components": component_scores,
                "_gaussian_fit": gaussian_fit,
                "_correlation": correlation,
                "_pixel_correlation": pixel_correlation,
                "_chamfer_curve": chamfer_curve,
                "_observed_edge": observed_edge,
                "_predicted_probability": predicted_probability,
                "_polygon_points_crop": polygon_points_crop,
                "_polar": polar,
                "_mask": mask,
                "_crop": crop,
            }
        )
        return common

    @staticmethod
    def public_result(result: dict) -> dict:
        return {
            key: value
            for key, value in result.items()
            if not key.startswith("_")
        }


def _plot_angle_timeline(poses, output: Path) -> None:
    canvas = np.full((430, 900, 3), 18, np.uint8)
    cv2.putText(
        canvas,
        "Gaussian-fitted cursor screen angle (0=right, +clockwise)",
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    detected = [pose for pose in poses if pose.get("detected")]
    if not detected:
        cv2.imwrite(str(output), canvas)
        return
    times = np.array([pose["session_time_ns"] for pose in detected], np.float64) / 1e9
    times -= times.min()
    angles = np.array([pose["angle_screen_deg"] for pose in detected])
    for degree in (0, 90, 180, 270, 360):
        y = 380 - int(degree / 360 * 320)
        cv2.line(canvas, (55, y), (875, y), (60, 60, 60), 1)
        cv2.putText(
            canvas,
            str(degree),
            (10, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (180, 180, 180),
            1,
        )
    duration = max(float(times.max()), 1.0e-6)
    for index in range(len(detected) - 1):
        if abs(float(circular_difference_degrees(angles[index + 1], angles[index]))) > 180:
            continue
        x1 = 55 + int(times[index] / duration * 820)
        x2 = 55 + int(times[index + 1] / duration * 820)
        y1 = 380 - int(angles[index] / 360 * 320)
        y2 = 380 - int(angles[index + 1] / 360 * 320)
        if abs(y2 - y1) < 150:
            cv2.line(canvas, (x1, y1), (x2, y2), (0, 220, 255), 1)
    cv2.imwrite(str(output), canvas)


def _plot_confidence_timeline(poses, output: Path) -> None:
    canvas = np.full((350, 900, 3), 18, np.uint8)
    cv2.putText(
        canvas,
        "Cursor pose confidence and centroid cross-check",
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    detected = [pose for pose in poses if pose.get("detected")]
    if detected:
        confidence = np.array([pose["confidence"] for pose in detected])
        errors = np.abs(
            np.array([pose["centroid_agreement_error_deg"] for pose in detected])
        )
        for index in range(len(detected) - 1):
            x1 = 55 + int(index / max(1, len(detected) - 1) * 820)
            x2 = 55 + int((index + 1) / max(1, len(detected) - 1) * 820)
            yc1 = 300 - int(confidence[index] * 240)
            yc2 = 300 - int(confidence[index + 1] * 240)
            ye1 = 300 - int(min(errors[index], 45) / 45 * 240)
            ye2 = 300 - int(min(errors[index + 1], 45) / 45 * 240)
            cv2.line(canvas, (x1, yc1), (x2, yc2), (0, 220, 80), 1)
            cv2.line(canvas, (x1, ye1), (x2, ye2), (80, 100, 255), 1)
    cv2.putText(canvas, "green=confidence / red=|polar-centroid error| (0..45 deg)", (55,330), cv2.FONT_HERSHEY_SIMPLEX, .45, (210,210,210), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), canvas)


def _plot_gaussian_fits(poses, output: Path) -> None:
    detected = [pose for pose in poses if pose.get("detected")]
    if not detected:
        return
    sample_indices = np.linspace(0, len(detected) - 1, min(6, len(detected))).astype(int)
    row_height, width = 145, 900
    canvas = np.full((row_height * len(sample_indices), width, 3), 18, np.uint8)
    x0, x1 = 62, width - 18
    for row_index, pose_index in enumerate(sample_indices):
        pose = detected[pose_index]
        fit = pose["_gaussian_fit"]
        response = pose["_correlation"].astype(np.float64)
        relative = np.arange(-70, 71, dtype=np.float64)
        sample_angles = (fit["center_deg"] + relative) % 360.0
        raw = np.interp(sample_angles, np.arange(360), response, period=360)
        fitted = fit["baseline"] + fit["amplitude"] * np.exp(
            -0.5 * (relative / fit["sigma_deg"]) ** 2
        )
        low = float(min(raw.min(), fitted.min()))
        high = float(max(raw.max(), fitted.max()))
        scale = max(high - low, 1.0e-6)
        top = row_index * row_height
        baseline_y = top + row_height - 18
        cv2.line(canvas, (x0, baseline_y), (x1, baseline_y), (65, 65, 65), 1)
        raw_points, fit_points = [], []
        for index in range(len(relative)):
            x = x0 + int(index / (len(relative) - 1) * (x1 - x0))
            raw_y = baseline_y - int((raw[index] - low) / scale * 90)
            fit_y = baseline_y - int((fitted[index] - low) / scale * 90)
            raw_points.append((x, raw_y))
            fit_points.append((x, fit_y))
        cv2.polylines(canvas, [np.asarray(raw_points)], False, (255, 220, 0), 1, cv2.LINE_AA)
        cv2.polylines(canvas, [np.asarray(fit_points)], False, (0, 220, 255), 2, cv2.LINE_AA)
        center_x = x0 + (x1 - x0) // 2
        cv2.line(canvas, (center_x, top + 30), (center_x, baseline_y), (255, 255, 255), 1)
        label = "frame {}  mean={:.2f} deg  sigma={:.2f} deg  std={:.3f} deg  R2={:.3f}".format(
            pose.get("frame_index"), fit["center_deg"], fit["sigma_deg"],
            fit["center_std_deg"], fit["r_squared"]
        )
        cv2.putText(canvas, label, (14, top + 20), cv2.FONT_HERSHEY_SIMPLEX, .44, (235,235,235), 1, cv2.LINE_AA)
    cv2.putText(canvas, "blue=polygon likelihood  yellow=Gaussian fit", (620, 20), cv2.FONT_HERSHEY_SIMPLEX, .38, (200,200,200), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), canvas)


def _plot_polar_samples(estimator, poses, output: Path) -> None:
    detected = [pose for pose in poses if pose.get("detected")]
    if not detected:
        return
    sample_indices = np.linspace(0, len(detected) - 1, min(5, len(detected))).astype(int)
    entries = [("canonical template", estimator.template_polar)]
    entries.extend(
        ("frame {}  fitted shift {:.2f} deg".format(
            detected[index].get("frame_index"), detected[index]["relative_rotation_deg"]
        ), detected[index]["_polar"])
        for index in sample_indices
    )
    row_height, width = 108, 760
    canvas = np.full((row_height * len(entries), width, 3), 18, np.uint8)
    for row_index, (label, polar) in enumerate(entries):
        top = row_index * row_height
        heatmap = _color_heatmap(polar.T)
        heatmap = cv2.resize(heatmap, (720, 72), interpolation=cv2.INTER_NEAREST)
        canvas[top + 28 : top + 100, 20:740] = heatmap
        cv2.putText(canvas, label, (20, top + 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (235,235,235), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), canvas)


def _pose_overlays(poses, output: Path) -> None:
    detected = [pose for pose in poses if pose.get("detected")]
    if not detected:
        return
    indices = np.linspace(0, len(detected) - 1, min(6, len(detected))).astype(int)
    rows = []
    for pose_index in indices:
        pose = detected[pose_index]
        crop = pose["_crop"].copy()
        pivot = np.array([pose["polar_origin_x"], pose["polar_origin_y"]])
        angle = math.radians(pose["angle_screen_deg"])
        direction = np.array([math.cos(angle), math.sin(angle)])
        # Keep all annotation outside the cursor so the source pixels remain
        # directly reviewable. The fitted polygon has its own evidence image.
        arrow_start = pivot + direction * 16.0
        arrow_end = pivot + direction * 38.0
        cv2.arrowedLine(
            crop,
            tuple(np.round(arrow_start).astype(int)),
            tuple(np.round(arrow_end).astype(int)),
            (0,255,255), 2, cv2.LINE_AA, tipLength=0.25
        )
        cv2.putText(
            crop, "{:.1f} deg  conf {:.2f}".format(pose["angle_screen_deg"], pose["confidence"]),
            (4, crop.shape[0] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
            (255,255,255), 1, cv2.LINE_AA
        )
        rows.append(crop)
    cv2.imwrite(str(output), np.vstack(rows))

def estimate_cursor_pose_frames(
    calibration_path: Path,
    frames: np.ndarray,
    output_path: Path,
    frame_indices: Optional[Sequence[int]] = None,
    session_times_ns: Optional[Sequence[int]] = None,
    provenance: Optional[dict] = None,
    gaussian_fit_method: str = "vectorized_grid",
) -> dict:
    """Estimate and persist raw pose measurements for a labeled frame sequence."""
    estimator = CursorPoseEstimator(
        calibration_path,
        gaussian_fit_method=gaussian_fit_method,
    )
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    if frame_indices is None:
        frame_indices = list(range(len(frames)))
    if session_times_ns is None:
        session_times_ns = [int(index * 1e9 / 30.0) for index in range(len(frames))]
    poses = []
    pose_durations_ns = []
    for frame, frame_index, session_time_ns in zip(
        frames, frame_indices, session_times_ns
    ):
        started_ns = time.perf_counter_ns()
        pose = estimator.estimate(
            frame,
            int(frame_index),
            int(session_time_ns),
        )
        pose_durations_ns.append(time.perf_counter_ns() - started_ns)
        poses.append(pose)
    public = [estimator.public_result(pose) for pose in poses]
    with (output_path / "cursor_poses.jsonl").open("w", encoding="utf-8") as stream:
        for pose in public:
            stream.write(json.dumps(pose, separators=(",", ":")) + "\n")
    detected = [pose for pose in poses if pose["detected"]]
    if not detected:
        raise RuntimeError("Cursor pose was not detected in any frame")
    correlations = np.stack([pose["_correlation"] for pose in detected])
    correlation_image = _color_heatmap(correlations)
    correlation_image = cv2.resize(
        correlation_image,
        (720, max(360, 2 * len(correlations))),
        interpolation=cv2.INTER_NEAREST,
    )
    for row, pose in enumerate(detected):
        shift = pose["relative_rotation_deg"] % 360.0
        x = int(round(shift / 360.0 * (correlation_image.shape[1] - 1)))
        y = int(round(row / max(1, len(detected) - 1) * (correlation_image.shape[0] - 1)))
        cv2.circle(correlation_image, (x, y), 1, (255, 255, 255), -1)
    cv2.imwrite(str(output_path / "cursor_pose_correlation.png"), correlation_image)
    _plot_angle_timeline(poses, output_path / "cursor_pose_angles.png")
    _plot_gaussian_fits(poses, output_path / "cursor_pose_gaussian_fits.png")
    _plot_polar_samples(estimator, poses, output_path / "cursor_pose_polar_samples.png")
    _plot_confidence_timeline(poses, output_path / "cursor_pose_confidence.png")
    _pose_overlays(poses, output_path / "cursor_pose_overlays.png")
    confidence = np.array([pose["confidence"] for pose in detected])
    agreement = np.array(
        [pose["centroid_agreement_error_deg"] for pose in detected]
    )
    iou = np.array([pose["template_aligned_iou"] for pose in detected])
    peak = np.array([pose["angular_likelihood_peak"] for pose in detected])
    margin = np.array([pose["angular_likelihood_margin"] for pose in detected])
    gaussian_sigma = np.array([pose["gaussian_sigma_deg"] for pose in detected])
    gaussian_std = np.array([pose["gaussian_center_std_deg"] for pose in detected])
    gaussian_r2 = np.array([pose["gaussian_fit_r_squared"] for pose in detected])
    polygon_chamfer = np.array(
        [pose["polygon_symmetric_chamfer_px"] for pose in detected]
    )
    polygon_pixel_error = np.array(
        [pose["polygon_pixel_agreement_error_deg"] for pose in detected]
    )
    raw_fit_difference = np.abs(
        circular_difference_degrees(
            np.array([pose["raw_angular_likelihood_peak_deg"] for pose in detected]),
            np.array([pose["gaussian_center_deg"] for pose in detected]),
        )
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "review_required",
        "provenance": provenance or {},
        "angle_convention": "screen degrees: 0=right, +clockwise",
        "world_heading_status": "unresolved",
        "polar_origin": "fitted_cursor_rotation_center",
        "polar_origin_xy": [float(estimator.pivot[0]), float(estimator.pivot[1])],
        "total_frames": int(len(frames)),
        "detected_frames": int(len(detected)),
        "detection_rate": float(len(detected) / len(frames)),
        "median_confidence": float(np.median(confidence)),
        "p10_confidence": float(np.percentile(confidence, 10)),
        "confidence_level": _confidence_level(float(np.median(confidence))),
        "pose_model": "symmetry_constrained_rigid_polygon",
        "gaussian_fit_method": estimator.gaussian_fit_method,
        "pose_estimation_benchmark": timing_summary_ms(
            pose_durations_ns,
            "observed per-frame estimator wall time; model load and evidence excluded",
        ),
        "median_angular_likelihood_peak": float(np.median(peak)),
        "median_polygon_symmetric_chamfer_px": float(np.median(polygon_chamfer)),
        "p90_polygon_symmetric_chamfer_px": float(np.percentile(polygon_chamfer, 90)),
        "median_polygon_pixel_agreement_abs_deg": float(
            np.median(np.abs(polygon_pixel_error))
        ),
        "median_polar_correlation_peak": float(
            np.median([pose["polar_correlation_peak"] for pose in detected])
        ),
        "median_angular_likelihood_margin": float(np.median(margin)),
        "median_polar_peak_margin": float(np.median([pose["polar_peak_margin"] for pose in detected])),
        "median_gaussian_sigma_deg": float(np.median(gaussian_sigma)),
        "median_gaussian_center_std_deg": float(np.median(gaussian_std)),
        "p90_gaussian_center_std_deg": float(np.percentile(gaussian_std, 90)),
        "median_gaussian_fit_r_squared": float(np.median(gaussian_r2)),
        "p10_gaussian_fit_r_squared": float(np.percentile(gaussian_r2, 10)),
        "median_raw_peak_to_gaussian_center_deg": float(np.median(raw_fit_difference)),
        "median_template_aligned_iou": float(np.median(iou)),
        "p10_template_aligned_iou": float(np.percentile(iou, 10)),
        "polar_centroid_agreement_median_abs_deg": float(
            np.median(np.abs(agreement))
        ),
        "polar_centroid_agreement_rmse_deg": float(
            np.sqrt(np.mean(agreement ** 2))
        ),
        "angular_coverage_10deg_bins": float(
            len(
                np.unique(
                    np.floor(
                        np.array([pose["angle_screen_deg"] for pose in detected])
                        / 10.0
                    ).astype(int)
                )
            )
            / 36.0
        ),
        "measurement_file": "cursor_poses.jsonl",
        "evidence": [
            {
                "name": "cursor_pose_correlation.png",
                "title": "Movement symmetric-polygon angular likelihood",
                "category": "cursor_pose",
            },
            {
                "name": "cursor_pose_angles.png",
                "title": "Gaussian-fitted cursor screen-angle timeline",
                "category": "cursor_pose",
            },
            {
                "name": "cursor_pose_gaussian_fits.png",
                "title": "Correlation lobes and circular Gaussian fits",
                "category": "cursor_pose",
            },
            {
                "name": "cursor_pose_polar_samples.png",
                "title": "Cursor-centered polar template and movement samples",
                "category": "cursor_pose",
            },
            {
                "name": "cursor_pose_confidence.png",
                "title": "Pose confidence and centroid cross-check",
                "category": "cursor_pose",
            },
            {
                "name": "cursor_pose_overlays.png",
                "title": "Raw cursor samples with exterior direction arrows",
                "category": "cursor_pose",
            },
        ],
    }
    (output_path / "cursor_pose_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def _read_session_frames(
    session_path: Path, interval: Sequence[float], crop_xywh: Sequence[int]
):
    reader = SessionReader(session_path)
    video_path = reader.video_path("main")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("Cannot open cursor-pose video")
    capture.set(cv2.CAP_PROP_POS_MSEC, float(interval[0]) * 1000.0)
    x, y, width, height = map(int, crop_xywh)
    frames, indices, times = [], [], []
    frame_records = reader.frames_by_stream.get("main", [])
    records_by_index = {
        int(record["frame_index"]): record for record in frame_records
    }
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded_index = max(
                0, int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
            )
            record = records_by_index.get(decoded_index)
            if record is not None:
                time_s = float(record["session_time_ns"]) / 1e9
            else:
                time_s = decoded_index / max(fps, 1.0e-6)
            if time_s > float(interval[1]) + 1e-6:
                break
            if time_s + 1.0e-6 < float(interval[0]):
                continue
            frames.append(frame[y : y + height, x : x + width].copy())
            indices.append(decoded_index)
            times.append(
                int(record["session_time_ns"])
                if record is not None
                else int(time_s * 1e9)
            )
    finally:
        capture.release()
    if not frames:
        raise ValueError("Cursor-pose interval contains no frames")
    return np.stack(frames), indices, times, reader, video_path

def estimate_cursor_pose_session(
    calibration_path: Path,
    session_path: Path,
    output_path: Path,
    interval: Sequence[float],
    gaussian_fit_method: str = "vectorized_grid",
) -> dict:
    calibration_file = (
        Path(calibration_path) / "calibration.json"
        if Path(calibration_path).is_dir()
        else Path(calibration_path)
    )
    calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
    frames, indices, times, reader, video_path = _read_session_frames(
        Path(session_path),
        interval,
        calibration["config"]["crop_xywh"],
    )
    return estimate_cursor_pose_frames(
        calibration_file,
        frames,
        output_path,
        frame_indices=indices,
        session_times_ns=times,
        provenance={
            "session_path": str(Path(session_path).resolve()),
            "session_id": reader.manifest.get("session_id"),
            "video_path": str(video_path),
            "interval_s": list(map(float, interval)),
            "calibration_path": str(calibration_file.resolve()),
        },
        gaussian_fit_method=gaussian_fit_method,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("session", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--interval", nargs=2, type=float, required=True)
    parser.add_argument(
        "--gaussian-fit-method",
        choices=CursorPoseEstimator.GAUSSIAN_FIT_METHODS,
        default="vectorized_grid",
    )
    args = parser.parse_args()
    summary = estimate_cursor_pose_session(
        args.calibration,
        args.session,
        args.output,
        args.interval,
        gaussian_fit_method=args.gaussian_fit_method,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
