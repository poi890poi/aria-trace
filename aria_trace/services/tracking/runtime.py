"""Two-rate live localization using global map fixes and local visual motion."""

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from aria_trace.services.tracking import FusionConfig, Pose2D, PoseFusionGate
from rig_runtime.services.vision import KltAngularYawEstimator, YawEstimate, camera_matrix

from rig_runtime.services.calibration.cursor.pose import CursorPoseEstimator
from rig_runtime.services.calibration.cursor.worker import CursorPoseProcessExecutor
from rig_runtime.services.calibration.minimap.transition import TransitionController
from rig_runtime.services.calibration.minimap.verification import estimate_masked_shift
from rig_runtime.services.calibration.minimap.spatial import (
    minimap_crop_space,
    normalize_minimap_geometry,
)


def _gradient(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def _angle_difference_deg(first: float, second: float) -> float:
    return ((float(first) - float(second) + 180.0) % 360.0) - 180.0


def _circular_mean_deg(values) -> float:
    radians = np.radians(np.asarray(values, dtype=np.float64))
    return math.degrees(math.atan2(float(np.mean(np.sin(radians))), float(np.mean(np.cos(radians)))))


@dataclass
class GlobalFix:
    x: float
    y: float
    yaw_deg: float
    scale: float
    score: float
    margin: float
    elapsed_ms: float
    valid: bool = True
    rejection_reasons: tuple = ()
    ratio_match_count: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    reprojection_p95_px: Optional[float] = None
    center_agreement_px: Optional[float] = None
    alternatives: tuple = ()
    search_bounds_xyxy: Optional[tuple] = None
    search_area_fraction: float = 1.0
    diagnostics: Optional[dict] = None


class GlobalMapLocalizer:
    """Feature-proposed, correlation-verified localization on a normalized map."""

    supports_bounded_search = True

    def __init__(
        self,
        mosaic: np.ndarray,
        coverage: Optional[np.ndarray] = None,
        localization_to_original_3x3=None,
    ) -> None:
        if mosaic is None or mosaic.size == 0:
            raise ValueError("Global map mosaic is empty")
        self.mosaic = mosaic.copy()
        self.map_gradient = _gradient(mosaic)
        if coverage is None:
            coverage = np.full(mosaic.shape[:2], 255, np.uint8)
        if coverage.shape[:2] != mosaic.shape[:2]:
            raise ValueError("Localization coverage and mosaic dimensions differ")
        self.coverage = np.uint8(coverage > 0) * 255
        self.localization_to_original = np.asarray(
            localization_to_original_3x3
            if localization_to_original_3x3 is not None
            else np.eye(3),
            dtype=np.float64,
        )
        if self.localization_to_original.shape != (3, 3):
            raise ValueError("Localization-to-original transform must be 3x3")
        self.original_to_localization = np.linalg.inv(self.localization_to_original)
        self.sift = cv2.SIFT_create(
            nfeatures=8000, contrastThreshold=0.005, edgeThreshold=15
        )
        map_mask = cv2.erode(self.coverage, np.ones((5, 5), np.uint8))
        self.map_points, self.map_descriptors = self.sift.detectAndCompute(
            cv2.cvtColor(self.mosaic, cv2.COLOR_BGR2GRAY), map_mask
        )
        if self.map_descriptors is None or len(self.map_points) < 6:
            raise ValueError("Localization mosaic has too few usable SIFT features")
        self._cancel = threading.Event()

    def close(self) -> None:
        self._cancel.set()

    @staticmethod
    def _transform(template, mask, scale: float, angle_deg: float):
        height, width = template.shape[:2]
        scaled_size = (
            max(16, int(round(width * scale))),
            max(16, int(round(height * scale))),
        )
        scaled = cv2.resize(template, scaled_size, interpolation=cv2.INTER_LINEAR)
        scaled_mask = cv2.resize(mask, scaled_size, interpolation=cv2.INTER_NEAREST)
        sh, sw = scaled.shape[:2]
        matrix = cv2.getRotationMatrix2D(((sw - 1) / 2.0, (sh - 1) / 2.0), angle_deg, 1.0)
        rotated = cv2.warpAffine(scaled, matrix, (sw, sh), flags=cv2.INTER_LINEAR)
        rotated_mask = cv2.warpAffine(scaled_mask, matrix, (sw, sh), flags=cv2.INTER_NEAREST)
        return rotated, rotated_mask

    def _original_xy(self, point_xy):
        point = np.asarray([float(point_xy[0]), float(point_xy[1]), 1.0])
        mapped = self.localization_to_original.dot(point)
        return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])

    def _localization_xy(self, point_xy):
        point = np.asarray([float(point_xy[0]), float(point_xy[1]), 1.0])
        mapped = self.original_to_localization.dot(point)
        return float(mapped[0] / mapped[2]), float(mapped[1] / mapped[2])

    def _invalid(self, started, reasons, **metrics):
        def finite_or_none(name):
            value = metrics.get(name)
            if value is None:
                return None
            value = float(value)
            return value if math.isfinite(value) else None

        return GlobalFix(
            0.0,
            0.0,
            0.0,
            0.0,
            float(metrics.get("score", 0.0)),
            float(metrics.get("margin", 0.0)),
            (time.perf_counter() - started) * 1000.0,
            valid=False,
            rejection_reasons=tuple(reasons),
            ratio_match_count=int(metrics.get("ratio_match_count", 0)),
            inlier_count=int(metrics.get("inlier_count", 0)),
            inlier_ratio=float(metrics.get("inlier_ratio", 0.0)),
            reprojection_p95_px=finite_or_none("reprojection_p95_px"),
            center_agreement_px=finite_or_none("center_agreement_px"),
        )

    def localize(
        self,
        observation: np.ndarray,
        mask: np.ndarray,
        yaw_prior_deg: Optional[float] = None,
        search_center_xy=None,
        search_radius_px: Optional[float] = None,
    ) -> GlobalFix:
        started = time.perf_counter()
        if self._cancel.is_set():
            raise RuntimeError("Global localization canceled")
        observation_gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        points, descriptors = self.sift.detectAndCompute(observation_gray, mask)
        if descriptors is None or len(points) < 6:
            return self._invalid(started, ("too-few-observation-features",))
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
            descriptors, self.map_descriptors, k=2
        )
        matches = [
            first
            for pair in pairs
            if len(pair) == 2
            for first, second in [pair]
            if first.distance < 0.80 * second.distance
        ]
        ratio_count = len(matches)
        if ratio_count < 6:
            return self._invalid(
                started,
                ("too-few-ratio-matches",),
                ratio_match_count=ratio_count,
            )
        source = np.float32([points[item.queryIdx].pt for item in matches])
        target = np.float32([self.map_points[item.trainIdx].pt for item in matches])
        affine, inlier_mask = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=20000,
            confidence=0.999,
        )
        if affine is None or inlier_mask is None:
            return self._invalid(
                started,
                ("no-consistent-similarity",),
                ratio_match_count=ratio_count,
            )
        inliers = inlier_mask.ravel().astype(bool)
        inlier_count = int(np.count_nonzero(inliers))
        inlier_ratio = float(np.mean(inliers))
        if inlier_count:
            predicted = cv2.transform(source.reshape((-1, 1, 2)), affine).reshape((-1, 2))
            errors = np.linalg.norm(predicted - target, axis=1)
            reprojection_p95 = float(np.percentile(errors[inliers], 95))
        else:
            reprojection_p95 = float("inf")
        scale = math.hypot(float(affine[0, 0]), float(affine[0, 1]))
        angle = math.degrees(math.atan2(float(affine[1, 0]), float(affine[0, 0])))
        angle = ((angle + 180.0) % 360.0) - 180.0
        center = cv2.transform(
            np.float32([[[observation.shape[1] / 2.0, observation.shape[0] / 2.0]]]),
            affine,
        ).reshape(2)

        # These mandatory geometry checks cannot be rescued by correlation.
        # Reject before scanning the map or building diagnostic rasters.
        geometry_reasons = []
        if inlier_count < 6:
            geometry_reasons.append("too-few-geometric-inliers")
        if inlier_ratio < 0.60:
            geometry_reasons.append("low-inlier-ratio")
        if reprojection_p95 > 3.0:
            geometry_reasons.append("high-reprojection-error")
        if not 0.75 <= scale <= 1.30:
            geometry_reasons.append("scale-out-of-range")
        if geometry_reasons:
            return self._invalid(
                started, geometry_reasons,
                ratio_match_count=ratio_count, inlier_count=inlier_count,
                inlier_ratio=inlier_ratio, reprojection_p95_px=reprojection_p95,
            )
        transformed, transformed_mask = self._transform(
            _gradient(observation), mask, scale, angle
        )
        height, width = transformed.shape[:2]
        if height >= self.map_gradient.shape[0] or width >= self.map_gradient.shape[1]:
            return self._invalid(
                started,
                ("transformed-observation-exceeds-map",),
                ratio_match_count=ratio_count,
                inlier_count=inlier_count,
                inlier_ratio=inlier_ratio,
                reprojection_p95_px=reprojection_p95,
            )
        search_left = 0
        search_top = 0
        search_right = self.map_gradient.shape[1]
        search_bottom = self.map_gradient.shape[0]
        if search_center_xy is not None and search_radius_px is not None:
            center_x, center_y = self._localization_xy(search_center_xy)
            scale_x = math.hypot(
                self.original_to_localization[0, 0],
                self.original_to_localization[1, 0],
            )
            scale_y = math.hypot(
                self.original_to_localization[0, 1],
                self.original_to_localization[1, 1],
            )
            radius = max(
                16.0,
                float(search_radius_px) * (scale_x + scale_y) / 2.0,
            )
            search_left = max(0, int(math.floor(center_x - radius - width / 2.0)))
            search_top = max(0, int(math.floor(center_y - radius - height / 2.0)))
            search_right = min(
                self.map_gradient.shape[1],
                int(math.ceil(center_x + radius + width / 2.0)),
            )
            search_bottom = min(
                self.map_gradient.shape[0],
                int(math.ceil(center_y + radius + height / 2.0)),
            )
        search_gradient = self.map_gradient[
            search_top:search_bottom, search_left:search_right
        ]
        if height >= search_gradient.shape[0] or width >= search_gradient.shape[1]:
            search_left = search_top = 0
            search_right = self.map_gradient.shape[1]
            search_bottom = self.map_gradient.shape[0]
            search_gradient = self.map_gradient
        search_area_fraction = float(
            search_gradient.shape[0]
            * search_gradient.shape[1]
            / (self.map_gradient.shape[0] * self.map_gradient.shape[1])
        )
        response = cv2.matchTemplate(
            search_gradient,
            transformed,
            cv2.TM_CCORR_NORMED,
            mask=transformed_mask,
        )
        response = np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0)
        peaks = []
        suppressed = response.copy()
        suppression_radius = max(8, min(width, height) // 3)
        for _ in range(3):
            _, peak_score, _, location = cv2.minMaxLoc(suppressed)
            peak_center = (
                search_left + location[0] + width / 2.0,
                search_top + location[1] + height / 2.0,
            )
            peaks.append((float(peak_score), peak_center))
            cv2.circle(suppressed, location, suppression_radius, -1.0, -1)
        score = peaks[0][0]
        margin = score - peaks[1][0]
        correlation_center = np.asarray(peaks[0][1], dtype=np.float64)
        center_agreement = float(np.linalg.norm(correlation_center - center))
        center_x = int(np.clip(round(correlation_center[0]), 0, self.coverage.shape[1] - 1))
        center_y = int(np.clip(round(correlation_center[1]), 0, self.coverage.shape[0] - 1))
        reasons = []
        if score < 0.55:
            reasons.append("low-correlation")
        if margin < 0.06:
            reasons.append("ambiguous-correlation")
        if center_agreement > 8.0:
            reasons.append("feature-correlation-disagreement")
        if not self.coverage[center_y, center_x]:
            reasons.append("outside-observed-coverage")
        alternatives = []
        for peak_score, peak_center in peaks:
            original_x, original_y = self._original_xy(peak_center)
            alternatives.append(
                {"x": original_x, "y": original_y, "score": peak_score}
            )
        original_x, original_y = self._original_xy(correlation_center)
        feature_original_x, feature_original_y = self._original_xy(center)
        feature_center_in_bounds = bool(
            0.0 <= center[0] < self.coverage.shape[1]
            and 0.0 <= center[1] < self.coverage.shape[0]
        )
        feature_center_covered = False
        if feature_center_in_bounds:
            feature_center_x = int(
                np.clip(round(center[0]), 0, self.coverage.shape[1] - 1)
            )
            feature_center_y = int(
                np.clip(round(center[1]), 0, self.coverage.shape[0] - 1)
            )
            feature_center_covered = bool(
                self.coverage[feature_center_y, feature_center_x]
            )
        map_scale_x = math.hypot(
            self.localization_to_original[0, 0], self.localization_to_original[1, 0]
        )
        map_scale_y = math.hypot(
            self.localization_to_original[0, 1], self.localization_to_original[1, 1]
        )
        original_scale = scale * (map_scale_x + map_scale_y) / 2.0
        response_u8 = cv2.normalize(
            response, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        correlation_heatmap = cv2.applyColorMap(
            response_u8, cv2.COLORMAP_TURBO
        )
        search_region = self.mosaic[
            search_top:search_bottom, search_left:search_right
        ].copy()
        candidate_overlay = search_region.copy()
        colors = ((80, 230, 120), (70, 210, 245), (90, 90, 245))
        for rank, ((peak_score, peak_center), color) in enumerate(
            zip(peaks, colors), start=1
        ):
            local_center = (
                int(round(peak_center[0] - search_left)),
                int(round(peak_center[1] - search_top)),
            )
            cv2.rectangle(
                candidate_overlay,
                (
                    int(round(local_center[0] - width / 2.0)),
                    int(round(local_center[1] - height / 2.0)),
                ),
                (
                    int(round(local_center[0] + width / 2.0)),
                    int(round(local_center[1] + height / 2.0)),
                ),
                color,
                2,
            )
            cv2.putText(
                candidate_overlay,
                "{} {:.3f}".format(rank, peak_score),
                (local_center[0] + 4, local_center[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        feature_center = (
            int(round(center[0] - search_left)),
            int(round(center[1] - search_top)),
        )
        cv2.drawMarker(
            candidate_overlay,
            feature_center,
            (255, 120, 60),
            cv2.MARKER_CROSS,
            18,
            2,
        )
        transformed_gradient = cv2.normalize(
            transformed, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        diagnostics = {
            "observation": observation.copy(),
            "mask": mask.copy(),
            "transformed_gradient": transformed_gradient,
            "search_region": search_region,
            "correlation_heatmap": correlation_heatmap,
            "candidate_overlay": candidate_overlay,
            # This remains diagnostic evidence: GlobalMapLocalizer acceptance
            # still requires correlation verification. Offline consumers may
            # independently require repeated geometric consensus.
            "feature_center_original_xy": (
                float(feature_original_x),
                float(feature_original_y),
            ),
            "feature_center_covered": feature_center_covered,
        }
        return GlobalFix(
            original_x,
            original_y,
            angle,
            original_scale,
            score,
            margin,
            (time.perf_counter() - started) * 1000.0,
            valid=not reasons,
            rejection_reasons=tuple(reasons),
            ratio_match_count=ratio_count,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            reprojection_p95_px=(
                reprojection_p95 if math.isfinite(reprojection_p95) else None
            ),
            center_agreement_px=(
                center_agreement if math.isfinite(center_agreement) else None
            ),
            alternatives=tuple(alternatives),
            search_bounds_xyxy=(
                int(search_left),
                int(search_top),
                int(search_right),
                int(search_bottom),
            ),
            search_area_fraction=search_area_fraction,
            diagnostics=diagnostics,
        )


class MinimapExtractor:
    def __init__(self, crop_xywh, calibration: dict) -> None:
        self.crop_xywh = tuple(int(value) for value in crop_xywh)
        _, _, width, height = self.crop_xywh
        calibration = normalize_minimap_geometry(
            calibration,
            minimap_crop_space([width, height]),
            allow_legacy=True,
        )
        boundary = calibration["outer_boundary"]
        self.center = (
            float(boundary.get("center_x", self.crop_xywh[2] / 2.0)),
            float(boundary.get("center_y", self.crop_xywh[3] / 2.0)),
        )
        self.radius = float(boundary.get("radius", min(self.crop_xywh[2:]) * 0.4))
        cx, cy = self.center
        radius = int(round(self.radius))
        left = max(0, int(round(cx)) - radius)
        top = max(0, int(round(cy)) - radius)
        right = min(width, int(round(cx)) + radius)
        bottom = min(height, int(round(cy)) + radius)
        self.observation_bounds = (left, top, right, bottom)
        observation_height = bottom - top
        observation_width = right - left
        self.mask = np.zeros((observation_height, observation_width), np.uint8)
        cv2.circle(
            self.mask,
            (observation_width // 2, observation_height // 2),
            max(1, min(observation_width, observation_height) // 2 - 2),
            255,
            -1,
        )
        cv2.circle(
            self.mask,
            (observation_width // 2, observation_height // 2),
            max(5, int(round(radius * 0.20))),
            0,
            -1,
        )

    def crop(self, frame: np.ndarray) -> np.ndarray:
        x, y, width, height = self.crop_xywh
        crop = frame[y : y + height, x : x + width]
        if crop.shape[:2] != (height, width):
            raise RuntimeError("Live frame does not contain the calibrated mini-map crop")
        return crop

    def extract(self, frame: np.ndarray):
        crop = self.crop(frame)
        left, top, right, bottom = self.observation_bounds
        observation = crop[top:bottom, left:right].copy()
        return observation, self.mask


class TwoRateRealtimeTracker:
    """Fuse low-rate absolute map fixes with high-rate visual deltas."""

    def __init__(
        self,
        mosaic: np.ndarray,
        minimap_config: dict,
        minimap_calibration: dict,
        scene_yaw_calibration: dict,
        global_interval_s: float = 1.0,
        localizer: Optional[GlobalMapLocalizer] = None,
        initial_consensus_count: int = 2,
        cursor_pose_estimator: Optional[CursorPoseEstimator] = None,
        cursor_interval_s: float = 0.25,
        temporal_pose_search: bool = False,
        pose_confidence_min: float = 0.45,
        local_response_min: float = 0.25,
        local_shift_max_fraction: float = 0.18,
        relocalize_after_rejections: int = 6,
        recovery_consensus_count: int = 2,
        global_candidate_advisor=None,
        route_visual_tracker=None,
        cursor_pose_process_config=None,
        representation_interval_s: float = 0.25,
    ) -> None:
        self.extractor = MinimapExtractor(
            minimap_config["crop_xywh"], minimap_calibration
        )
        self.localizer = localizer or GlobalMapLocalizer(mosaic)
        focal_ratio = float(scene_yaw_calibration["focal_ratio"])
        self.focal_ratio = focal_ratio
        scene_config = scene_yaw_calibration.get("config") or {}
        self.scene_excluded_rects = tuple(
            tuple(row)
            for row in (scene_config.get("excluded_rects") or ())
        )
        self.scene_estimator = None
        self.global_interval_ns = int(float(global_interval_s) * 1.0e9)
        self.last_global_ns = None
        self.previous_minimap = None
        self._latest_minimap_observation = None
        self.local_response_min = float(local_response_min)
        self.local_shift_max_px = max(
            3.0,
            float(min(self.extractor.mask.shape[:2]))
            * float(local_shift_max_fraction),
        )
        self.relocalize_after_rejections = max(
            1, int(relocalize_after_rejections)
        )
        self.recovery_consensus_count = max(2, int(recovery_consensus_count))
        self._local_rejections = 0
        self._recovery_request_active = False
        self._recovery_hypotheses = []
        transition_model = getattr(self.localizer, "transition_model", None)
        runtime_transition = (transition_model or {}).get("runtime") or {}
        self.transition_controller = (
            TransitionController(
                transition_model,
                confirmation_count=int(
                    runtime_transition.get("confirmation_count", 2)
                ),
            )
            if transition_model is not None
            else None
        )
        self._active_map_mode_id = None
        self._last_map_transition = None
        self._last_representation_observation = None
        self._representation_error = None
        self.representation_interval_ns = int(
            max(0.0, float(representation_interval_s)) * 1.0e9
        )
        self.last_representation_ns = None
        self.fusion = PoseFusionGate(
            FusionConfig(
                initial_position_sigma_m=3.0,
                initial_yaw_sigma_deg=3.0,
                position_sigma_per_step_m=0.25,
                position_sigma_per_meter=0.04,
                prediction_position_floor_m=35.0,
                prediction_position_limit_m=180.0,
                coarse_position_limit_m=220.0,
                corrected_position_sigma_m=4.0,
                cautious_position_sigma_m=18.0,
                relocalize_position_sigma_m=40.0,
                stop_position_sigma_m=100.0,
            )
        )
        self.map_scale = 1.0
        self.sequence = 0
        self.trail = []
        self._global_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="aria-global-fix"
        )
        self._global_future = None
        self._last_global_fix = None
        self._last_global_search = None
        self._global_error = None
        self._last_global_diagnostics = None
        self.global_candidate_advisor = global_candidate_advisor
        self.route_visual_tracker = route_visual_tracker
        if route_visual_tracker is None and callable(getattr(self.localizer, "refine_near", None)):
            # Atlas matching does not require demonstrated states. Keep the
            # relative-motion fallback for localizers without this capability.
            from aria_trace.services.localization.route.tracker import RouteVisualTracker

            self.route_visual_tracker = RouteVisualTracker(
                None, self.localizer, score_min=0.0, local_radius_px=12.0,
            )
        self._last_route_tracking = None
        self._last_route_assistance = None
        observe_modes = getattr(self.localizer, "observe_modes", None)
        if self.transition_controller is not None and callable(observe_modes):
            self._representation_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="aria-map-representation"
            )
        else:
            self._representation_executor = None
        self._representation_future = None
        self._representation_position_sigma = 0.0
        self.initial_consensus_count = max(1, int(initial_consensus_count))
        self._initial_hypotheses = []
        if cursor_pose_estimator is not None and cursor_pose_process_config is not None:
            raise ValueError(
                "Choose either an in-process cursor estimator or process config"
            )
        self.cursor_pose_estimator = cursor_pose_estimator
        self.cursor_interval_ns = int(float(cursor_interval_s) * 1.0e9)
        self.last_cursor_ns = None
        if cursor_pose_process_config is not None:
            process_options = dict(cursor_pose_process_config)
            process_options.pop("calibration_metadata", None)
            self._cursor_executor = CursorPoseProcessExecutor(
                **process_options
            )
            self._cursor_executor_kind = "process"
        elif self.cursor_pose_estimator is not None:
            self._cursor_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="aria-cursor-pose"
            )
            self._cursor_executor_kind = "thread"
        else:
            self._cursor_executor = None
            self._cursor_executor_kind = None
        self._cursor_future = None
        self._cursor_submit_host_time_ns = None
        self._last_cursor_pose = None
        self._last_cursor_measurement = None
        self._cursor_error = None
        self.temporal_pose_search = bool(temporal_pose_search)
        self.pose_confidence_min = float(pose_confidence_min)
        calibration = getattr(self.cursor_pose_estimator, "calibration", {}) or {}
        if cursor_pose_process_config is not None:
            calibration = dict(
                cursor_pose_process_config.get("calibration_metadata") or {}
            )
        dynamics = calibration.get("cursor_temporal_dynamics") or {}
        self.cursor_motion_envelope = dynamics.get(
            "recommended_runtime_envelope", {}
        )
        self._cursor_last_angle = None
        self._cursor_last_time_ns = None
        self._cursor_angular_velocity_deg_s = 0.0
        self._cursor_rejections = 0
        self._cursor_tracking_state = "uninitialized"
        self._last_cursor_search = None

    def close(self) -> None:
        self._global_executor.shutdown(wait=False)
        if self._representation_executor is not None:
            self._representation_executor.shutdown(wait=False)
        if self._cursor_executor is not None:
            self._cursor_executor.shutdown(wait=False)
        close_localizer = getattr(self.localizer, "close", None)
        if close_localizer is not None:
            close_localizer()

    def take_global_diagnostics(self):
        diagnostics = self._last_global_diagnostics
        self._last_global_diagnostics = None
        return diagnostics

    def _ensure_scene_estimator(self, frame):
        if self.scene_estimator is None:
            height, width = frame.shape[:2]
            self.scene_estimator = KltAngularYawEstimator(
                camera_matrix(width, height, self.focal_ratio),
                max_corners=1000,
                min_tracks=20,
                use_essential_gate=True,
                excluded_rects=self.scene_excluded_rects,
            )

    @staticmethod
    def _rotate(image, angle_deg):
        height, width = image.shape[:2]
        matrix = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), angle_deg, 1.0)
        return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR)

    def _cursor_search(self, timestamp_ns: int):
        if (
            not self.temporal_pose_search
            or self._cursor_last_angle is None
            or self._cursor_last_time_ns is None
        ):
            return None, None
        elapsed_s = (timestamp_ns - self._cursor_last_time_ns) / 1.0e9
        if elapsed_s <= 0.0 or elapsed_s > 2.0:
            return None, None
        envelope = self.cursor_motion_envelope
        calibrated_turn_rate = float(
            envelope.get("calibrated_turn_rate_p99_deg_s") or 720.0
        )
        normal_turn_rate = float(
            envelope.get("normal_turn_rate_p99_deg_s")
            or calibrated_turn_rate
        )
        turn_rate = (
            normal_turn_rate
            if self._cursor_tracking_state == "stable"
            else calibrated_turn_rate
        )
        acceleration = float(
            envelope.get("calibrated_angular_acceleration_p99_deg_s2") or 1440.0
        )
        ordinary_jump = float(
            envelope.get("ordinary_heading_jump_p99_deg") or 2.0
        )
        half_width = max(
            6.0,
            ordinary_jump + turn_rate * elapsed_s + 0.5 * acceleration * elapsed_s ** 2,
        )
        half_width *= 2.0 ** min(self._cursor_rejections, 3)
        if half_width >= 170.0:
            return None, None
        predicted = (
            self._cursor_last_angle
            + self._cursor_angular_velocity_deg_s * elapsed_s
        ) % 360.0
        return float(predicted), float(half_width)

    def _global_search(self):
        if self.fusion._state is None:
            return None, None
        state = self.fusion.state
        radius = float(np.clip(state.position_sigma_m * 3.0, 100.0, 1200.0))
        return (float(state.pose.x), float(state.pose.y)), radius

    def _localize_global(self, minimap, mask, yaw_prior, center, radius):
        proposal = None
        if self.global_candidate_advisor is not None:
            try:
                proposal = self.global_candidate_advisor.propose(minimap, mask)
            except Exception as exc:
                self._last_route_assistance = {
                    "policy": "candidate-window-only",
                    "status": "advisor-error",
                    "error": "{}: {}".format(type(exc).__name__, exc),
                }
            if proposal is not None:
                advised = self.localizer.localize(
                    minimap,
                    mask,
                    yaw_prior,
                    search_center_xy=proposal["center_xy"],
                    search_radius_px=proposal["radius_px"],
                )
                self._last_route_assistance = dict(proposal)
                self._last_route_assistance["bounded_fix_valid"] = bool(
                    advised.valid
                )
                if advised.valid:
                    self._last_route_assistance["status"] = "verified"
                    return advised
                self._last_route_assistance["status"] = (
                    "bounded-miss-fell-back"
                )
        if getattr(self.localizer, "supports_bounded_search", False):
            return self.localizer.localize(
                minimap,
                mask,
                yaw_prior,
                search_center_xy=center,
                search_radius_px=radius,
            )
        return self.localizer.localize(minimap, mask, yaw_prior)

    @staticmethod
    def _fixes_agree(first: GlobalFix, second: GlobalFix) -> bool:
        first_mode = TwoRateRealtimeTracker._fix_mode(first)
        second_mode = TwoRateRealtimeTracker._fix_mode(second)
        if first_mode and second_mode and first_mode != second_mode:
            return False
        position_delta = math.hypot(second.x - first.x, second.y - first.y)
        yaw_delta = abs(_angle_difference_deg(second.yaw_deg, first.yaw_deg))
        scale_delta = abs(second.scale - first.scale) / max(first.scale, 1.0e-6)
        return position_delta <= 30.0 and yaw_delta <= 15.0 and scale_delta <= 0.12

    @staticmethod
    def _fix_mode(fix: GlobalFix):
        return (
            ((fix.diagnostics or {}).get("map_layer") or {}).get(
                "selected_mode_id"
            )
        )

    @staticmethod
    def _fix_mode_likelihoods(fix: GlobalFix):
        return dict(
            ((fix.diagnostics or {}).get("map_layer") or {}).get(
                "mode_likelihoods"
            )
            or {}
        )

    def _activate_map_mode(self, mode_id: str, *, update_scale: bool) -> None:
        self._active_map_mode_id = str(mode_id)
        if self.transition_controller is not None:
            self.transition_controller.set_active_mode(self._active_map_mode_id)
        setter = getattr(self.localizer, "set_active_mode", None)
        if callable(setter):
            setter(self._active_map_mode_id)
        if update_scale:
            scale_for_mode = getattr(self.localizer, "map_scale_for_mode", None)
            if callable(scale_for_mode):
                self.map_scale = float(scale_for_mode(self._active_map_mode_id))

    def _initialize_map_mode(self, fixes) -> None:
        modes = [self._fix_mode(item) for item in fixes]
        modes = [str(item) for item in modes if item]
        if not modes:
            return
        self._activate_map_mode(modes[-1], update_scale=True)

    def _consume_representation_observation(self, timestamp_ns: int) -> bool:
        """Apply representation evidence without granting it pose authority."""

        if self._representation_future is None or not self._representation_future.done():
            return False
        try:
            observation = self._representation_future.result()
            self._representation_error = None
        except Exception as exc:
            self._representation_error = "{}: {}".format(type(exc).__name__, exc)
            self._representation_future = None
            return False
        self._representation_future = None
        controller_result = None
        if observation.get("valid") and self.transition_controller is not None:
            canonical_xy = observation.get("canonical_xy_read_only")
            controller_result = self.transition_controller.update(
                observation.get("likelihoods") or {},
                position_uncertainty_px=self._representation_position_sigma,
                canonical_xy=(
                    tuple(float(value) for value in canonical_xy)
                    if canonical_xy is not None
                    else None
                ),
            )
        self._last_representation_observation = dict(observation)
        self._last_representation_observation["controller"] = controller_result
        route_transition = self.route_visual_tracker
        if controller_result and route_transition is not None:
            arm = getattr(route_transition, "arm_trained_transition", None)
            cancel = getattr(route_transition, "cancel_trained_transition", None)
            if (
                controller_result.get("within_observed_zone")
                and controller_result.get("transition_zone_id") is not None
                and callable(arm)
            ):
                arm(
                    self._active_map_mode_id,
                    controller_result.get("competing_mode_id"),
                )
            elif (
                not controller_result.get("transition_armed")
                and not controller_result.get("switched")
                and callable(cancel)
            ):
                cancel()
        if not controller_result or not controller_result.get("switched"):
            return True
        previous_mode = self._active_map_mode_id
        next_mode = str(controller_result["active_mode_id"])
        self._activate_map_mode(next_mode, update_scale=True)
        confirm_route_transition = getattr(
            self.route_visual_tracker,
            "confirm_trained_transition_layer",
            None,
        )
        if callable(confirm_route_transition):
            confirm_route_transition(next_mode)
        self.previous_minimap = None
        self._last_map_transition = {
            "from_mode_id": previous_mode,
            "to_mode_id": next_mode,
            "host_time_ns": timestamp_ns,
            "position_policy": "held-continuous-pose",
            "evidence_source": "continuous-local-representation-observer",
            "reset_local_reference": True,
            "mode_likelihoods": dict(observation.get("likelihoods") or {}),
            "raw_correlation_scores": dict(
                observation.get("raw_correlation_scores") or {}
            ),
            "observation_elapsed_ms": observation.get("elapsed_ms"),
        }
        return True

    def _recovery_requested(self) -> bool:
        if self.fusion._state is None:
            return False
        if self._local_rejections >= self.relocalize_after_rejections:
            self._recovery_request_active = True
        if self.fusion.state.mode in ("RELOCALIZE", "STOP"):
            self._recovery_request_active = True
        return self._recovery_request_active

    def _clear_recovery_request(self) -> None:
        self._recovery_request_active = False

    def latest_minimap_observation(self) -> Optional[np.ndarray]:
        """Expose the current frame's extracted mini-map for read-only evidence."""

        return self._latest_minimap_observation

    @staticmethod
    def _mean_fix(rows) -> Pose2D:
        return Pose2D(
            float(np.mean([item.x for item in rows])),
            float(np.mean([item.y for item in rows])),
            _circular_mean_deg([item.yaw_deg for item in rows]),
        )

    def update(self, frame: np.ndarray, host_time_ns: Optional[int] = None) -> dict:
        started = time.perf_counter()
        timestamp_ns = int(host_time_ns or time.perf_counter_ns())
        if self.route_visual_tracker is not None:
            # Route-assisted XY is measured directly against the north-fixed map.
            # Its local-motion branch is bypassed below, so full-frame scene KLT
            # has no consumer and must not occupy the live control path.
            yaw = YawEstimate(
                0.0,
                0.0,
                0,
                0,
                0.0,
                0.0,
                "bypassed:route-map-correlation",
            )
        else:
            self._ensure_scene_estimator(frame)
            yaw = self.scene_estimator.update(frame)
        minimap, mask = self.extractor.extract(frame)
        self._latest_minimap_observation = minimap
        representation_observation_fresh = self._consume_representation_observation(
            timestamp_ns
        )
        local_shift = (0.0, 0.0)
        local_response = 0.0
        local_accepted = False
        local_decision = "reference-initialized"
        if self.route_visual_tracker is not None:
            compensation_sign = 0.0
            local_decision = "bypassed:route-map-correlation"
        elif self.previous_minimap is not None:
            trials = []
            signs = (0.0,)
            if yaw.confidence >= 0.20 and abs(yaw.delta_deg) >= 0.05:
                signs = (0.0, -1.0, 1.0)
            for sign in signs:
                compensated = self._rotate(minimap, sign * yaw.delta_deg)
                shift, response = estimate_masked_shift(
                    self.previous_minimap, compensated, mask
                )
                trials.append((response, shift, sign))
            best = max(trials)
            zero = next(item for item in trials if item[2] == 0.0)
            # Do not turn scene-camera motion into map rotation unless it
            # materially improves the mini-map registration itself.
            selected = zero if zero[0] >= best[0] - 0.02 else best
            local_response, local_shift, compensation_sign = selected
            local_distance = math.hypot(*local_shift)
            if local_response < self.local_response_min:
                local_decision = "rejected:weak-correlation"
            elif local_distance > self.local_shift_max_px:
                local_decision = "rejected:implausible-displacement"
            else:
                local_accepted = True
                local_decision = "accepted"
        else:
            compensation_sign = 0.0
        self.previous_minimap = minimap
        alignment_delta_deg = (
            compensation_sign * yaw.delta_deg if local_accepted else 0.0
        )

        local_motion_applied = False
        local_quality = 0.0
        if local_accepted:
            local_quality = float(
                np.clip(
                    (local_response - self.local_response_min)
                    / max(1.0 - self.local_response_min, 1.0e-6),
                    0.0,
                    1.0,
                )
            )
        if self.fusion._state is not None and self.sequence and local_accepted:
            shift_x, shift_y = local_shift
            local_motion = (-shift_x * self.map_scale, -shift_y * self.map_scale)
            self.fusion.predict(
                local_motion,
                alignment_delta_deg,
                measurement_quality=local_quality,
            )
            local_motion_applied = True
            self._local_rejections = 0
            self._recovery_hypotheses = []
        elif (
            self.fusion._state is not None
            and self.sequence
            and self.route_visual_tracker is None
        ):
            self._local_rejections += 1

        cursor_due = (
            self.last_cursor_ns is None
            or timestamp_ns - self.last_cursor_ns >= self.cursor_interval_ns
        )
        if (
            cursor_due
            and self._cursor_future is None
            and self._cursor_executor is not None
        ):
            angle_prior, search_half_width = self._cursor_search(timestamp_ns)
            self._last_cursor_search = {
                "enabled": self.temporal_pose_search,
                "angle_prior_deg": angle_prior,
                "half_width_deg": search_half_width,
                "rejection_count": self._cursor_rejections,
                "state": self._cursor_tracking_state,
            }
            cursor_crop = self.extractor.crop(frame).copy()
            self._cursor_submit_host_time_ns = time.perf_counter_ns()
            if self._cursor_executor_kind == "process":
                self._cursor_future = self._cursor_executor.submit(
                    cursor_crop,
                    None,
                    timestamp_ns,
                    angle_prior,
                    search_half_width,
                )
            elif angle_prior is None:
                self._cursor_future = self._cursor_executor.submit(
                    self.cursor_pose_estimator.estimate,
                    cursor_crop,
                    None,
                    timestamp_ns,
                )
            else:
                self._cursor_future = self._cursor_executor.submit(
                    self.cursor_pose_estimator.estimate,
                    cursor_crop,
                    None,
                    timestamp_ns,
                    angle_prior,
                    search_half_width,
                )
            self.last_cursor_ns = timestamp_ns

        global_fix = None
        decision = None
        if self._global_future is not None and self._global_future.done():
            try:
                global_fix = self._global_future.result()
                self._global_error = None
            except Exception as exc:
                self._global_error = "{}: {}".format(type(exc).__name__, exc)
            self._global_future = None
        if global_fix is not None:
            self._last_global_diagnostics = global_fix.diagnostics
            fusion_metrics = None
            if not global_fix.valid:
                self._initial_hypotheses = []
                decision = "rejected-quality:" + ",".join(
                    global_fix.rejection_reasons
                )
            else:
                if self.fusion._state is None:
                    if self._initial_hypotheses:
                        previous = self._initial_hypotheses[-1]
                        if not self._fixes_agree(previous, global_fix):
                            self._initial_hypotheses = []
                    self._initial_hypotheses.append(global_fix)
                    count = len(self._initial_hypotheses)
                    if count >= self.initial_consensus_count:
                        rows = self._initial_hypotheses[-self.initial_consensus_count :]
                        initialized = self._mean_fix(rows)
                        self.map_scale = float(np.mean([item.scale for item in rows]))
                        self.fusion.initialize(initialized)
                        self._initialize_map_mode(rows)
                        self._initial_hypotheses = []
                        decision = "initialized-consensus"
                        fusion_metrics = {
                            "accepted": True,
                            "reason": decision,
                            "predicted_position_innovation_map_px": None,
                            "predicted_yaw_innovation_deg": None,
                            "applied_position_change_map_px": 0.0,
                            "applied_yaw_change_deg": 0.0,
                        }
                    else:
                        decision = "awaiting-consensus:{}/{}".format(
                            count, self.initial_consensus_count
                        )
                        fusion_metrics = {
                            "accepted": False,
                            "reason": decision,
                        }
                else:
                    recovery_requested = self._recovery_requested()
                    if not recovery_requested:
                        decision = "ignored:locked-continuity"
                        fusion_metrics = {
                            "accepted": False,
                            "reason": decision,
                        }
                    else:
                        if self._recovery_hypotheses and not self._fixes_agree(
                            self._recovery_hypotheses[-1], global_fix
                        ):
                            self._recovery_hypotheses = []
                        self._recovery_hypotheses.append(global_fix)
                        count = len(self._recovery_hypotheses)
                        if count < self.recovery_consensus_count:
                            decision = "awaiting-recovery-consensus:{}/{}".format(
                                count, self.recovery_consensus_count
                            )
                            fusion_metrics = {
                                "accepted": False,
                                "reason": decision,
                            }
                        else:
                            rows = self._recovery_hypotheses[
                                -self.recovery_consensus_count :
                            ]
                            recovery_mode = self._fix_mode(rows[-1])
                            mode_changed = bool(
                                recovery_mode
                                and self._active_map_mode_id
                                and recovery_mode != self._active_map_mode_id
                            )
                            if mode_changed:
                                decision = "rejected:map-mode-mismatch"
                                self._recovery_hypotheses = []
                                fusion_metrics = {
                                    "accepted": False,
                                    "reason": decision,
                                }
                                rows = None
                            if rows is not None:
                                recovery_pose = self._mean_fix(rows)
                                correction = self.fusion.consider_absolute(recovery_pose)
                                decision = (
                                    "recovered-consensus"
                                    if correction.accepted
                                    else "rejected:" + correction.reason
                                )
                                if correction.accepted:
                                    if self._active_map_mode_id is None:
                                        self.map_scale = float(
                                            np.mean([item.scale for item in rows])
                                        )
                                    if self.route_visual_tracker is not None:
                                        recovered_state = self.fusion.state
                                        self.route_visual_tracker.seed(
                                            recovered_state.pose.x,
                                            recovered_state.pose.y,
                                        )
                                    self._local_rejections = 0
                                    self._clear_recovery_request()
                                self._recovery_hypotheses = []
                                fusion_metrics = {
                                    "accepted": correction.accepted,
                                    "reason": decision,
                                    "predicted_position_innovation_map_px": (
                                        correction.predicted_position_innovation_m
                                    ),
                                    "predicted_yaw_innovation_deg": (
                                        correction.predicted_yaw_innovation_deg
                                    ),
                                    "applied_position_change_map_px": (
                                        correction.applied_position_change_m
                                    ),
                                    "applied_yaw_change_deg": (
                                        correction.applied_yaw_change_deg
                                    ),
                                }
            self._last_global_fix = {
                "x": global_fix.x,
                "y": global_fix.y,
                "yaw_deg": global_fix.yaw_deg,
                "scale": global_fix.scale,
                "score": global_fix.score,
                "margin": global_fix.margin,
                "elapsed_ms": global_fix.elapsed_ms,
                "decision": decision,
                "valid": global_fix.valid,
                "rejection_reasons": list(global_fix.rejection_reasons),
                "ratio_match_count": global_fix.ratio_match_count,
                "inlier_count": global_fix.inlier_count,
                "inlier_ratio": global_fix.inlier_ratio,
                "reprojection_p95_localization_px": global_fix.reprojection_p95_px,
                "feature_correlation_center_agreement_px": global_fix.center_agreement_px,
                "alternatives": list(global_fix.alternatives),
                "search_bounds_xyxy": list(global_fix.search_bounds_xyxy)
                if global_fix.search_bounds_xyxy is not None
                else None,
                "search_area_fraction": global_fix.search_area_fraction,
                "fusion": fusion_metrics,
                "map_mode_id": self._fix_mode(global_fix),
                "mode_likelihoods": self._fix_mode_likelihoods(global_fix),
                "host_time_ns": timestamp_ns,
            }

        route_tracking_fresh = False
        if self.route_visual_tracker is not None and self.fusion._state is not None:
            if self.route_visual_tracker.previous_xy is None:
                state = self.fusion.state
                self.route_visual_tracker.seed(state.pose.x, state.pose.y)
            try:
                route_result = self.route_visual_tracker.track(
                    minimap, mask, timestamp_ns=timestamp_ns
                )
                self._last_route_tracking = dict(route_result)
                route_tracking_fresh = True
                if route_result.get("measurement_accepted"):
                    score = float(route_result.get("score") or 0.0)
                    quality = float(np.clip(score, 0.0, 1.0))
                    self.fusion.accept_position_measurement(
                        route_result["x"], route_result["y"], quality
                    )
                    self._local_rejections = 0
                else:
                    if not route_result.get("transition_waiting"):
                        self._local_rejections += 1
            except Exception as exc:
                self._last_route_tracking = {
                    "measurement_accepted": False,
                    "pose_available": True,
                    "held": True,
                    "decision": "held:route-visual-error",
                    "error": "{}: {}".format(type(exc).__name__, exc),
                    "route_role": "bounded-search-proposal-only",
                }
                route_tracking_fresh = True
                self._local_rejections += 1
        global_search_needed = (
            self.fusion._state is None or self._recovery_requested()
        )
        due = (
            self.last_global_ns is None
            or timestamp_ns - self.last_global_ns >= self.global_interval_ns
        )
        if global_search_needed and due and self._global_future is None:
            yaw_prior = self.fusion.state.pose.yaw_deg if self.fusion._state is not None else None
            search_center, search_radius = self._global_search()
            self._last_global_search = {
                "center_xy": list(search_center) if search_center else None,
                "radius_px": search_radius,
            }
            self._global_future = self._global_executor.submit(
                self._localize_global,
                minimap,
                mask,
                yaw_prior,
                search_center,
                search_radius,
            )
            self.last_global_ns = timestamp_ns
        representation_interval_ns = self.representation_interval_ns
        representation_controller = (
            (self._last_representation_observation or {}).get("controller") or {}
        )
        if representation_controller.get("transition_armed"):
            # Two visual confirmations should not each wait a quarter second
            # while current-layer position estimates are already deteriorating.
            representation_interval_ns = min(representation_interval_ns, int(1e9 / 30))
        representation_due = (
            self.last_representation_ns is None
            or timestamp_ns - self.last_representation_ns
            >= representation_interval_ns
        )
        representation_spatially_relevant = False
        if self.fusion._state is not None and self.transition_controller is not None:
            state = self.fusion.state
            representation_spatially_relevant = (
                self.transition_controller.observation_required(
                    (float(state.pose.x), float(state.pose.y)),
                    float(state.position_sigma_m),
                )
            )
        if (
            self.fusion._state is not None
            and self._active_map_mode_id is not None
            and representation_spatially_relevant
            and representation_due
            and self._representation_future is None
            and self._representation_executor is not None
        ):
            state = self.fusion.state
            canonical_xy = (float(state.pose.x), float(state.pose.y))
            search_radius = float(
                np.clip(state.position_sigma_m * 3.0, 40.0, 150.0)
            )
            self._representation_position_sigma = float(state.position_sigma_m)
            self._representation_future = self._representation_executor.submit(
                self.localizer.observe_modes,
                minimap.copy(),
                mask.copy(),
                canonical_xy,
                search_radius,
            )
            self.last_representation_ns = timestamp_ns
        # Overlap cursor processing with XY work, then wait only within the
        # remaining source-to-control budget. Late results stay asynchronous.
        if self._cursor_future is not None and not self._cursor_future.done():
            remaining_s = max(0.0, (timestamp_ns + 1e9/30 - time.perf_counter_ns())/1e9)
            if remaining_s:
                try:
                    self._cursor_future.result(timeout=remaining_s)
                except Exception:
                    # Timeouts stay pending; completed errors are recorded below.
                    pass
        cursor_pose_fresh = False
        if self._cursor_future is not None and self._cursor_future.done():
            try:
                result = self._cursor_future.result()
                public_result = (
                    result
                    if self._cursor_executor_kind == "process"
                    else getattr(
                        self.cursor_pose_estimator,
                        "public_result",
                        lambda value: value,
                    )(result)
                )
                published_host_time_ns = time.perf_counter_ns()
                source_frame_time_ns = public_result.get("session_time_ns")
                capture_to_publish_ms = (
                    max(
                        0.0,
                        (
                            published_host_time_ns - int(source_frame_time_ns)
                        )
                        / 1.0e6,
                    )
                    if source_frame_time_ns is not None
                    else None
                )
                submit_to_publish_ms = (
                    max(
                        0.0,
                        (
                            published_host_time_ns
                            - int(self._cursor_submit_host_time_ns)
                        )
                        / 1.0e6,
                    )
                    if self._cursor_submit_host_time_ns is not None
                    else None
                )
                public_result["published_host_time_ns"] = published_host_time_ns
                public_result["capture_to_publish_ms"] = capture_to_publish_ms
                public_result["submit_to_publish_ms"] = submit_to_publish_ms
                accepted = bool(
                    public_result.get("detected")
                    and float(public_result.get("confidence") or 0.0)
                    >= self.pose_confidence_min
                )
                decision = (
                    "accepted"
                    if accepted
                    else "rejected:cursor-detection-or-confidence"
                )
                result_time_ns = int(
                    public_result.get("session_time_ns") or timestamp_ns
                )
                result_angle = (
                    float(public_result["angle_screen_deg"])
                    if accepted
                    else None
                )
                measured_rate = None
                hard_rate = float(
                    self.cursor_motion_envelope.get(
                        "calibrated_turn_rate_p99_deg_s", 1440.0
                    )
                )
                if (
                    accepted
                    and self._cursor_last_angle is not None
                    and self._cursor_last_time_ns is not None
                ):
                    elapsed_s = (
                        result_time_ns - self._cursor_last_time_ns
                    ) / 1.0e9
                    if 0.005 <= elapsed_s <= 2.0:
                        measured_rate = _angle_difference_deg(
                            result_angle, self._cursor_last_angle
                        ) / elapsed_s
                        if abs(measured_rate) > hard_rate:
                            accepted = False
                            decision = "rejected:implausible-angular-rate"
                public_result["accepted"] = accepted
                public_result["decision"] = decision
                public_result["fresh_measurement"] = accepted
                public_result["held"] = bool(
                    not accepted and self._last_cursor_pose is not None
                )
                public_result["held_angle_screen_deg"] = (
                    float(self._last_cursor_pose["angle_screen_deg"])
                    if public_result["held"]
                    else None
                )
                public_result["measured_angular_rate_deg_s"] = measured_rate
                public_result["angular_rate_limit_deg_s"] = hard_rate
                public_result["capture_to_publish_within_33_3ms"] = bool(
                    accepted
                    and capture_to_publish_ms is not None
                    and capture_to_publish_ms <= (1000.0 / 30.0)
                )
                public_result["capture_to_publish_within_66_7ms"] = bool(
                    accepted
                    and capture_to_publish_ms is not None
                    and capture_to_publish_ms <= (1000.0 / 15.0)
                )
                self._last_cursor_measurement = public_result
                if accepted:
                    self._last_cursor_pose = public_result
                    if measured_rate is not None:
                        self._cursor_angular_velocity_deg_s = (
                            0.5 * self._cursor_angular_velocity_deg_s
                            + 0.5 * measured_rate
                        )
                        normal_threshold = float(
                            self.cursor_motion_envelope.get(
                                "normal_turn_rate_p95_deg_s",
                                hard_rate * 0.25,
                            )
                        )
                        self._cursor_tracking_state = (
                            "turning"
                            if abs(measured_rate) > normal_threshold
                            else "stable"
                        )
                    else:
                        self._cursor_tracking_state = "stable"
                    self._cursor_last_angle = result_angle
                    self._cursor_last_time_ns = result_time_ns
                    self._cursor_rejections = 0
                else:
                    self._cursor_rejections += 1
                    self._cursor_tracking_state = "recovering"
                self._cursor_error = None
                cursor_pose_fresh = True
            except Exception as exc:
                self._cursor_error = "{}: {}".format(type(exc).__name__, exc)
                self._cursor_rejections += 1
                self._cursor_tracking_state = "recovering"
            self._cursor_future = None
            self._cursor_submit_host_time_ns = None
        self.sequence += 1
        cursor_pose_output = (
            dict(self._last_cursor_measurement)
            if self._last_cursor_measurement
            else None
        )
        cursor_measurement_fresh_accepted = bool(
            cursor_pose_fresh
            and cursor_pose_output
            and cursor_pose_output.get("accepted")
        )
        cursor_age_ms = None
        if cursor_pose_output and cursor_pose_output.get("session_time_ns") is not None:
            cursor_age_ms = max(
                0.0,
                (timestamp_ns - int(cursor_pose_output["session_time_ns"])) / 1.0e6,
            )
            cursor_pose_output["age_ms"] = cursor_age_ms
        accepted_cursor_age_ms = None
        if (
            self._last_cursor_pose
            and self._last_cursor_pose.get("session_time_ns") is not None
        ):
            accepted_cursor_age_ms = max(
                0.0,
                (
                    timestamp_ns
                    - int(self._last_cursor_pose["session_time_ns"])
                )
                / 1.0e6,
            )
        if self.fusion._state is None:
            pose = None
            mode = "INITIALIZING"
            position_sigma = None
            yaw_sigma = None
        else:
            state = self.fusion.state
            cursor_screen_deg = None
            player_heading_deg = None
            if (
                self._last_cursor_pose
                and self._last_cursor_pose.get("accepted")
                and accepted_cursor_age_ms is not None
                and accepted_cursor_age_ms <= 2000.0
            ):
                cursor_screen_deg = float(
                    self._last_cursor_pose["angle_screen_deg"]
                )
                player_heading_deg = float(
                    (state.pose.yaw_deg + cursor_screen_deg) % 360.0
                )
            pose = {
                "x": state.pose.x,
                "y": state.pose.y,
                "yaw_deg": player_heading_deg,
                "player_heading_map_deg": player_heading_deg,
                "map_alignment_deg": state.pose.yaw_deg,
                "cursor_screen_deg": cursor_screen_deg,
                "heading_source": (
                    "calibrated_cursor_plus_map_alignment"
                    if player_heading_deg is not None
                    else "unavailable"
                ),
            }
            if self._local_rejections >= self.relocalize_after_rejections:
                mode = "RELOCALIZING"
            elif self._local_rejections:
                mode = "HOLD"
            else:
                mode = state.mode
            position_sigma = state.position_sigma_m
            yaw_sigma = state.yaw_sigma_deg
            self.trail.append((state.pose.x, state.pose.y))
            self.trail = self.trail[-300:]
        control_ready_host_time_ns = time.perf_counter_ns()
        capture_to_control_ready_ms = max(
            0.0,
            (control_ready_host_time_ns - timestamp_ns) / 1.0e6,
        )
        xy_measurement_fresh_accepted = bool(
            (
                route_tracking_fresh
                and self._last_route_tracking
                and self._last_route_tracking.get("measurement_accepted")
            )
            if self.route_visual_tracker is not None
            else local_accepted
        )
        return {
            "sequence": self.sequence,
            "host_time_ns": timestamp_ns,
            "control_ready_host_time_ns": control_ready_host_time_ns,
            "capture_to_control_ready_ms": capture_to_control_ready_ms,
            "xy_measurement_fresh_accepted": xy_measurement_fresh_accepted,
            "control_output_available": pose is not None,
            "mode": mode,
            "pose": pose,
            "position_sigma_map_px": position_sigma,
            "yaw_sigma_deg": yaw_sigma,
            "scene_yaw": {
                "delta_deg": yaw.delta_deg,
                "confidence": yaw.confidence,
                "tracks": yaw.tracks,
                "inliers": yaw.inliers,
                "status": yaw.status,
            },
            "local_motion": {
                "map_content_shift_xy_px": list(local_shift),
                "response": float(local_response),
                "quality": local_quality,
                "accepted": local_accepted,
                "applied": local_motion_applied,
                "decision": local_decision,
                "rejection_streak": self._local_rejections,
                "recovery_requested": self._recovery_request_active,
                "response_min": self.local_response_min,
                "shift_limit_px": self.local_shift_max_px,
                "rotation_compensation_sign": compensation_sign,
                "map_alignment_delta_deg": alignment_delta_deg,
            },
            "route_tracking": dict(self._last_route_tracking)
            if self._last_route_tracking
            else None,
            "route_tracking_fresh": route_tracking_fresh,
            "route_assistance": dict(self._last_route_assistance)
            if self._last_route_assistance
            else None,
            "active_map_mode_id": self._active_map_mode_id,
            "map_scale": float(self.map_scale),
            "map_transition": dict(self._last_map_transition)
            if self._last_map_transition
            else None,
            "map_representation_observation": dict(
                self._last_representation_observation
            )
            if self._last_representation_observation
            else None,
            "map_representation_observation_fresh": (
                representation_observation_fresh
            ),
            "map_representation_observation_running": (
                self._representation_future is not None
            ),
            "map_representation_observation_spatially_relevant": (
                representation_spatially_relevant
            ),
            "map_representation_error": self._representation_error,
            "cursor_pose": cursor_pose_output,
            "cursor_pose_fresh": cursor_pose_fresh,
            "cursor_pose_measurement_fresh_accepted": (
                cursor_measurement_fresh_accepted
            ),
            "cursor_pose_deadline_33_3ms_met": bool(
                cursor_measurement_fresh_accepted
                and cursor_pose_output.get("capture_to_publish_within_33_3ms")
            ),
            "cursor_pose_deadline_66_7ms_met": bool(
                cursor_measurement_fresh_accepted
                and cursor_pose_output.get("capture_to_publish_within_66_7ms")
            ),
            "cursor_pose_running": self._cursor_future is not None,
            "cursor_executor": self._cursor_executor_kind,
            "cursor_error": self._cursor_error,
            "cursor_temporal_search": dict(self._last_cursor_search)
            if self._last_cursor_search
            else None,
            "cursor_tracking_state": self._cursor_tracking_state,
            "global_fix": dict(self._last_global_fix) if self._last_global_fix else None,
            "global_fix_fresh": global_fix is not None,
            "global_localization_running": self._global_future is not None,
            "global_error": self._global_error,
            "update_elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "trail": [list(item) for item in self.trail],
        }


def render_map_overlay(
    mosaic: np.ndarray,
    state: dict,
    size=(520, 360),
    route_points=None,
) -> np.ndarray:
    """Render a local observed-map viewport with pose, heading, trail, and text."""
    width, height = int(size[0]), int(size[1])
    canvas = np.full((height, width, 3), 12, np.uint8)
    pose = state.get("pose") or {}
    global_fix = state.get("global_fix") or {}
    panel_y = height - 80
    if not pose:
        map_h, map_w = mosaic.shape[:2]
        scale = min(width / float(map_w), panel_y / float(map_h))
        scaled_size = (
            max(1, int(round(map_w * scale))),
            max(1, int(round(map_h * scale))),
        )
        overview = cv2.resize(mosaic, scaled_size, interpolation=cv2.INTER_AREA)
        offset_x = (width - scaled_size[0]) // 2
        offset_y = (panel_y - scaled_size[1]) // 2
        canvas[
            offset_y : offset_y + scaled_size[1],
            offset_x : offset_x + scaled_size[0],
        ] = overview
        if route_points is not None:
            points = np.asarray(route_points, dtype=np.float64).reshape((-1, 2))
            if len(points) >= 2:
                projected = np.rint(
                    points * scale + np.asarray([offset_x, offset_y])
                ).astype(np.int32)
                cv2.polylines(
                    canvas,
                    [projected.reshape((-1, 1, 2))],
                    False,
                    (235, 95, 245),
                    2,
                    cv2.LINE_AA,
                )
        decision = str(global_fix.get("decision") or "waiting-for-first-candidate")
        if global_fix and global_fix.get("x") is not None:
            candidate = (
                offset_x + int(round(float(global_fix["x"]) * scale)),
                offset_y + int(round(float(global_fix["y"]) * scale)),
            )
            if decision.startswith("rejected"):
                cv2.line(
                    canvas,
                    (candidate[0] - 8, candidate[1] - 8),
                    (candidate[0] + 8, candidate[1] + 8),
                    (70, 70, 255),
                    3,
                    cv2.LINE_AA,
                )
                cv2.line(
                    canvas,
                    (candidate[0] - 8, candidate[1] + 8),
                    (candidate[0] + 8, candidate[1] - 8),
                    (70, 70, 255),
                    3,
                    cv2.LINE_AA,
                )
            else:
                cv2.circle(canvas, candidate, 8, (80, 205, 245), 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (0, panel_y), (width, height), (8, 16, 24), -1)
        cv2.putText(
            canvas,
            "LOCALIZING - no accepted pose",
            (10, panel_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (80, 220, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            decision[:72],
            (10, panel_y + 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (80, 80, 255) if decision.startswith("rejected") else (225, 235, 245),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "score {:.3f}  margin {:.3f}  inliers {}/{}".format(
                float(global_fix.get("score") or 0.0),
                float(global_fix.get("margin") or 0.0),
                int(global_fix.get("inlier_count") or 0),
                int(global_fix.get("ratio_match_count") or 0),
            ),
            (10, panel_y + 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (225, 235, 245),
            1,
            cv2.LINE_AA,
        )
        return canvas
    x, y = float(pose["x"]), float(pose["y"])
    map_h, map_w = mosaic.shape[:2]
    view_h = max(80, height - 90)
    view_w = width
    left = int(np.clip(round(x - view_w / 2), 0, max(0, map_w - view_w)))
    top = int(np.clip(round(y - view_h / 2), 0, max(0, map_h - view_h)))
    right, bottom = min(map_w, left + view_w), min(map_h, top + view_h)
    view = mosaic[top:bottom, left:right]
    canvas[: view.shape[0], : view.shape[1]] = view
    if route_points is not None:
        points = np.asarray(route_points, dtype=np.float64).reshape((-1, 2))
        if len(points) >= 2:
            projected = np.rint(points - np.asarray([left, top])).astype(
                np.int32
            )
            cv2.polylines(
                canvas,
                [projected.reshape((-1, 1, 2))],
                False,
                (235, 95, 245),
                3,
                cv2.LINE_AA,
            )
            for index in range(8, len(projected), 12):
                start = tuple(projected[index - 1])
                end = tuple(projected[index])
                if (
                    0 <= end[0] < view_w
                    and 0 <= end[1] < view_h
                    and np.linalg.norm(projected[index] - projected[index - 1]) > 1
                ):
                    cv2.arrowedLine(
                        canvas,
                        start,
                        end,
                        (235, 95, 245),
                        2,
                        cv2.LINE_AA,
                        tipLength=0.45,
                    )
    for tx, ty in state.get("trail") or ():
        point = (int(round(tx - left)), int(round(ty - top)))
        if 0 <= point[0] < view_w and 0 <= point[1] < view_h:
            cv2.circle(canvas, point, 1, (80, 210, 240), -1)
    global_decision = str(global_fix.get("decision") or "")
    if global_decision.startswith("rejected"):
        candidate = (
            int(round(float(global_fix.get("x") or 0.0) - left)),
            int(round(float(global_fix.get("y") or 0.0) - top)),
        )
        if 0 <= candidate[0] < view_w and 0 <= candidate[1] < view_h:
            cv2.line(
                canvas,
                (candidate[0] - 7, candidate[1] - 7),
                (candidate[0] + 7, candidate[1] + 7),
                (70, 70, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.line(
                canvas,
                (candidate[0] - 7, candidate[1] + 7),
                (candidate[0] + 7, candidate[1] - 7),
                (70, 70, 255),
                2,
                cv2.LINE_AA,
            )
    center = (int(round(x - left)), int(round(y - top)))
    player_heading = pose.get("player_heading_map_deg", pose.get("yaw_deg"))
    cv2.circle(canvas, center, 8, (245, 205, 60), 2, cv2.LINE_AA)
    if player_heading is not None:
        yaw = math.radians(float(player_heading))
        tip = (
            center[0] + int(math.cos(yaw) * 28),
            center[1] + int(math.sin(yaw) * 28),
        )
        cv2.arrowedLine(
            canvas, center, tip, (245, 205, 60), 3, cv2.LINE_AA, tipLength=0.35
        )
    cv2.rectangle(canvas, (0, panel_y), (width, height), (8, 16, 24), -1)
    local = state.get("local_motion") or {}
    scene = state.get("scene_yaw") or {}
    map_alignment = pose.get("map_alignment_deg", pose.get("yaw_deg"))
    heading_text = (
        "{:+.1f}".format(float(player_heading))
        if player_heading is not None
        else "unavailable"
    )
    lines = [
        "{}  x {:.1f}  y {:.1f}  player heading {} deg".format(
            state.get("mode", "?"), x, y, heading_text
        ),
        "map alignment {:+.1f} deg  cursor {}".format(
            float(map_alignment or 0.0),
            "{:+.1f} deg".format(float(pose["cursor_screen_deg"]))
            if pose.get("cursor_screen_deg") is not None
            else "unavailable",
        ),
        "global {:.3f} margin {:.3f}  local {:.3f}  yaw conf {:.3f}".format(float(global_fix.get("score") or 0), float(global_fix.get("margin") or 0), float(local.get("response") or 0), float(scene.get("confidence") or 0)),
        "update {:.1f} ms  global {:.1f} ms  sigma {:.1f}px / {:.1f}deg".format(float(state.get("update_elapsed_ms") or 0), float(global_fix.get("elapsed_ms") or 0), float(state.get("position_sigma_map_px") or 0), float(state.get("yaw_sigma_deg") or 0)),
    ]
    for index, line in enumerate(lines):
        cv2.putText(canvas, line, (10, panel_y + 17 + index * 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (225, 235, 245), 1, cv2.LINE_AA)
    return canvas


def render_minimap_route_overlay(
    route_points,
    state: dict,
    minimap_calibration: dict,
    crop_xywh,
) -> np.ndarray:
    """Project a demonstrated canonical route into live mini-map pixels."""

    _, _, width, height = tuple(map(int, crop_xywh))
    overlay = np.zeros((height, width, 4), np.uint8)
    pose = state.get("pose") or {}
    if route_points is None or not pose:
        return overlay
    scale = float(state.get("map_scale") or 0.0)
    alignment = pose.get("map_alignment_deg", pose.get("yaw_deg"))
    if scale <= 0.0 or alignment is None:
        return overlay
    points = np.asarray(route_points, dtype=np.float64).reshape((-1, 2))
    if len(points) < 2 or not np.all(np.isfinite(points)):
        return overlay
    minimap_calibration = normalize_minimap_geometry(
        minimap_calibration,
        minimap_crop_space([width, height]),
        allow_legacy=True,
    )
    center_model = minimap_calibration.get("rotation_center") or {}
    boundary = minimap_calibration["outer_boundary"]
    center = np.asarray(
        [
            float(center_model.get("x", boundary.get("center_x", width / 2.0))),
            float(center_model.get("y", boundary.get("center_y", height / 2.0))),
        ],
        dtype=np.float64,
    )
    radius = float(boundary.get("radius") or min(width, height) / 2.0)
    relative = points - np.asarray(
        [float(pose["x"]), float(pose["y"])], dtype=np.float64
    )
    angle = math.radians(float(alignment))
    cosine, sine = math.cos(angle), math.sin(angle)
    # Inverse of PoseFusionGate's local-to-canonical rotation.
    local = np.column_stack(
        (
            cosine * relative[:, 0] + sine * relative[:, 1],
            -sine * relative[:, 0] + cosine * relative[:, 1],
        )
    )
    projected = np.rint(center + local / scale).astype(np.int32)
    cv2.polylines(
        overlay,
        [projected.reshape((-1, 1, 2))],
        False,
        (235, 80, 250, 220),
        3,
        cv2.LINE_AA,
    )
    for index in range(6, len(projected), 10):
        start = tuple(projected[index - 1])
        end = tuple(projected[index])
        if np.linalg.norm(projected[index] - projected[index - 1]) > 1:
            cv2.arrowedLine(
                overlay,
                start,
                end,
                (235, 80, 250, 235),
                2,
                cv2.LINE_AA,
                tipLength=0.55,
            )
    circular_mask = np.zeros((height, width), np.uint8)
    cv2.circle(
        circular_mask,
        tuple(np.rint(center).astype(int)),
        max(1, int(round(radius - 2.0))),
        255,
        -1,
        cv2.LINE_AA,
    )
    overlay[:, :, 3] = cv2.bitwise_and(overlay[:, :, 3], circular_mask)
    return overlay


def render_route_trace_video_frame(
    frame: np.ndarray,
    mosaic: np.ndarray,
    state: dict,
    route_points,
    minimap_calibration: dict,
    crop_xywh,
    *,
    panel_opacity: float = 0.68,
) -> np.ndarray:
    """Compose only a game source frame and the route-tracing overlays."""

    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Route video needs one BGR game frame")
    output = frame.copy()
    frame_height, frame_width = output.shape[:2]
    crop_x, crop_y, crop_width, crop_height = tuple(map(int, crop_xywh))
    route_overlay = render_minimap_route_overlay(
        route_points,
        state,
        minimap_calibration,
        crop_xywh,
    )
    left = max(0, crop_x)
    top = max(0, crop_y)
    right = min(frame_width, crop_x + crop_width)
    bottom = min(frame_height, crop_y + crop_height)
    if right > left and bottom > top:
        overlay_left = left - crop_x
        overlay_top = top - crop_y
        overlay = route_overlay[
            overlay_top : overlay_top + (bottom - top),
            overlay_left : overlay_left + (right - left),
        ]
        alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
        base = output[top:bottom, left:right].astype(np.float32)
        foreground = overlay[:, :, :3].astype(np.float32)
        output[top:bottom, left:right] = np.clip(
            foreground * alpha + base * (1.0 - alpha), 0, 255
        ).astype(np.uint8)

    panel_width = max(180, min(360, int(round(frame_width * 0.24))))
    panel_height = max(120, int(round(panel_width * 2.0 / 3.0)))
    panel = render_map_overlay(
        mosaic,
        state,
        size=(panel_width, panel_height),
        route_points=route_points,
    )
    margin = max(6, int(round(min(frame_width, frame_height) * 0.012)))
    panel_left = max(0, frame_width - panel_width - margin)
    panel_top = margin
    panel_right = min(frame_width, panel_left + panel_width)
    panel_bottom = min(frame_height, panel_top + panel_height)
    panel = panel[: panel_bottom - panel_top, : panel_right - panel_left]
    opacity = float(np.clip(panel_opacity, 0.0, 1.0))
    output[panel_top:panel_bottom, panel_left:panel_right] = cv2.addWeighted(
        panel,
        opacity,
        output[panel_top:panel_bottom, panel_left:panel_right],
        1.0 - opacity,
        0.0,
    )
    return output
