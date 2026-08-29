"""Two-rate live localization using global map fixes and local visual motion."""

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from poc.pose_fusion import FusionConfig, Pose2D, PoseFusionGate
from poc.yaw_estimation import KltAngularYawEstimator, camera_matrix

from .cursor_pose import CursorPoseEstimator
from .minimap_verification import estimate_masked_shift


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


class GlobalMapLocalizer:
    """Feature-proposed, correlation-verified localization on a normalized map."""

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
        response = cv2.matchTemplate(
            self.map_gradient,
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
            peak_center = (location[0] + width / 2.0, location[1] + height / 2.0)
            peaks.append((float(peak_score), peak_center))
            cv2.circle(suppressed, location, suppression_radius, -1.0, -1)
        score = peaks[0][0]
        margin = score - peaks[1][0]
        correlation_center = np.asarray(peaks[0][1], dtype=np.float64)
        center_agreement = float(np.linalg.norm(correlation_center - center))
        center_x = int(np.clip(round(correlation_center[0]), 0, self.coverage.shape[1] - 1))
        center_y = int(np.clip(round(correlation_center[1]), 0, self.coverage.shape[0] - 1))
        reasons = []
        if inlier_count < 6:
            reasons.append("too-few-geometric-inliers")
        if inlier_ratio < 0.60:
            reasons.append("low-inlier-ratio")
        if reprojection_p95 > 3.0:
            reasons.append("high-reprojection-error")
        if not 0.75 <= scale <= 1.30:
            reasons.append("scale-out-of-range")
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
        map_scale_x = math.hypot(
            self.localization_to_original[0, 0], self.localization_to_original[1, 0]
        )
        map_scale_y = math.hypot(
            self.localization_to_original[0, 1], self.localization_to_original[1, 1]
        )
        original_scale = scale * (map_scale_x + map_scale_y) / 2.0
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
        )


class MinimapExtractor:
    def __init__(self, crop_xywh, calibration: dict) -> None:
        self.crop_xywh = tuple(int(value) for value in crop_xywh)
        boundary = calibration.get("outer_boundary") or {}
        self.center = (
            float(boundary.get("center_x", self.crop_xywh[2] / 2.0)),
            float(boundary.get("center_y", self.crop_xywh[3] / 2.0)),
        )
        self.radius = float(boundary.get("radius", min(self.crop_xywh[2:]) * 0.4))

    def crop(self, frame: np.ndarray) -> np.ndarray:
        x, y, width, height = self.crop_xywh
        crop = frame[y : y + height, x : x + width]
        if crop.shape[:2] != (height, width):
            raise RuntimeError("Live frame does not contain the calibrated mini-map crop")
        return crop

    def extract(self, frame: np.ndarray):
        crop = self.crop(frame)
        _, _, width, height = self.crop_xywh
        cx, cy = self.center
        radius = int(round(self.radius))
        left = max(0, int(round(cx)) - radius)
        top = max(0, int(round(cy)) - radius)
        right = min(width, int(round(cx)) + radius)
        bottom = min(height, int(round(cy)) + radius)
        observation = crop[top:bottom, left:right].copy()
        oh, ow = observation.shape[:2]
        mask = np.zeros((oh, ow), np.uint8)
        cv2.circle(mask, (ow // 2, oh // 2), max(1, min(ow, oh) // 2 - 2), 255, -1)
        cv2.circle(mask, (ow // 2, oh // 2), max(5, int(round(radius * 0.20))), 0, -1)
        return observation, mask


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
        self._global_executor = ThreadPoolExecutor(max_workers=1)
        self._global_future = None
        self._last_global_fix = None
        self._global_error = None
        self.initial_consensus_count = max(1, int(initial_consensus_count))
        self._initial_hypotheses = []
        self.cursor_pose_estimator = cursor_pose_estimator
        self.cursor_interval_ns = int(float(cursor_interval_s) * 1.0e9)
        self.last_cursor_ns = None
        self._cursor_executor = (
            ThreadPoolExecutor(max_workers=1)
            if self.cursor_pose_estimator is not None
            else None
        )
        self._cursor_future = None
        self._last_cursor_pose = None
        self._cursor_error = None
        self.temporal_pose_search = bool(temporal_pose_search)
        self.pose_confidence_min = float(pose_confidence_min)
        calibration = getattr(self.cursor_pose_estimator, "calibration", {}) or {}
        dynamics = calibration.get("cursor_temporal_dynamics") or {}
        self.cursor_motion_envelope = dynamics.get(
            "recommended_runtime_envelope", {}
        )
        self._cursor_last_angle = None
        self._cursor_last_time_ns = None
        self._cursor_angular_velocity_deg_s = 0.0
        self._cursor_rejections = 0
        self._last_cursor_search = None

    def close(self) -> None:
        close_localizer = getattr(self.localizer, "close", None)
        if close_localizer is not None:
            close_localizer()
        self._global_executor.shutdown(wait=False)
        if self._cursor_executor is not None:
            self._cursor_executor.shutdown(wait=False)

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
        turn_rate = float(
            envelope.get("calibrated_turn_rate_p99_deg_s") or 720.0
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

    def update(self, frame: np.ndarray, host_time_ns: Optional[int] = None) -> dict:
        started = time.perf_counter()
        timestamp_ns = int(host_time_ns or time.perf_counter_ns())
        self._ensure_scene_estimator(frame)
        yaw = self.scene_estimator.update(frame)
        minimap, mask = self.extractor.extract(frame)
        local_shift = (0.0, 0.0)
        local_response = 0.0
        if self.previous_minimap is not None:
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
        else:
            compensation_sign = 0.0
        self.previous_minimap = minimap
        alignment_delta_deg = compensation_sign * yaw.delta_deg

        if self.fusion._state is not None and self.sequence:
            shift_x, shift_y = local_shift
            local_motion = (-shift_x * self.map_scale, -shift_y * self.map_scale)
            self.fusion.predict(local_motion, alignment_delta_deg)

        cursor_pose_fresh = False
        if self._cursor_future is not None and self._cursor_future.done():
            try:
                result = self._cursor_future.result()
                public_result = getattr(
                    self.cursor_pose_estimator, "public_result", lambda value: value
                )(result)
                accepted = bool(
                    public_result.get("detected")
                    and float(public_result.get("confidence") or 0.0)
                    >= self.pose_confidence_min
                )
                public_result["accepted"] = accepted
                public_result["decision"] = (
                    "accepted"
                    if accepted
                    else "rejected:cursor-detection-or-confidence"
                )
                self._last_cursor_pose = public_result
                if accepted:
                    result_time_ns = int(
                        public_result.get("session_time_ns") or timestamp_ns
                    )
                    result_angle = float(public_result["angle_screen_deg"])
                    if (
                        self._cursor_last_angle is not None
                        and self._cursor_last_time_ns is not None
                    ):
                        elapsed_s = (
                            result_time_ns - self._cursor_last_time_ns
                        ) / 1.0e9
                        if 0.005 <= elapsed_s <= 2.0:
                            measured_rate = _angle_difference_deg(
                                result_angle, self._cursor_last_angle
                            ) / elapsed_s
                            hard_rate = float(
                                self.cursor_motion_envelope.get(
                                    "calibrated_turn_rate_p99_deg_s", 1440.0
                                )
                            )
                            measured_rate = float(
                                np.clip(measured_rate, -hard_rate, hard_rate)
                            )
                            self._cursor_angular_velocity_deg_s = (
                                0.5 * self._cursor_angular_velocity_deg_s
                                + 0.5 * measured_rate
                            )
                    self._cursor_last_angle = result_angle
                    self._cursor_last_time_ns = result_time_ns
                    self._cursor_rejections = 0
                else:
                    self._cursor_rejections += 1
                self._cursor_error = None
                cursor_pose_fresh = True
            except Exception as exc:
                self._cursor_error = "{}: {}".format(type(exc).__name__, exc)
            self._cursor_future = None
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
            }
            cursor_crop = self.extractor.crop(frame).copy()
            if angle_prior is None:
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
            if not global_fix.valid:
                self._initial_hypotheses = []
                decision = "rejected-quality:" + ",".join(
                    global_fix.rejection_reasons
                )
            else:
                hypothesis = Pose2D(global_fix.x, global_fix.y, global_fix.yaw_deg)
                if self.fusion._state is None:
                    if self._initial_hypotheses:
                        previous = self._initial_hypotheses[-1]
                        position_delta = math.hypot(
                            global_fix.x - previous.x, global_fix.y - previous.y
                        )
                        yaw_delta = abs(
                            _angle_difference_deg(global_fix.yaw_deg, previous.yaw_deg)
                        )
                        scale_delta = abs(global_fix.scale - previous.scale) / max(
                            previous.scale, 1.0e-6
                        )
                        if (
                            position_delta > 30.0
                            or yaw_delta > 15.0
                            or scale_delta > 0.12
                        ):
                            self._initial_hypotheses = []
                    self._initial_hypotheses.append(global_fix)
                    count = len(self._initial_hypotheses)
                    if count >= self.initial_consensus_count:
                        rows = self._initial_hypotheses[-self.initial_consensus_count :]
                        initialized = Pose2D(
                            float(np.mean([item.x for item in rows])),
                            float(np.mean([item.y for item in rows])),
                            _circular_mean_deg([item.yaw_deg for item in rows]),
                        )
                        self.map_scale = float(np.mean([item.scale for item in rows]))
                        self.fusion.initialize(initialized)
                        self._initial_hypotheses = []
                        decision = "initialized-consensus"
                    else:
                        decision = "awaiting-consensus:{}/{}".format(
                            count, self.initial_consensus_count
                        )
                else:
                    correction = self.fusion.consider_absolute(hypothesis)
                    decision = (
                        correction.reason
                        if correction.accepted
                        else "rejected:" + correction.reason
                    )
                    if correction.accepted:
                        self.map_scale = global_fix.scale
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
                "host_time_ns": timestamp_ns,
            }
        due = self.last_global_ns is None or timestamp_ns - self.last_global_ns >= self.global_interval_ns
        if due and self._global_future is None:
            yaw_prior = self.fusion.state.pose.yaw_deg if self.fusion._state is not None else None
            self._global_future = self._global_executor.submit(
                self.localizer.localize,
                minimap.copy(),
                mask.copy(),
                yaw_prior,
            )
            self.last_global_ns = timestamp_ns
        self.sequence += 1
        cursor_pose_output = (
            dict(self._last_cursor_pose) if self._last_cursor_pose else None
        )
        cursor_age_ms = None
        if cursor_pose_output and cursor_pose_output.get("session_time_ns") is not None:
            cursor_age_ms = max(
                0.0,
                (timestamp_ns - int(cursor_pose_output["session_time_ns"])) / 1.0e6,
            )
            cursor_pose_output["age_ms"] = cursor_age_ms
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
                and cursor_age_ms is not None
                and cursor_age_ms <= 2000.0
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
            mode = state.mode
            position_sigma = state.position_sigma_m
            yaw_sigma = state.yaw_sigma_deg
            self.trail.append((state.pose.x, state.pose.y))
            self.trail = self.trail[-300:]
        return {
            "sequence": self.sequence,
            "host_time_ns": timestamp_ns,
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
                "rotation_compensation_sign": compensation_sign,
                "map_alignment_delta_deg": alignment_delta_deg,
            },
            "cursor_pose": cursor_pose_output,
            "cursor_pose_fresh": cursor_pose_fresh,
            "cursor_pose_running": self._cursor_future is not None,
            "cursor_error": self._cursor_error,
            "cursor_temporal_search": dict(self._last_cursor_search)
            if self._last_cursor_search
            else None,
            "global_fix": dict(self._last_global_fix) if self._last_global_fix else None,
            "global_fix_fresh": global_fix is not None,
            "global_localization_running": self._global_future is not None,
            "global_error": self._global_error,
            "update_elapsed_ms": (time.perf_counter() - started) * 1000.0,
            "trail": [list(item) for item in self.trail],
        }


def render_map_overlay(mosaic: np.ndarray, state: dict, size=(520, 360)) -> np.ndarray:
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
