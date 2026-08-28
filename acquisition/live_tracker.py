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

from .minimap_verification import estimate_masked_shift


def _gradient(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


@dataclass
class GlobalFix:
    x: float
    y: float
    yaw_deg: float
    scale: float
    score: float
    margin: float
    elapsed_ms: float


class GlobalMapLocalizer:
    """High-cost exhaustive masked template localization against a map mosaic."""

    def __init__(
        self,
        mosaic: np.ndarray,
        scales=(0.65, 0.8, 1.0, 1.2, 1.45),
        coarse_angle_step_deg: float = 15.0,
        refine_angle_step_deg: float = 2.0,
    ) -> None:
        if mosaic is None or mosaic.size == 0:
            raise ValueError("Global map mosaic is empty")
        self.mosaic = mosaic.copy()
        self.map_gradient = _gradient(mosaic)
        self.scales = tuple(float(value) for value in scales)
        self.coarse_angle_step_deg = float(coarse_angle_step_deg)
        self.refine_angle_step_deg = float(refine_angle_step_deg)
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

    def _evaluate(self, template, mask, scale: float, angle: float):
        transformed, transformed_mask = self._transform(template, mask, scale, angle)
        height, width = transformed.shape[:2]
        if height >= self.map_gradient.shape[0] or width >= self.map_gradient.shape[1]:
            return None
        if np.count_nonzero(transformed_mask) < 64:
            return None
        result = cv2.matchTemplate(
            self.map_gradient,
            transformed,
            cv2.TM_CCORR_NORMED,
            mask=transformed_mask,
        )
        result = np.nan_to_num(result, nan=-1.0, posinf=-1.0, neginf=-1.0)
        _, score, _, location = cv2.minMaxLoc(result)
        suppressed = result.copy()
        radius = max(8, min(width, height) // 3)
        cv2.circle(suppressed, location, radius, -1.0, -1)
        _, second, _, _ = cv2.minMaxLoc(suppressed)
        return {
            "x": float(location[0] + width / 2.0),
            "y": float(location[1] + height / 2.0),
            "scale": scale,
            "angle": angle,
            "score": float(score),
            "margin": float(score - second),
        }

    def localize(
        self,
        observation: np.ndarray,
        mask: np.ndarray,
        yaw_prior_deg: Optional[float] = None,
    ) -> GlobalFix:
        started = time.perf_counter()
        template = _gradient(observation)
        if yaw_prior_deg is None:
            angles = np.arange(-180.0, 180.0, self.coarse_angle_step_deg)
        else:
            angles = np.arange(
                yaw_prior_deg - 30.0,
                yaw_prior_deg + 30.0 + 0.1,
                self.coarse_angle_step_deg,
            )
        candidates = []
        for scale in self.scales:
            for angle in angles:
                if self._cancel.is_set():
                    raise RuntimeError("Global localization canceled")
                value = self._evaluate(template, mask, scale, float(angle))
                if value is not None:
                    candidates.append(value)
        if not candidates:
            raise RuntimeError("No global localization template fits the map mosaic")
        coarse = max(candidates, key=lambda item: item["score"])
        scale_values = sorted(
            set(
                max(0.1, coarse["scale"] * factor)
                for factor in (0.92, 0.96, 1.0, 1.04, 1.08)
            )
        )
        refined = []
        for scale in scale_values:
            for angle in np.arange(
                coarse["angle"] - self.coarse_angle_step_deg,
                coarse["angle"] + self.coarse_angle_step_deg + 0.1,
                self.refine_angle_step_deg,
            ):
                if self._cancel.is_set():
                    raise RuntimeError("Global localization canceled")
                value = self._evaluate(template, mask, scale, float(angle))
                if value is not None:
                    refined.append(value)
        best = max(refined or candidates, key=lambda item: item["score"])
        return GlobalFix(
            best["x"],
            best["y"],
            ((best["angle"] + 180.0) % 360.0) - 180.0,
            best["scale"],
            best["score"],
            best["margin"],
            (time.perf_counter() - started) * 1000.0,
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

    def extract(self, frame: np.ndarray):
        x, y, width, height = self.crop_xywh
        crop = frame[y : y + height, x : x + width]
        if crop.shape[:2] != (height, width):
            raise RuntimeError("Live frame does not contain the calibrated mini-map crop")
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

    def close(self) -> None:
        close_localizer = getattr(self.localizer, "close", None)
        if close_localizer is not None:
            close_localizer()
        self._global_executor.shutdown(wait=False)

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
            for sign in (-1.0, 1.0):
                compensated = self._rotate(minimap, sign * yaw.delta_deg)
                shift, response = estimate_masked_shift(
                    self.previous_minimap, compensated, mask
                )
                trials.append((response, shift, sign))
            local_response, local_shift, compensation_sign = max(trials)
        else:
            compensation_sign = 0.0
        self.previous_minimap = minimap

        if self.fusion._state is not None and self.sequence:
            shift_x, shift_y = local_shift
            local_motion = (-shift_x * self.map_scale, -shift_y * self.map_scale)
            self.fusion.predict(local_motion, yaw.delta_deg)

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
            self.map_scale = global_fix.scale
            hypothesis = Pose2D(global_fix.x, global_fix.y, global_fix.yaw_deg)
            if self.fusion._state is None:
                self.fusion.initialize(hypothesis)
                decision = "initialized"
            else:
                correction = self.fusion.consider_absolute(hypothesis)
                decision = correction.reason if correction.accepted else "rejected:" + correction.reason
            self._last_global_fix = {
                "x": global_fix.x,
                "y": global_fix.y,
                "yaw_deg": global_fix.yaw_deg,
                "scale": global_fix.scale,
                "score": global_fix.score,
                "margin": global_fix.margin,
                "elapsed_ms": global_fix.elapsed_ms,
                "decision": decision,
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
        if self.fusion._state is None:
            pose = None
            mode = "INITIALIZING"
            position_sigma = None
            yaw_sigma = None
        else:
            state = self.fusion.state
            pose = {"x": state.pose.x, "y": state.pose.y, "yaw_deg": state.pose.yaw_deg}
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
            },
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
    if not pose:
        cv2.putText(canvas, "LOCALIZING...", (22, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 220, 230), 2, cv2.LINE_AA)
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
    center = (int(round(x - left)), int(round(y - top)))
    yaw = math.radians(float(pose["yaw_deg"]))
    tip = (center[0] + int(math.cos(yaw) * 28), center[1] + int(math.sin(yaw) * 28))
    cv2.circle(canvas, center, 8, (60, 235, 110), 2, cv2.LINE_AA)
    cv2.arrowedLine(canvas, center, tip, (60, 235, 110), 3, cv2.LINE_AA, tipLength=0.35)
    panel_y = height - 80
    cv2.rectangle(canvas, (0, panel_y), (width, height), (8, 16, 24), -1)
    global_fix = state.get("global_fix") or {}
    local = state.get("local_motion") or {}
    scene = state.get("scene_yaw") or {}
    lines = [
        "{}  x {:.1f}  y {:.1f}  yaw {:+.1f} deg".format(state.get("mode", "?"), x, y, float(pose["yaw_deg"])),
        "global {:.3f} margin {:.3f}  local {:.3f}  yaw conf {:.3f}".format(float(global_fix.get("score") or 0), float(global_fix.get("margin") or 0), float(local.get("response") or 0), float(scene.get("confidence") or 0)),
        "update {:.1f} ms  global {:.1f} ms  sigma {:.1f}px / {:.1f}deg".format(float(state.get("update_elapsed_ms") or 0), float(global_fix.get("elapsed_ms") or 0), float(state.get("position_sigma_map_px") or 0), float(state.get("yaw_sigma_deg") or 0)),
    ]
    for index, line in enumerate(lines):
        cv2.putText(canvas, line, (10, panel_y + 20 + index * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (225, 235, 245), 1, cv2.LINE_AA)
    return canvas
