"""Staged HIK-camera/Android-display rig calibration workflow."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np

from ...commented_yaml import (
    HIK_CONFIG_COMMENTS,
    HIK_CONFIG_HEADER,
    write_commented_yaml,
)
from ..app.device_adapters import CameraConfiguration
from ..app.phone_target import LocalPhoneTargetServer, PhoneTargetAdapter, Presentation
from ..bundle import build_calibration, write_calibration_bundle
from ..data_matrix_readability import grade_data_matrix_decode, render_data_matrix_target
from ..geometry import CharucoLayout, detect_charuco_correspondences, estimate_screen_geometry
from ..image_quality import measure_slanted_edge_esfr
from .algorithms import (
    BlackLevelObservation,
    ExposureObservation,
    camera_roi_for_screen_region,
    camera_visible_screen_region,
    charuco_orientation_evidence,
    choose_black_level,
    choose_exposure,
    detect_focus_pose_frame,
    estimate_focus_target_pose,
    laplacian_sharpness,
    manual_white_balance_ratios,
    refresh_quantized_exposure_us,
    temporal_white_statistics,
    temporal_black_statistics,
    white_statistics,
)
from .driver import HikMvsCameraAdapter
from .display import AdbDisplayTarget
from .patterns import (
    camera_white_mask,
    focus_edge_regions,
    focus_frame_rect,
    focus_pattern,
    tinted,
    white_patch,
)
from .phone import AdbPhoneSession, PhoneMetrics


Progress = Callable[[str], None]
CompleteGrader = Callable[[np.ndarray, str, Mapping[str, Any]], Mapping[str, Any]]

DATA_MATRIX_ACCEPTANCE_RATE = 0.95
DATA_MATRIX_MINIMUM_TRIALS = 20
DATA_MATRIX_DEFAULT_TRIALS = 40
DATA_MATRIX_MAX_PATTERNS_PER_SCREEN = 8
PANEL_FONT_SCALE = 0.50
PANEL_LINE_STEP_PX = 22


def _default_progress(message: str) -> None:
    print(message, flush=True)


def screen_filling_charuco_layout(screen_size_px: Sequence[int]) -> CharucoLayout:
    """Choose square counts matching the screen aspect without stretching cells."""

    width, height = map(int, screen_size_px)
    if min(width, height) <= 0:
        raise ValueError("Screen size must be positive")
    ratio = Fraction(width, height).limit_denominator(20)
    squares_x, squares_y = int(ratio.numerator), int(ratio.denominator)
    scale = max(1, (5 + min(squares_x, squares_y) - 1) // min(squares_x, squares_y))
    squares_x *= scale
    squares_y *= scale
    return CharucoLayout(
        (width, height),
        squares_x=squares_x,
        squares_y=squares_y,
        margin_px=(0, 0),
    )


def cross_source_alignment_evidence(
    adb_crop: np.ndarray,
    hik_rectified: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Compare already-aligned ADB and HIK views without fitting a transform."""

    if adb_crop is None or hik_rectified is None:
        raise ValueError("Cross-source images are required")
    if adb_crop.shape[:2] != hik_rectified.shape[:2]:
        raise ValueError(
            "Cross-source image sizes differ: {} versus {}".format(
                adb_crop.shape[:2], hik_rectified.shape[:2]
            )
        )
    if valid_mask.shape[:2] != adb_crop.shape[:2]:
        raise ValueError("Cross-source valid mask has the wrong size")
    selected = np.asarray(valid_mask) > 0
    if int(np.count_nonzero(selected)) < 64:
        raise ValueError("Cross-source valid mask contains too few pixels")

    adb_gray = cv2.cvtColor(adb_crop, cv2.COLOR_BGR2GRAY)
    hik_gray = cv2.cvtColor(hik_rectified, cv2.COLOR_BGR2GRAY)
    adb_blur = cv2.GaussianBlur(adb_gray, (5, 5), 0.8)
    hik_blur = cv2.GaussianBlur(hik_gray, (5, 5), 0.8)
    adb_values = adb_blur[selected].astype(np.float64)
    hik_values = hik_blur[selected].astype(np.float64)
    if min(float(np.std(adb_values)), float(np.std(hik_values))) <= 1.0e-6:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(adb_values, hik_values)[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0

    adb_threshold = float(np.percentile(adb_values, 50))
    hik_threshold = float(np.percentile(hik_values, 50))
    adb_binary = adb_blur >= adb_threshold
    hik_binary = hik_blur >= hik_threshold
    binary_agreement = float(np.mean(adb_binary[selected] == hik_binary[selected]))

    adb_edges = cv2.Canny(adb_blur, 40, 120) > 0
    hik_edges = cv2.Canny(hik_blur, 40, 120) > 0
    kernel = np.ones((3, 3), np.uint8)
    adb_dilated = cv2.dilate(adb_edges.astype(np.uint8), kernel) > 0
    hik_dilated = cv2.dilate(hik_edges.astype(np.uint8), kernel) > 0
    adb_edge_count = int(np.count_nonzero(adb_edges & selected))
    hik_edge_count = int(np.count_nonzero(hik_edges & selected))
    adb_supported = (
        float(np.count_nonzero(adb_edges & hik_dilated & selected)) / adb_edge_count
        if adb_edge_count
        else 0.0
    )
    hik_supported = (
        float(np.count_nonzero(hik_edges & adb_dilated & selected)) / hik_edge_count
        if hik_edge_count
        else 0.0
    )
    edge_overlap = 0.5 * (adb_supported + hik_supported)
    confidence = float(
        np.clip(
            0.50 * max(0.0, correlation)
            + 0.25 * binary_agreement
            + 0.25 * edge_overlap,
            0.0,
            1.0,
        )
    )

    adb_normalized = cv2.normalize(adb_blur, None, 0, 255, cv2.NORM_MINMAX)
    hik_normalized = cv2.normalize(hik_blur, None, 0, 255, cv2.NORM_MINMAX)
    difference = cv2.absdiff(adb_normalized, hik_normalized)
    heatmap = cv2.applyColorMap(difference, cv2.COLORMAP_TURBO)
    overlay = np.zeros((*adb_gray.shape, 3), np.uint8)
    overlay[:, :, 2] = adb_edges.astype(np.uint8) * 255
    overlay[:, :, 0] = hik_edges.astype(np.uint8) * 255
    overlay[:, :, 1] = hik_edges.astype(np.uint8) * 255
    invalid = ~selected
    heatmap[invalid] = 0
    overlay[invalid] = 0
    side_by_side = np.hstack([adb_crop, hik_rectified])
    metrics = {
        "confidence": confidence,
        "grayscale_correlation": correlation,
        "binary_agreement": binary_agreement,
        "edge_overlap": edge_overlap,
        "adb_edge_supported_fraction": adb_supported,
        "hik_edge_supported_fraction": hik_supported,
        "valid_pixel_fraction": float(np.mean(selected)),
    }
    images = {
        "adb_visible_crop.png": adb_crop,
        "hik_rectified.png": hik_rectified,
        "edge_overlay_adb_red_hik_cyan.png": overlay,
        "normalized_difference_heatmap.png": heatmap,
        "side_by_side_adb_then_hik.png": side_by_side,
        "valid_mask.png": np.asarray(valid_mask, dtype=np.uint8),
    }
    return metrics, images


def _load_callable(specification: str):
    module_name, separator, attribute = str(specification).partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Plugin must be in module:callable form")
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError("Plugin {} is not callable".format(specification))
    return value


@dataclass(frozen=True)
class HikCalibrationOptions:
    camera_id: str
    phone_serial: str
    output_directory: Path
    camera_width_px: int = 2448
    camera_height_px: int = 2048
    camera_fps: float = 30.0
    target_port: int = 8765
    display_component: Optional[str] = None
    operation_timeout_seconds: float = 8.0
    refresh_hz_override: Optional[float] = None
    maximum_shutter_multiplier: int = 2
    maximum_exposure_periods: int = 2
    exposure_noise_frames: int = 4
    geometry_frames: int = 12
    settle_frames: int = 3
    headless: bool = False
    save_without_prompt: bool = False
    grade_data_matrix: bool = False
    data_matrix_trials_per_size: int = DATA_MATRIX_DEFAULT_TRIALS
    data_matrix_initial_module_px: int = 1
    complete_grader_plugin: Optional[str] = None

    def __post_init__(self) -> None:
        if self.maximum_shutter_multiplier not in (2, 3):
            raise ValueError("Maximum shutter multiplier must be 2 or 3")
        if self.maximum_exposure_periods not in (1, 2, 3):
            raise ValueError("Maximum exposure periods must be 1, 2, or 3")
        if min(
            self.camera_width_px,
            self.camera_height_px,
            self.geometry_frames,
            self.exposure_noise_frames,
        ) <= 0:
            raise ValueError("Camera dimensions and geometry frame count must be positive")
        if self.data_matrix_trials_per_size < DATA_MATRIX_MINIMUM_TRIALS:
            raise ValueError(
                "Data Matrix trial count must be at least {}"
                .format(DATA_MATRIX_MINIMUM_TRIALS)
            )
        if self.operation_timeout_seconds <= 0:
            raise ValueError("Operation timeout must be positive")


class HikRigCalibrationSession:
    """Own the camera, target server, and phone for one bounded calibration."""

    def __init__(
        self,
        options: HikCalibrationOptions,
        camera: Optional[HikMvsCameraAdapter] = None,
        phone: Optional[AdbPhoneSession] = None,
        target: Optional[PhoneTargetAdapter] = None,
        progress: Progress = _default_progress,
    ) -> None:
        self.options = options
        self.camera = camera or HikMvsCameraAdapter()
        self.phone = phone or AdbPhoneSession(options.phone_serial)
        self.target = target or AdbDisplayTarget(
            self.phone,
            component=options.display_component,
            presentation_timeout_seconds=options.operation_timeout_seconds,
        )
        self.progress = progress
        self.phone_metrics: Optional[PhoneMetrics] = None
        self.phone_display_brightness: Optional[Dict[str, Any]] = None
        self.charuco_layout: Optional[CharucoLayout] = None
        self.viewer_metrics: Dict[str, Any] = {}
        self.camera_metadata: Dict[str, Any] = {}
        self.camera_controls: Dict[str, Any] = {}
        self.correspondences: Optional[Dict[str, Any]] = None
        self.geometry = None
        self.visible_region: Optional[Dict[str, Any]] = None
        self.orientation_evidence: Optional[Dict[str, Any]] = None
        self.white_mask: Optional[np.ndarray] = None
        self.exposure: Optional[ExposureObservation] = None
        self.exposure_observations: List[ExposureObservation] = []
        self.black_level: Optional[int] = None
        self.black_level_observations: List[BlackLevelObservation] = []
        self.white_balance: Optional[Dict[str, Any]] = None
        self.final_white_statistics: Optional[Dict[str, Any]] = None
        self.white_balance_attempts: List[Dict[str, Any]] = []
        self.calibration_warnings: List[str] = []
        self.auto_imaging_seed: Optional[Dict[str, Any]] = None
        self.auto_target_image: Optional[np.ndarray] = None
        self.auto_result_frame: Optional[np.ndarray] = None
        self.auto_phone_screenshot: Optional[np.ndarray] = None
        self.cv_verification: Optional[Dict[str, Any]] = None
        self.transport_benchmark: Optional[Dict[str, Any]] = None
        self.latency_benchmark: Optional[Dict[str, Any]] = None
        self.cross_source_check: Optional[Dict[str, Any]] = None
        self.hardware_roi: Optional[List[int]] = None
        self.focus_history: List[Dict[str, Any]] = []
        self.focus_pose_history: List[Dict[str, Any]] = []
        self._focus_focal_length_px: Optional[float] = None
        self._focus_focal_candidates_px: List[float] = []
        self._focus_geometry_changed = False
        self.data_matrix_result: Optional[Dict[str, Any]] = None
        self.data_matrix_evidence_directory: Optional[Path] = None
        self.data_matrix_failure_evidence: List[Dict[str, Any]] = []
        self.last_frame: Optional[np.ndarray] = None
        self._preview_window = "HIK calibration - live camera"
        self._preview_created = False
        self._preview_disabled = bool(options.headless)
        self._preview_stage = "Starting"
        self._preview_settings: Dict[str, Any] = {}
        self._opened = False
        self._saved = False

    @staticmethod
    def _required(value, name: str):
        if value is None:
            raise RuntimeError("Calibration stage requires {}".format(name))
        return value

    def _warn(self, message: str) -> None:
        value = str(message)
        self.calibration_warnings.append(value)
        self.progress("Warning: {}".format(value))

    def _capture_balanced_white(
        self, white_mask: np.ndarray, source: str
    ) -> Dict[str, Any]:
        self._capture_settled(white_mask)
        frames = [
            self.camera.read().image
            for _ in range(int(self.options.exposure_noise_frames))
        ]
        self.last_frame = frames[-1].copy()
        statistics = temporal_white_statistics(frames, white_mask)
        mean = float(np.mean(statistics["mean_bgr"]))
        maximum_clipped = float(max(statistics["clipped_fraction_bgr"]))
        target = 255.0 * 0.90
        tolerance = 255.0 * 0.02
        return {
            **statistics,
            "mean_dn": mean,
            "maximum_clipped_fraction": maximum_clipped,
            "target_dn": target,
            "tolerance_dn": tolerance,
            "within_preferred_range": (
                abs(mean - target) <= tolerance and maximum_clipped <= 0.05
            ),
            "source": str(source),
            "verified_after_white_balance": True,
        }

    @staticmethod
    def _exposure_observation_dict(
        row: ExposureObservation,
    ) -> Dict[str, Any]:
        return {
            **row.__dict__,
            "brightness": row.brightness,
            "white_balance_reference_brightness": (
                row.white_balance_reference_brightness
            ),
            "maximum_clipped_fraction": row.maximum_clipped_fraction,
            "exposure_refresh_periods": row.exposure_refresh_periods,
        }

    @staticmethod
    def _data_matrix_cannot_qualify(
        passed: int,
        completed: int,
        planned: int,
        required_rate: float = DATA_MATRIX_ACCEPTANCE_RATE,
    ) -> bool:
        """Return true once perfect remaining trials cannot reach the target."""

        passed, completed, planned = int(passed), int(completed), int(planned)
        if planned <= 0 or completed < 0 or completed > planned:
            raise ValueError("Data Matrix trial counts are invalid")
        if passed < 0 or passed > completed:
            raise ValueError("Data Matrix pass count is invalid")
        maximum_final_rate = (passed + planned - completed) / float(planned)
        return maximum_final_rate + 1.0e-12 < float(required_rate)

    @staticmethod
    def _data_matrix_decode_succeeded(row: Mapping[str, Any]) -> bool:
        """Return the operational exact-payload decode result for one pattern."""

        exact = bool(row.get("exact_payload_decoded", True))
        if "decode_success" in row:
            decoded = bool(row["decode_success"])
        elif "reference_decode_succeeded" in row:
            decoded = bool(row["reference_decode_succeeded"])
        else:
            decoded = float(row.get("grade", 0.0)) >= 4.0
        return bool(exact and decoded)

    @staticmethod
    def _desktop_work_area() -> Sequence[int]:
        """Return the usable desktop size, excluding the Windows taskbar."""

        try:
            import ctypes
            from ctypes import wintypes

            rectangle = wintypes.RECT()
            if ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(rectangle), 0):
                return [
                    max(640, int(rectangle.right - rectangle.left)),
                    max(480, int(rectangle.bottom - rectangle.top)),
                ]
        except (AttributeError, OSError):
            pass
        return [1280, 720]

    @staticmethod
    def _fit_complete_view(
        image: np.ndarray, box_width: int, box_height: int
    ) -> np.ndarray:
        """Fit the complete image inside a fixed pane without cropping."""

        if image is None or image.size == 0:
            raise ValueError("Complete-view image must be non-empty")
        box_width, box_height = int(box_width), int(box_height)
        if min(box_width, box_height) <= 0:
            raise ValueError("Complete-view pane must be positive")
        scale = min(
            box_width / float(image.shape[1]),
            box_height / float(image.shape[0]),
        )
        width = max(1, int(round(image.shape[1] * scale)))
        height = max(1, int(round(image.shape[0] * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST
        resized = cv2.resize(image, (width, height), interpolation=interpolation)
        canvas = np.full((box_height, box_width, 3), 24, np.uint8)
        left = (box_width - width) // 2
        top = (box_height - height) // 2
        canvas[top : top + height, left : left + width] = resized
        return canvas

    def _set_preview_stage(self, stage: str, **settings: Any) -> None:
        self._preview_stage = str(stage)
        self._preview_settings = dict(settings)

    @staticmethod
    def _format_metric(value: Any, digits: int = 2) -> str:
        """Format an optional finite measurement without failing the live UI."""

        try:
            number = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if not np.isfinite(number):
            return "n/a"
        return ("{:.%df}" % int(digits)).format(number)

    @staticmethod
    def _wrap_panel_lines(
        lines: Sequence[str],
        maximum_width_px: int,
        font_scale: float,
        thickness: int = 1,
    ) -> List[str]:
        """Wrap OpenCV text using its measured pixel width, preserving paragraphs."""

        maximum_width_px = max(40, int(maximum_width_px))
        wrapped: List[str] = []
        for value in lines:
            words = []
            for word in str(value).split():
                remainder = word
                while remainder:
                    fitting = ""
                    for character in remainder:
                        candidate = fitting + character
                        width = cv2.getTextSize(
                            candidate,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            float(font_scale),
                            int(thickness),
                        )[0][0]
                        if fitting and width > maximum_width_px:
                            break
                        fitting = candidate
                    words.append(fitting)
                    remainder = remainder[len(fitting) :]
            if not words:
                wrapped.append("")
                continue
            current = words[0]
            for word in words[1:]:
                candidate = current + " " + word
                width = cv2.getTextSize(
                    candidate, cv2.FONT_HERSHEY_SIMPLEX, float(font_scale), int(thickness)
                )[0][0]
                if width <= maximum_width_px:
                    current = candidate
                else:
                    wrapped.append(current)
                    current = word
            wrapped.append(current)
        return wrapped

    @classmethod
    def _draw_text_panel(
        cls,
        canvas: np.ndarray,
        panel_left: int,
        panel_width: int,
        lines: Sequence[str],
    ) -> List[str]:
        """Draw a fixed-scale text panel and clip overflow without eye-straining zoom."""

        left = int(panel_left) + 14
        usable_width = max(40, int(panel_width) - 28)
        selected_scale = PANEL_FONT_SCALE
        selected_step = PANEL_LINE_STEP_PX
        selected_lines = cls._wrap_panel_lines(
            lines, usable_width, selected_scale
        )
        maximum_lines = max(1, (canvas.shape[0] - 12) // selected_step)
        selected_lines = selected_lines[:maximum_lines]
        for index, line in enumerate(selected_lines):
            cv2.putText(
                canvas,
                line,
                (left, 24 + index * selected_step),
                cv2.FONT_HERSHEY_SIMPLEX,
                selected_scale,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )
        return selected_lines

    def _preview_lines(
        self,
        frame: np.ndarray,
        sample_metadata: Optional[Mapping[str, Any]] = None,
        **settings: Any
    ) -> List[str]:
        values = dict(self._preview_settings)
        values.update(settings)
        try:
            values.update(dict(self.camera.imaging_state()))
        except Exception:
            pass
        metadata = dict(sample_metadata or {})
        exposure_us = values.get("exposure_us")
        lines = [
            self._preview_stage,
            "{}  {}".format(
                self.camera_metadata.get("model", "HIK camera"),
                self.camera_metadata.get("serial", self.options.camera_id),
            ),
            "{}x{}  {:.2f} fps".format(
                frame.shape[1], frame.shape[0], float(self.camera_metadata.get("fps", 0.0))
            ),
            "Exposure mode: {}".format(values.get("exposure_mode", "manual")),
        ]
        if exposure_us is not None:
            shutter_hz = 1.0e6 / max(float(exposure_us), 1.0)
            lines.append("Exposure: {:.1f} us  ({:.1f} Hz)".format(float(exposure_us), shutter_hz))
        if values.get("gain") is not None:
            lines.append("Gain: {:.3f}".format(float(values["gain"])))
        if values.get("black_level") is not None:
            lines.append("Black level: {}".format(values["black_level"]))
        elif self.black_level is not None:
            lines.append("Black level: {}".format(self.black_level))
        white_balance = values.get("white_balance") or self.white_balance
        if white_balance:
            lines.append(
                "WB R/G/B: {}/{}/{}".format(
                    white_balance.get("ratio_red", "-"),
                    white_balance.get("ratio_green", "-"),
                    white_balance.get("ratio_blue", "-"),
                )
            )
        if values.get("charuco_corners") is not None:
            lines.append("ChArUco corners: {}".format(values["charuco_corners"]))
        lines.extend(
            [
                "Image mean/p95/max: {:.1f} / {:.1f} / {}".format(
                    float(np.mean(frame)), float(np.percentile(frame, 95)), int(np.max(frame))
                ),
                "Frame: {}".format(metadata.get("frame_number", metadata.get("frame_num", "-"))),
                "Close this window to hide preview.",
            ]
        )
        return lines

    def _preview_update(
        self,
        frame: np.ndarray,
        sample_metadata: Optional[Mapping[str, Any]] = None,
        **settings: Any
    ) -> None:
        """Show an unobstructed fit overview with settings in a side panel."""

        if self._preview_disabled:
            return
        try:
            if not self._preview_created:
                cv2.namedWindow(self._preview_window, cv2.WINDOW_NORMAL)
                self._preview_created = True
                self.progress(
                    "Live camera preview opened; close it to hide preview without cancelling calibration."
                )
            elif cv2.getWindowProperty(self._preview_window, cv2.WND_PROP_VISIBLE) < 1:
                self._close_preview(disable=True)
                return
            work_width, work_height = self._desktop_work_area()
            panel_width = min(400, max(300, int(work_width * 0.30)))
            maximum_width = max(320, int(work_width * 0.88) - panel_width)
            maximum_height = max(240, int(work_height * 0.82))
            scale = min(
                1.0,
                maximum_width / float(frame.shape[1]),
                maximum_height / float(frame.shape[0]),
            )
            image_width = max(1, int(round(frame.shape[1] * scale)))
            image_height = max(1, int(round(frame.shape[0] * scale)))
            image = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_AREA)
            canvas = np.full((image_height, image_width + panel_width, 3), 24, np.uint8)
            canvas[:, :image_width] = image
            self._draw_text_panel(
                canvas,
                image_width,
                panel_width,
                self._preview_lines(frame, sample_metadata, **settings),
            )
            cv2.resizeWindow(self._preview_window, canvas.shape[1], canvas.shape[0])
            cv2.imshow(self._preview_window, canvas)
            cv2.waitKey(1)
        except cv2.error as exc:
            self._preview_disabled = True
            self._preview_created = False
            self.progress("Live preview disabled after OpenCV window error: {}".format(exc))

    def _close_preview(self, disable: bool = False) -> None:
        if self._preview_created:
            try:
                cv2.destroyWindow(self._preview_window)
            except cv2.error:
                pass
        self._preview_created = False
        if disable:
            self._preview_disabled = True

    def _capture_settled(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Wait for frame brightness to converge after a camera/display change."""

        deadline = time.monotonic() + float(self.options.operation_timeout_seconds)
        window = []
        minimum = max(4, int(self.options.settle_frames))
        sample = None
        while time.monotonic() < deadline:
            sample = self.camera.read()
            image = sample.image
            self._preview_update(image, sample.metadata)
            if mask is not None:
                selected = np.asarray(mask) > 0
                brightness = float(np.mean(image[selected]))
            else:
                brightness = float(np.mean(image))
            window.append(brightness)
            window = window[-minimum:]
            if len(window) == minimum:
                tolerance = max(2.0, abs(float(np.mean(window))) * 0.02)
                if max(window) - min(window) <= tolerance:
                    self.last_frame = image.copy()
                    return image
        if sample is not None:
            self.last_frame = sample.image.copy()
        raise RuntimeError(
            "Camera frames did not stabilize within {:.1f}s; recent mean range {}"
            .format(self.options.operation_timeout_seconds, window)
        )

    def _capture_data_matrix_frame(self) -> np.ndarray:
        """Drain target-change backlog before accepting a settled grading frame."""

        discard_count = max(8, int(self.options.settle_frames) * 2)
        for _ in range(discard_count):
            sample = self.camera.read()
            self._preview_update(sample.image, sample.metadata)
        return self._capture_settled()

    @staticmethod
    def _data_matrix_screen_crop(
        symbol_rect_screen_xywh: Sequence[int],
        visible_region_xywh: Sequence[int],
        angle_deg: float,
        module_width_display_px: int,
        screen_size_px: Sequence[int],
    ) -> Sequence[int]:
        """Bound a rotated symbol with a quiet margin in phone coordinates."""

        x, y, width, height = map(int, symbol_rect_screen_xywh)
        region_x, region_y, region_width, region_height = map(
            int, visible_region_xywh
        )
        corners = np.asarray(
            [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
            dtype=np.float64,
        )
        if float(angle_deg):
            rotation = cv2.getRotationMatrix2D(
                (region_x + region_width / 2.0, region_y + region_height / 2.0),
                float(angle_deg),
                1.0,
            )
            corners = cv2.transform(corners.reshape((-1, 1, 2)), rotation).reshape(
                (-1, 2)
            )
        # The rendered raster contains the required one-module Data Matrix quiet
        # zone. Keep one additional module for interpolation/crop tolerance; the
        # remainder of its batch cell is white as well.
        margin = max(4, int(module_width_display_px))
        screen_width, screen_height = map(int, screen_size_px)
        left = max(0, int(np.floor(np.min(corners[:, 0]))) - margin)
        top = max(0, int(np.floor(np.min(corners[:, 1]))) - margin)
        right = min(screen_width, int(np.ceil(np.max(corners[:, 0]))) + margin)
        bottom = min(screen_height, int(np.ceil(np.max(corners[:, 1]))) + margin)
        if right <= left or bottom <= top:
            raise RuntimeError("Rotated Data Matrix crop is empty")
        return [left, top, right - left, bottom - top]

    def _compose_data_matrix_batch(
        self,
        screen_size_px: Sequence[int],
        visible_region_xywh: Sequence[int],
        module_width_display_px: int,
        specifications: Sequence[Mapping[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Pack as many independent grading symbols as fit in one phone image."""

        screen_width, screen_height = map(int, screen_size_px)
        region_x, region_y, region_width, region_height = map(
            int, visible_region_xywh
        )
        candidates = list(specifications)[:DATA_MATRIX_MAX_PATTERNS_PER_SCREEN]
        for batch_count in range(len(candidates), 0, -1):
            selected = candidates[:batch_count]
            for columns in range(1, batch_count + 1):
                rows = int(np.ceil(batch_count / float(columns)))
                if (rows - 1) * columns >= batch_count:
                    continue
                x_edges = [
                    region_x + (region_width * index) // columns
                    for index in range(columns + 1)
                ]
                y_edges = [
                    region_y + (region_height * index) // rows
                    for index in range(rows + 1)
                ]
                stimulus = np.full(
                    (screen_height, screen_width, 3), 127, dtype=np.uint8
                )
                items = []
                valid = True
                for item_index, specification in enumerate(selected):
                    column = item_index % columns
                    row = item_index // columns
                    cell = [
                        x_edges[column],
                        y_edges[row],
                        x_edges[column + 1] - x_edges[column],
                        y_edges[row + 1] - y_edges[row],
                    ]
                    trial = int(specification["trial_index"])
                    angle = float(specification["angle_deg"])
                    color = tuple(specification["color_bgr"])
                    intensity = float(specification["intensity"])
                    payload = "A{:X}{:X}".format(
                        int(module_width_display_px), trial
                    )
                    try:
                        target = render_data_matrix_target(
                            screen_size_px,
                            cell,
                            payload,
                            module_width_display_px,
                            quiet_zone_modules=1,
                            trial_id="dm-m{}-{:04d}".format(
                                module_width_display_px, trial
                            ),
                        )
                        decode_rect = self._data_matrix_screen_crop(
                            target.symbol_rect_screen_xywh,
                            cell,
                            angle,
                            module_width_display_px,
                            screen_size_px,
                        )
                    except (RuntimeError, ValueError):
                        valid = False
                        break
                    decode_x, decode_y, decode_width, decode_height = decode_rect
                    cell_x, cell_y, cell_width, cell_height = cell
                    if batch_count > 1 and not (
                        decode_x >= cell_x
                        and decode_y >= cell_y
                        and decode_x + decode_width <= cell_x + cell_width
                        and decode_y + decode_height <= cell_y + cell_height
                    ):
                        valid = False
                        break
                    patch = target.image[
                        cell_y : cell_y + cell_height,
                        cell_x : cell_x + cell_width,
                    ]
                    patch = tinted(patch, color, intensity)
                    if angle:
                        rotation = cv2.getRotationMatrix2D(
                            (cell_width / 2.0, cell_height / 2.0), angle, 1.0
                        )
                        border = tuple(
                            int(value)
                            for value in tinted(
                                np.full((1, 1, 3), 127, dtype=np.uint8),
                                color,
                                intensity,
                            )[0, 0]
                        )
                        patch = cv2.warpAffine(
                            patch,
                            rotation,
                            (cell_width, cell_height),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=border,
                        )
                    stimulus[
                        cell_y : cell_y + cell_height,
                        cell_x : cell_x + cell_width,
                    ] = patch
                    items.append(
                        {
                            "trial_index": trial,
                            "payload": payload,
                            "trial_id": target.trial_id,
                            "angle_deg": angle,
                            "color_bgr": list(color),
                            "intensity": intensity,
                            "cell_rect_screen_xywh": list(cell),
                            "decode_rect_screen_xywh": list(decode_rect),
                        }
                    )
                if valid and len(items) == batch_count:
                    return {
                        "image": stimulus,
                        "items": items,
                        "columns": columns,
                        "rows": rows,
                    }
        return None

    def _ensure_data_matrix_evidence_directory(self) -> Path:
        """Create a reviewable sibling artifact without requiring calibration save."""

        if self.data_matrix_evidence_directory is not None:
            return self.data_matrix_evidence_directory
        output = Path(self.options.output_directory).resolve()
        evidence = output.parent / "{}-data-matrix-evidence-{}-{}".format(
            output.name,
            time.strftime("%Y%m%d-%H%M%S"),
            uuid.uuid4().hex[:8],
        )
        evidence.mkdir(parents=True, exist_ok=False)
        self.data_matrix_evidence_directory = evidence
        return evidence

    @staticmethod
    def _data_matrix_camera_polygon(
        screen_rect_xywh: Sequence[int],
        screen_to_camera_3x3: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Project a phone-screen decoder crop back into the original camera frame."""

        x, y, width, height = map(float, screen_rect_xywh)
        screen_corners = np.asarray(
            [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
            dtype=np.float32,
        )
        return cv2.perspectiveTransform(
            screen_corners.reshape((1, -1, 2)),
            np.asarray(screen_to_camera_3x3, dtype=np.float64),
        ).reshape((-1, 2))

    def _write_data_matrix_failure_evidence(
        self,
        module_px: int,
        presentation_index: int,
        camera_frame: np.ndarray,
        rectified_camera_frame: np.ndarray,
        target_image: np.ndarray,
        failures: Sequence[Dict[str, Any]],
    ) -> Path:
        """Mark failed symbols in camera space and persist their decoder inputs."""

        if not failures:
            raise ValueError("At least one failed Data Matrix pattern is required")
        geometry = self._required(self.geometry, "screen geometry")
        evidence = self._ensure_data_matrix_evidence_directory()
        stem = "m{:03d}-screen-{:03d}".format(
            int(module_px), int(presentation_index) + 1
        )
        annotated = camera_frame.copy()
        presentation_entries = []
        for failure in failures:
            row = failure["row"]
            trial_index = int(failure["trial_index"])
            polygon = self._data_matrix_camera_polygon(
                row["decode_rect_screen_xywh"], geometry.inverse_matrix_3x3
            )
            polygon_i = np.rint(polygon).astype(np.int32)
            cv2.polylines(
                annotated,
                [polygon_i.reshape((-1, 1, 2))],
                True,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )
            reason = str(
                row.get("failure_reason")
                or failure.get("failure_reason")
                or "exact_payload_decode_failed"
            )
            label = "FAIL m{} pattern {}: {}".format(
                int(module_px), trial_index + 1, reason
            )
            anchor_x = max(
                4, min(int(np.min(polygon_i[:, 0])), annotated.shape[1] - 5)
            )
            anchor_y = max(
                22,
                min(int(np.min(polygon_i[:, 1])) - 6, annotated.shape[0] - 5),
            )
            text_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )[0]
            cv2.rectangle(
                annotated,
                (anchor_x - 2, max(0, anchor_y - text_size[1] - 5)),
                (
                    min(annotated.shape[1] - 1, anchor_x + text_size[0] + 3),
                    anchor_y + 4,
                ),
                (0, 0, 0),
                cv2.FILLED,
            )
            cv2.putText(
                annotated,
                label,
                (anchor_x, anchor_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            left = max(0, int(np.floor(np.min(polygon[:, 0]))) - 8)
            top = max(0, int(np.floor(np.min(polygon[:, 1]))) - 8)
            right = min(
                camera_frame.shape[1], int(np.ceil(np.max(polygon[:, 0]))) + 9
            )
            bottom = min(
                camera_frame.shape[0], int(np.ceil(np.max(polygon[:, 1]))) + 9
            )
            pattern_stem = "{}-pattern-{:03d}".format(stem, trial_index + 1)
            raw_crop_name = "{}-raw-camera-crop.png".format(pattern_stem)
            decoder_crop_name = "{}-rectified-decoder-crop.png".format(
                pattern_stem
            )
            if right > left and bottom > top:
                if not cv2.imwrite(
                    str(evidence / raw_crop_name),
                    camera_frame[top:bottom, left:right],
                ):
                    raise OSError("Could not save {}".format(raw_crop_name))
            else:
                raw_crop_name = None
            if not cv2.imwrite(
                str(evidence / decoder_crop_name), failure["decode_image"]
            ):
                raise OSError("Could not save {}".format(decoder_crop_name))
            row["camera_polygon_xy"] = polygon.astype(float).tolist()
            row["failure_evidence_files"] = {
                "annotated_camera_frame": "{}-annotated-camera.png".format(stem),
                "raw_camera_crop": raw_crop_name,
                "rectified_decoder_crop": decoder_crop_name,
                "rectified_camera_frame": "{}-rectified-camera.png".format(stem),
                "display_target": "{}-display-target.png".format(stem),
            }
            entry = {
                "module_width_display_px": int(module_px),
                "presentation_index": int(presentation_index),
                "trial_index": trial_index,
                "payload": str(failure.get("payload", "")),
                "failure_reason": reason,
                "camera_polygon_xy": row["camera_polygon_xy"],
                "decode_rect_screen_xywh": list(row["decode_rect_screen_xywh"]),
                "angle_deg": float(row.get("angle_deg", 0.0)),
                "color_bgr": list(row.get("color_bgr", [])),
                "intensity": float(row.get("intensity", 1.0)),
                "decoded_payloads": list(row.get("decoded_payloads", [])),
                "files": dict(row["failure_evidence_files"]),
            }
            self.data_matrix_failure_evidence.append(entry)
            presentation_entries.append(entry)

        images = {
            "{}-annotated-camera.png".format(stem): annotated,
            "{}-rectified-camera.png".format(stem): rectified_camera_frame,
            "{}-display-target.png".format(stem): target_image,
        }
        for name, image in images.items():
            if not cv2.imwrite(str(evidence / name), image):
                raise OSError("Could not save {}".format(name))
        (evidence / "index.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "meaning": (
                        "Red polygons mark exact-payload decode failures in the "
                        "original HIK camera frame."
                    ),
                    "camera": dict(self.camera_metadata),
                    "phone": self.phone_metrics.to_dict()
                    if self.phone_metrics
                    else None,
                    "failure_count": len(self.data_matrix_failure_evidence),
                    "failures": self.data_matrix_failure_evidence,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        if self._preview_created:
            self._preview_update(
                annotated,
                {"data_matrix_decode_failures": len(presentation_entries)},
            )
        return evidence

    def _wait_auto_imaging(self) -> Dict[str, Any]:
        """Wait for hardware auto exposure/gain and image brightness to converge."""

        deadline = time.monotonic() + float(self.options.operation_timeout_seconds)
        rows = []
        frames_observed = 0
        start = time.monotonic_ns()
        while time.monotonic() < deadline:
            sample = self.camera.read()
            frames_observed += 1
            state = dict(self.camera.imaging_state())
            self._set_preview_stage(
                "Auto exposure/gain bootstrap", exposure_mode="camera auto"
            )
            self._preview_update(sample.image, sample.metadata, **state)
            rows.append(
                {
                    "exposure_us": float(state["exposure_us"]),
                    "gain": float(state["gain"]),
                    "mean": float(np.mean(sample.image)),
                }
            )
            rows = rows[-8:]
            self.last_frame = sample.image.copy()
            if len(rows) == 8:
                exposure = [row["exposure_us"] for row in rows]
                gain = [row["gain"] for row in rows]
                means = [row["mean"] for row in rows]
                if (
                    max(exposure) - min(exposure) <= max(2.0, np.mean(exposure) * 0.01)
                    and max(gain) - min(gain) <= 0.1
                    and max(means) - min(means) <= max(2.0, np.mean(means) * 0.02)
                ):
                    return {
                        "frames_observed": frames_observed,
                        "elapsed_ms": (time.monotonic_ns() - start) / 1.0e6,
                        "exposure_us": exposure[-1],
                        "gain": gain[-1],
                        "mean_bgr_dn": means[-1],
                        "mode": "camera_auto_temporary",
                    }
        raise RuntimeError(
            "Camera auto exposure/gain did not converge within {:.1f}s; recent states {}"
            .format(self.options.operation_timeout_seconds, rows)
        )

    def _wait_painted(self, presentation: Presentation, timeout_seconds: float = 5.0) -> None:
        phone_metrics = self._required(self.phone_metrics, "phone metrics")
        expected_size = list(map(int, phone_metrics.screen_size_px))
        deadline = time.monotonic() + float(timeout_seconds)
        while time.monotonic() < deadline:
            acknowledgements = self.target.telemetry().get("acknowledgements", [])
            if any(
                int(item.get("revision", -1)) == presentation.revision
                and bool(item.get("painted"))
                and bool(item.get("fullscreen"))
                and [int(item.get("canvas_width", -1)), int(item.get("canvas_height", -1))]
                == expected_size
                for item in acknowledgements
            ):
                return
            time.sleep(0.04)
        raise RuntimeError("Phone did not acknowledge target revision {}".format(presentation.revision))

    def open(self) -> None:
        if self._opened:
            return
        self.progress("Reading phone and HIK camera specifications...")
        self.phone_metrics = self.phone.metrics(self.options.refresh_hz_override)
        detected_rotation = int(self.phone_metrics.orientation_quarter_turns) * 90
        self.progress(
            "Phone: {} {} ({}), {}x{} px at {:.3f} Hz; rotation {} degrees ({}), "
            "natural raster {}x{} px.".format(
                self.phone_metrics.manufacturer,
                self.phone_metrics.model,
                self.phone_metrics.serial,
                self.phone_metrics.screen_size_px[0],
                self.phone_metrics.screen_size_px[1],
                self.phone_metrics.refresh_hz,
                detected_rotation,
                self.phone_metrics.to_dict()["orientation_name"],
                self.phone_metrics.natural_screen_size_px[0],
                self.phone_metrics.natural_screen_size_px[1],
            )
        )
        phone_scale = self.phone_metrics.to_dict()
        if phone_scale.get("physical_pixel_pitch_mm_xy"):
            self.progress(
                "Phone physical scale: pitch {:.6f} x {:.6f} mm/px, active {:.2f} x {:.2f} mm "
                "({}).".format(
                    phone_scale["physical_pixel_pitch_mm_xy"][0],
                    phone_scale["physical_pixel_pitch_mm_xy"][1],
                    phone_scale["physical_size_mm"][0],
                    phone_scale["physical_size_mm"][1],
                    phone_scale["physical_size_source"],
                )
            )
        else:
            self.progress(
                "Phone physical pixel pitch is unavailable; focus MTF will remain in cycles/display-pixel."
            )
        layout = screen_filling_charuco_layout(self.phone_metrics.screen_size_px)
        self.charuco_layout = layout
        self.progress(
            "ChArUco atlas: {}x{} complete squares, {}x{} px board on {}x{} px display."
            .format(
                layout.squares_x,
                layout.squares_y,
                layout.board_size_px[0],
                layout.board_size_px[1],
                layout.screen_size_px[0],
                layout.screen_size_px[1],
            )
        )
        try:
            if isinstance(self.target, LocalPhoneTargetServer):
                self.target.start(layout)
                self.phone.wake_and_hold(
                    self.target.bound_port,
                    self.phone_metrics.screen_size_px,
                    self.phone_metrics.orientation_quarter_turns,
                )
            else:
                self.phone.wake_and_hold_display(
                    self.phone_metrics.orientation_quarter_turns
                )
                self.target.start(layout)
            self.phone_display_brightness = dict(self.phone.display_brightness_state())
            self.progress(
                "Phone calibration brightness: manual {}/{}; original setting will be restored."
                .format(
                    self.phone_display_brightness.get("brightness_value"),
                    self.phone_display_brightness.get("declared_maximum"),
                )
            )
            self._wait_fullscreen_canvas()
            self.camera_metadata = dict(
                self.camera.open(
                    CameraConfiguration(
                        device_id=self.options.camera_id,
                        width_px=self.options.camera_width_px,
                        height_px=self.options.camera_height_px,
                        fps=self.options.camera_fps,
                        backend="hik_mvs",
                    )
                )
            )
            self.camera_controls = dict(self.camera.controls())
            self.progress(
                "Bootstrapping ChArUco visibility with camera auto exposure/gain; "
                "final imaging remains mask-calibrated manual control..."
            )
            self.camera.set_auto_imaging()
            self.camera_metadata["geometry_bootstrap"] = self._wait_auto_imaging()
            self.progress(
                "Camera: {} {} ({}), {}x{} px at {:.3f} fps.".format(
                    self.camera_metadata.get("transport", "HIK"),
                    self.camera_metadata.get("model", "camera"),
                    self.camera_metadata.get("serial", self.options.camera_id),
                    self.camera_metadata["width_px"],
                    self.camera_metadata["height_px"],
                    self.camera_metadata["fps"],
                )
            )
            self._opened = True
        except Exception:
            self.close()
            raise

    def _wait_fullscreen_canvas(self, timeout_seconds: float = 8.0) -> None:
        phone_metrics = self._required(self.phone_metrics, "phone metrics")
        expected = list(map(int, phone_metrics.screen_size_px))
        deadline = time.monotonic() + float(timeout_seconds)
        observed = None
        observed_browser: Dict[str, Any] = {}
        next_fullscreen_request = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            telemetry = self.target.telemetry()
            browser = telemetry.get("viewer") or telemetry.get("browser", {})
            observed_browser = dict(browser)
            if browser.get("canvas_width") and browser.get("canvas_height"):
                observed = [int(browser["canvas_width"]), int(browser["canvas_height"])]
                if (
                    all(abs(a - b) <= 2 for a, b in zip(observed, expected))
                    and bool(browser.get("fullscreen"))
                ):
                    self.viewer_metrics = {
                        **dict(browser),
                        "adapter_id": browser.get("adapter_id", "android_http_view"),
                        "activity": browser.get("activity", self.phone.viewer_activity),
                    }
                    return
                now = time.monotonic()
                if (
                    isinstance(self.target, LocalPhoneTargetServer)
                    and not bool(browser.get("fullscreen"))
                    and now >= next_fullscreen_request
                ):
                    self.phone.request_fullscreen(expected, observed)
                    next_fullscreen_request = now + 0.75
            time.sleep(0.05)
        raise RuntimeError(
            "Phone target did not enter fullscreen at {} px: observed {}. "
            "Viewer evidence: {}. Dismiss viewer/browser prompts and retry.".format(
                expected, observed, observed_browser
            )
        )

    def calibrate_geometry(self) -> None:
        phone_metrics = self._required(self.phone_metrics, "phone metrics")
        self.progress("Showing ChArUco atlas and locating the camera-visible phone area...")
        self._set_preview_stage("ChArUco geometry", exposure_mode="camera auto")
        layout = self._required(self.charuco_layout, "screen-filling ChArUco layout")
        presentation = self.target.present_charuco()
        self._wait_painted(presentation)
        candidates = []
        for _ in range(int(self.options.geometry_frames)):
            sample = self.camera.read()
            frame = sample.image
            self.last_frame = frame.copy()
            try:
                detected = detect_charuco_correspondences(frame, layout)
                candidates.append(detected)
                self._preview_update(
                    frame,
                    sample.metadata,
                    charuco_corners=int(detected["corner_count"]),
                )
            except RuntimeError:
                self._preview_update(frame, sample.metadata, charuco_corners=0)
                continue
        if not candidates:
            raise RuntimeError("No usable ChArUco correspondence set was captured")
        self.correspondences = max(candidates, key=lambda row: int(row["corner_count"]))
        camera_size = [
            int(self.camera_metadata["width_px"]),
            int(self.camera_metadata["height_px"]),
        ]
        self.geometry = estimate_screen_geometry(
            self.correspondences["camera_points_xy"],
            self.correspondences["screen_points_xy"],
            camera_size,
            phone_metrics.screen_size_px,
        )
        self.visible_region = camera_visible_screen_region(
            self.geometry.viewport_polygon_screen_xy,
            phone_metrics.screen_size_px,
        )
        x, y, width, height = self.visible_region["xywh"]
        self.orientation_evidence = charuco_orientation_evidence(
            self.geometry.inverse_matrix_3x3,
            [x + (width - 1) / 2.0, y + (height - 1) / 2.0],
            probe_distance_px=max(16.0, min(width, height) / 4.0),
        )
        self.white_mask = camera_white_mask(
            camera_size,
            self.phone_metrics.screen_size_px,
            self.visible_region["xywh"],
            self.geometry.inverse_matrix_3x3,
        )
        self.progress(
            "Camera-visible phone region: x={} y={} w={} h={} px; screen IoU {:.3f}; "
            "ChArUco app-up is {:.2f} degrees clockwise from camera-up.".format(
                x,
                y,
                width,
                height,
                self.visible_region["screen_view_iou"],
                self.orientation_evidence["camera_up_to_app_up_clockwise_degrees"],
            )
        )

    def _observe(self, multiplier: float, exposure_us: float, gain: float) -> ExposureObservation:
        white_mask = self._required(self.white_mask, "white-area camera mask")
        self._set_preview_stage(
            "Exposure candidate {:.2f} panel periods ({:.3g}x refresh shutter rate)".format(
                1.0 / float(multiplier), float(multiplier)
            ),
            exposure_mode="manual candidate",
            exposure_us=exposure_us,
            gain=gain,
        )
        effective = self.camera.set_manual_imaging(exposure_us, gain)
        if not isinstance(effective, Mapping):
            effective = {"exposure_us": exposure_us, "gain": gain}
        if effective.get("fps"):
            self.camera_metadata["fps"] = float(effective["fps"])
        self._capture_settled(white_mask)
        frames = [
            self.camera.read().image
            for _ in range(int(self.options.exposure_noise_frames))
        ]
        self.last_frame = frames[-1].copy()
        statistics = temporal_white_statistics(frames, white_mask)
        row = ExposureObservation(
            shutter_refresh_multiplier=float(multiplier),
            exposure_us=float(effective.get("exposure_us", exposure_us)),
            gain=float(effective.get("gain", gain)),
            mean_bgr=tuple(statistics["mean_bgr"]),
            clipped_fraction_bgr=tuple(statistics["clipped_fraction_bgr"]),
            temporal_noise_bgr=tuple(statistics["temporal_noise_bgr"]),
        )
        self.exposure_observations.append(row)
        return row

    def calibrate_black_level(self) -> None:
        """Evaluate only the sensor black pedestal; skip cleanly if unsupported."""

        white_mask = self._required(self.white_mask, "camera-visible phone mask")
        phone_metrics = self._required(self.phone_metrics, "phone metrics")
        control = dict(self.camera.black_level_control())
        if not control.get("available"):
            self.progress("Black level is not writable on this camera; leaving it unchanged.")
            return
        minimum = int(control["minimum"])
        current = int(control["current"])
        increment = max(1, int(control.get("increment", 1)))
        halfway = minimum + ((current - minimum) // (2 * increment)) * increment
        candidates = sorted(set([minimum, halfway, current]))
        bootstrap = dict(self.camera_metadata.get("geometry_bootstrap", {}))
        provisional_exposure = float(bootstrap.get("exposure_us", 1000.0))
        provisional_gain = float(bootstrap.get("gain", 0.0))
        self.camera.set_manual_imaging(provisional_exposure, provisional_gain)
        shown = self.target.present_image(
            np.zeros((phone_metrics.screen_size_px[1], phone_metrics.screen_size_px[0], 3), np.uint8),
            "Black-level target",
        )
        self._wait_painted(shown)
        self.progress(
            "Checking {} significant black-level candidates (not the Guru feature tree)..."
            .format(len(candidates))
        )
        for candidate in candidates:
            self._set_preview_stage(
                "Black-level candidate {}".format(candidate),
                exposure_mode="manual provisional",
                exposure_us=provisional_exposure,
                gain=provisional_gain,
                black_level=candidate,
            )
            effective = self.camera.set_black_level(candidate)
            self._capture_settled(white_mask)
            frames = [
                self.camera.read().image
                for _ in range(int(self.options.exposure_noise_frames))
            ]
            statistics = temporal_black_statistics(frames, white_mask)
            self.black_level_observations.append(
                BlackLevelObservation(
                    level=int(effective),
                    mean_bgr=tuple(statistics["mean_bgr"]),
                    zero_fraction_bgr=tuple(statistics["zero_fraction_bgr"]),
                    temporal_noise_bgr=tuple(statistics["temporal_noise_bgr"]),
                )
            )
        selected = choose_black_level(self.black_level_observations)
        self.black_level = int(self.camera.set_black_level(selected.level))
        self._preview_settings["black_level"] = self.black_level
        self.progress(
            "Black level: {} (max zero fraction {:.3%}, temporal noise {:.3f} DN RMS)."
            .format(
                self.black_level,
                selected.maximum_zero_fraction,
                selected.temporal_noise_rms_dn,
            )
        )

    def calibrate_once_auto_imaging(self) -> None:
        """Use HIK one-shot AE/gain/AWB on a controlled neutral target as a seed."""

        phone_metrics = self._required(self.phone_metrics, "phone metrics")
        visible_region = self._required(self.visible_region, "camera-visible screen region")
        geometry = self._required(self.geometry, "screen geometry")
        auto_camera_roi = camera_roi_for_screen_region(
            visible_region["xywh"],
            geometry.inverse_matrix_3x3,
            [int(self.camera_metadata["width_px"]), int(self.camera_metadata["height_px"])],
            margin_px=0,
        )
        try:
            auto_function_aoi = dict(
                self.camera.configure_auto_function_roi(auto_camera_roi)
            )
            self.progress(
                "HIK auto AOI1 (exposure/gain) and AOI2 (WB) use camera ROI x={} y={} w={} h={}."
                .format(*auto_camera_roi)
            )
        except Exception as exc:
            auto_function_aoi = {
                "status": "unavailable",
                "requested_camera_roi_xywh": auto_camera_roi,
                "error": str(exc),
            }
            self.progress(
                "HIK auto-function AOI is unavailable; one-shot auto will use the camera default area: {}"
                .format(exc)
            )
        auto_target = white_patch(
            phone_metrics.screen_size_px,
            visible_region["xywh"],
            intensity_bgr=(128, 128, 128),
        )
        self.auto_target_image = auto_target.copy()
        shown = self.target.present_image(
            auto_target,
            "Neutral gray camera-auto target",
        )
        self._wait_painted(shown)
        target_screenshot = getattr(self.target, "last_screenshot", None)
        self.auto_phone_screenshot = (
            target_screenshot.copy() if target_screenshot is not None else None
        )
        maximum_auto_exposure_us = refresh_quantized_exposure_us(
            phone_metrics.refresh_hz,
            1.0 / float(self.options.maximum_exposure_periods),
        )
        auto_limits = dict(
            self.camera.configure_once_auto_limits(maximum_auto_exposure_us)
        )
        self.progress(
            "Running HIK one-shot auto exposure, gain, and white balance on neutral gray "
            "(target DN 128; exposure limit {:.1f} us)..."
            .format(float(auto_limits["exposure_upper_us"]))
        )
        started = time.monotonic_ns()
        initial = dict(self.camera.set_once_auto_imaging())
        deadline = time.monotonic() + max(
            12.0, float(self.options.operation_timeout_seconds) * 2.0
        )
        rows = []
        completed_modes = None
        while time.monotonic() < deadline:
            sample = self.camera.read()
            state = dict(self.camera.imaging_state())
            modes = dict(self.camera.auto_imaging_modes())
            mean = float(np.mean(sample.image))
            rows.append(
                {
                    "exposure_us": float(state["exposure_us"]),
                    "gain": float(state["gain"]),
                    "mean_bgr_dn": mean,
                    "modes": modes,
                }
            )
            rows = rows[-6:]
            self.last_frame = sample.image.copy()
            self._set_preview_stage(
                "HIK one-shot auto exposure/gain/WB",
                exposure_mode="camera one-shot",
                exposure_us=state["exposure_us"],
                gain=state["gain"],
            )
            self._preview_update(sample.image, sample.metadata, **state)
            modes_complete = all(value == "Off" for value in modes.values())
            values_stable = False
            if len(rows) >= 4:
                recent = rows[-4:]
                exposures = [row["exposure_us"] for row in recent]
                gains = [row["gain"] for row in recent]
                means = [row["mean_bgr_dn"] for row in recent]
                values_stable = (
                    max(exposures) - min(exposures)
                    <= max(2.0, float(np.mean(exposures)) * 0.005)
                    and max(gains) - min(gains) <= 0.05
                    and max(means) - min(means)
                    <= max(1.0, float(np.mean(means)) * 0.01)
                )
            if modes_complete and values_stable:
                completed_modes = modes
                break
        if completed_modes is None or not rows:
            raise RuntimeError(
                "HIK one-shot auto controls did not finish and stabilize; recent states {}"
                .format(rows)
            )
        final = rows[-1]
        self.auto_result_frame = self.last_frame.copy() if self.last_frame is not None else None
        white_balance = dict(self.camera.white_balance_state())
        self.camera.set_manual_imaging(final["exposure_us"], final["gain"])
        self.camera.set_white_balance(
            white_balance["ratio_red"],
            white_balance["ratio_green"],
            white_balance["ratio_blue"],
        )
        self.auto_imaging_seed = {
            "target": "neutral_gray_128_inside_camera_visible_phone_region",
            "initial": initial,
            "elapsed_ms": (time.monotonic_ns() - started) / 1.0e6,
            "exposure_us": final["exposure_us"],
            "gain": final["gain"],
            "mean_bgr_dn": final["mean_bgr_dn"],
            "white_balance": white_balance,
            "auto_function_aoi": auto_function_aoi,
            "auto_limits": auto_limits,
            "completed_modes": completed_modes,
            "recent_states": rows,
        }
        self.progress(
            "HIK one-shot result: {:.1f} us, gain {:.3f}, WB R/G/B {}/{}/{}."
            .format(
                final["exposure_us"],
                final["gain"],
                white_balance["ratio_red"],
                white_balance["ratio_green"],
                white_balance["ratio_blue"],
            )
        )

    def calibrate_exposure(self) -> None:
        phone_metrics = self._required(self.phone_metrics, "phone metrics")
        visible_region = self._required(self.visible_region, "camera-visible screen region")
        self.progress("Calibrating refresh-quantized exposure and gain...")
        self._set_preview_stage("Preparing exposure target", exposure_mode="manual")
        shown = self.target.present_image(
            white_patch(phone_metrics.screen_size_px, visible_region["xywh"]),
            "Exposure white patch",
        )
        self._wait_painted(shown)
        # Lock the camera's completed one-shot WB while exposure candidates are
        # measured; no auto state machine may move the clipping boundary here.
        seed = self._required(self.auto_imaging_seed, "HIK one-shot auto seed")
        seed_white_balance = dict(seed["white_balance"])
        self.camera.set_white_balance(
            seed_white_balance["ratio_red"],
            seed_white_balance["ratio_green"],
            seed_white_balance["ratio_blue"],
        )
        gain_limits = self.camera.controls().get("gain", {})
        gain_min = float(gain_limits.get("minimum", 0.0))
        gain_max = float(gain_limits.get("maximum", max(24.0, gain_min)))
        target = 255.0 * 0.90
        multipliers = [
            1.0 / float(periods)
            for periods in range(
                int(self.options.maximum_exposure_periods), 1, -1
            )
        ]
        multipliers.extend(
            float(value)
            for value in range(
                1, int(self.options.maximum_shutter_multiplier) + 1
            )
        )
        nearest_multiplier = min(
            multipliers,
            key=lambda value: abs(
                refresh_quantized_exposure_us(phone_metrics.refresh_hz, value)
                - float(seed["exposure_us"])
            ),
        )
        multipliers.sort(key=lambda value: (value != nearest_multiplier, value))
        nearest_exposure_us = refresh_quantized_exposure_us(
            phone_metrics.refresh_hz, nearest_multiplier
        )
        seed["refresh_quantization"] = {
            "source_exposure_us": float(seed["exposure_us"]),
            "nearest_shutter_refresh_multiplier": float(nearest_multiplier),
            "nearest_exposure_refresh_periods": 1.0 / float(nearest_multiplier),
            "nearest_exposure_us": float(nearest_exposure_us),
            "absolute_error_us": abs(float(seed["exposure_us"]) - nearest_exposure_us),
            "fallback_candidates_evaluated": len(multipliers) - 1,
        }
        self.progress(
            "One-shot exposure {:.1f} us quantizes nearest to {:.1f} us "
            "({:.3g}x refresh shutter rate, {:.2f} panel periods); "
            "checking allowed alternatives for clipping/noise safety."
            .format(
                float(seed["exposure_us"]),
                nearest_exposure_us,
                nearest_multiplier,
                1.0 / float(nearest_multiplier),
            )
        )
        for multiplier in multipliers:
            exposure_us = refresh_quantized_exposure_us(
                phone_metrics.refresh_hz, multiplier
            )
            low, high = gain_min, gain_max
            for _ in range(8):
                gain = (low + high) * 0.5
                row = self._observe(multiplier, exposure_us, gain)
                if (
                    row.white_balance_reference_brightness < target
                    and row.maximum_clipped_fraction <= 0.05
                ):
                    low = gain
                else:
                    high = gain
            self._observe(multiplier, exposure_us, low)
            self._observe(multiplier, exposure_us, high)
        self.exposure = choose_exposure(self.exposure_observations)
        selected_effective = self.camera.set_manual_imaging(
            self.exposure.exposure_us, self.exposure.gain
        )
        if isinstance(selected_effective, Mapping) and selected_effective.get("fps"):
            self.camera_metadata["fps"] = float(selected_effective["fps"])
        self._capture_settled(self._required(self.white_mask, "white-area camera mask"))
        self._set_preview_stage(
            "Selected exposure",
            exposure_mode="manual locked",
            exposure_us=self.exposure.exposure_us,
            gain=self.exposure.gain,
        )
        self.progress(
            "Exposure: {:.1f} us ({:.1f} Hz = {:.3g}x panel refresh, {:.2f} panel periods), "
            "gain {:.3f}, pre-WB mean {:.1f}, WB reference {:.1f}, "
            "temporal noise {:.3f} DN RMS.".format(
                self.exposure.exposure_us,
                self.exposure.shutter_rate_hz,
                self.exposure.shutter_refresh_multiplier,
                self.exposure.exposure_refresh_periods,
                self.exposure.gain,
                self.exposure.brightness,
                self.exposure.white_balance_reference_brightness,
                self.exposure.temporal_noise_rms_dn,
            )
        )
        minimum_acceptable = target - 255.0 * 0.02
        if self.exposure.white_balance_reference_brightness < minimum_acceptable:
            self._warn(
                "Selected exposure is below the preferred white level "
                "({:.1f}/255 versus {:.1f}/255); keeping the best measured manual lock."
                .format(
                    self.exposure.white_balance_reference_brightness,
                    minimum_acceptable,
                )
            )
        if self.exposure.maximum_clipped_fraction > 0.05:
            self._warn(
                "Every exposure candidate exceeded preferred clipping; keeping the least-clipped "
                "candidate at {:.3%}.".format(
                    self.exposure.maximum_clipped_fraction
                )
            )

    def calibrate_white_balance(self) -> None:
        exposure = self._required(self.exposure, "exposure calibration")
        white_mask = self._required(self.white_mask, "white-area camera mask")
        seed = dict(
            self._required(self.auto_imaging_seed, "HIK one-shot auto seed")[
                "white_balance"
            ]
        )
        self.progress(
            "Checking one residual white-balance correction; camera one-shot WB is the fallback..."
        )
        quarter = exposure.exposure_us / 4.0
        self._set_preview_stage(
            "White balance at quarter exposure",
            exposure_mode="manual WB measurement",
            exposure_us=quarter,
            gain=exposure.gain,
        )
        candidate = None
        candidate_statistics = None
        candidate_error = None
        try:
            self.camera.set_manual_imaging(quarter, exposure.gain)
            image = self._capture_settled(white_mask)
            correction = manual_white_balance_ratios(image, white_mask)
            candidate = {
                "ratio_red": int(
                    np.clip(
                        round(seed["ratio_red"] * correction["ratio_red"] / 1000.0),
                        1,
                        4095,
                    )
                ),
                "ratio_green": int(
                    np.clip(
                        round(seed["ratio_green"] * correction["ratio_green"] / 1000.0),
                        1,
                        4095,
                    )
                ),
                "ratio_blue": int(
                    np.clip(
                        round(seed["ratio_blue"] * correction["ratio_blue"] / 1000.0),
                        1,
                        4095,
                    )
                ),
                "one_shot_seed": seed,
                "quarter_exposure_residual_correction": correction,
                "method": "hik_one_shot_awb_then_blurred_white_residual",
            }
            effective_wb = self.camera.set_white_balance(
                candidate["ratio_red"],
                candidate["ratio_green"],
                candidate["ratio_blue"],
            )
            if isinstance(effective_wb, Mapping):
                candidate.update(effective_wb)
        except Exception as exc:
            candidate_error = str(exc)
        finally:
            self.camera.set_manual_imaging(exposure.exposure_us, exposure.gain)
        if candidate is not None and candidate_error is None:
            candidate_statistics = self._capture_balanced_white(
                white_mask, "residual_white_balance_candidate"
            )
            self.white_balance_attempts.append(
                {"white_balance": dict(candidate), "statistics": candidate_statistics}
            )
        if (
            candidate is not None
            and candidate_error is None
            and candidate_statistics is not None
            and candidate_statistics["within_preferred_range"]
        ):
            self.white_balance = candidate
            self.final_white_statistics = candidate_statistics
        else:
            effective_seed = self.camera.set_white_balance(
                seed["ratio_red"], seed["ratio_green"], seed["ratio_blue"]
            )
            self.camera.set_manual_imaging(exposure.exposure_us, exposure.gain)
            self.white_balance = {
                **seed,
                "method": "hik_one_shot_awb_fallback",
                "residual_attempt_error": candidate_error,
            }
            if isinstance(effective_seed, Mapping):
                self.white_balance.update(effective_seed)
            self.final_white_statistics = self._capture_balanced_white(
                white_mask, "hik_one_shot_awb_fallback"
            )
            self.white_balance_attempts.append(
                {
                    "white_balance": dict(self.white_balance),
                    "statistics": self.final_white_statistics,
                }
            )
            if candidate_error:
                self._warn(
                    "Residual white balance was unavailable ({}); kept HIK one-shot WB."
                    .format(candidate_error)
                )
            else:
                self._warn(
                    "Residual white balance increased clipping to {:.3%}; kept HIK one-shot WB."
                    .format(
                        candidate_statistics["maximum_clipped_fraction"]
                        if candidate_statistics is not None
                        else 0.0
                    )
                )
        final_mean = float(self.final_white_statistics["mean_dn"])
        final_maximum_clipped = float(
            self.final_white_statistics["maximum_clipped_fraction"]
        )
        if not self.final_white_statistics["within_preferred_range"]:
            self._warn(
                "Locked camera imaging is usable but outside the preferred white target: "
                "mean {:.1f}/255, maximum channel clipping {:.3%}."
                .format(final_mean, final_maximum_clipped)
            )
        self._set_preview_stage(
            "White balance locked",
            exposure_mode="manual locked",
            exposure_us=exposure.exposure_us,
            gain=exposure.gain,
            white_balance=self.white_balance,
        )
        self.progress(
            "White balance ratios R/G/B: {}/{}/{}; verified white mean {:.1f}, "
            "maximum channel clipping {:.3%}.".format(
                self.white_balance["ratio_red"],
                self.white_balance["ratio_green"],
                self.white_balance["ratio_blue"],
                final_mean,
                final_maximum_clipped,
            )
        )

    def verify_final_imaging(self) -> None:
        """Re-run ChArUco detection under final locked imaging settings as evidence."""

        phone_metrics = self._required(self.phone_metrics, "phone metrics")
        geometry = self._required(self.geometry, "screen geometry")
        layout = self._required(self.charuco_layout, "screen-filling ChArUco layout")
        self._set_preview_stage(
            "Final ChArUco verification",
            exposure_mode="manual locked",
            exposure_us=self.exposure.exposure_us if self.exposure else None,
            gain=self.exposure.gain if self.exposure else None,
        )
        shown = self.target.present_charuco()
        self._wait_painted(shown)
        corner_counts = []
        errors = []
        attempted = int(self.options.geometry_frames)
        for _ in range(attempted):
            sample = self.camera.read()
            frame = sample.image
            self.last_frame = frame.copy()
            try:
                detected = detect_charuco_correspondences(frame, layout)
            except RuntimeError:
                corner_counts.append(0)
                self._preview_update(frame, sample.metadata, charuco_corners=0)
                continue
            self._preview_update(
                frame,
                sample.metadata,
                charuco_corners=int(detected["corner_count"]),
            )
            camera_points = np.asarray(detected["camera_points_xy"], dtype=np.float64)
            screen_points = np.asarray(detected["screen_points_xy"], dtype=np.float64)
            predicted = cv2.perspectiveTransform(
                screen_points.reshape(-1, 1, 2),
                np.asarray(geometry.inverse_matrix_3x3, dtype=np.float64),
            ).reshape(-1, 2)
            corner_counts.append(int(len(camera_points)))
            errors.extend(np.linalg.norm(camera_points - predicted, axis=1).tolist())
        successful = sum(value > 0 for value in corner_counts)
        self.cv_verification = {
            "attempted_frames": attempted,
            "successful_frames": successful,
            "detection_rate": successful / float(attempted),
            "corner_counts": corner_counts,
            "reprojection_error_camera_px_p50": float(np.percentile(errors, 50)) if errors else None,
            "reprojection_error_camera_px_p95": float(np.percentile(errors, 95)) if errors else None,
            "algorithm": "existing_charuco_detector_and_saved_homography",
        }
        if not errors:
            raise RuntimeError(
                "Final manual imaging produced no ChArUco detections; calibration is not usable"
            )
        self.progress(
            "Final imaging verification: {}/{} ChArUco frames, reprojection p95 {:.3f} camera px."
            .format(successful, attempted, self.cv_verification["reprojection_error_camera_px_p95"])
        )

    @staticmethod
    def _distribution_ms(values: Sequence[float]) -> Dict[str, Optional[float]]:
        rows = np.asarray(list(values), dtype=np.float64)
        if rows.size == 0:
            return {"count": 0, "p50": None, "p95": None, "maximum": None}
        return {
            "count": int(rows.size),
            "p50": float(np.percentile(rows, 50)),
            "p95": float(np.percentile(rows, 95)),
            "maximum": float(np.max(rows)),
        }

    def _measure_signal_transition(
        self,
        state: str,
        mask: np.ndarray,
        threshold: float,
        rising: bool,
        trial_index: int,
    ) -> Dict[str, Any]:
        """Measure request-to-first-stable-camera-frame while Display changes."""

        holder: Dict[str, Any] = {}
        request_time_ns = time.monotonic_ns()

        def present() -> None:
            try:
                presentation = self.target.present_signal(
                    state, "latency-{}-{}".format(trial_index, state)
                )
                self._wait_painted(
                    presentation, timeout_seconds=self.options.operation_timeout_seconds
                )
                holder["presentation"] = presentation
            except Exception as exc:
                holder["error"] = str(exc)
            finally:
                holder["finished_time_ns"] = time.monotonic_ns()

        worker = threading.Thread(target=present, name="hik-display-transition")
        worker.daemon = True
        worker.start()
        deadline = time.monotonic() + max(
            12.0, float(self.options.operation_timeout_seconds) + 4.0
        )
        crossing_times: List[int] = []
        observed_means: List[float] = []
        stable_time_ns = None
        while time.monotonic() < deadline:
            sample = self.camera.read()
            self.last_frame = sample.image.copy()
            mean = float(np.mean(sample.image[np.asarray(mask) > 0]))
            observed_means.append(mean)
            crossed = mean >= threshold if rising else mean <= threshold
            if crossed:
                crossing_times.append(int(sample.receive_time_ns))
                if len(crossing_times) >= 3:
                    stable_time_ns = crossing_times[-3]
                    break
            else:
                crossing_times = []
        worker.join(max(0.0, deadline - time.monotonic()))
        if worker.is_alive():
            holder["error"] = "Android Display transition did not finish before timeout"
        presentation = holder.get("presentation")
        issued_time_ns = (
            int(presentation.issued_time_ns) if presentation is not None else request_time_ns
        )
        acknowledgement_time_ns = None
        if presentation is not None:
            for item in reversed(self.target.telemetry().get("acknowledgements", [])):
                if int(item.get("revision", -1)) == int(presentation.revision):
                    acknowledgement_time_ns = int(
                        item.get("server_receive_time_ns", holder.get("finished_time_ns", 0))
                    )
                    break
        accepted = stable_time_ns is not None and not holder.get("error")
        return {
            "trial": int(trial_index),
            "state": state,
            "direction": "rising" if rising else "falling",
            "accepted": bool(accepted),
            "error": holder.get("error"),
            "threshold_bgr_dn": float(threshold),
            "observed_mean_bgr_dn_min": float(min(observed_means)) if observed_means else None,
            "observed_mean_bgr_dn_max": float(max(observed_means)) if observed_means else None,
            "request_time_ns": issued_time_ns,
            "display_ack_time_ns": acknowledgement_time_ns,
            "first_stable_camera_time_ns": stable_time_ns,
            "request_to_first_stable_ms": (
                (stable_time_ns - issued_time_ns) / 1.0e6
                if stable_time_ns is not None
                else None
            ),
            "display_ack_to_first_stable_ms": (
                (stable_time_ns - acknowledgement_time_ns) / 1.0e6
                if stable_time_ns is not None and acknowledgement_time_ns is not None
                else None
            ),
        }

    def benchmark_final_stream(self) -> None:
        """Apply final hardware ROI and save compact transport/latency evidence."""

        if self.transport_benchmark is not None:
            return
        geometry = self._required(self.geometry, "screen geometry")
        visible_region = self._required(self.visible_region, "camera-visible screen region")
        white_mask = self._required(self.white_mask, "camera-visible phone mask")
        full_size = [
            int(self.camera_metadata["width_px"]),
            int(self.camera_metadata["height_px"]),
        ]
        requested = camera_roi_for_screen_region(
            visible_region["xywh"], geometry.inverse_matrix_3x3, full_size
        )
        expected = self.camera.align_roi(requested)
        effective = list(map(int, self.camera.set_roi(expected)))
        if effective != list(map(int, expected)):
            raise RuntimeError(
                "HIK effective ROI {} differs from calibrated {}".format(effective, expected)
            )
        self.hardware_roi = effective
        x, y, width, height = effective
        roi_mask = white_mask[y : y + height, x : x + width]
        if roi_mask.shape != (height, width):
            raise RuntimeError("Hardware ROI mask does not match the returned camera frame")
        self.progress(
            "Benchmarking final {}x{} hardware ROI and display-to-camera response..."
            .format(width, height)
        )
        read_durations_ms = []
        receive_times_ns = []
        for _ in range(24):
            started = time.monotonic_ns()
            sample = self.camera.read()
            finished = time.monotonic_ns()
            if sample.image.shape[:2] != (height, width):
                raise RuntimeError(
                    "HIK ROI returned frame shape {}, expected {}"
                    .format(sample.image.shape[:2], (height, width))
                )
            read_durations_ms.append((finished - started) / 1.0e6)
            receive_times_ns.append(int(sample.receive_time_ns))
            self.last_frame = sample.image.copy()
        intervals_ms = [
            (right - left) / 1.0e6
            for left, right in zip(receive_times_ns, receive_times_ns[1:])
        ]
        pixel_format = (
            self.camera_controls.get("genicam", {}).get("PixelFormat", {}).get("value")
        )
        bits_per_pixel = None
        if isinstance(pixel_format, (int, float)):
            candidate = (int(pixel_format) >> 16) & 0xFF
            if 1 <= candidate <= 64:
                bits_per_pixel = candidate
        full_pixels = int(full_size[0] * full_size[1])
        roi_pixels = int(width * height)
        self.transport_benchmark = {
            "endpoint": "adapter_read_after_hardware_roi",
            "hardware_roi_xywh": effective,
            "full_sensor_pixels": full_pixels,
            "roi_pixels": roi_pixels,
            "pixel_reduction_fraction": 1.0 - roi_pixels / float(full_pixels),
            "sensor_payload_bits_per_pixel": bits_per_pixel,
            "estimated_full_payload_bytes": (
                full_pixels * bits_per_pixel / 8.0 if bits_per_pixel else None
            ),
            "estimated_roi_payload_bytes": (
                roi_pixels * bits_per_pixel / 8.0 if bits_per_pixel else None
            ),
            "adapter_read_duration_ms": self._distribution_ms(read_durations_ms),
            "frame_interval_ms": self._distribution_ms(intervals_ms),
        }

        black = self.target.present_signal("black", "latency-baseline-black")
        self._wait_painted(black, timeout_seconds=self.options.operation_timeout_seconds)
        black_image = self._capture_settled(roi_mask)
        black_mean = float(np.mean(black_image[roi_mask > 0]))
        white = self.target.present_signal("white", "latency-baseline-white")
        self._wait_painted(white, timeout_seconds=self.options.operation_timeout_seconds)
        white_image = self._capture_settled(roi_mask)
        white_mean = float(np.mean(white_image[roi_mask > 0]))
        contrast = white_mean - black_mean
        trials = []
        if contrast > 10.0:
            current = "white"
            threshold = (black_mean + white_mean) * 0.5
            for trial_index in range(3):
                next_state = "black" if current == "white" else "white"
                row = self._measure_signal_transition(
                    next_state,
                    roi_mask,
                    threshold,
                    rising=next_state == "white",
                    trial_index=trial_index + 1,
                )
                trials.append(row)
                current = next_state
        accepted_ms = [
            float(row["request_to_first_stable_ms"])
            for row in trials
            if row.get("accepted") and row.get("request_to_first_stable_ms") is not None
        ]
        self.latency_benchmark = {
            "endpoint": "host_display_request_to_first_of_three_stable_camera_frames",
            "clock": "host_monotonic",
            "reference_only": True,
            "black_mean_bgr_dn": black_mean,
            "white_mean_bgr_dn": white_mean,
            "contrast_bgr_dn": contrast,
            "trials": trials,
            "request_to_first_stable_ms": self._distribution_ms(accepted_ms),
            "status": (
                "measured" if accepted_ms else "insufficient_transition_evidence"
            ),
        }
        self.progress(
            "ROI reduces sensor payload by {:.1%}; read p50 {:.2f} ms; "
            "request-to-camera latency {}.".format(
                self.transport_benchmark["pixel_reduction_fraction"],
                self.transport_benchmark["adapter_read_duration_ms"]["p50"],
                (
                    "p50 {:.2f} ms".format(
                        self.latency_benchmark["request_to_first_stable_ms"]["p50"]
                    )
                    if accepted_ms
                    else "was not measurable"
                ),
            )
        )

    def _focus_measurement(self, frame: np.ndarray) -> Dict[str, Any]:
        visible_region = self._required(self.visible_region, "camera-visible screen region")
        geometry = self._required(self.geometry, "screen geometry")
        phone_metrics = self._required(self.phone_metrics, "phone metrics")
        roi = camera_roi_for_screen_region(
            visible_region["xywh"],
            geometry.inverse_matrix_3x3,
            [frame.shape[1], frame.shape[0]],
            margin_px=0,
        )
        roi_x, roi_y, roi_width, roi_height = roi
        focus_crop = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]
        row: Dict[str, Any] = {
            "laplacian": laplacian_sharpness(focus_crop),
            "time_ns": time.monotonic_ns(),
            "camera_roi_xywh": roi,
            "standard": "ISO 12233:2024 slanted-edge e-SFR (non-certified)",
            "edges": [],
        }
        pitch = phone_metrics.to_dict().get("physical_pixel_pitch_mm_xy")
        for edge in focus_edge_regions(visible_region["xywh"]):
            evidence = dict(edge)
            try:
                esfr, _ = measure_slanted_edge_esfr(
                    frame,
                    geometry.matrix_3x3,
                    edge["rect_screen_xywh"],
                    edge_angle_deg=edge["angle_deg"],
                    geometry_confidence=geometry.confidence,
                    display_pixel_pitch_mm_xy=pitch,
                )
                evidence.update(
                    {
                        "mtf50_cycles_per_display_pixel": esfr["display_referred"]["mtf50"],
                        "mtf10_cycles_per_display_pixel": esfr["display_referred"]["mtf10"],
                        "confidence": esfr["confidence"],
                    }
                )
                physical = esfr.get("display_physical")
                if physical:
                    evidence.update(
                        {
                            "mtf50_lp_per_mm": physical["mtf50"],
                            "mtf10_lp_per_mm": physical["mtf10"],
                            "pixel_pitch_mm_along_edge_normal": physical[
                                "pixel_pitch_mm_along_edge_normal"
                            ],
                            "physical_scale_source": phone_metrics.physical_scale_source,
                        }
                    )
            except (ValueError, RuntimeError) as exc:
                evidence["error"] = str(exc)
            row["edges"].append(evidence)
        for output, source in (
            ("mtf50", "mtf50_cycles_per_display_pixel"),
            ("mtf10", "mtf10_cycles_per_display_pixel"),
            ("mtf50_lp_per_mm", "mtf50_lp_per_mm"),
            ("mtf10_lp_per_mm", "mtf10_lp_per_mm"),
        ):
            values = [
                float(edge[source])
                for edge in row["edges"]
                if edge.get(source) is not None
            ]
            row[output] = min(values) if len(values) == 4 else None
        row["usable_edge_count"] = sum(
            edge.get("mtf50_cycles_per_display_pixel") is not None
            for edge in row["edges"]
        )
        if row["mtf50"] is None:
            row["mtf_error"] = "No focus-chart edge produced a usable e-SFR result"
        return row

    def focus_loop(self) -> str:
        """Run native-pixel focus UI. Return save, data-matrix, or quit action."""

        phone_metrics = self._required(self.phone_metrics, "phone metrics")
        visible_region = self._required(self.visible_region, "camera-visible screen region")
        geometry = self._required(self.geometry, "screen geometry")
        physical_pitch = phone_metrics.to_dict().get("physical_pixel_pitch_mm_xy")
        frame_x, frame_y, frame_width, frame_height = focus_frame_rect(
            visible_region["xywh"]
        )
        frame_screen_quad = np.asarray(
            [
                [frame_x, frame_y],
                [frame_x + frame_width - 1, frame_y],
                [frame_x + frame_width - 1, frame_y + frame_height - 1],
                [frame_x, frame_y + frame_height - 1],
            ],
            dtype=np.float64,
        )
        expected_frame_quad = cv2.perspectiveTransform(
            frame_screen_quad.reshape((-1, 1, 2)),
            np.asarray(geometry.inverse_matrix_3x3, dtype=np.float64),
        ).reshape((-1, 2))
        geometry_change_threshold_px = max(
            8.0,
            3.0
            * float(
                (self.cv_verification or {}).get(
                    "reprojection_error_camera_px_p95", 0.0
                )
                or 0.0
            ),
        )
        shown = self.target.present_image(
            focus_pattern(phone_metrics.screen_size_px, visible_region["xywh"]),
            "Native focus and ISO 12233 slanted edge",
        )
        self._wait_painted(shown)
        self._focus_geometry_changed = False
        self._close_preview(disable=True)
        self.progress(
            "Focus: adjust against the framed target; R recalibrates after moving the rig, "
            "S saves, D tests Data Matrix decoding, Q/Esc exits."
        )
        window = "HIK focus - native 1:1"
        window_created = False
        try:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            window_created = True
            while True:
                frame = self.camera.read().image
                self.last_frame = frame.copy()
                measurement = self._focus_measurement(frame)
                self.focus_history.append(measurement)
                pose = None
                pose_error = None
                frame_displacement_px = None
                detected_quad = None
                try:
                    detected_frame = detect_focus_pose_frame(frame, expected_frame_quad)
                    detected_quad = np.asarray(
                        detected_frame["camera_quad_xy"], dtype=np.float64
                    )
                    frame_displacement_px = float(
                        np.mean(np.linalg.norm(detected_quad - expected_frame_quad, axis=1))
                    )
                    if frame_displacement_px > geometry_change_threshold_px:
                        self._focus_geometry_changed = True
                    if physical_pitch:
                        pose = estimate_focus_target_pose(
                            detected_quad,
                            [
                                (frame_width - 1) * float(physical_pitch[0]),
                                (frame_height - 1) * float(physical_pitch[1]),
                            ],
                            [frame.shape[1], frame.shape[0]],
                            focal_length_px=self._focus_focal_length_px,
                        )
                        if self._focus_focal_length_px is None:
                            self._focus_focal_candidates_px.append(
                                float(pose["focal_length_px"])
                            )
                            self._focus_focal_candidates_px = self._focus_focal_candidates_px[-20:]
                            if len(self._focus_focal_candidates_px) >= 8:
                                candidates = np.asarray(
                                    self._focus_focal_candidates_px, dtype=np.float64
                                )
                                median = float(np.median(candidates))
                                relative_mad = float(
                                    np.median(np.abs(candidates - median))
                                    / max(abs(median), 1.0)
                                )
                                if relative_mad <= 0.10:
                                    self._focus_focal_length_px = median
                                    pose["focal_length_px"] = median
                                    pose["focal_length_source"] = (
                                        "median_of_stable_orthogonal_target_estimates"
                                    )
                        pose["frame_displacement_from_calibration_px"] = frame_displacement_px
                        pose["time_ns"] = time.monotonic_ns()
                        self.focus_pose_history.append(dict(pose))
                except (ValueError, RuntimeError) as exc:
                    pose_error = str(exc)
                maxima = {
                    key: max(
                        (float(row[key]) for row in self.focus_history if row.get(key) is not None),
                        default=float("nan"),
                    )
                    for key in (
                        "laplacian",
                        "mtf50",
                        "mtf10",
                        "mtf50_lp_per_mm",
                        "mtf10_lp_per_mm",
                    )
                }
                lines = [
                    "Complete positioning view at left; native 1:1 crops: TL / TR / BL / BR",
                    "Laplacian {:.2f}  max {:.2f}".format(measurement["laplacian"], maxima["laplacian"]),
                    "MTF50 min(4 edges) {}  max {} cy/display-px".format(
                        self._format_metric(measurement.get("mtf50"), 4),
                        self._format_metric(maxima.get("mtf50"), 4),
                    ),
                    "MTF10 min(4 edges) {}  max {} | S save, D decode test, Q quit".format(
                        self._format_metric(measurement.get("mtf10"), 4),
                        self._format_metric(maxima.get("mtf10"), 4),
                    ),
                ]
                if pose is not None:
                    lines.extend(
                        [
                            "Camera target pose: pitch {} deg; yaw {} deg".format(
                                self._format_metric(pose.get("pitch_deg")),
                                self._format_metric(pose.get("yaw_deg")),
                            ),
                            "Phone rotation: {} deg clockwise from camera-up".format(
                                self._format_metric(
                                    pose.get(
                                        "phone_rotation_clockwise_from_camera_up_deg"
                                    )
                                )
                            ),
                            "Lens-to-panel perpendicular distance: {} mm".format(
                                self._format_metric(
                                    pose.get("lens_to_panel_distance_mm"), 1
                                )
                            ),
                            "Pose p95 {} px; frame moved {} px from calibrated geometry".format(
                                self._format_metric(
                                    pose.get("reprojection_p95_camera_px")
                                ),
                                self._format_metric(frame_displacement_px),
                            ),
                        ]
                    )
                    if (
                        frame_displacement_px is not None
                        and frame_displacement_px > geometry_change_threshold_px
                    ):
                        lines.append(
                            "Rig moved: press R to rerun geometry and imaging before saving."
                        )
                else:
                    lines.append(
                        "Camera target pose unavailable: {}".format(
                            pose_error or "physical phone pixel pitch is unavailable"
                        )
                    )
                if (
                    measurement.get("mtf50_lp_per_mm") is not None
                    or measurement.get("mtf10_lp_per_mm") is not None
                ):
                    lines.append(
                        "Physical MTF50 {} max {}; MTF10 {} max {} lp/mm (Android-reported pitch)".format(
                            self._format_metric(measurement.get("mtf50_lp_per_mm")),
                            self._format_metric(maxima.get("mtf50_lp_per_mm")),
                            self._format_metric(measurement.get("mtf10_lp_per_mm")),
                            self._format_metric(maxima.get("mtf10_lp_per_mm")),
                        )
                    )
                work_width, work_height = self._desktop_work_area()
                panel_width = min(480, max(360, int(work_width * 0.28)))
                available_width = max(320, int(work_width * 0.92) - panel_width)
                available_height = max(240, int(work_height * 0.86))
                overview_width = min(
                    420, max(240, int(round(available_width * 0.32)))
                )
                grid_available_width = max(240, available_width - overview_width - 6)
                tile_limit_width = max(120, (grid_available_width - 6) // 2)
                tile_limit_height = max(120, (available_height - 6) // 2)
                crops = []
                for edge in focus_edge_regions(visible_region["xywh"]):
                    edge_roi = camera_roi_for_screen_region(
                        edge["rect_screen_xywh"],
                        geometry.inverse_matrix_3x3,
                        [frame.shape[1], frame.shape[0]],
                        margin_px=0,
                    )
                    x, y, width, height = edge_roi
                    crop = frame[y : y + height, x : x + width]
                    crop_height = min(crop.shape[0], tile_limit_height)
                    crop_width = min(crop.shape[1], tile_limit_width)
                    crop_y = max(0, (crop.shape[0] - crop_height) // 2)
                    crop_x = max(0, (crop.shape[1] - crop_width) // 2)
                    crops.append(crop[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width])
                tile_width = max(crop.shape[1] for crop in crops)
                tile_height = max(crop.shape[0] for crop in crops)
                grid_width = tile_width * 2 + 6
                image_width = overview_width + 6 + grid_width
                image_height = tile_height * 2 + 6
                view = np.full((image_height, image_width + panel_width, 3), 24, np.uint8)
                overview_frame = frame.copy()
                cv2.polylines(
                    overview_frame,
                    [np.rint(expected_frame_quad).astype(np.int32)],
                    True,
                    (255, 160, 0),
                    2,
                    cv2.LINE_AA,
                )
                if detected_quad is not None:
                    cv2.polylines(
                        overview_frame,
                        [np.rint(detected_quad).astype(np.int32)],
                        True,
                        (0, 220, 255),
                        2,
                        cv2.LINE_AA,
                    )
                overview = self._fit_complete_view(
                    overview_frame, overview_width, image_height
                )
                view[:, :overview_width] = overview
                cv2.putText(
                    view,
                    "FULL VIEW",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                for index, crop in enumerate(crops):
                    left = overview_width + 6 + (index % 2) * (tile_width + 6)
                    top = (index // 2) * (tile_height + 6)
                    view[top : top + crop.shape[0], left : left + crop.shape[1]] = crop
                self._draw_text_panel(view, image_width, panel_width, lines)
                cv2.resizeWindow(window, view.shape[1], view.shape[0])
                cv2.imshow(window, view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("s"), ord("S")):
                    if self._focus_geometry_changed:
                        self.progress(
                            "Save blocked because the rig moved; press R to recalibrate geometry and imaging."
                        )
                        continue
                    return "save"
                if key in (ord("d"), ord("D")):
                    return "data_matrix"
                if key in (ord("r"), ord("R")):
                    return "recalibrate"
                if key in (27, ord("q"), ord("Q")):
                    return "quit"
        finally:
            if window_created:
                cv2.destroyWindow(window)

    def grade_data_matrix(self) -> Dict[str, Any]:
        """Measure batched exact-payload decode success at a 95% threshold."""

        phone_metrics = self._required(self.phone_metrics, "phone metrics")
        visible_region = self._required(self.visible_region, "camera-visible screen region")
        geometry = self._required(self.geometry, "screen geometry")
        complete_grader: Optional[CompleteGrader] = None
        if self.options.complete_grader_plugin:
            complete_grader = _load_callable(self.options.complete_grader_plugin)
        conditions = [
            (0.0, (1.0, 1.0, 1.0), 1.0),
            (15.0, (1.0, 0.92, 0.82), 0.85),
            (-15.0, (0.82, 0.92, 1.0), 0.70),
            (30.0, (0.92, 1.0, 0.82), 0.55),
        ]
        trials_per_size = int(self.options.data_matrix_trials_per_size)
        module_px = int(self.options.data_matrix_initial_module_px)
        maximum_module = max(1, min(visible_region["xywh"][2:]))
        per_size = []
        required_rate = DATA_MATRIX_ACCEPTANCE_RATE
        self._preview_disabled = False
        self.progress(
            "Data Matrix decode test started; {} controlled patterns per size, "
            "packed up to {} per phone screen; pass threshold {:.0%}."
            .format(
                trials_per_size,
                DATA_MATRIX_MAX_PATTERNS_PER_SCREEN,
                required_rate,
            )
        )
        while module_px <= maximum_module:
            rows = []
            early_rejected = False
            presentation_count = 0
            batch_sizes = []
            trial = 0
            while trial < trials_per_size:
                specifications = []
                for candidate in range(
                    trial,
                    min(
                        trials_per_size,
                        trial + DATA_MATRIX_MAX_PATTERNS_PER_SCREEN,
                    ),
                ):
                    angle, color, intensity = conditions[candidate % len(conditions)]
                    specifications.append(
                        {
                            "trial_index": candidate,
                            "angle_deg": angle,
                            "color_bgr": color,
                            "intensity": intensity,
                        }
                    )
                batch = self._compose_data_matrix_batch(
                    phone_metrics.screen_size_px,
                    visible_region["xywh"],
                    module_px,
                    specifications,
                )
                if batch is None:
                    break
                items = list(batch["items"])
                first_trial = int(items[0]["trial_index"])
                last_trial = int(items[-1]["trial_index"])
                label = "dm-m{}-batch-{:03d}".format(
                    module_px, presentation_count
                )
                shown = self.target.present_image(batch["image"], label)
                self._wait_painted(shown)
                self._set_preview_stage(
                    "Data Matrix decode test: module {} px, screen {}, patterns {}-{}/{}"
                    .format(
                        module_px,
                        presentation_count + 1,
                        first_trial + 1,
                        last_trial + 1,
                        trials_per_size,
                    ),
                    exposure_mode="manual locked",
                    exposure_us=self.exposure.exposure_us if self.exposure else None,
                    gain=self.exposure.gain if self.exposure else None,
                )
                frame = self._capture_data_matrix_frame()
                screen_frame = cv2.warpPerspective(
                    frame,
                    np.asarray(geometry.matrix_3x3, dtype=np.float64),
                    tuple(phone_metrics.screen_size_px),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(127, 127, 127),
                )
                presentation_count += 1
                batch_sizes.append(len(items))
                presentation_failures = []
                for pattern_index, item in enumerate(items):
                    decode_rect = item["decode_rect_screen_xywh"]
                    x, y, width, height = map(int, decode_rect)
                    decode_image = screen_frame[y : y + height, x : x + width]
                    metadata = {
                        "angle_deg": float(item["angle_deg"]),
                        "color_bgr": list(item["color_bgr"]),
                        "intensity": float(item["intensity"]),
                        "decode_input": "rectified_rotated_symbol_crop",
                        "decode_rect_screen_xywh": list(decode_rect),
                        "cell_rect_screen_xywh": list(
                            item["cell_rect_screen_xywh"]
                        ),
                        "presentation_index": presentation_count - 1,
                        "pattern_index_in_presentation": pattern_index,
                        "patterns_in_presentation": len(items),
                        "discarded_camera_frames_after_target_change": max(
                            8, int(self.options.settle_frames) * 2
                        ),
                    }
                    try:
                        grade = dict(
                            complete_grader(frame, item["payload"], metadata)
                            if complete_grader
                            else grade_data_matrix_decode(
                                decode_image, item["payload"]
                            )
                        )
                    except Exception as exc:
                        failed_trial = int(item["trial_index"])
                        error_row = {
                            **metadata,
                            "decode_success": False,
                            "exact_payload_decoded": False,
                            "failure_reason": "decoder_error: {}".format(exc),
                        }
                        try:
                            self._write_data_matrix_failure_evidence(
                                module_px,
                                presentation_count - 1,
                                frame,
                                screen_frame,
                                batch["image"],
                                [
                                    {
                                        "row": error_row,
                                        "trial_index": failed_trial,
                                        "payload": item["payload"],
                                        "decode_image": decode_image.copy(),
                                        "failure_reason": error_row["failure_reason"],
                                    }
                                ],
                            )
                        except (OSError, ValueError, cv2.error) as evidence_exc:
                            self._warn(
                                "Could not save Data Matrix failure evidence: {}"
                                .format(evidence_exc)
                            )
                        message = (
                            "Data Matrix decode test is unavailable at module {} px, "
                            "pattern {}: {}. Returning to calibration without "
                            "discarding the session."
                            .format(module_px, failed_trial, exc)
                        )
                        self._warn(message)
                        self.data_matrix_result = {
                            "measurement": "exact_payload_decode_success",
                            "status": "unavailable",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "failed_module_width_display_px": module_px,
                            "failed_trial_index": failed_trial,
                            "completed_sizes": per_size,
                            "partial_current_size_trials": rows,
                            "partial_current_size_presentation_count": (
                                presentation_count
                            ),
                            "failure_evidence_directory": (
                                str(self.data_matrix_evidence_directory)
                                if self.data_matrix_evidence_directory
                                else None
                            ),
                        }
                        return self.data_matrix_result
                    if not complete_grader:
                        for formal_grade_key in (
                            "standard",
                            "parameter",
                            "grade",
                            "grade_letter",
                            "grade_scale",
                        ):
                            grade.pop(formal_grade_key, None)
                    grade.update(metadata)
                    grade["decode_success"] = self._data_matrix_decode_succeeded(
                        grade
                    )
                    rows.append(grade)
                    if not grade["decode_success"]:
                        presentation_failures.append(
                            {
                                "row": grade,
                                "trial_index": int(item["trial_index"]),
                                "payload": item["payload"],
                                "decode_image": decode_image.copy(),
                            }
                        )
                    trial = int(item["trial_index"]) + 1
                    current_success_count = sum(
                        1
                        for row in rows
                        if self._data_matrix_decode_succeeded(row)
                    )
                    if self._data_matrix_cannot_qualify(
                        current_success_count,
                        len(rows),
                        trials_per_size,
                        required_rate,
                    ):
                        early_rejected = True
                        failures = len(rows) - current_success_count
                        self.progress(
                            "Data Matrix module {} px skipped after {} patterns "
                            "and {} failures; {:.0%} is no longer reachable."
                            .format(
                                module_px,
                                len(rows),
                                failures,
                                required_rate,
                            )
                        )
                        break
                if presentation_failures:
                    try:
                        evidence = self._write_data_matrix_failure_evidence(
                            module_px,
                            presentation_count - 1,
                            frame,
                            screen_frame,
                            batch["image"],
                            presentation_failures,
                        )
                        self.progress(
                            "Marked {} failed Data Matrix pattern(s); evidence: {}"
                            .format(len(presentation_failures), evidence)
                        )
                    except (OSError, ValueError, cv2.error) as exc:
                        self._warn(
                            "Could not save Data Matrix failure evidence: {}".format(exc)
                        )
                if early_rejected:
                    break
            if not rows:
                break
            success_count = sum(
                1 for row in rows if self._data_matrix_decode_succeeded(row)
            )
            rate = success_count / float(len(rows))
            size_row = {
                "module_width_display_px": module_px,
                "trial_count": len(rows),
                "planned_trial_count": trials_per_size,
                "presentation_count": presentation_count,
                "patterns_per_presentation": batch_sizes,
                "decode_success_count": success_count,
                "decode_success_rate": rate,
                "maximum_possible_final_rate": (
                    success_count + trials_per_size - len(rows)
                )
                / float(trials_per_size),
                "early_rejected": early_rejected,
                "qualified": (
                    len(rows) == trials_per_size and rate >= required_rate
                ),
                "trials": rows,
            }
            per_size.append(size_row)
            self.progress(
                "Data Matrix module {} px: {:.1%} decoded ({}/{} patterns)."
                .format(module_px, rate, success_count, len(rows))
            )
            if size_row["qualified"]:
                break
            module_px *= 2
        self.data_matrix_result = {
            "measurement": "exact_payload_decode_success",
            "acceptance": "observed_exact_payload_decode_success_rate_at_least_0.95",
            "required_decode_success_rate": required_rate,
            "iso_iec_15415_note": (
                "Decode parameter only; not a complete ISO/IEC 15415 symbol grade"
            ),
            "minimum_patterns_per_size": DATA_MATRIX_MINIMUM_TRIALS,
            "planned_patterns_per_size": trials_per_size,
            "maximum_patterns_per_presentation": (
                DATA_MATRIX_MAX_PATTERNS_PER_SCREEN
            ),
            "implementation_conformance": (
                "external_complete_grader" if complete_grader else "Decode_parameter_only_not_complete_ISO_IEC_15415_verifier"
            ),
            "failure_evidence_directory": (
                str(self.data_matrix_evidence_directory)
                if self.data_matrix_evidence_directory
                else None
            ),
            "failure_evidence_count": len(self.data_matrix_failure_evidence),
            "qualified_module_width_display_px": (
                per_size[-1]["module_width_display_px"]
                if per_size and per_size[-1]["qualified"]
                else None
            ),
            "per_size": per_size,
        }
        return self.data_matrix_result

    def _rectification_maps(self, matrix: np.ndarray, output_size: Sequence[int]):
        width, height = map(int, output_size)
        inverse = np.linalg.inv(np.asarray(matrix, dtype=np.float64))
        yy, xx = np.mgrid[0:height, 0:width]
        points = np.column_stack([xx.reshape(-1), yy.reshape(-1)]).astype(np.float64)
        homogeneous = np.column_stack([points, np.ones(len(points))]).dot(inverse.T)
        source = homogeneous[:, :2] / homogeneous[:, 2:3]
        return source[:, 0].reshape((height, width)).astype(np.float32), source[:, 1].reshape((height, width)).astype(np.float32)

    def _save_cross_source_check(
        self,
        output_directory: Path,
        rectification_maps: tuple[np.ndarray, np.ndarray],
        valid_mask: np.ndarray,
        visible_region: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Write a best-effort ADB/HIK alignment check; never raise."""

        evidence_directory = Path(output_directory) / "cross_source_check"
        result: Dict[str, Any] = {
            "schema_version": 1,
            "status": "unavailable",
            "non_gating": True,
            "method": "saved_rig_geometry_adb_crop_vs_rectified_hik_charuco",
            "confidence": None,
            "evidence_directory": "cross_source_check",
            "evidence_files": {},
        }
        try:
            evidence_directory.mkdir(parents=True, exist_ok=True)
            if not self._opened:
                raise RuntimeError("camera session is not open")
            presentation = self.target.present_charuco()
            self._wait_painted(
                presentation, timeout_seconds=self.options.operation_timeout_seconds
            )
            screenshot = getattr(self.target, "last_screenshot", None)
            if screenshot is None:
                raise RuntimeError("ADB screenshot is unavailable")
            camera_frame = self._capture_settled()
            roi = self.hardware_roi or [0, 0, camera_frame.shape[1], camera_frame.shape[0]]
            roi_x, roi_y, _, _ = map(int, roi)
            map_x, map_y = rectification_maps
            hik_rectified = cv2.remap(
                camera_frame,
                np.asarray(map_x, dtype=np.float32) - float(roi_x),
                np.asarray(map_y, dtype=np.float32) - float(roi_y),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            x, y, width, height = map(int, visible_region["xywh"])
            adb_crop = screenshot[y : y + height, x : x + width].copy()
            if adb_crop.shape[:2] != (height, width):
                raise RuntimeError(
                    "ADB screenshot does not contain the camera-visible phone region"
                )
            metrics, images = cross_source_alignment_evidence(
                adb_crop, hik_rectified, valid_mask
            )
            written = {}
            for name, image in images.items():
                if cv2.imwrite(str(evidence_directory / name), image):
                    written[name.rsplit(".", 1)[0]] = name
            result.update(
                {
                    "status": "measured",
                    **metrics,
                    "source_spaces": {
                        "adb": "phone display pixels cropped by camera_visible_screen_region",
                        "hik": "hardware ROI rectified by saved rig geometry",
                    },
                    "camera_visible_screen_region_xywh": [x, y, width, height],
                    "camera_hardware_roi_xywh": list(map(int, roi)),
                    "evidence_files": written,
                }
            )
            self.progress(
                "Cross-source alignment check: confidence {:.3f}; evidence saved with calibration."
                .format(result["confidence"])
            )
        except Exception as exc:
            result["error"] = str(exc)
            try:
                self.progress(
                    "Cross-source alignment check unavailable (non-gating): {}".format(exc)
                )
            except Exception:
                pass
        try:
            evidence_directory.mkdir(parents=True, exist_ok=True)
            (evidence_directory / "cross_source_check.json").write_text(
                json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
            )
            write_commented_yaml(
                evidence_directory / "cross_source_check.yaml",
                result,
                header=(
                    "# Non-gating ADB/HIK coordinate-alignment evidence.\n"
                    "# Confidence summarizes the saved rig mapping; no transform is fitted here."
                ),
                section_comments={
                    "source_spaces": "Coordinate systems compared by this diagnostic.",
                    "evidence_files": "Relative review images saved beside this result.",
                },
            )
        except Exception as exc:
            result["evidence_write_error"] = str(exc)
        self.cross_source_check = result
        return result

    def save(self) -> Path:
        if self._saved:
            return Path(self.options.output_directory)
        phone_metrics = self._required(self.phone_metrics, "phone metrics")
        correspondences = self._required(self.correspondences, "ChArUco correspondences")
        geometry = self._required(self.geometry, "screen geometry")
        visible_region = self._required(self.visible_region, "camera-visible screen region")
        exposure = self._required(self.exposure, "exposure calibration")
        white_balance = self._required(self.white_balance, "white-balance calibration")
        orientation_evidence = self._required(
            self.orientation_evidence, "ChArUco orientation evidence"
        )
        output = Path(self.options.output_directory).resolve()
        if output.exists():
            raise FileExistsError("Calibration output already exists: {}".format(output))
        if self._opened:
            self.benchmark_final_stream()
        temporary = output.parent / ".{}.tmp-{}".format(output.name, uuid.uuid4().hex)
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            phone_value = {
                **phone_metrics.to_dict(),
                "calibration_display_brightness": self.phone_display_brightness,
                "viewer": dict(self.viewer_metrics),
            }
            x, y, width, height = visible_region["xywh"]
            full_size = [int(self.camera_metadata["width_px"]), int(self.camera_metadata["height_px"])]
            camera_roi = self.hardware_roi or self.camera.align_roi(
                camera_roi_for_screen_region(
                    visible_region["xywh"], geometry.inverse_matrix_3x3, full_size
                )
            )
            calibration, _, valid_mask = build_calibration(
                calibration_id="hik-{}".format(uuid.uuid4().hex),
                camera_points_xy=correspondences["camera_points_xy"],
                screen_points_xy=correspondences["screen_points_xy"],
                camera_size_px=full_size,
                screen_size_px=phone_metrics.screen_size_px,
                input_frame_id="hik://{}/full_sensor".format(self.options.camera_id),
                canonical_screen_frame_id="android://{}/display".format(self.options.phone_serial),
                output_origin_screen_xy=[x, y],
                output_size_px=[width, height],
                rig={
                    "camera": {**dict(self.camera_metadata), "controls": dict(self.camera_controls)},
                    "phone": phone_value,
                },
                image_quality={"focus_history": self.focus_history, "confidence": geometry.confidence},
                data_matrix_decode=self.data_matrix_result or {},
                # Acceptance remains a separate review/gating action; saving
                # evidence must never silently promote the artifact.
                status="warning",
            )
            calibration["normalization"]["input_space"] = "camera_sensor_bgr_px"
            calibration["normalization"]["orientation"] = {
                **dict(orientation_evidence),
                "output_x_axis": "app_right",
                "output_y_axis": "app_down",
            }
            normalization_matrix = np.asarray(calibration["normalization"]["matrix_3x3"], dtype=np.float64)
            maps = self._rectification_maps(normalization_matrix, [width, height])
            try:
                cross_source_check = self._save_cross_source_check(
                    temporary, maps, valid_mask, visible_region
                )
            except Exception as exc:
                # This diagnostic is deliberately outside calibration acceptance.
                # Even an implementation or filesystem error cannot reject an
                # otherwise complete rig calibration.
                cross_source_check = {
                    "schema_version": 1,
                    "status": "unavailable",
                    "non_gating": True,
                    "method": "saved_rig_geometry_adb_crop_vs_rectified_hik_charuco",
                    "confidence": None,
                    "error": str(exc),
                    "evidence_directory": "cross_source_check",
                    "evidence_files": {},
                }
            calibration["evidence"]["cross_source_check"] = {
                "status": cross_source_check.get("status"),
                "confidence": cross_source_check.get("confidence"),
                "non_gating": True,
                "result_file": "cross_source_check/cross_source_check.json",
            }
            write_calibration_bundle(temporary, calibration, valid_mask, maps)
            config = {
                "schema_version": 1,
                "camera": {
                    "adapter_id": "hik_mvs",
                    "device_id": self.options.camera_id,
                    "metadata": dict(self.camera_metadata),
                    "controls": dict(self.camera_controls),
                    "full_sensor_mode": {
                        "width_px": full_size[0],
                        "height_px": full_size[1],
                        "fps": float(self.camera_metadata.get("fps", self.options.camera_fps)),
                    },
                    "hardware_roi_xywh": camera_roi,
                },
                "phone": phone_value,
                "imaging": {
                    "exposure_us": exposure.exposure_us,
                    "shutter_refresh_multiplier": exposure.shutter_refresh_multiplier,
                    "shutter_rate_hz": exposure.shutter_rate_hz,
                    "exposure_refresh_periods": exposure.exposure_refresh_periods,
                    "temporal_noise_bgr": list(exposure.temporal_noise_bgr),
                    "gain": exposure.gain,
                    "black_level": self.black_level,
                    "white_balance": dict(white_balance),
                    "final_balanced_white": self.final_white_statistics,
                    "white_balance_attempts": self.white_balance_attempts,
                    "calibration_warnings": self.calibration_warnings,
                },
                "geometry": {
                    "charuco_layout": {
                        "squares_x": int(self.charuco_layout.squares_x),
                        "squares_y": int(self.charuco_layout.squares_y),
                        "margin_px": list(self.charuco_layout.margin_px),
                        "board_size_px": list(self.charuco_layout.board_size_px),
                    },
                    "full_sensor_camera_to_screen_3x3": geometry.matrix_3x3.tolist(),
                    "screen_to_full_sensor_camera_3x3": geometry.inverse_matrix_3x3.tolist(),
                    "camera_visible_screen_region": dict(visible_region),
                },
                "normalization": {
                    "full_sensor_camera_to_output_3x3": normalization_matrix.tolist(),
                    "output_size_px": [width, height],
                    "origin_screen_xy": [x, y],
                    "dense_map_file": "rectification_maps.npz",
                    "orientation": {
                        **dict(orientation_evidence),
                        "output_x_axis": "app_right",
                        "output_y_axis": "app_down",
                    },
                },
                "results": {
                    "hik_one_shot_auto_seed": self.auto_imaging_seed,
                    "exposure_observations": [
                        self._exposure_observation_dict(row)
                        for row in self.exposure_observations
                    ],
                    "final_balanced_white": self.final_white_statistics,
                    "black_level_observations": [
                        row.__dict__ for row in self.black_level_observations
                    ],
                    "cv_verification": self.cv_verification,
                    "transport_benchmark": self.transport_benchmark,
                    "latency_benchmark": self.latency_benchmark,
                    "cross_source_check": cross_source_check,
                    "focus_history": self.focus_history,
                    "focus_pose_history": self.focus_pose_history,
                    "data_matrix": self.data_matrix_result,
                },
            }
            (temporary / "hik_camera_calibration.json").write_text(
                json.dumps(config, indent=2, sort_keys=True), encoding="utf-8"
            )
            write_commented_yaml(
                temporary / "hik_camera_calibration.yaml",
                config,
                header=HIK_CONFIG_HEADER,
                section_comments=HIK_CONFIG_COMMENTS,
            )
            if self.last_frame is not None:
                cv2.imwrite(str(temporary / "last_camera_frame.png"), self.last_frame)
            os.replace(str(temporary), str(output))
            self._saved = True
            self.progress("Saved calibration bundle: {}".format(output))
            return output
        except Exception:
            shutil.rmtree(str(temporary), ignore_errors=True)
            raise

    def run(self) -> Optional[Path]:
        self.open()
        try:
            try:
                self.calibrate_geometry()
                self.calibrate_black_level()
                self.calibrate_once_auto_imaging()
                self.calibrate_exposure()
                self.calibrate_white_balance()
                self.verify_final_imaging()
                if self.options.grade_data_matrix:
                    self.grade_data_matrix()
                if self.options.headless:
                    return self.save() if self.options.save_without_prompt else None
                while True:
                    action = self.focus_loop()
                    if action == "data_matrix":
                        self.grade_data_matrix()
                        continue
                    if action == "recalibrate":
                        self.progress(
                            "Recalibrating geometry and locked imaging after rig movement..."
                        )
                        self._preview_disabled = False
                        self.exposure_observations = []
                        self.final_white_statistics = None
                        self.white_balance_attempts = []
                        self.calibration_warnings = []
                        self.cv_verification = None
                        self.calibrate_geometry()
                        self.calibrate_once_auto_imaging()
                        self.calibrate_exposure()
                        self.calibrate_white_balance()
                        self.verify_final_imaging()
                        continue
                    if action == "save":
                        return self.save()
                    return None
            except Exception as exc:
                evidence = self._write_failure_evidence(exc)
                if evidence is not None:
                    self.progress("Failure evidence saved for review: {}".format(evidence))
                raise
        finally:
            self.close()

    def _write_failure_evidence(self, error: Exception) -> Optional[Path]:
        """Save review images only; never create a failed calibration bundle."""

        target_image = getattr(self.target, "last_target", None)
        phone_image = getattr(self.target, "last_screenshot", None)
        if self.last_frame is None and target_image is None and phone_image is None:
            return None
        root = Path(self.options.output_directory).resolve().parent
        evidence = root / "{}-failure-evidence-{}".format(
            Path(self.options.output_directory).name,
            time.strftime("%Y%m%d-%H%M%S"),
        )
        evidence.mkdir(parents=True, exist_ok=False)
        if target_image is not None:
            cv2.imwrite(str(evidence / "display-target.png"), target_image)
        if phone_image is not None:
            cv2.imwrite(str(evidence / "phone-screenshot.png"), phone_image)
        if self.last_frame is not None:
            cv2.imwrite(str(evidence / "raw-hik-frame.png"), self.last_frame)
        if self.auto_target_image is not None:
            cv2.imwrite(str(evidence / "auto-neutral-target.png"), self.auto_target_image)
        if self.auto_phone_screenshot is not None:
            cv2.imwrite(
                str(evidence / "auto-neutral-phone-screenshot.png"),
                self.auto_phone_screenshot,
            )
        if self.auto_result_frame is not None:
            cv2.imwrite(str(evidence / "auto-neutral-hik-frame.png"), self.auto_result_frame)
        (evidence / "failure.json").write_text(
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "camera": dict(self.camera_metadata),
                    "phone": self.phone_metrics.to_dict() if self.phone_metrics else None,
                    "phone_calibration_display_brightness": self.phone_display_brightness,
                    "viewer": dict(self.viewer_metrics),
                    "hik_one_shot_auto_seed": self.auto_imaging_seed,
                    "exposure_observations": [
                        self._exposure_observation_dict(row)
                        for row in self.exposure_observations
                    ],
                    "final_balanced_white": self.final_white_statistics,
                    "white_balance_attempts": self.white_balance_attempts,
                    "calibration_warnings": self.calibration_warnings,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return evidence

    def close(self) -> None:
        errors = []
        self._close_preview(disable=True)
        try:
            self.camera.close()
        except Exception as exc:
            errors.append("camera: {}".format(exc))
        try:
            self.target.stop()
        except Exception as exc:
            errors.append("target: {}".format(exc))
        try:
            self.phone.cleanup(turn_display_off=True)
        except Exception as exc:
            errors.append("phone: {}".format(exc))
        self._opened = False
        if errors:
            self.progress("Cleanup warning: {}".format("; ".join(errors)))
