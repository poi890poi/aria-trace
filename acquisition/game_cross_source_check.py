"""Non-gating ADB/HIK alignment evidence for dual-source game capture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np

from .rig_calibration.hik.workflow import cross_source_alignment_evidence
from .rig_calibration.hik.spaces import RigCalibratedSpaceConverter


def natural_crop_to_logical(
    crop_xywh: Sequence[int],
    natural_size_px: Sequence[int],
    quarter_turns_clockwise: int,
) -> list[int]:
    """Map a phone-natural rectangle into the current logical display."""

    x, y, width, height = map(int, crop_xywh)
    natural_width, natural_height = map(int, natural_size_px)
    turns = int(quarter_turns_clockwise) % 4
    if turns == 0:
        return [x, y, width, height]
    if turns == 1:
        return [natural_height - y - height, x, height, width]
    if turns == 2:
        return [
            natural_width - x - width,
            natural_height - y - height,
            width,
            height,
        ]
    return [y, natural_width - x - width, height, width]


def load_game_alignment_geometry(
    calibration_file: Path,
    phone_surface_orientation: Mapping[str, object],
) -> dict:
    """Load the saved rig crop/mask and express them in game logical space."""

    calibration_file = Path(calibration_file)
    config = json.loads(calibration_file.read_text(encoding="utf-8"))
    normalization = config["normalization"]
    width, height = map(int, normalization["output_size_px"])
    converter = RigCalibratedSpaceConverter(
        config,
        int(
            phone_surface_orientation.get(
                "quarter_turns_clockwise_from_natural", 0
            )
        ),
    )
    mask_file = normalization.get("valid_mask_file", "valid_screen_mask.png")
    mask = cv2.imread(
        str(calibration_file.parent / str(mask_file)), cv2.IMREAD_GRAYSCALE
    )
    if mask is None:
        raise RuntimeError("Rig calibration valid-screen mask is missing")
    if mask.shape[:2] != (height, width):
        raise RuntimeError(
            "Rig valid-screen mask is {}x{}, expected {}x{}".format(
                mask.shape[1], mask.shape[0], width, height
            )
        )
    return {
        "logical_crop_xywh": converter.camera_adapter_bounds_in_adb_xywh(),
        "natural_crop_xywh": [
            *map(int, normalization["origin_screen_xy"]),
            width,
            height,
        ],
        "valid_mask": converter.camera_adapter_image_to_adb_orientation(mask),
        "android_surface_quarter_turns_clockwise_from_natural": (
            converter.adb_surface_quarter_turns_clockwise_from_natural
        ),
        "output_image_quarter_turns_clockwise_from_phone_natural": (
            converter.output_image_quarter_turns_clockwise_from_phone_natural
        ),
        "space_conversion": converter.describe(),
    }


class GameCrossSourceEvidenceRecorder:
    """Sample synchronized game frames and reuse the rig alignment checker.

    This recorder never gates or invalidates capture. It records the best
    observed comparison because animated game content and display latency can
    make an individual synchronized pair less representative.
    """

    def __init__(
        self,
        calibration_file: Path,
        phone_surface_orientation: Mapping[str, object],
        *,
        adb_stream_id: str = "android_phone",
        hik_stream_id: str = "hik_phone",
        sample_period_seconds: float = 1.0,
        maximum_pair_delta_ms: float = 250.0,
    ) -> None:
        self.calibration_file = Path(calibration_file)
        self.geometry = load_game_alignment_geometry(
            self.calibration_file, phone_surface_orientation
        )
        self.adb_stream_id = str(adb_stream_id)
        self.hik_stream_id = str(hik_stream_id)
        self.sample_period_ns = int(float(sample_period_seconds) * 1.0e9)
        self.maximum_pair_delta_ns = int(float(maximum_pair_delta_ms) * 1.0e6)
        self._path: Optional[Path] = None
        self._latest_adb = None
        self._last_sample_ns: Optional[int] = None
        self._evaluated_pairs = 0
        self._best_metrics = None
        self._best_images = None
        self._best_pair_delta_ms = None
        self._warning = None

    def start(self, session_path: Path, _session_id: str, _origin_ns: int) -> None:
        self._path = Path(session_path) / "cross_source_check"

    def process(self, packet, _frame_index: int, _session_time_ns: int) -> None:
        if packet.stream_id == self.adb_stream_id:
            self._latest_adb = (
                int(packet.host_capture_time_ns),
                packet.image.copy(),
            )
            return
        if packet.stream_id != self.hik_stream_id or self._latest_adb is None:
            return
        capture_ns = int(packet.host_capture_time_ns)
        if (
            self._last_sample_ns is not None
            and capture_ns - self._last_sample_ns < self.sample_period_ns
        ):
            return
        adb_ns, adb_image = self._latest_adb
        delta_ns = abs(capture_ns - adb_ns)
        if delta_ns > self.maximum_pair_delta_ns:
            return
        self._last_sample_ns = capture_ns
        x, y, width, height = self.geometry["logical_crop_xywh"]
        if (
            min(x, y, width, height) < 0
            or x + width > adb_image.shape[1]
            or y + height > adb_image.shape[0]
        ):
            self._warning = "ADB frame does not contain the saved rig-visible crop"
            return
        adb_crop = adb_image[y : y + height, x : x + width].copy()
        hik_image = packet.image[:height, :width].copy()
        if hik_image.shape[:2] != adb_crop.shape[:2]:
            self._warning = "Calibrated HIK frame does not match the rig-visible crop"
            return
        try:
            metrics, images = cross_source_alignment_evidence(
                adb_crop,
                hik_image,
                self.geometry["valid_mask"],
            )
        except Exception as exc:
            self._warning = "{}: {}".format(type(exc).__name__, exc)
            return
        self._evaluated_pairs += 1
        if (
            self._best_metrics is None
            or float(metrics["confidence"])
            > float(self._best_metrics["confidence"])
        ):
            self._best_metrics = dict(metrics)
            self._best_images = dict(images)
            self._best_pair_delta_ms = delta_ns / 1.0e6

    def close(self, status: str = "complete") -> None:
        if self._path is None:
            return
        self._path.mkdir(parents=True, exist_ok=True)
        result = {
            "status": "measured" if self._best_metrics is not None else "unavailable",
            "capture_status": str(status),
            "method": "saved_rig_geometry_then_cross_source_alignment_evidence",
            "non_gating": True,
            "rig_calibration": str(self.calibration_file.resolve()),
            "adb_stream_id": self.adb_stream_id,
            "hik_stream_id": self.hik_stream_id,
            "logical_adb_crop_xywh": self.geometry["logical_crop_xywh"],
            "evaluated_pairs": self._evaluated_pairs,
            "best_pair_delta_ms": self._best_pair_delta_ms,
            "metrics": self._best_metrics,
            "warning": self._warning,
        }
        (self._path / "summary.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        for name, image in (self._best_images or {}).items():
            cv2.imwrite(str(self._path / name), image)

    def describe(self) -> dict:
        result = {
            "type": type(self).__name__,
            "method": "reuse_rig_cross_source_alignment_evidence_on_game_frames",
            "non_gating": True,
            "path": "cross_source_check/summary.json",
            "logical_adb_crop_xywh": self.geometry["logical_crop_xywh"],
            "evaluated_pairs": self._evaluated_pairs,
        }
        if self._best_metrics is not None:
            result["best_metrics"] = dict(self._best_metrics)
            result["best_pair_delta_ms"] = self._best_pair_delta_ms
        if self._warning:
            result["warning"] = self._warning
        return result
