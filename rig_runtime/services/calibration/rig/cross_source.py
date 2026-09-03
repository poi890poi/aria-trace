"""Non-gating ADB/HIK alignment evidence for dual-source game capture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np

from rig_runtime.evidence.rig_alignment import (
    cross_source_alignment_evidence,
    cross_source_alignment_warning,
)
from .hik.spaces import RigCalibratedSpaceConverter


def _load_optional_valid_mask(
    calibration_file: Path,
    normalization: Mapping[str, object],
    expected_size_px: Sequence[int],
) -> tuple[Optional[np.ndarray], dict]:
    """Load validity evidence without making its absence block acquisition."""

    width, height = map(int, expected_size_px)
    filename = str(
        normalization.get("valid_mask_file") or "valid_screen_mask.png"
    )
    path = Path(calibration_file).parent / filename
    status = {
        "file": filename,
        "expected_size_px": [width, height],
        "required_for_acquisition": False,
    }
    if not path.is_file():
        return None, {**status, "status": "missing"}
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None, {**status, "status": "unreadable"}
    if mask.shape[:2] != (height, width):
        return None, {
            **status,
            "status": "shape_mismatch",
            "actual_size_px": [int(mask.shape[1]), int(mask.shape[0])],
        }
    return mask, {**status, "status": "available"}


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
    mask, mask_status = _load_optional_valid_mask(
        calibration_file, normalization, [width, height]
    )
    return {
        "logical_crop_xywh": converter.camera_adapter_bounds_in_adb_xywh(),
        "natural_crop_xywh": [
            *map(int, normalization["origin_screen_xy"]),
            width,
            height,
        ],
        "valid_mask": (
            converter.camera_adapter_image_to_adb_orientation(mask)
            if mask is not None
            else None
        ),
        "valid_mask_status": mask_status,
        "android_surface_quarter_turns_clockwise_from_natural": (
            converter.adb_surface_quarter_turns_clockwise_from_natural
        ),
        "output_image_quarter_turns_clockwise_from_calibration_display": (
            converter.output_image_quarter_turns_clockwise_from_calibration_display
        ),
        "space_conversion": converter.describe(),
    }


def match_game_camera_orientation(
    adb_image: np.ndarray,
    hik_calibration_display_image: np.ndarray,
    calibration_file: Path,
    *,
    android_reported_quarter_turns: Optional[int] = None,
    preferred_confidence: float = 0.50,
    preferred_margin: float = 0.08,
) -> tuple[dict, dict]:
    """Select HIK output orientation solely from ADB/HIK image evidence.

    The HIK input must be the rig-normalized visible phone region in the
    calibration display's app-up/app-right space. Candidates represent the
    current ADB surface orientation. The converter alone supplies the relative
    image rotation and coordinate transform for each candidate.
    """

    calibration_file = Path(calibration_file)
    config = json.loads(calibration_file.read_text(encoding="utf-8"))
    normalization = config["normalization"]
    natural_width, natural_height = map(
        int, config["phone"]["natural_screen_size_px"]
    )
    expected_width, expected_height = map(
        int, normalization["output_size_px"]
    )
    if hik_calibration_display_image.shape[:2] != (
        expected_height,
        expected_width,
    ):
        raise RuntimeError(
            "HIK orientation evidence is {}x{}, expected rig-normalized {}x{}"
            .format(
                hik_calibration_display_image.shape[1],
                hik_calibration_display_image.shape[0],
                expected_width,
                expected_height,
            )
        )
    valid_mask, valid_mask_status = _load_optional_valid_mask(
        calibration_file, normalization, [expected_width, expected_height]
    )
    comparison_mask = (
        valid_mask
        if valid_mask is not None
        else np.full((expected_height, expected_width), 255, dtype=np.uint8)
    )

    adb_height, adb_width = adb_image.shape[:2]
    candidates = []
    evidence_images = {
        "first_adb_game_image.png": adb_image.copy(),
        "first_hik_rig_normalized_calibration_display.png": (
            hik_calibration_display_image.copy()
        ),
    }
    scored = []
    for surface_turns in range(4):
        converter = RigCalibratedSpaceConverter(config, surface_turns)
        image_turns = int(
            converter.output_image_quarter_turns_clockwise_from_calibration_display
        )
        candidate = {
            "adb_surface_quarter_turns_clockwise_from_phone_natural": surface_turns,
            "adb_surface_degrees_clockwise_from_phone_natural": surface_turns * 90,
            "camera_adapter_image_quarter_turns_clockwise_from_calibration_display": image_turns,
            "camera_adapter_image_degrees_clockwise_from_calibration_display": image_turns * 90,
            "expected_adb_size_px": list(converter.adb_size_px),
            "logical_adb_crop_xywh": (
                converter.camera_adapter_bounds_in_adb_xywh()
            ),
            "metrics": None,
        }
        if converter.adb_size_px != (adb_width, adb_height):
            candidate["status"] = "not_scored_adb_raster_size_mismatch"
            candidates.append(candidate)
            continue
        x, y, width, height = candidate["logical_adb_crop_xywh"]
        if (
            min(x, y, width, height) < 0
            or x + width > adb_width
            or y + height > adb_height
        ):
            candidate["status"] = "not_scored_crop_outside_adb_image"
            candidates.append(candidate)
            continue
        adb_crop = adb_image[y : y + height, x : x + width].copy()
        hik_candidate = converter.camera_adapter_image_to_adb_orientation(
            hik_calibration_display_image
        )
        candidate_mask = converter.camera_adapter_image_to_adb_orientation(
            comparison_mask
        )
        if hik_candidate.shape[:2] != adb_crop.shape[:2]:
            candidate["status"] = "not_scored_image_size_mismatch"
            candidates.append(candidate)
            continue
        metrics, images = cross_source_alignment_evidence(
            adb_crop, hik_candidate, candidate_mask
        )
        candidate["status"] = "scored"
        candidate["metrics"] = dict(metrics)
        candidates.append(candidate)
        scored.append(candidate)
        prefix = "candidate_surface_{}_adapter_{}deg_".format(
            surface_turns, image_turns * 90
        )
        evidence_images[prefix + "adb_crop.png"] = adb_crop
        evidence_images[prefix + "hik.png"] = hik_candidate
        for name, image in images.items():
            evidence_images[prefix + name] = image

    if not scored:
        raise RuntimeError(
            "No HIK orientation candidate matches the first ADB raster {}x{}; "
            "phone natural raster is {}x{}".format(
                adb_width,
                adb_height,
                natural_width,
                natural_height,
            )
        )
    ranked = sorted(
        scored,
        key=lambda value: float(value["metrics"]["confidence"]),
        reverse=True,
    )
    best = ranked[0]
    best_confidence = float(best["metrics"]["confidence"])
    runner_up_confidence = (
        float(ranked[1]["metrics"]["confidence"])
        if len(ranked) > 1
        else None
    )
    margin = (
        best_confidence - runner_up_confidence
        if runner_up_confidence is not None
        else None
    )
    preferred = best_confidence >= float(preferred_confidence) and (
        margin is None or margin >= float(preferred_margin)
    )
    selected_surface_turns = int(
        best["adb_surface_quarter_turns_clockwise_from_phone_natural"]
    )
    selected_image_turns = int(
        best["camera_adapter_image_quarter_turns_clockwise_from_calibration_display"]
    )
    warnings = []
    if valid_mask is None:
        warnings.append(
            "Rig valid-screen mask is {}; orientation used an explicitly "
            "unmasked full-output comparison.".format(valid_mask_status["status"])
        )
    if not preferred:
        warnings.append(
            "The best image-evidence candidate was applied, but the match is ambiguous."
        )
    summary = {
        "schema_version": 1,
        "status": "selected" if preferred else "selected_low_confidence",
        "selection_basis": "first_game_adb_and_hik_image_evidence_only",
        "runtime_operation": "space_converter_discrete_quarter_turn_no_interpolation",
        "rectification_note": (
            "A rectified HIK image is required only for this evidence check; "
            "the selected quarter-turn can be applied to an unrectified ROI stream."
        ),
        "rig_calibration": str(calibration_file.resolve()),
        "first_adb_size_px": [adb_width, adb_height],
        "hik_rig_normalized_calibration_display_size_px": [
            expected_width,
            expected_height,
        ],
        "rig_calibration_display_quarter_turns_clockwise_from_natural": int(
            config.get("phone", {}).get("orientation_quarter_turns", 0)
        ) % 4,
        "android_reported_quarter_turns_clockwise_from_natural": (
            int(android_reported_quarter_turns) % 4
            if android_reported_quarter_turns is not None
            else None
        ),
        "selected_adb_surface_quarter_turns_clockwise_from_phone_natural": (
            selected_surface_turns
        ),
        "selected_adb_surface_degrees_clockwise_from_phone_natural": (
            selected_surface_turns * 90
        ),
        "selected_camera_adapter_image_quarter_turns_clockwise_from_calibration_display": (
            selected_image_turns
        ),
        "selected_camera_adapter_image_degrees_clockwise_from_calibration_display": (
            selected_image_turns * 90
        ),
        "selected_confidence": best_confidence,
        "runner_up_confidence": runner_up_confidence,
        "confidence_margin": margin,
        "preferred_confidence": float(preferred_confidence),
        "preferred_margin": float(preferred_margin),
        "valid_screen_mask": {
            **valid_mask_status,
            "orientation_comparison": (
                "saved_valid_mask"
                if valid_mask is not None
                else "full_output_unmasked_fallback"
            ),
        },
        "warning": " ".join(warnings) if warnings else None,
        "warnings": warnings,
        "candidates": candidates,
    }
    return summary, evidence_images


def orient_hik_source_from_first_adb_frame(
    adb_source,
    hik_source,
    calibration_file: Path,
    *,
    android_reported_quarter_turns: Optional[int] = None,
) -> tuple[dict, dict]:
    """Start reusable sources and orient HIK from their first game images."""

    hik_source.set_output_orientation(0)
    adb_source.start()
    hik_source.start()
    adb_packet = adb_source.read()
    if adb_packet is None:
        raise RuntimeError("ADB source ended before its first game image")
    hik_packet = hik_source.read()
    if hik_packet is None:
        raise RuntimeError("HIK source ended before its first game image")
    hik_calibration_display = hik_source.alignment_evidence_image(hik_packet)
    summary, images = match_game_camera_orientation(
        adb_packet.image,
        hik_calibration_display,
        calibration_file,
        android_reported_quarter_turns=android_reported_quarter_turns,
    )
    summary["first_frame_pair_delta_ms"] = abs(
        int(hik_packet.host_capture_time_ns)
        - int(adb_packet.host_capture_time_ns)
    ) / 1.0e6
    selected_image_turns = int(
        summary[
            "selected_camera_adapter_image_quarter_turns_clockwise_from_calibration_display"
        ]
    )
    hik_source.set_output_orientation(
        selected_image_turns,
        {
            "status": summary["status"],
            "selection_basis": summary["selection_basis"],
            "selected_confidence": summary["selected_confidence"],
            "confidence_margin": summary["confidence_margin"],
        },
    )
    return summary, images


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
        orientation_match: Optional[Mapping[str, object]] = None,
        orientation_evidence_images: Optional[Mapping[str, np.ndarray]] = None,
    ) -> None:
        self.calibration_file = Path(calibration_file)
        self.orientation_match = (
            dict(orientation_match) if orientation_match is not None else None
        )
        self.orientation_evidence_images = {
            str(name): image.copy()
            for name, image in (orientation_evidence_images or {}).items()
        }
        effective_surface = dict(phone_surface_orientation)
        if self.orientation_match is not None:
            selected = self.orientation_match.get(
                "selected_adb_surface_quarter_turns_clockwise_from_phone_natural"
            )
            if selected is not None:
                effective_surface["quarter_turns_clockwise_from_natural"] = (
                    int(selected) % 4
                )
        self.geometry = load_game_alignment_geometry(
            self.calibration_file, effective_surface
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
        self._warning = (
            None
            if self.geometry["valid_mask"] is not None
            else "Cross-source quality evidence skipped because the rig "
            "valid-screen mask is {}.".format(
                self.geometry["valid_mask_status"]["status"]
            )
        )

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
        if self.geometry["valid_mask"] is None:
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
        spatial_warning = cross_source_alignment_warning(metrics)
        if spatial_warning:
            self._warning = spatial_warning
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
            "valid_screen_mask": self.geometry["valid_mask_status"],
            "orientation_match": self.orientation_match,
        }
        (self._path / "summary.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        for name, image in (self._best_images or {}).items():
            cv2.imwrite(str(self._path / name), image)
        if self.orientation_match is not None:
            orientation_path = self._path / "orientation_match"
            orientation_path.mkdir(parents=True, exist_ok=True)
            (orientation_path / "summary.json").write_text(
                json.dumps(self.orientation_match, indent=2), encoding="utf-8"
            )
            for name, image in self.orientation_evidence_images.items():
                cv2.imwrite(str(orientation_path / name), image)

    def describe(self) -> dict:
        result = {
            "type": type(self).__name__,
            "method": "reuse_rig_cross_source_alignment_evidence_on_game_frames",
            "non_gating": True,
            "path": "cross_source_check/summary.json",
            "logical_adb_crop_xywh": self.geometry["logical_crop_xywh"],
            "evaluated_pairs": self._evaluated_pairs,
            "valid_screen_mask": self.geometry["valid_mask_status"],
        }
        if self._best_metrics is not None:
            result["best_metrics"] = dict(self._best_metrics)
            result["best_pair_delta_ms"] = self._best_pair_delta_ms
        if self._warning:
            result["warning"] = self._warning
        if self.orientation_match is not None:
            result["orientation_match"] = {
                "status": self.orientation_match.get("status"),
                "selection_basis": self.orientation_match.get("selection_basis"),
                "selected_adb_surface_quarter_turns_clockwise_from_phone_natural": (
                    self.orientation_match.get(
                        "selected_adb_surface_quarter_turns_clockwise_from_phone_natural"
                    )
                ),
                "selected_camera_adapter_image_quarter_turns_clockwise_from_calibration_display": (
                    self.orientation_match.get(
                        "selected_camera_adapter_image_quarter_turns_clockwise_from_calibration_display"
                    )
                ),
                "selected_confidence": self.orientation_match.get(
                    "selected_confidence"
                ),
                "path": "cross_source_check/orientation_match/summary.json",
            }
        return result
