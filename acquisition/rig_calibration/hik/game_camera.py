"""Low-latency HIK streams configured by rig and mini-map calibration results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import cv2
import numpy as np

from ..app.device_adapters import CameraConfiguration
from ..contracts import FrameSample
from .algorithms import camera_roi_for_screen_region, compose_hardware_roi_homography
from .driver import HikMvsCameraAdapter


PathLike = Union[str, Path]


def _load_json(value: PathLike, names: Sequence[str]) -> tuple[Path, dict]:
    path = Path(value)
    if path.is_dir():
        matches = [path / name for name in names if (path / name).is_file()]
        if not matches:
            raise FileNotFoundError(
                "Calibration directory contains none of: {}".format(
                    ", ".join(names)
                )
            )
        path = matches[0]
    document = json.loads(path.read_text(encoding="utf-8"))
    current_revision = document.get("current_revision")
    if current_revision:
        revision = Path(current_revision)
        if not revision.is_absolute():
            revision = path.parent / revision
        return _load_json(revision, names)
    artifact = (document.get("artifacts") or {}).get("minimap_calibration")
    if artifact:
        artifact_path = Path(artifact)
        if not artifact_path.is_absolute():
            artifact_path = path.parent / artifact_path
        return _load_json(artifact_path, names)
    return path, document


def _translation(x: float, y: float) -> np.ndarray:
    return np.asarray(
        [[1.0, 0.0, x], [0.0, 1.0, y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _validate_crop(value: Sequence[int], label: str) -> list[int]:
    if len(value) != 4:
        raise ValueError("{} must contain x, y, width, height".format(label))
    crop = list(map(int, value))
    if crop[0] < 0 or crop[1] < 0 or crop[2] <= 0 or crop[3] <= 0:
        raise ValueError("{} is invalid: {}".format(label, crop))
    return crop


def _source_crop_to_canonical_phone(calibration: Mapping[str, object]) -> list[int]:
    """Resolve a mini-map crop into the rig's natural phone raster.

    New calibration results should write ``canonical_phone_crop_xywh``. A
    source crop is accepted only when its origin in that same coordinate space
    is explicit. Android logical-display crops are rotated back to the natural
    raster using the recorded surface orientation.
    """

    direct = calibration.get("canonical_phone_crop_xywh")
    if direct is not None:
        return _validate_crop(direct, "canonical_phone_crop_xywh")

    crop_value = calibration.get("crop_xywh")
    source_space = calibration.get("source_space") or {}
    if crop_value is None or not isinstance(source_space, Mapping):
        raise ValueError(
            "Mini-map calibration must provide canonical_phone_crop_xywh"
        )
    crop = _validate_crop(crop_value, "crop_xywh")
    origin = source_space.get("origin_in_canonical_phone_xy")
    if origin is None:
        raise ValueError(
            "Ambiguous mini-map crop: source_space.origin_in_canonical_phone_xy "
            "or canonical_phone_crop_xywh is required"
        )
    if len(origin) != 2:
        raise ValueError("source-space origin must contain x and y")

    image_source = str(calibration.get("image_source") or "")
    orientation = calibration.get("phone_surface_orientation") or {}
    quarter_turns = int(
        orientation.get("quarter_turns_clockwise_from_natural", 0)
        if isinstance(orientation, Mapping)
        else 0
    ) % 4
    natural_size = (
        orientation.get("natural_size_px")
        if isinstance(orientation, Mapping)
        else None
    )
    if image_source == "android_scrcpy" and quarter_turns:
        if natural_size is None or len(natural_size) != 2:
            raise ValueError(
                "Rotated Android crop requires phone_surface_orientation."
                "natural_size_px"
            )
        natural_width, natural_height = map(int, natural_size)
        x, y, width, height = crop
        if quarter_turns == 1:
            crop = [y, natural_height - x - width, height, width]
        elif quarter_turns == 2:
            crop = [
                natural_width - x - width,
                natural_height - y - height,
                width,
                height,
            ]
        else:
            crop = [natural_width - y - height, x, height, width]

    crop[0] += int(origin[0])
    crop[1] += int(origin[1])
    return _validate_crop(crop, "resolved canonical phone crop")


@dataclass(frozen=True)
class HikGameFrameSet:
    """Synchronized products derived from exactly one HIK acquisition."""

    time_ns: int
    receive_time_ns: int
    frame_number: Optional[int]
    streams: Mapping[str, np.ndarray]
    metadata: Mapping[str, object]


class ProfiledHikGameCamera:
    """Read a full-camera stream, a mini-map stream, or both.

    ``minimap`` applies a hardware ROI to reduce camera and USB throughput.
    ``dual`` acquires one full frame and derives its mini-map product from that
    frame, so both outputs always share the same clock and frame number.
    """

    MODES = ("minimap", "full", "dual")

    def __init__(
        self,
        rig_calibration: PathLike,
        minimap_calibration: PathLike,
        *,
        mode: str = "minimap",
        rectify_minimap: bool = True,
        adapter: Optional[HikMvsCameraAdapter] = None,
        minimap_margin_px: int = 6,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError("HIK game stream mode must be minimap, full, or dual")
        self.rig_path, self.rig = _load_json(
            rig_calibration, ("hik_camera_calibration.json",)
        )
        self.minimap_path, self.minimap = _load_json(
            minimap_calibration,
            ("minimap_calibration.json", "calibration.json"),
        )
        self.mode = mode
        self.rectify_minimap = bool(rectify_minimap)
        self.minimap_margin_px = int(minimap_margin_px)
        self.adapter = adapter or HikMvsCameraAdapter(
            sdk_python_path=self.rig.get("mvs_python_path")
        )
        self._opened = False
        self._effective_roi: Optional[list[int]] = None
        self._minimap_sensor_roi: Optional[list[int]] = None
        self._minimap_matrix: Optional[np.ndarray] = None
        self._minimap_size: Optional[tuple[int, int]] = None

    def _screen_crop(self) -> list[int]:
        crop = _source_crop_to_canonical_phone(self.minimap)
        phone_size = self.rig.get("phone", {}).get("natural_screen_size_px")
        if phone_size is None:
            phone_size = self.rig.get("phone", {}).get("natural_size_px")
        if phone_size is None:
            phone_size = self.rig.get("normalization", {}).get("phone_size_px")
        if phone_size is not None:
            width, height = map(int, phone_size)
            if crop[0] + crop[2] > width or crop[1] + crop[3] > height:
                raise ValueError(
                    "Canonical mini-map crop {} exceeds phone raster {}x{}".format(
                        crop, width, height
                    )
                )
        return crop

    def _sensor_size(self) -> list[int]:
        camera = self.rig["camera"]
        features = ((camera.get("controls") or {}).get("genicam") or {})
        width = (features.get("SensorWidth") or {}).get("value")
        height = (features.get("SensorHeight") or {}).get("value")
        mode = camera["full_sensor_mode"]
        return [
            int(width if width is not None else mode["width_px"]),
            int(height if height is not None else mode["height_px"]),
        ]

    def _full_mode_roi(self) -> list[int]:
        width, height = self._sensor_size()
        return [0, 0, width, height]

    def _apply_locked_imaging(self) -> None:
        imaging = self.rig["imaging"]
        if imaging.get("black_level") is not None:
            self.adapter.set_black_level(int(imaging["black_level"]))
        self.adapter.set_manual_imaging(imaging["exposure_us"], imaging["gain"])
        wb = imaging["white_balance"]
        self.adapter.set_white_balance(
            wb["ratio_red"], wb["ratio_green"], wb["ratio_blue"]
        )

    def open(self) -> "ProfiledHikGameCamera":
        if self._opened:
            return self
        camera = self.rig["camera"]
        full_mode = camera["full_sensor_mode"]
        self.adapter.open(
            CameraConfiguration(
                device_id=str(camera["device_id"]),
                width_px=int(full_mode["width_px"]),
                height_px=int(full_mode["height_px"]),
                fps=float(full_mode["fps"]),
                backend="hik_mvs",
            )
        )
        try:
            self._apply_locked_imaging()
            screen_crop = self._screen_crop()
            requested_minimap_roi = camera_roi_for_screen_region(
                screen_crop,
                self.rig["geometry"]["screen_to_full_sensor_camera_3x3"],
                self._sensor_size(),
                margin_px=self.minimap_margin_px,
            )
            self._minimap_sensor_roi = list(
                self.adapter.align_roi(requested_minimap_roi)
            )
            requested_roi = (
                self._minimap_sensor_roi
                if self.mode == "minimap"
                else self._full_mode_roi()
            )
            self._effective_roi = list(self.adapter.set_roi(requested_roi))

            camera_to_screen = np.asarray(
                self.rig["geometry"]["full_sensor_camera_to_screen_3x3"],
                dtype=np.float64,
            )
            acquisition_to_screen = compose_hardware_roi_homography(
                camera_to_screen, self._effective_roi
            )
            crop_x, crop_y, crop_width, crop_height = screen_crop
            self._minimap_matrix = _translation(-crop_x, -crop_y).dot(
                acquisition_to_screen
            )
            self._minimap_size = (crop_width, crop_height)
            self._opened = True
        except Exception:
            self.adapter.close()
            raise
        return self

    def isOpened(self) -> bool:
        return self._opened

    def _minimap_from_acquisition(self, image: np.ndarray) -> np.ndarray:
        if self.rectify_minimap:
            return cv2.warpPerspective(
                image,
                self._minimap_matrix,
                self._minimap_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        if self.mode == "minimap":
            return image
        acquisition_x, acquisition_y, _, _ = self._effective_roi
        mini_x, mini_y, mini_width, mini_height = self._minimap_sensor_roi
        left = mini_x - acquisition_x
        top = mini_y - acquisition_y
        return image[top : top + mini_height, left : left + mini_width].copy()

    def read_streams(self) -> HikGameFrameSet:
        if not self._opened:
            raise RuntimeError("Profiled HIK game camera is not open")
        sample = self.adapter.read()
        if self.mode == "minimap":
            streams: Dict[str, np.ndarray] = {
                "minimap": self._minimap_from_acquisition(sample.image)
            }
        elif self.mode == "full":
            streams = {"full": sample.image}
        else:
            streams = {
                "full": sample.image,
                "minimap": self._minimap_from_acquisition(sample.image),
            }
        frame_number = sample.metadata.get("frame_number")
        return HikGameFrameSet(
            time_ns=int(sample.time_ns),
            receive_time_ns=int(sample.receive_time_ns or sample.time_ns),
            frame_number=(int(frame_number) if frame_number is not None else None),
            streams=streams,
            metadata={
                **dict(sample.metadata),
                "mode": self.mode,
                "rectified_minimap": self.rectify_minimap,
                "acquisition_roi_xywh": list(self._effective_roi),
                "minimap_sensor_roi_xywh": list(self._minimap_sensor_roi),
                "one_acquisition_for_all_streams": True,
            },
        )

    def read_sample(self, stream_id: Optional[str] = None) -> FrameSample:
        frame_set = self.read_streams()
        selected = stream_id or (
            "minimap" if "minimap" in frame_set.streams else "full"
        )
        if selected not in frame_set.streams:
            raise KeyError(
                "Stream {!r} is unavailable in {} mode".format(selected, self.mode)
            )
        return FrameSample(
            image=frame_set.streams[selected],
            time_ns=frame_set.time_ns,
            receive_time_ns=frame_set.receive_time_ns,
            source_id="hik_game:{}".format(selected),
            metadata={**dict(frame_set.metadata), "stream_id": selected},
        )

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if not self._opened:
            return False, None
        try:
            return True, self.read_sample().image
        except Exception:
            return False, None

    def release(self) -> None:
        self.adapter.close()
        self._opened = False

    def __enter__(self) -> "ProfiledHikGameCamera":
        return self.open()

    def __exit__(self, *_exc) -> None:
        self.release()


__all__ = ["HikGameFrameSet", "ProfiledHikGameCamera"]
