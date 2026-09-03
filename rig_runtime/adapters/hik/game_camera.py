"""Low-latency HIK streams configured by rig and mini-map calibration results."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import cv2
import numpy as np

from rig_runtime.adapters.rig.devices import CameraConfiguration
from rig_runtime.domain.spatial import require_spatial_geometry
from rig_runtime.services.calibration.rig.contracts import FrameSample
from rig_runtime.services.calibration.rig.distortion import (
    distort_pixel_points,
    distorted_screen_region_roi,
)
from rig_runtime.services.calibration.rig.geometry import transform_points
from rig_runtime.services.calibration.rig.hik.algorithms import (
    camera_adapter_roi_to_output_homography,
    camera_roi_for_screen_region,
    compose_hardware_roi_homography,
)
from .driver import (
    HikMvsCameraAdapter,
    quarter_turn_output_geometry,
    rotate_quarter_turns_clockwise,
    rotate_xywh_in_parent,
)


PathLike = Union[str, Path]


class MinimapRoiUnavailableError(RuntimeError):
    """The calibrated mini-map cannot be acquired from the current rig view."""


def _projected_screen_crop_is_complete(
    screen_crop_xywh: Sequence[int],
    screen_to_camera_3x3: Sequence[Sequence[float]],
    camera_size_px: Sequence[int],
    *,
    lens_model: Optional[Mapping[str, object]] = None,
) -> bool:
    """Return whether every sampled crop-boundary point lies on the sensor."""

    x, y, width, height = map(float, screen_crop_xywh)
    count = 17 if lens_model else 2
    xs = np.linspace(x, x + width, count)
    ys = np.linspace(y, y + height, count)
    boundary = np.vstack(
        [
            np.column_stack([xs, np.full(count, y)]),
            np.column_stack([np.full(count, x + width), ys]),
            np.column_stack([xs[::-1], np.full(count, y + height)]),
            np.column_stack([np.full(count, x), ys[::-1]]),
        ]
    )
    projected = transform_points(boundary, np.asarray(screen_to_camera_3x3))
    if lens_model:
        projected = distort_pixel_points(projected, lens_model)
    if not np.all(np.isfinite(projected)):
        return False
    camera_width, camera_height = map(int, camera_size_px)
    return bool(
        np.all(projected[:, 0] >= 0.0)
        and np.all(projected[:, 1] >= 0.0)
        # XYWH rectangles use an exclusive right/bottom boundary throughout
        # the rig ROI code, so a boundary exactly on width/height is visible.
        and np.all(projected[:, 0] <= float(camera_width))
        and np.all(projected[:, 1] <= float(camera_height))
    )


def _load_json(value: PathLike, names: Sequence[str]) -> tuple[Path, dict]:
    path = Path(value)
    if path.is_dir():
        raise ValueError(
            "Directory-based calibration selection is obsolete; pass the exact "
            "immutable revision file"
        )
    if not path.is_file():
        raise FileNotFoundError(
            "Calibration file does not exist: {} (expected {})".format(
                path, ", ".join(names)
            )
        )
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("current_revision") or (document.get("artifacts") or {}).get(
        "minimap_calibration"
    ):
        raise ValueError(
            "Mutable current/artifact profile pointers are obsolete; resolve an "
            "active immutable revision through ProfileRegistry"
        )
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
    stream_metadata: Mapping[str, Mapping[str, object]] = field(default_factory=dict)


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
        apply_game_color: bool = True,
        bayer_conversion: Optional[Mapping[str, object]] = None,
        output_quarter_turns_clockwise: int = 0,
        runtime_surface_quarter_turns_clockwise_from_natural: Optional[int] = None,
        best_effort_initialization: bool = False,
        mask_policy: str = "none",
    ) -> None:
        if mode not in self.MODES:
            raise ValueError("HIK game stream mode must be minimap, full, or dual")
        self.rig_path, self.rig = _load_json(
            rig_calibration, ("hik_camera_calibration.json",)
        )
        self.minimap_path, loaded_minimap = _load_json(
            minimap_calibration,
            ("profile.json", "minimap_calibration.json", "calibration.json"),
        )
        payload = loaded_minimap.get("payload")
        self.minimap = (
            {**loaded_minimap, **dict(payload)}
            if isinstance(payload, Mapping)
            else loaded_minimap
        )
        self.mode = mode
        self.rectify_minimap = bool(rectify_minimap)
        self.minimap_margin_px = int(minimap_margin_px)
        self.apply_game_color = bool(apply_game_color)
        self.bayer_conversion = dict(bayer_conversion or {})
        self.output_quarter_turns_clockwise = (
            int(output_quarter_turns_clockwise) % 4
        )
        self.runtime_surface_quarter_turns_clockwise_from_natural = (
            int(runtime_surface_quarter_turns_clockwise_from_natural) % 4
            if runtime_surface_quarter_turns_clockwise_from_natural is not None
            else None
        )
        self.best_effort_initialization = bool(best_effort_initialization)
        self.mask_policy = str(mask_policy)
        if self.mask_policy not in ("none", "minimap_circle"):
            raise ValueError("Mask policy must be none or minimap_circle")
        if self.mask_policy != "none" and self.mode not in ("minimap", "dual"):
            raise ValueError("Mini-map masking requires minimap or dual mode")
        if self.mask_policy != "none" and not self.rectify_minimap:
            raise ValueError("Mini-map masking requires rectification")
        self.adapter = adapter or HikMvsCameraAdapter(
            sdk_python_path=self.rig.get("mvs_python_path")
        )
        self._opened = False
        self._effective_roi: Optional[list[int]] = None
        self._minimap_sensor_roi: Optional[list[int]] = None
        self._minimap_matrix: Optional[np.ndarray] = None
        self._minimap_size: Optional[tuple[int, int]] = None
        self._full_matrix: Optional[np.ndarray] = None
        self._full_size: Optional[tuple[int, int]] = None
        self._full_map_x: Optional[np.ndarray] = None
        self._full_map_y: Optional[np.ndarray] = None
        self._minimap_map_x: Optional[np.ndarray] = None
        self._minimap_map_y: Optional[np.ndarray] = None
        self._minimap_in_full_xywh: Optional[list[int]] = None
        self._screen_crop_xywh: Optional[list[int]] = None
        self._screen_crop_selection: Dict[str, object] = {}
        self._normalization_origin_xy = (0.0, 0.0)
        self._screen_units_per_output_pixel_xy = (1.0, 1.0)
        self._minimap_mask_precomposed = False
        self._upright_to_base_full = np.eye(3, dtype=np.float64)
        self._upright_to_base_minimap = np.eye(3, dtype=np.float64)
        self._last_stream_metadata: Dict[str, Mapping[str, object]] = {}

    def _base_screen_crop(self) -> list[int]:
        crop = _source_crop_to_canonical_phone(self.minimap)
        phone_size = self.rig.get("phone", {}).get("natural_screen_size_px")
        if phone_size is None:
            phone_size = self.rig.get("phone", {}).get("natural_size_px")
        if phone_size is None:
            phone_size = self.rig.get("normalization", {}).get("phone_size_px")
        if phone_size is not None:
            width, height = map(int, phone_size)
            rotation_center = self.minimap.get("rotation_center")
            if isinstance(rotation_center, Mapping):
                point = require_spatial_geometry(
                    rotation_center,
                    "point",
                    expected_space_id="android_phone_natural_display_pixels",
                )
                if point["space"]["size_px"] != [width, height]:
                    raise ValueError(
                        "Rotation-center raster {} does not match phone raster {}x{}".format(
                            point["space"]["size_px"], width, height
                        )
                    )
                crop[0] = int(round(float(point["x"]) - crop[2] / 2.0))
                crop[1] = int(round(float(point["y"]) - crop[3] / 2.0))
                crop[0] = min(max(0, crop[0]), width - crop[2])
                crop[1] = min(max(0, crop[1]), height - crop[3])
            if crop[0] + crop[2] > width or crop[1] + crop[3] > height:
                raise ValueError(
                    "Canonical mini-map crop {} exceeds phone raster {}x{}".format(
                        crop, width, height
                    )
                )
        return crop

    def _phone_natural_size(self) -> Optional[list[int]]:
        phone_size = self.rig.get("phone", {}).get("natural_screen_size_px")
        if phone_size is None:
            phone_size = self.rig.get("phone", {}).get("natural_size_px")
        if phone_size is None:
            phone_size = self.rig.get("normalization", {}).get("phone_size_px")
        return list(map(int, phone_size)) if phone_size is not None else None

    def _surface_orientation_is_known(self) -> bool:
        surface = self.minimap.get("phone_surface_orientation") or {}
        return bool(
            self.runtime_surface_quarter_turns_clockwise_from_natural is not None
            or (
                isinstance(surface, Mapping)
                and "quarter_turns_clockwise_from_natural" in surface
            )
        )

    def _output_turns_for_surface(self, surface_turns: int) -> int:
        phone = self.rig.get("phone") or {}
        viewer = phone.get("viewer") or {}
        calibration_display_turns = int(
            phone.get(
                "orientation_quarter_turns",
                viewer.get("canonical_orientation_quarter_turns", 0),
            )
        ) % 4
        return (int(surface_turns) - calibration_display_turns) % 4

    def _screen_crop_candidates(self) -> list[dict]:
        """Express the saved game crop at each possible Android surface turn.

        Android rotation telemetry is useful as an ordering hint, not proof: an
        app or compatibility layer can render in a differently transformed
        surface. The hardware-ROI projection therefore evaluates all four
        quarter-turn hypotheses and keeps the deterministic one first.
        """

        base = self._base_screen_crop()
        natural_size = self._phone_natural_size()
        if natural_size is None:
            return [{"xywh": base, "surface_quarter_turns": None, "preferred": True}]
        natural_width, natural_height = natural_size
        stored_surface = self.minimap.get("phone_surface_orientation") or {}
        stored_turns = int(
            stored_surface.get("quarter_turns_clockwise_from_natural", 0)
            if isinstance(stored_surface, Mapping)
            else 0
        ) % 4
        preferred_turns = (
            self.runtime_surface_quarter_turns_clockwise_from_natural
            if self.runtime_surface_quarter_turns_clockwise_from_natural is not None
            else stored_turns
        )
        matrices = (
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, -1, natural_height], [1, 0, 0], [0, 0, 1]],
            [[-1, 0, natural_width], [0, -1, natural_height], [0, 0, 1]],
            [[0, 1, 0], [-1, 0, natural_width], [0, 0, 1]],
        )
        source_to_logical = np.asarray(matrices[stored_turns], dtype=np.float64)
        candidates = []
        orientation_order = [preferred_turns]
        if self.best_effort_initialization:
            # Mobile games normally remain landscape. A 180-degree reversal is
            # therefore both more likely and less destructive than guessing a
            # portrait/landscape transition; test the odd turns only afterward.
            orientation_order.extend(
                [
                    (preferred_turns + 2) % 4,
                    (preferred_turns + 1) % 4,
                    (preferred_turns + 3) % 4,
                ]
            )
        for turns in orientation_order:
            candidate_to_logical = np.asarray(matrices[turns], dtype=np.float64)
            matrix = np.linalg.inv(candidate_to_logical).dot(source_to_logical)
            x, y, width, height = base
            corners = np.asarray(
                [
                    [x, y, 1.0],
                    [x + width, y, 1.0],
                    [x + width, y + height, 1.0],
                    [x, y + height, 1.0],
                ],
                dtype=np.float64,
            )
            mapped = (matrix.dot(corners.T)).T
            mapped = mapped[:, :2] / mapped[:, 2:3]
            left = int(round(float(np.min(mapped[:, 0]))))
            top = int(round(float(np.min(mapped[:, 1]))))
            right = int(round(float(np.max(mapped[:, 0]))))
            bottom = int(round(float(np.max(mapped[:, 1]))))
            crop = [left, top, right - left, bottom - top]
            phone_bounds_valid = not (
                crop[0] < 0
                or crop[1] < 0
                or crop[2] <= 0
                or crop[3] <= 0
                or crop[0] + crop[2] > natural_width
                or crop[1] + crop[3] > natural_height
            )
            candidates.append(
                {
                    "xywh": crop,
                    "surface_quarter_turns": turns,
                    "preferred": turns == preferred_turns,
                    "phone_bounds_valid": phone_bounds_valid,
                }
            )
        return candidates

    def _screen_crop(self) -> list[int]:
        return list(self._screen_crop_xywh or self._base_screen_crop())

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
        return list(map(int, self.rig["camera"]["hardware_roi_xywh"]))

    def _apply_locked_imaging(self) -> None:
        imaging = self.rig["imaging"]
        if imaging.get("black_level") is not None:
            self.adapter.set_black_level(int(imaging["black_level"]))
        self.adapter.set_manual_imaging(imaging["exposure_us"], imaging["gain"])
        wb = imaging["white_balance"]
        self.adapter.set_white_balance(
            wb["ratio_red"], wb["ratio_green"], wb["ratio_blue"]
        )
        conversion = (
            self.bayer_conversion
            or self.minimap.get("hik_bayer_conversion")
            or {}
        )
        if (
            self.apply_game_color
            and
            isinstance(conversion, Mapping)
            and conversion.get("status") == "selected"
        ):
            self.adapter.set_bayer_conversion(
                float(conversion["gamma"]),
                conversion["ccm_rgb_3x3"],
            )

    @staticmethod
    def _transform_xy(matrix: np.ndarray, point_xy: Sequence[float]) -> np.ndarray:
        point = np.asarray([float(point_xy[0]), float(point_xy[1]), 1.0])
        transformed = np.asarray(matrix, dtype=np.float64).dot(point)
        return transformed[:2] / transformed[2]

    def _canonical_phone_to_upright_full_xy(
        self, point_xy: Sequence[float]
    ) -> np.ndarray:
        origin_x, origin_y = self._normalization_origin_xy
        scale_x, scale_y = self._screen_units_per_output_pixel_xy
        base = np.asarray(
            [
                (float(point_xy[0]) - origin_x) / scale_x,
                (float(point_xy[1]) - origin_y) / scale_y,
            ],
            dtype=np.float64,
        )
        return self._transform_xy(np.linalg.inv(self._upright_to_base_full), base)

    def _canonical_phone_vector_to_upright_full_xy(
        self, anchor_xy: Sequence[float], vector_xy: Sequence[float]
    ) -> np.ndarray:
        """Transform a direction as an anchored ray through the actual output map."""

        first = self._canonical_phone_to_upright_full_xy(anchor_xy)
        second = self._canonical_phone_to_upright_full_xy(
            [
                float(anchor_xy[0]) + float(vector_xy[0]),
                float(anchor_xy[1]) + float(vector_xy[1]),
            ]
        )
        result = second - first
        norm = float(np.linalg.norm(result))
        if norm <= 1.0e-9:
            raise ValueError("Canonical mini-map orientation collapsed in output space")
        return result / norm

    def _cursor_envelope_size_xy(self) -> Optional[list[float]]:
        geometry = dict(self.minimap.get("cursor_geometry") or {})
        value = geometry.get("rotating_cursor_envelope_diameter_px")
        if value is None:
            return None
        diameter = float(value)
        scale_x, scale_y = self._screen_units_per_output_pixel_xy
        size = [diameter / scale_x, diameter / scale_y]
        if self.output_quarter_turns_clockwise % 2:
            size.reverse()
        return [float(value) for value in size]

    def _precompose_minimap_mask(self) -> None:
        if self.mask_policy == "none":
            return
        if self._minimap_map_x is None or self._minimap_map_y is None:
            raise RuntimeError(
                "Mini-map circle masking requires a prebuilt dense rectification map"
            )
        boundary = require_spatial_geometry(
            self.minimap.get("outer_boundary"),
            "circle",
            expected_space_id="android_phone_natural_display_pixels",
        )
        center_full = self._canonical_phone_to_upright_full_xy(
            [boundary["center_x"], boundary["center_y"]]
        )
        crop_x, crop_y, _, _ = self._minimap_in_full_xywh
        center = center_full - np.asarray([crop_x, crop_y], dtype=np.float64)
        scale_x, scale_y = self._screen_units_per_output_pixel_xy
        radii = [float(boundary["radius"]) / scale_x, float(boundary["radius"]) / scale_y]
        if self.output_quarter_turns_clockwise % 2:
            radii.reverse()
        mask = np.zeros(self._minimap_map_x.shape, dtype=np.uint8)
        cv2.ellipse(
            mask,
            tuple(np.round(center).astype(int)),
            tuple(max(1, int(round(value))) for value in radii),
            0.0,
            0.0,
            360.0,
            255,
            -1,
            cv2.LINE_8,
        )
        self._minimap_map_x = self._minimap_map_x.copy()
        self._minimap_map_y = self._minimap_map_y.copy()
        self._minimap_map_x[mask == 0] = -1.0
        self._minimap_map_y[mask == 0] = -1.0
        self._minimap_mask_precomposed = True

    def get_cursor_geometry(self, stream_id: str = "minimap") -> Mapping[str, object]:
        """Return calibrated cursor geometry in canonical and runtime spaces."""

        canonical = dict(self.minimap.get("cursor_geometry") or {})
        if not canonical:
            return {}
        if not self._opened:
            raise RuntimeError("Camera must be open to query runtime cursor geometry")
        selected = str(stream_id)
        if selected not in ("minimap", "full", "canonical_phone"):
            raise ValueError("Cursor geometry stream must be minimap, full, or canonical_phone")
        result = {
            "schema_version": "1.0",
            "available": True,
            "canonical_phone": copy.deepcopy(canonical),
            "stream_id": selected,
        }
        if selected == "canonical_phone":
            return result
        if not self.rectify_minimap:
            result.update(
                available_in_stream_space=False,
                reason="Unrectified camera ROI is projective; use canonical_phone geometry",
            )
            return result
        center_geometry = canonical.get("rotation_center") or self.minimap.get(
            "rotation_center"
        )
        if not isinstance(center_geometry, Mapping):
            result.update(
                available_in_stream_space=False,
                reason=(
                    "Static cursor evidence has no verified rotation center; "
                    "canonical_phone contains the observed static size"
                ),
            )
            return result
        center = require_spatial_geometry(
            center_geometry,
            "point",
            expected_space_id="android_phone_natural_display_pixels",
        )
        center_full = self._canonical_phone_to_upright_full_xy(
            [center["x"], center["y"]]
        )
        if selected == "minimap":
            crop_x, crop_y, _, _ = self._minimap_in_full_xywh
            center_runtime = center_full - np.asarray([crop_x, crop_y])
        else:
            center_runtime = center_full
        size_xy = self._cursor_envelope_size_xy()
        metadata = self._last_stream_metadata.get(selected) or {}
        result.update(
            available_in_stream_space=True,
            center_xy_px=[float(center_runtime[0]), float(center_runtime[1])],
            image_space=copy.deepcopy(dict(metadata.get("image_space") or {})),
        )
        if size_xy is not None:
            result.update(
                rotating_cursor_envelope_size_xy_px=size_xy,
                rotating_cursor_envelope_diameter_px=float(max(size_xy)),
            )
        return result

    def get_minimap_geometry(
        self, stream_id: str = "minimap"
    ) -> Mapping[str, object]:
        """Return the calibrated mini-map boundary in canonical and runtime spaces."""

        canonical = self.minimap.get("outer_boundary")
        if not isinstance(canonical, Mapping):
            return {}
        boundary = require_spatial_geometry(
            canonical,
            "circle",
            expected_space_id="android_phone_natural_display_pixels",
        )
        if not self._opened:
            raise RuntimeError("Camera must be open to query runtime mini-map geometry")
        selected = str(stream_id)
        if selected not in ("minimap", "full", "canonical_phone"):
            raise ValueError(
                "Mini-map geometry stream must be minimap, full, or canonical_phone"
            )
        result = {
            "schema_version": "1.0",
            "available": True,
            "canonical_phone": copy.deepcopy(dict(boundary)),
            "stream_id": selected,
        }
        if selected == "canonical_phone":
            return result
        if not self.rectify_minimap:
            result.update(
                available_in_stream_space=False,
                reason=(
                    "Unrectified camera ROI is projective; use canonical_phone geometry"
                ),
            )
            return result
        center_full = self._canonical_phone_to_upright_full_xy(
            [boundary["center_x"], boundary["center_y"]]
        )
        if selected == "minimap":
            crop_x, crop_y, _, _ = self._minimap_in_full_xywh
            center_runtime = center_full - np.asarray([crop_x, crop_y])
        else:
            center_runtime = center_full
        scale_x, scale_y = self._screen_units_per_output_pixel_xy
        diameter_xy = [
            2.0 * float(boundary["radius"]) / scale_x,
            2.0 * float(boundary["radius"]) / scale_y,
        ]
        if self.output_quarter_turns_clockwise % 2:
            diameter_xy.reverse()
        metadata = self._last_stream_metadata.get(selected) or {}
        result.update(
            available_in_stream_space=True,
            center_xy_px=[float(center_runtime[0]), float(center_runtime[1])],
            boundary_size_xy_px=[float(value) for value in diameter_xy],
            radius_px=float(max(diameter_xy) / 2.0),
            image_space=copy.deepcopy(dict(metadata.get("image_space") or {})),
        )
        orientation_frame = boundary.get("orientation_frame")
        if isinstance(orientation_frame, Mapping):
            runtime_frame = {
                **copy.deepcopy(dict(orientation_frame)),
                "up_unit_xy": self._canonical_phone_vector_to_upright_full_xy(
                    [boundary["center_x"], boundary["center_y"]],
                    orientation_frame["up_unit_xy"],
                ).tolist(),
                "right_unit_xy": self._canonical_phone_vector_to_upright_full_xy(
                    [boundary["center_x"], boundary["center_y"]],
                    orientation_frame["right_unit_xy"],
                ).tolist(),
            }
            result["orientation_frame"] = runtime_frame
            result["orientation_error_degrees"] = float(
                np.degrees(
                    np.arctan2(
                        runtime_frame["up_unit_xy"][0],
                        -runtime_frame["up_unit_xy"][1],
                    )
                )
            )
        return result

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
            coordinate_schema = int(
                (self.rig.get("coordinate_spaces") or {}).get("schema_version", 0)
            )
            lens_model = (self.rig.get("optics") or {}).get("lens_model") or {}
            candidates = self._screen_crop_candidates()
            screen_crop = list(candidates[0]["xywh"])
            previous_output_turns = self.output_quarter_turns_clockwise
            if self.mode != "full":
                valid_candidates = []
                failures = []
                for candidate in candidates:
                    if not candidate.get("phone_bounds_valid", True):
                        failures.append(
                            {
                                "surface_quarter_turns": candidate[
                                    "surface_quarter_turns"
                                ],
                                "crop_xywh": list(candidate["xywh"]),
                                "reason": "candidate crop is outside the canonical phone raster",
                            }
                        )
                        continue
                    if not _projected_screen_crop_is_complete(
                        candidate["xywh"],
                        self.rig["geometry"]["screen_to_full_sensor_camera_3x3"],
                        self._sensor_size(),
                        lens_model=(lens_model if coordinate_schema >= 3 else None),
                    ):
                        failures.append(
                            {
                                "surface_quarter_turns": candidate[
                                    "surface_quarter_turns"
                                ],
                                "crop_xywh": list(candidate["xywh"]),
                                "reason": (
                                    "projected complete mini-map is not fully "
                                    "visible on the HIK sensor"
                                ),
                            }
                        )
                        continue
                    try:
                        if coordinate_schema >= 3:
                            requested = distorted_screen_region_roi(
                                self._sensor_size(),
                                candidate["xywh"],
                                self.rig["geometry"]["screen_to_full_sensor_camera_3x3"],
                                lens_model,
                                margin_px=self.minimap_margin_px,
                            )
                        else:
                            requested = camera_roi_for_screen_region(
                                candidate["xywh"],
                                self.rig["geometry"]["screen_to_full_sensor_camera_3x3"],
                                self._sensor_size(),
                                margin_px=self.minimap_margin_px,
                            )
                    except RuntimeError as exc:
                        failures.append(
                            {
                                "surface_quarter_turns": candidate[
                                    "surface_quarter_turns"
                                ],
                                "crop_xywh": list(candidate["xywh"]),
                                "reason": str(exc),
                            }
                        )
                        continue
                    valid_candidates.append((candidate, list(requested)))
                if not valid_candidates:
                    raise MinimapRoiUnavailableError(
                        "No complete mini-map crop from {} Android/game orientation "
                        "hypothesis/hypotheses is fully visible on the calibrated "
                        "HIK sensor: {}".format(len(candidates), failures)
                    )
                selected, requested_minimap_roi = next(
                    (
                        item
                        for item in valid_candidates
                        if bool(item[0].get("preferred"))
                    ),
                    valid_candidates[0],
                )
                screen_crop = list(selected["xywh"])
                recovered_output_turns = previous_output_turns
                if (
                    self.best_effort_initialization
                    and self._surface_orientation_is_known()
                    and selected.get("surface_quarter_turns") is not None
                ):
                    recovered_output_turns = self._output_turns_for_surface(
                        int(selected["surface_quarter_turns"])
                    )
                self.output_quarter_turns_clockwise = recovered_output_turns
                self._minimap_sensor_roi = list(
                    self.adapter.align_roi(requested_minimap_roi)
                )
                self._screen_crop_selection = {
                    "status": (
                        "deterministic_orientation_intersection"
                        if selected.get("preferred")
                        else "four_orientation_intersection_fallback"
                    ),
                    "selected_surface_quarter_turns": selected[
                        "surface_quarter_turns"
                    ],
                    "previous_output_quarter_turns": previous_output_turns,
                    "selected_output_quarter_turns": recovered_output_turns,
                    "orientation_recovered": (
                        recovered_output_turns != previous_output_turns
                    ),
                    "selection_basis": (
                        "saved_game_surface_coordinates"
                        if selected.get("preferred")
                        else "visible_projected_minimap_coordinates"
                    ),
                    "runtime_cost": "initialization_only",
                    "evaluated_candidates": len(candidates),
                    "valid_candidates": len(valid_candidates),
                    "failed_candidates": failures,
                }
            else:
                self._minimap_sensor_roi = None
                preferred_surface_turns = candidates[0].get("surface_quarter_turns")
                recovered_output_turns = previous_output_turns
                if (
                    self.best_effort_initialization
                    and self._surface_orientation_is_known()
                    and preferred_surface_turns is not None
                ):
                    recovered_output_turns = self._output_turns_for_surface(
                        int(preferred_surface_turns)
                    )
                self.output_quarter_turns_clockwise = recovered_output_turns
                self._screen_crop_selection = {
                    "status": (
                        "coordinate_projected_game_orientation"
                        if self._surface_orientation_is_known()
                        else "not_required_for_full_stream"
                    ),
                    "evaluated_candidates": 0,
                    "selected_surface_quarter_turns": preferred_surface_turns,
                    "previous_output_quarter_turns": previous_output_turns,
                    "selected_output_quarter_turns": recovered_output_turns,
                    "orientation_recovered": (
                        recovered_output_turns != previous_output_turns
                    ),
                    "selection_basis": "saved_game_surface_coordinates",
                    "runtime_cost": "initialization_only",
                }
            self._screen_crop_xywh = list(screen_crop)
            requested_roi = (
                self._minimap_sensor_roi
                if self.mode == "minimap"
                else self._full_mode_roi()
            )
            self._effective_roi = list(self.adapter.set_roi(requested_roi))

            crop_x, crop_y, crop_width, crop_height = screen_crop
            if coordinate_schema < 3:
                camera_to_screen = np.asarray(
                    self.rig["geometry"]["full_sensor_camera_to_screen_3x3"],
                    dtype=np.float64,
                )
                acquisition_to_screen = compose_hardware_roi_homography(
                    camera_to_screen, self._effective_roi
                )
                self._minimap_matrix = _translation(-crop_x, -crop_y).dot(
                    acquisition_to_screen
                )
            self._minimap_size = (crop_width, crop_height)

            normalization = self.rig["normalization"]
            self._full_matrix = (
                None
                if coordinate_schema >= 3
                else camera_adapter_roi_to_output_homography(
                    self.rig, self._effective_roi
                )
            )
            self._full_size = tuple(map(int, normalization["output_size_px"]))
            origin_x, origin_y = map(
                float, normalization.get("origin_screen_xy", [0, 0])
            )
            scale_x, scale_y = map(
                float,
                normalization.get("screen_units_per_output_pixel_xy", [1, 1]),
            )
            self._normalization_origin_xy = (origin_x, origin_y)
            self._screen_units_per_output_pixel_xy = (scale_x, scale_y)
            self._minimap_in_full_xywh = [
                int(round((crop_x - origin_x) / scale_x)),
                int(round((crop_y - origin_y) / scale_y)),
                int(round(crop_width / scale_x)),
                int(round(crop_height / scale_y)),
            ]
            dense_file = normalization.get("dense_map_file")
            if dense_file and self.rectify_minimap:
                dense_path = self.rig_path.parent / str(dense_file)
                if dense_path.is_file():
                    with np.load(str(dense_path)) as dense:
                        self._full_map_x = np.asarray(
                            dense["map_x"], dtype=np.float32
                        ) - float(self._effective_roi[0])
                        self._full_map_y = np.asarray(
                            dense["map_y"], dtype=np.float32
                        ) - float(self._effective_roi[1])
            if coordinate_schema >= 3 and self.rectify_minimap:
                if self._full_map_x is None or self._full_map_y is None:
                    raise RuntimeError(
                        "Distortion-corrected HIK game stream requires its dense remap"
                    )
                if self.mode != "full":
                    mini_x, mini_y, mini_width, mini_height = self._minimap_in_full_xywh
                    if (
                        mini_x < 0
                        or mini_y < 0
                        or mini_x + mini_width > self._full_map_x.shape[1]
                        or mini_y + mini_height > self._full_map_x.shape[0]
                    ):
                        raise MinimapRoiUnavailableError(
                            "Mini-map crop {} exceeds the saved dense rectification map"
                            .format(self._minimap_in_full_xywh)
                        )
                    self._minimap_map_x = self._full_map_x[
                        mini_y : mini_y + mini_height,
                        mini_x : mini_x + mini_width,
                    ]
                    self._minimap_map_y = self._full_map_y[
                        mini_y : mini_y + mini_height,
                        mini_x : mini_x + mini_width,
                    ]
            if self.rectify_minimap and self.output_quarter_turns_clockwise:
                turns = self.output_quarter_turns_clockwise
                base_full_size = self._full_size
                full_rotation, self._full_size = quarter_turn_output_geometry(
                    base_full_size, turns
                )
                self._upright_to_base_full = np.linalg.inv(full_rotation)
                self._minimap_in_full_xywh = rotate_xywh_in_parent(
                    self._minimap_in_full_xywh, base_full_size, turns
                )
                if self._full_map_x is not None:
                    self._full_map_x = rotate_quarter_turns_clockwise(
                        self._full_map_x, turns
                    )
                    self._full_map_y = rotate_quarter_turns_clockwise(
                        self._full_map_y, turns
                    )
                elif self._full_matrix is not None:
                    self._full_matrix = full_rotation.dot(self._full_matrix)

                if self.mode != "full":
                    minimap_rotation, self._minimap_size = quarter_turn_output_geometry(
                        self._minimap_size, turns
                    )
                    self._upright_to_base_minimap = np.linalg.inv(minimap_rotation)
                    if self._minimap_map_x is not None:
                        self._minimap_map_x = rotate_quarter_turns_clockwise(
                            self._minimap_map_x, turns
                        )
                        self._minimap_map_y = rotate_quarter_turns_clockwise(
                            self._minimap_map_y, turns
                        )
                    elif self._minimap_matrix is not None:
                        self._minimap_matrix = minimap_rotation.dot(
                            self._minimap_matrix
                        )
            elif self.output_quarter_turns_clockwise:
                turns = self.output_quarter_turns_clockwise
                self._upright_to_base_full = np.linalg.inv(
                    quarter_turn_output_geometry(
                        (int(self._effective_roi[2]), int(self._effective_roi[3])),
                        turns,
                    )[0]
                )
                if self._minimap_sensor_roi is not None:
                    self._upright_to_base_minimap = np.linalg.inv(
                        quarter_turn_output_geometry(
                            (
                                int(self._minimap_sensor_roi[2]),
                                int(self._minimap_sensor_roi[3]),
                            ),
                            turns,
                        )[0]
                    )
            self._precompose_minimap_mask()
            self._opened = True
        except Exception:
            self.adapter.close()
            raise
        return self

    def isOpened(self) -> bool:
        return self._opened

    def initialization_orientation_recovery(self) -> Mapping[str, object]:
        """Return the initialization-only crop/orientation decision."""

        return copy.deepcopy(self._screen_crop_selection)

    def geometry_postmortem(self) -> Mapping[str, object]:
        """Describe runtime ROI/map composition for retrospective diagnosis."""

        requested_roi = (
            list(self._minimap_sensor_roi)
            if self.mode == "minimap" and self._minimap_sensor_roi is not None
            else list(self._full_mode_roi())
        )
        effective_roi = (
            list(self._effective_roi) if self._effective_roi is not None else None
        )
        map_arrays = [
            value for value in (
                self._full_map_x,
                self._full_map_y,
                self._minimap_map_x,
                self._minimap_map_y,
            ) if value is not None
        ]
        finite_fraction = (
            float(np.mean(np.logical_and.reduce([
                np.isfinite(value) for value in map_arrays
                if value.shape == map_arrays[0].shape
            ])))
            if map_arrays else None
        )
        return {
            "schema_version": "1.0",
            "status": "observed" if self._opened else "not_open",
            "non_gating": True,
            "reader": type(self).__name__,
            "mode": self.mode,
            "rectification_enabled": bool(self.rectify_minimap),
            "requested_hardware_roi_xywh": requested_roi,
            "effective_hardware_roi_xywh": effective_roi,
            "effective_roi_matches_request": effective_roi == requested_roi,
            "screen_crop_xywh": (
                list(self._screen_crop_xywh)
                if self._screen_crop_xywh is not None else None
            ),
            "minimap_in_full_xywh": (
                list(self._minimap_in_full_xywh)
                if self._minimap_in_full_xywh is not None else None
            ),
            "full_output_size_px": list(self._full_size),
            "minimap_output_size_px": list(self._minimap_size),
            "output_quarter_turns_clockwise": int(
                self.output_quarter_turns_clockwise
            ),
            "dense_map_array_count": len(map_arrays),
            "dense_map_finite_fraction": finite_fraction,
            "orientation_initialization": copy.deepcopy(
                self._screen_crop_selection
            ),
            "calibration_geometry_confidence": (
                (self.rig.get("results") or {}).get("cv_verification")
                or (self.rig.get("image_quality") or {}).get("confidence")
            ),
        }

    def _minimap_from_acquisition(self, image: np.ndarray) -> np.ndarray:
        if self.rectify_minimap:
            if self._minimap_map_x is not None and self._minimap_map_y is not None:
                return cv2.remap(
                    image,
                    self._minimap_map_x,
                    self._minimap_map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
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

    def _full_from_acquisition(self, image: np.ndarray) -> np.ndarray:
        if not self.rectify_minimap:
            return image
        if self._full_map_x is not None and self._full_map_y is not None:
            return cv2.remap(
                image,
                self._full_map_x,
                self._full_map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
        return cv2.warpPerspective(
            image,
            self._full_matrix,
            self._full_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def _minimap_from_full(self, full: np.ndarray) -> np.ndarray:
        x, y, width, height = self._minimap_in_full_xywh
        if x < 0 or y < 0 or x + width > full.shape[1] or y + height > full.shape[0]:
            raise ValueError(
                "Mini-map crop {} exceeds normalized full output {}x{}".format(
                    self._minimap_in_full_xywh, full.shape[1], full.shape[0]
                )
            )
        return full[y : y + height, x : x + width].copy()

    def read_streams(self) -> HikGameFrameSet:
        if not self._opened:
            raise RuntimeError("Profiled HIK game camera is not open")
        sample = self.adapter.read()
        if self.mode == "minimap":
            streams: Dict[str, np.ndarray] = {
                "minimap": self._minimap_from_acquisition(sample.image)
            }
        elif self.mode == "full":
            streams = {"full": self._full_from_acquisition(sample.image)}
        else:
            full = self._full_from_acquisition(sample.image)
            streams = {
                "full": full,
                "minimap": (
                    self._minimap_from_full(full)
                    if self.rectify_minimap
                    else self._minimap_from_acquisition(sample.image)
                ),
            }
        if not self.rectify_minimap and self.output_quarter_turns_clockwise:
            streams = {
                name: rotate_quarter_turns_clockwise(
                    image, self.output_quarter_turns_clockwise
                )
                for name, image in streams.items()
            }
        frame_number = sample.metadata.get("frame_number")
        common = {
            **dict(sample.metadata),
            "mode": self.mode,
            "rectified_minimap": self.rectify_minimap,
            "rectified_full": bool(
                self.rectify_minimap and self.mode != "minimap"
            ),
            "acquisition_roi_xywh": list(self._effective_roi),
            "minimap_sensor_roi_xywh": (
                list(self._minimap_sensor_roi)
                if self._minimap_sensor_roi is not None
                else None
            ),
            "minimap_crop_orientation_selection": copy.deepcopy(
                self._screen_crop_selection
            ),
            "full_output_normalized_by_base_rig": bool(
                self.rectify_minimap and self.mode != "minimap"
            ),
            "minimap_crop_in_full_output_xywh": (
                list(self._minimap_in_full_xywh)
                if self.rectify_minimap else None
            ),
            "one_acquisition_for_all_streams": True,
            "mask_policy": self.mask_policy,
            "minimap_mask_precomposed_in_rectification_map": bool(
                self._minimap_mask_precomposed
            ),
            "minimap_mask_off_pixel_map_coordinate_xy": (
                [-1.0, -1.0] if self._minimap_mask_precomposed else None
            ),
            "minimap_mask_geometry_source": (
                "verified_radial_circle_fit_seeded_by_hough_from_temporal_and_average_evidence"
                if self._minimap_mask_precomposed else None
            ),
            "game_upright_quarter_turns_clockwise": (
                self.output_quarter_turns_clockwise
            ),
            "game_upright_runtime_operation": (
                "precomposed_rectification_lookup"
                if self.rectify_minimap and self.output_quarter_turns_clockwise
                else (
                    "discrete_quarter_turn_no_interpolation"
                    if self.output_quarter_turns_clockwise else "none"
                )
            ),
        }
        stream_metadata: Dict[str, Mapping[str, object]] = {}
        distortion_corrected = bool(
            self.rig.get("normalization", {}).get(
                "lens_correction_in_dense_map", False
            )
        )
        for name, image in streams.items():
            height, width = image.shape[:2]
            if name == "full" and self.rectify_minimap:
                image_space = {
                    "schema_version": "1.0",
                    "space_id": (
                        "hik_game_upright_rectified_visible_phone_pixels"
                        if self.output_quarter_turns_clockwise
                        else "hik_rig_rectified_visible_phone_pixels"
                    ),
                    "stored_size_px": [int(width), int(height)],
                    "parent_space_id": "hik_full_sensor_camera_pixels",
                    "source_roi_in_parent_xywh": list(self._effective_roi),
                    "transform_reference": (
                        "hik_camera_calibration.json#normalization.dense_map_file"
                        if distortion_corrected
                        else "hik_camera_calibration.json#normalization."
                        "full_sensor_camera_to_output_3x3"
                    ),
                    "lens_distortion_corrected": distortion_corrected,
                    "runtime_resampling_passes": 1,
                    "orientation": (
                        "game_surface_up"
                        if self.output_quarter_turns_clockwise else "phone_app_up"
                    ),
                    "local_to_phone_app_up_output_3x3": (
                        self._upright_to_base_full.tolist()
                    ),
                    "color_order": "BGR",
                }
            elif name == "minimap" and self.rectify_minimap:
                image_space = {
                    "schema_version": "1.0",
                    "space_id": "hik_phone_game_normalized_minimap_pixels",
                    "stored_size_px": [int(width), int(height)],
                    "parent_space_id": "android_phone_natural_display_pixels",
                    "roi_in_parent_xywh": list(self._screen_crop()),
                    "transform_reference": (
                        "hik_camera_calibration.json#normalization.dense_map_file"
                        if distortion_corrected
                        else "hik_camera_calibration.json#normalization."
                        "full_sensor_camera_to_output_3x3"
                    ),
                    "lens_distortion_corrected": distortion_corrected,
                    "runtime_resampling_passes": 1,
                    "orientation": (
                        "game_surface_up"
                        if self.output_quarter_turns_clockwise else "phone_app_up"
                    ),
                    "local_to_parent_3x3": (
                        np.asarray(
                            [
                                [1.0, 0.0, float(self._screen_crop()[0])],
                                [0.0, 1.0, float(self._screen_crop()[1])],
                                [0.0, 0.0, 1.0],
                            ],
                            dtype=np.float64,
                        ).dot(self._upright_to_base_minimap).tolist()
                    ),
                    "color_order": "BGR",
                }
            else:
                roi = (
                    list(self._effective_roi)
                    if name == "full" or self.mode == "minimap"
                    else list(self._minimap_sensor_roi)
                )
                image_space = {
                    "schema_version": "1.0",
                    "space_id": (
                        "hik_game_upright_camera_adapter_roi_pixels"
                        if self.output_quarter_turns_clockwise
                        else "hik_camera_adapter_roi_image_pixels"
                    ),
                    "stored_size_px": [int(width), int(height)],
                    "parent_space_id": "hik_full_sensor_camera_pixels",
                    "roi_in_parent_xywh": roi,
                    "local_to_parent_3x3": (
                        np.asarray(
                            [
                                [1.0, 0.0, float(roi[0])],
                                [0.0, 1.0, float(roi[1])],
                                [0.0, 0.0, 1.0],
                            ],
                            dtype=np.float64,
                        ).dot(
                            self._upright_to_base_full
                            if name == "full"
                            else self._upright_to_base_minimap
                        ).tolist()
                    ),
                    "orientation": (
                        "game_surface_up"
                        if self.output_quarter_turns_clockwise else "hik_camera_native"
                    ),
                    "color_order": "BGR",
                }
            stream_metadata[name] = {
                **common,
                "stream_id": name,
                "image_space": image_space,
            }
        self._last_stream_metadata = dict(stream_metadata)
        return HikGameFrameSet(
            time_ns=int(sample.time_ns),
            receive_time_ns=int(sample.receive_time_ns or sample.time_ns),
            frame_number=(int(frame_number) if frame_number is not None else None),
            streams=streams,
            metadata=common,
            stream_metadata=stream_metadata,
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
            metadata=dict(frame_set.stream_metadata[selected]),
        )

    def get_iris_frame_metadata(
        self, stream_id: Optional[str] = None
    ) -> Mapping[str, object]:
        """Return per-stream metadata without changing HIK-compatible reads."""

        if stream_id is None:
            if len(self._last_stream_metadata) == 1:
                return copy.deepcopy(next(iter(self._last_stream_metadata.values())))
            return copy.deepcopy({
                name: dict(value)
                for name, value in self._last_stream_metadata.items()
            })
        return copy.deepcopy(
            self._last_stream_metadata.get(str(stream_id), {})
        )

    def get_aria_frame_metadata(
        self, stream_id: Optional[str] = None
    ) -> Mapping[str, object]:
        """Compatibility alias for :meth:`get_iris_frame_metadata`."""

        return self.get_iris_frame_metadata(stream_id)

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


__all__ = [
    "HikGameFrameSet",
    "MinimapRoiUnavailableError",
    "ProfiledHikGameCamera",
]
