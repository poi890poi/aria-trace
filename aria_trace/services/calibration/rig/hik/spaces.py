"""Coordinate conversion for rig-rectified HIK and Android display images."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import cv2
import numpy as np

from ..geometry import transform_points


CalibrationSource = Union[str, Path, Mapping[str, Any]]


def _load_calibration(source: CalibrationSource) -> tuple[dict, str]:
    if isinstance(source, Mapping):
        return dict(source), "embedded_mapping"
    path = Path(source).resolve()
    return json.loads(path.read_text(encoding="utf-8")), str(path)


def _matrix_list(matrix: np.ndarray) -> list:
    cleaned = np.asarray(matrix, dtype=np.float64).copy()
    cleaned[np.abs(cleaned) < 1.0e-12] = 0.0
    return cleaned.tolist()


class RigCalibratedSpaceConverter:
    """Map adapter-output pixels to/from the current ADB logical raster.

    The adapter output is expressed in the logical display raster used during
    rig calibration (app-up/app-right), not necessarily phone-natural space.
    A session can use another Android surface orientation, so every conversion
    goes calibration-display -> phone-natural -> current ADB explicitly.
    """

    def __init__(
        self,
        calibration: CalibrationSource,
        adb_surface_quarter_turns_clockwise_from_natural: int = 0,
    ) -> None:
        self.calibration, self.calibration_reference = _load_calibration(calibration)
        normalization = self.calibration["normalization"]
        phone = self.calibration["phone"]
        self.adapter_size_px = tuple(map(int, normalization["output_size_px"]))
        self.phone_natural_size_px = tuple(
            map(int, phone["natural_screen_size_px"])
        )
        self.calibration_display_size_px = tuple(
            map(int, phone.get("screen_size_px", self.phone_natural_size_px))
        )
        viewer = phone.get("viewer") or {}
        self.calibration_display_quarter_turns_clockwise_from_natural = int(
            phone.get(
                "orientation_quarter_turns",
                viewer.get("canonical_orientation_quarter_turns", 0),
            )
        ) % 4
        self.origin_calibration_display_xy = tuple(
            map(float, normalization["origin_screen_xy"])
        )
        self.phone_units_per_adapter_pixel_xy = tuple(
            map(
                float,
                normalization.get(
                    "screen_units_per_output_pixel_xy", [1.0, 1.0]
                ),
            )
        )
        if min(self.adapter_size_px + self.phone_natural_size_px) <= 0:
            raise ValueError("Coordinate-space image sizes must be positive")
        if min(self.phone_units_per_adapter_pixel_xy) <= 0:
            raise ValueError("Adapter-to-phone pixel scale must be positive")
        natural_width, natural_height = self.phone_natural_size_px
        expected_calibration_size = (
            (natural_height, natural_width)
            if self.calibration_display_quarter_turns_clockwise_from_natural % 2
            else (natural_width, natural_height)
        )
        if self.calibration_display_size_px != expected_calibration_size:
            raise ValueError(
                "Rig calibration display size {} does not match natural size {} "
                "at quarter-turn {}".format(
                    self.calibration_display_size_px,
                    self.phone_natural_size_px,
                    self.calibration_display_quarter_turns_clockwise_from_natural,
                )
            )
        self.adb_surface_quarter_turns_clockwise_from_natural = (
            int(adb_surface_quarter_turns_clockwise_from_natural) % 4
        )
        self.output_image_quarter_turns_clockwise_from_calibration_display = (
            self.adb_surface_quarter_turns_clockwise_from_natural
            - self.calibration_display_quarter_turns_clockwise_from_natural
        ) % 4
        self.adapter_to_calibration_display_3x3 = np.asarray(
            [
                [
                    self.phone_units_per_adapter_pixel_xy[0],
                    0.0,
                    self.origin_calibration_display_xy[0],
                ],
                [
                    0.0,
                    self.phone_units_per_adapter_pixel_xy[1],
                    self.origin_calibration_display_xy[1],
                ],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.phone_natural_to_calibration_display_3x3 = (
            self._natural_to_logical_matrix(
                self.phone_natural_size_px,
                self.calibration_display_quarter_turns_clockwise_from_natural,
            )
        )
        self.calibration_display_to_phone_natural_3x3 = np.linalg.inv(
            self.phone_natural_to_calibration_display_3x3
        )
        self.adapter_to_phone_natural_3x3 = (
            self.calibration_display_to_phone_natural_3x3
            @ self.adapter_to_calibration_display_3x3
        )
        self.phone_natural_to_adb_3x3 = self._natural_to_logical_matrix(
            self.phone_natural_size_px,
            self.adb_surface_quarter_turns_clockwise_from_natural,
        )
        self.adapter_to_adb_3x3 = (
            self.phone_natural_to_adb_3x3
            @ self.adapter_to_phone_natural_3x3
        )
        self.adb_to_adapter_3x3 = np.linalg.inv(self.adapter_to_adb_3x3)
        self.adapter_to_output_image_3x3 = self._natural_to_logical_matrix(
            self.adapter_size_px,
            self.output_image_quarter_turns_clockwise_from_calibration_display,
        )
        self.output_image_to_adapter_3x3 = np.linalg.inv(
            self.adapter_to_output_image_3x3
        )

    @staticmethod
    def _natural_to_logical_matrix(
        natural_size_px: Sequence[int], turns_clockwise: int
    ) -> np.ndarray:
        width, height = map(int, natural_size_px)
        turns = int(turns_clockwise) % 4
        matrices = (
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, -1, height - 1], [1, 0, 0], [0, 0, 1]],
            [[-1, 0, width - 1], [0, -1, height - 1], [0, 0, 1]],
            [[0, 1, 0], [-1, 0, width - 1], [0, 0, 1]],
        )
        return np.asarray(matrices[turns], dtype=np.float64)

    @property
    def adb_size_px(self) -> tuple[int, int]:
        width, height = self.phone_natural_size_px
        if self.adb_surface_quarter_turns_clockwise_from_natural % 2:
            return height, width
        return width, height

    @property
    def output_image_size_px(self) -> tuple[int, int]:
        width, height = self.adapter_size_px
        if self.output_image_quarter_turns_clockwise_from_calibration_display % 2:
            return height, width
        return width, height

    def camera_adapter_to_adb_points(
        self, points_xy: Sequence[Sequence[float]]
    ) -> np.ndarray:
        return transform_points(points_xy, self.adapter_to_adb_3x3)

    def adb_to_camera_adapter_points(
        self, points_xy: Sequence[Sequence[float]]
    ) -> np.ndarray:
        return transform_points(points_xy, self.adb_to_adapter_3x3)

    def camera_adapter_image_to_adb_orientation(
        self, image: np.ndarray
    ) -> np.ndarray:
        turns = self.output_image_quarter_turns_clockwise_from_calibration_display
        if turns == 0:
            return image
        if turns == 1:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if turns == 2:
            return cv2.rotate(image, cv2.ROTATE_180)
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def camera_adapter_bounds_in_adb_xywh(self) -> list[int]:
        width, height = self.adapter_size_px
        corners = self.camera_adapter_to_adb_points(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        )
        minimum = np.rint(np.min(corners, axis=0)).astype(int)
        maximum = np.rint(np.max(corners, axis=0)).astype(int)
        return [
            int(minimum[0]),
            int(minimum[1]),
            int(maximum[0] - minimum[0] + 1),
            int(maximum[1] - minimum[1] + 1),
        ]

    def describe(self) -> dict:
        return {
            "schema_version": "1.0",
            "rig_calibration": self.calibration_reference,
            "coordinate_convention": {
                "coordinates": "pixel_center_xy",
                "origin": "top_left_pixel_center_is_[0,0]",
                "axes": "+X_right_+Y_down",
                "matrices": "column_homogeneous_[x,y,1]",
            },
            "spaces": {
                "camera_adapter": {
                    "id": "hik_rig_rectified_visible_phone_pixels",
                    "size_px": list(self.adapter_size_px),
                    "image_orientation": "rig_calibration_display_app_up_app_right",
                },
                "calibration_display": {
                    "id": "android_calibration_logical_display_pixels",
                    "size_px": list(self.calibration_display_size_px),
                    "surface_quarter_turns_clockwise_from_natural": (
                        self.calibration_display_quarter_turns_clockwise_from_natural
                    ),
                },
                "phone_natural": {
                    "id": "android_phone_natural_display_pixels",
                    "size_px": list(self.phone_natural_size_px),
                },
                "adb": {
                    "id": "android_logical_display_pixels",
                    "size_px": list(self.adb_size_px),
                    "surface_quarter_turns_clockwise_from_natural": (
                        self.adb_surface_quarter_turns_clockwise_from_natural
                    ),
                },
            },
            "conversion": {
                "camera_adapter_to_adb_3x3": _matrix_list(
                    self.adapter_to_adb_3x3
                ),
                "adb_to_camera_adapter_3x3": _matrix_list(
                    self.adb_to_adapter_3x3
                ),
                "camera_adapter_to_phone_natural_3x3": (
                    _matrix_list(self.adapter_to_phone_natural_3x3)
                ),
                "camera_adapter_to_calibration_display_3x3": (
                    _matrix_list(self.adapter_to_calibration_display_3x3)
                ),
                "calibration_display_to_phone_natural_3x3": (
                    _matrix_list(self.calibration_display_to_phone_natural_3x3)
                ),
                "phone_natural_to_adb_3x3": (
                    _matrix_list(self.phone_natural_to_adb_3x3)
                ),
                "camera_adapter_to_output_image_3x3": (
                    _matrix_list(self.adapter_to_output_image_3x3)
                ),
                "output_image_to_camera_adapter_3x3": (
                    _matrix_list(self.output_image_to_adapter_3x3)
                ),
                "output_image_quarter_turns_clockwise_from_calibration_display": (
                    self.output_image_quarter_turns_clockwise_from_calibration_display
                ),
                "output_image_size_px": list(self.output_image_size_px),
                "camera_adapter_bounds_in_adb_xywh": (
                    self.camera_adapter_bounds_in_adb_xywh()
                ),
            },
        }
