"""Apply a reviewed rig calibration to camera frames and coordinates."""

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .contracts import matrix_3x3
from .geometry import transform_points


def build_rectification_maps(
    camera_matrix_3x3: Sequence[Sequence[float]],
    distortion_coefficients: Sequence[float],
    input_size_px: Sequence[int],
    new_camera_matrix_3x3: Optional[Sequence[Sequence[float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    width, height = map(int, input_size_px)
    if width <= 0 or height <= 0:
        raise ValueError("Input size must be positive")
    camera_matrix = matrix_3x3(camera_matrix_3x3)
    new_camera_matrix = (
        matrix_3x3(new_camera_matrix_3x3)
        if new_camera_matrix_3x3 is not None
        else camera_matrix
    )
    distortion = np.asarray(distortion_coefficients, dtype=np.float64).reshape(-1)
    if distortion.size < 4 or not np.all(np.isfinite(distortion)):
        raise ValueError("At least four finite distortion coefficients are required")
    return cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        new_camera_matrix,
        (width, height),
        cv2.CV_32FC1,
    )


class FrameNormalizer:
    """Normalize reviewed input pixels without guessing coordinate conventions."""

    def __init__(
        self,
        calibration: Mapping[str, Any],
        artifact_root: Optional[Path] = None,
    ) -> None:
        self.calibration = calibration
        self.artifact_root = Path(artifact_root) if artifact_root is not None else None
        normalization = calibration.get("normalization")
        if not isinstance(normalization, Mapping):
            raise ValueError("Calibration has no normalization block")
        self.normalization = normalization
        self.input_space = str(normalization.get("input_space", ""))
        if not self.input_space:
            raise ValueError("normalization.input_space is required")
        self.input_size = self._size(normalization.get("input_size_px"), "input_size_px")
        self.output_size = self._size(
            normalization.get("output_size_px"), "output_size_px"
        )
        self.matrix = matrix_3x3(normalization.get("matrix_3x3"))
        self.origin_screen_xy = self._pair(
            normalization.get("origin_screen_xy"), "origin_screen_xy"
        )
        self.screen_units_per_output_pixel_xy = self._pair(
            normalization.get("screen_units_per_output_pixel_xy"),
            "screen_units_per_output_pixel_xy",
        )
        if min(self.screen_units_per_output_pixel_xy) <= 0:
            raise ValueError("Output-to-screen scale must be positive")
        self.border_value = tuple(
            int(value) for value in normalization.get("border_value_bgr", [0, 0, 0])
        )
        if len(self.border_value) != 3:
            raise ValueError("border_value_bgr must contain three values")
        self._map_x = None
        self._map_y = None
        self._load_or_build_lens_maps()
        self.valid_mask = self._load_valid_mask()

    @staticmethod
    def _size(value: Any, name: str) -> Tuple[int, int]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("normalization.{} must contain width and height".format(name))
        result = tuple(map(int, value))
        if min(result) <= 0:
            raise ValueError("normalization.{} must be positive".format(name))
        return result

    @staticmethod
    def _pair(value: Any, name: str) -> Tuple[float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("normalization.{} must contain two values".format(name))
        result = tuple(map(float, value))
        if not np.all(np.isfinite(result)):
            raise ValueError("normalization.{} must be finite".format(name))
        return result

    def _resolve(self, relative: str) -> Path:
        if self.artifact_root is None:
            raise ValueError("Calibration references files but artifact_root was not supplied")
        path = (self.artifact_root / relative).resolve()
        root = self.artifact_root.resolve()
        try:
            path.relative_to(root)
        except ValueError:
            raise ValueError("Calibration file reference escapes artifact_root")
        return path

    def _load_or_build_lens_maps(self) -> None:
        lens = self.calibration.get("optics", {}).get("lens_model", {})
        source = lens.get("source", "unavailable")
        maps_file = lens.get("precomputed_maps_file")
        if maps_file:
            data = np.load(str(self._resolve(str(maps_file))), allow_pickle=False)
            self._map_x = np.asarray(data["map_x"], dtype=np.float32)
            self._map_y = np.asarray(data["map_y"], dtype=np.float32)
        elif source not in ("unavailable", "assumed", None):
            self._map_x, self._map_y = build_rectification_maps(
                lens.get("camera_matrix_3x3"),
                lens.get("distortion_coefficients"),
                self.input_size,
                lens.get("new_camera_matrix_3x3"),
            )
        if self._map_x is not None:
            expected_shape = (self.input_size[1], self.input_size[0])
            if self._map_x.shape != expected_shape or self._map_y.shape != expected_shape:
                raise ValueError("Lens maps do not match normalization input size")

    def _load_valid_mask(self) -> Optional[np.ndarray]:
        relative = self.normalization.get("valid_mask_file")
        if not relative:
            return None
        mask = cv2.imread(str(self._resolve(str(relative))), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError("Cannot read calibration valid mask")
        expected_shape = (self.output_size[1], self.output_size[0])
        if mask.shape != expected_shape:
            raise ValueError("Valid mask does not match normalization output size")
        return mask

    def _validate_image(self, image: np.ndarray) -> None:
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")
        if image.shape[1::-1] != self.input_size:
            raise ValueError(
                "Input image is {}x{}, expected {}x{}".format(
                    image.shape[1], image.shape[0], self.input_size[0], self.input_size[1]
                )
            )

    def undistort(self, raw_image: np.ndarray) -> np.ndarray:
        self._validate_image(raw_image)
        if self._map_x is None:
            return raw_image.copy()
        return cv2.remap(
            raw_image,
            self._map_x,
            self._map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.border_value,
        )

    def normalize(
        self, image: np.ndarray, input_space: str = "camera_undistorted_px"
    ) -> np.ndarray:
        if input_space != self.input_space:
            raise ValueError(
                "Input space {} does not match calibration {}".format(
                    input_space, self.input_space
                )
            )
        self._validate_image(image)
        normalized = cv2.warpPerspective(
            image,
            self.matrix,
            self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.border_value,
        )
        return normalized

    def normalize_raw(self, raw_image: np.ndarray) -> np.ndarray:
        return self.normalize(self.undistort(raw_image), self.input_space)

    def input_to_output_points(
        self, input_points_xy: Sequence[Sequence[float]]
    ) -> np.ndarray:
        return transform_points(input_points_xy, self.matrix)

    def output_to_screen_points(
        self, output_points_xy: Sequence[Sequence[float]]
    ) -> np.ndarray:
        points = np.asarray(output_points_xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Expected output XY points")
        return (
            points * np.asarray(self.screen_units_per_output_pixel_xy)
            + np.asarray(self.origin_screen_xy)
        )

    def input_to_screen_points(
        self, input_points_xy: Sequence[Sequence[float]]
    ) -> np.ndarray:
        return self.output_to_screen_points(self.input_to_output_points(input_points_xy))
