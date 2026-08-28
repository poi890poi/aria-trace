"""ISO/IEC 15415 Decode grading for displayed Data Matrix symbols.

This module intentionally implements only the standard's binary ``Decode``
parameter: grade 4/A when a Data Matrix is successfully decoded and its
codewords are valid, otherwise grade 0/F.  It does not invent an aggregate
threshold or claim to implement the complete ISO/IEC 15415 symbol grade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


PayloadDecoder = Callable[[np.ndarray], Iterable[str]]
ModuleEncoder = Callable[[str], np.ndarray]


@dataclass(frozen=True)
class DataMatrixObservation:
    """One settled camera image of one displayed Data Matrix stimulus."""

    image: np.ndarray
    expected_payload: str
    module_width_display_px: int
    trial_id: str = ""
    metadata: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if self.image is None or self.image.size == 0:
            raise ValueError("Data Matrix observation image must be non-empty")
        if not self.expected_payload:
            raise ValueError("Expected Data Matrix payload is required")
        if int(self.module_width_display_px) <= 0:
            raise ValueError("Module width must be a positive display-pixel count")


@dataclass(frozen=True)
class DataMatrixTarget:
    """A full-screen raster plus the exact stimulus description."""

    image: np.ndarray
    payload: str
    module_width_display_px: int
    symbol_modules_xy: Tuple[int, int]
    quiet_zone_modules: int
    symbol_rect_screen_xywh: Tuple[int, int, int, int]
    target_rect_screen_xywh: Tuple[int, int, int, int]
    trial_id: str


def _zxing_module():
    try:
        import zxingcpp  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Data Matrix encoding/decoding requires the optional zxing-cpp package"
        ) from exc
    return zxingcpp


def encode_data_matrix_modules(payload: str) -> np.ndarray:
    """Return the unscaled Data Matrix bitmap with one array cell per module."""

    if not payload:
        raise ValueError("Data Matrix payload is required")
    zxingcpp = _zxing_module()
    try:
        try:
            barcode = zxingcpp.create_barcode(
                payload,
                zxingcpp.BarcodeFormat.DataMatrix,
                force_square=True,
            )
        except TypeError:
            # zxing-cpp 2.x exposes the writer, but not its optional shape
            # keyword.  The default Data Matrix writer is square for the
            # payloads used by this calibration sweep.
            barcode = zxingcpp.create_barcode(
                payload,
                zxingcpp.BarcodeFormat.DataMatrix,
            )
        try:
            image = np.asarray(
                barcode.to_image(scale=1, add_hrt=False, add_quiet_zones=False)
            )
        except TypeError:
            image = np.asarray(barcode.to_image())
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            "The installed zxing-cpp lacks a compatible Data Matrix writer API"
        ) from exc
    if image.ndim != 2 or min(image.shape) < 8:
        raise RuntimeError("Data Matrix encoder returned an invalid module bitmap")
    return np.where(image < 128, 0, 255).astype(np.uint8)


def decode_data_matrix_payloads(image: np.ndarray) -> List[str]:
    """Decode Data Matrix symbols using the replaceable ZXing-C++ adapter."""

    if image is None or image.size == 0:
        raise ValueError("Data Matrix image must be non-empty")
    zxingcpp = _zxing_module()
    contiguous = np.ascontiguousarray(image)
    try:
        values = zxingcpp.read_barcodes(
            contiguous,
            formats=zxingcpp.BarcodeFormat.DataMatrix,
            try_rotate=True,
            try_downscale=False,
            try_invert=False,
            return_errors=False,
        )
    except TypeError:
        # zxing-cpp 2.x has the same decoder but no ``try_invert`` keyword.
        try:
            values = zxingcpp.read_barcodes(
                contiguous,
                formats=zxingcpp.BarcodeFormat.DataMatrix,
                try_rotate=True,
                try_downscale=False,
                return_errors=False,
            )
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                "The installed zxing-cpp lacks a compatible Data Matrix decoder API"
            ) from exc
    return [str(value.text) for value in values]


def grade_data_matrix_decode(
    image: np.ndarray,
    expected_payload: str,
    decoder: Optional[PayloadDecoder] = None,
) -> Dict[str, Any]:
    """Return the ISO/IEC 15415 Decode parameter grade for one image.

    The expected payload is a controlled-test synchronization check. If a
    different valid symbol is decoded, its Decode grade is still 4/A, but the
    observation is marked ineligible because it is not the requested target.
    """

    if image is None or image.size == 0:
        raise ValueError("Data Matrix image must be non-empty")
    if not expected_payload:
        raise ValueError("Expected Data Matrix payload is required")
    decode = decoder or decode_data_matrix_payloads
    decoded = [str(value) for value in decode(image)]
    reference_decoded = bool(decoded)
    exact = expected_payload in decoded
    return {
        "standard": "ISO/IEC 15415:2024",
        "parameter": "Decode",
        "grade": 4.0 if reference_decoded else 0.0,
        "grade_letter": "A" if reference_decoded else "F",
        "grade_scale": "4/A or 0/F",
        "reference_decode_succeeded": reference_decoded,
        "exact_payload_decoded": bool(exact),
        "eligible_for_requested_target": bool(exact or not reference_decoded),
        "expected_payload": str(expected_payload),
        "decoded_payloads": decoded,
        "decode_count": len(decoded),
        "failure_kind": (
            None if exact else "wrong_target_or_stale_frame" if decoded else "no_decode"
        ),
        "implementation_conformance": (
            "Decode_parameter_only_not_complete_ISO_IEC_15415_verifier"
        ),
    }


def _rect(value: Sequence[int], name: str) -> Tuple[int, int, int, int]:
    if len(value) != 4:
        raise ValueError("{} must contain x, y, width, height".format(name))
    x, y, width, height = map(int, value)
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError("{} is invalid".format(name))
    return x, y, width, height


def render_data_matrix_target(
    screen_size_px: Sequence[int],
    target_rect_screen_xywh: Sequence[int],
    payload: str,
    module_width_display_px: int,
    quiet_zone_modules: int = 1,
    encoder: Optional[ModuleEncoder] = None,
    trial_id: str = "",
) -> DataMatrixTarget:
    """Render one exact-pixel target centred in a fixed visible screen patch.

    Successive calls may change payload and module width while retaining the
    same ``target_rect_screen_xywh``.  This is the intended protocol for a
    camera that sees only a small part of the phone display.
    """

    if len(screen_size_px) != 2:
        raise ValueError("Screen size must contain width and height")
    screen_width, screen_height = map(int, screen_size_px)
    if min(screen_width, screen_height) <= 0:
        raise ValueError("Screen size must be positive")
    x, y, width, height = _rect(target_rect_screen_xywh, "Target rectangle")
    if x + width > screen_width or y + height > screen_height:
        raise ValueError("Target rectangle lies outside the display")
    module_width = int(module_width_display_px)
    quiet = int(quiet_zone_modules)
    if module_width <= 0 or quiet < 1:
        raise ValueError("Module width must be positive and quiet zone at least one module")

    modules = np.asarray((encoder or encode_data_matrix_modules)(payload))
    if modules.ndim == 3:
        modules = cv2.cvtColor(modules.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if modules.ndim != 2 or min(modules.shape) < 8:
        raise ValueError("Encoder must return a two-dimensional module bitmap")
    modules = np.where(modules < 128, 0, 255).astype(np.uint8)
    padded = cv2.copyMakeBorder(
        modules, quiet, quiet, quiet, quiet, cv2.BORDER_CONSTANT, value=255
    )
    raster = np.repeat(np.repeat(padded, module_width, axis=0), module_width, axis=1)
    raster_height, raster_width = raster.shape
    if raster_width > width or raster_height > height:
        raise ValueError(
            "Data Matrix needs {}x{} display px but the fixed target patch is only {}x{}"
            .format(raster_width, raster_height, width, height)
        )

    canvas = np.full((screen_height, screen_width, 3), 127, dtype=np.uint8)
    canvas[y : y + height, x : x + width] = 255
    left = x + (width - raster_width) // 2
    top = y + (height - raster_height) // 2
    canvas[top : top + raster_height, left : left + raster_width] = cv2.cvtColor(
        raster, cv2.COLOR_GRAY2BGR
    )
    return DataMatrixTarget(
        image=canvas,
        payload=str(payload),
        module_width_display_px=module_width,
        symbol_modules_xy=(int(modules.shape[1]), int(modules.shape[0])),
        quiet_zone_modules=quiet,
        symbol_rect_screen_xywh=(left, top, raster_width, raster_height),
        target_rect_screen_xywh=(x, y, width, height),
        trial_id=str(trial_id),
    )


def alternating_data_matrix_targets(
    screen_size_px: Sequence[int],
    target_rect_screen_xywh: Sequence[int],
    payloads: Sequence[str],
    module_widths_display_px: Sequence[int],
    quiet_zone_modules: int = 1,
    encoder: Optional[ModuleEncoder] = None,
) -> Iterator[DataMatrixTarget]:
    """Yield an interleaved sweep that reuses one fixed on-screen patch."""

    widths = sorted({int(value) for value in module_widths_display_px})
    if not payloads or not widths or min(widths) <= 0:
        raise ValueError("At least one payload and positive module width are required")
    expected_shape: Optional[Tuple[int, int]] = None
    for payload_index, payload in enumerate(payloads):
        for width in widths:
            target = render_data_matrix_target(
                screen_size_px,
                target_rect_screen_xywh,
                payload,
                width,
                quiet_zone_modules=quiet_zone_modules,
                encoder=encoder,
                trial_id="dm-{:04d}-x{}".format(payload_index, width),
            )
            if expected_shape is None:
                expected_shape = target.symbol_modules_xy
            elif target.symbol_modules_xy != expected_shape:
                raise ValueError(
                    "All sweep payloads must encode to the same Data Matrix dimensions"
                )
            yield target


def summarize_data_matrix_decode_sweep(
    observations: Iterable[DataMatrixObservation],
    decoder: Optional[PayloadDecoder] = None,
) -> Dict[str, Any]:
    """Grade an alternating sweep without creating a non-standard score.

    Results remain counts of standard 4/A and 0/F Decode grades at each
    declared module width. No pass percentage, average grade, or inferred
    resolution threshold is introduced.
    """

    rows = list(observations)
    if not rows:
        raise ValueError("At least one Data Matrix observation is required")
    trials: List[Dict[str, Any]] = []
    for index, observation in enumerate(rows):
        measured = grade_data_matrix_decode(
            observation.image, observation.expected_payload, decoder=decoder
        )
        measured.update(
            {
                "trial_id": observation.trial_id or "dm-{:05d}".format(index),
                "module_width_display_px": int(observation.module_width_display_px),
                "metadata": dict(observation.metadata or {}),
            }
        )
        trials.append(measured)

    summaries: List[Dict[str, Any]] = []
    for width in sorted({row["module_width_display_px"] for row in trials}):
        selected = [row for row in trials if row["module_width_display_px"] == width]
        eligible = [row for row in selected if row["eligible_for_requested_target"]]
        grade_a = sum(float(row["grade"]) == 4.0 for row in eligible)
        grade_f = sum(float(row["grade"]) == 0.0 for row in eligible)
        summaries.append(
            {
                "module_width_display_px": int(width),
                "observation_count": len(selected),
                "eligible_observation_count": len(eligible),
                "decode_grade_4_A_count": int(grade_a),
                "decode_grade_0_F_count": int(grade_f),
                "ineligible_wrong_target_count": sum(
                    not row["eligible_for_requested_target"] for row in selected
                ),
            }
        )
    return {
        "standard": "ISO/IEC 15415:2024",
        "symbology_standard": "ISO/IEC 16022:2024",
        "parameter": "Decode",
        "grade_scale": "4/A or 0/F",
        "meaning": "4/A = reference decode succeeds; 0/F = reference decode fails",
        "implementation": "ZXing-C++ Data Matrix decoder or caller-supplied compatible decoder",
        "implementation_conformance": "Decode_parameter_only_not_complete_ISO_IEC_15415_verifier",
        "aggregation": "none_standard_grades_reported_as_counts",
        "tested_module_widths_display_px": [
            int(item["module_width_display_px"]) for item in summaries
        ],
        "module_width_results": summaries,
        "trials": trials,
    }
