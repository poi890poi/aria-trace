"""Commented YAML persistence and validation for rig calibrations."""

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import yaml

from .contracts import matrix_3x3


CALIBRATION_SCHEMA_VERSION = "1.0"
_HEADER = """# AriaTrace camera-to-phone rig calibration.
#
# Consumer fast path:
#   1. Obtain a frame in normalization.input_space from the calibrated source.
#   2. Apply normalization.matrix_3x3 with normalization.output_size_px.
#   3. Interpret output pixel (0, 0) as normalization.origin_screen_xy and
#      multiply output coordinates by screen_units_per_output_pixel_xy.
#
# Coordinate convention: integer XY values address pixel centres; (0, 0) is
# the centre of the top-left pixel, +X points right, and +Y points down.
# Clock transforms and causal latency distributions are deliberately separate.
"""
_SECTION_COMMENTS = {
    "rig": "# Physical devices and the exact modes to which this artifact applies.",
    "optics": "# Lens and locked optical settings owned by the calibrated capture layer.",
    "normalization": "# Stable downstream contract: undistorted input pixels to normalized output.",
    "geometry": "# Measured overlap, reprojection, pose, and uncertainty evidence.",
    "required_roi": "# Opaque caller-supplied region; the calibration core does not interpret its label.",
    "image_quality": "# Display-referred ISO 12233 e-SFR/MTF evidence from pre-warp camera samples.",
    "data_matrix_decode": "# ISO/IEC 15415 Decode grades for displayed Data Matrix symbols.",
    "feature_matching": "# Ground-truth repeatability, matching score, MMA, coverage, and pose error.",
    "timing": "# Timestamp coordinates and measured causal delays are separate models.",
    "confidence": "# Quality summaries retain their underlying measurements and warnings.",
    "applicability": "# A mismatch requires validation or recalibration instead of silent reuse.",
    "evidence": "# Relative paths resolve beneath the artifact directory.",
}


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _pair(value: Any, name: str, positive: bool = False) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("{} must contain two values".format(name))
    result = list(map(float, value))
    if not np.all(np.isfinite(result)):
        raise ValueError("{} must contain finite values".format(name))
    if positive and min(result) <= 0:
        raise ValueError("{} must contain positive values".format(name))
    return result


def validate_calibration(value: Mapping[str, Any]) -> List[str]:
    """Validate the consumer contract and return non-fatal warnings."""

    if not isinstance(value, Mapping):
        raise ValueError("Calibration YAML root must be a mapping")
    if str(value.get("schema_version")) != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("Unsupported calibration schema version")
    if not value.get("calibration_id"):
        raise ValueError("calibration_id is required")
    if value.get("status") not in (
        "accepted",
        "warning",
        "failed_task_requirement",
    ):
        raise ValueError("Calibration status is invalid")
    normalization = value.get("normalization")
    if not isinstance(normalization, Mapping):
        raise ValueError("normalization block is required")
    required_strings = (
        "input_frame_id",
        "input_space",
        "output_frame_id",
        "canonical_screen_frame_id",
        "transform_direction",
    )
    for name in required_strings:
        if not normalization.get(name):
            raise ValueError("normalization.{} is required".format(name))
    if normalization["transform_direction"] != "input_pixel_to_output_pixel":
        raise ValueError("Unsupported normalization transform direction")
    _pair(normalization.get("input_size_px"), "normalization.input_size_px", True)
    _pair(normalization.get("output_size_px"), "normalization.output_size_px", True)
    _pair(normalization.get("origin_screen_xy"), "normalization.origin_screen_xy")
    _pair(
        normalization.get("screen_units_per_output_pixel_xy"),
        "normalization.screen_units_per_output_pixel_xy",
        True,
    )
    matrix_3x3(normalization.get("matrix_3x3"))
    if normalization.get("input_origin") != "top_left_pixel_center":
        raise ValueError("normalization.input_origin must be top_left_pixel_center")
    if normalization.get("output_origin") != "top_left_pixel_center":
        raise ValueError("normalization.output_origin must be top_left_pixel_center")

    warnings = []
    if value.get("status") != "accepted":
        warnings.append("calibration_is_not_accepted")
    assumptions = value.get("confidence", {}).get("assumptions", [])
    if assumptions:
        warnings.append("calibration_contains_assumptions")
    timing = value.get("timing", {})
    for name, latency in timing.items():
        if not isinstance(latency, Mapping) or "median_ns" not in latency:
            continue
        for field in ("median_ns", "p05_ns", "p95_ns"):
            if field in latency and float(latency[field]) < 0:
                raise ValueError("timing.{}.{} must be non-negative".format(name, field))
    return warnings


def _insert_section_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        key = line[:-1] if line.endswith(":") and not line.startswith(" ") else None
        if key in _SECTION_COMMENTS:
            if lines and lines[-1] != "":
                lines.append("")
            lines.append(_SECTION_COMMENTS[key])
        lines.append(line)
    return "\n".join(lines) + "\n"


def write_calibration_yaml(path: Path, value: Mapping[str, Any]) -> Path:
    """Atomically write an authoritative YAML artifact with contract comments."""

    plain = _plain(value)
    validate_calibration(plain)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(
        plain,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )
    text = _HEADER + "\n" + _insert_section_comments(body)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))
    return path


def load_calibration_yaml(path: Path) -> Dict[str, Any]:
    path = Path(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_calibration(value)
    return dict(value)
