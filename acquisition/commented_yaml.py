"""Small generic writer for human-readable, commented YAML companions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import yaml


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


def write_commented_yaml(
    path: Path,
    value: Mapping[str, Any],
    *,
    header: str,
    section_comments: Optional[Mapping[str, str]] = None,
) -> Path:
    """Atomically write YAML with a prose header and top-level comments."""

    comments = dict(section_comments or {})
    body = yaml.safe_dump(
        _plain(value),
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )
    lines = []
    for line in body.splitlines():
        key = line[:-1] if line.endswith(":") and not line.startswith(" ") else None
        if key in comments:
            if lines and lines[-1] != "":
                lines.append("")
            comment = str(comments[key]).strip()
            lines.extend(
                item if item.startswith("#") else "# " + item
                for item in comment.splitlines()
            )
        lines.append(line)
    text = str(header).rstrip() + "\n\n" + "\n".join(lines) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))
    return path


HIK_CONFIG_HEADER = """# AriaTrace HIK camera adapter configuration.
#
# This file owns camera controls, camera-to-phone geometry, and the normalized
# complete visible-phone output. Pixel coordinates use top-left origin, +X
# right, +Y down. Distances ending in _px are pixels; exposure is microseconds;
# timestamps ending in _ns are nanoseconds. The JSON companion has identical
# data for existing consumers."""

HIK_CONFIG_COMMENTS = {
    "camera": "Camera identity, effective full-sensor mode, controls, and transport ROI.",
    "phone": "Phone identity, raster, orientation, refresh, brightness, and physical scale.",
    "imaging": "Locked exposure, gain, black level, and white balance owned by this rig.",
    "geometry": "Measured camera-sensor to phone-display coordinate relationship.",
    "normalization": "Runtime mapping from HIK sensor pixels to the normalized visible-phone image.",
    "results": "Calibration and diagnostic evidence; these fields do not change runtime coordinates.",
}

PROFILE_HEADER = """# AriaTrace calibration profile.
#
# Profiles are immutable revisions. `current` files are pointers only. A
# profile's coordinate_space fields define where every XY/WH value lives;
# profiles from different scopes must not be combined without an explicit
# referenced coordinate mapping. The JSON companion has identical data."""

PROFILE_COMMENTS = {
    "profile": "Stable scope and identity of this reusable profile.",
    "reuse": "Compatibility and revision rules; evidence is never silently reused across scopes.",
    "artifacts": "Paths to machine-readable results and human-review evidence.",
    "notes": "Human-facing usage constraints not interpreted as runtime parameters.",
}
