"""Directional sampling-only MTF50 references, not measured hardware bounds.

See docs/iris-focus-sampling-theory-2026-09-08.md for the ideal independent-channel
low-pass model and its limitations. No focus measurement is an input here.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np

PANEL_LAYOUTS = ("unknown", "rgb-stripe", "diamond-pentile")
CAMERA_LAYOUTS = ("auto", "bayer", "full-grid")


def resolve_camera_sampling(requested: str, pixel_format: Any) -> tuple[str, str]:
    if requested not in CAMERA_LAYOUTS:
        raise ValueError("Unknown focus camera sampling layout")
    if requested != "auto":
        return requested, "configured"
    # These four standard PFNC Bayer8 values are independent of CFA origin.
    # Other numeric formats remain unknown rather than guessing from bit depth.
    if isinstance(pixel_format, (int, np.integer)) and int(pixel_format) in (
        0x01080008, 0x01080009, 0x0108000A, 0x0108000B,
    ):
        return "bayer", "PixelFormat"
    if isinstance(pixel_format, str) and re.fullmatch(
        r"Bayer(?:RG|GR|GB|BG)(?:8|10|12|14|16)(?:p|Packed)?", pixel_format
    ):
        return "bayer", "PixelFormat"
    # RGB/BGR and Mono transport formats do not identify the physical CFA.
    return "unknown", "unknown; alternatives"


def edge_sampling_theory(
    edge_angle_deg: float,
    jacobian_display_px_per_camera_px: Sequence[Sequence[float]],
    panel_layout: str = "unknown",
    camera_layout: str = "unknown",
) -> dict[str, Any]:
    """Return directional ideal cutoffs in cycles per display pixel.

    n is a display frequency direction; q=J.T@n is the same frequency in camera
    coordinates. Lattice translations (CFA origin, ROI offset) do not change the
    passband. Channel reconstruction is ideal and independent: no cross-channel
    priors, aperture blur, lens blur, gamma, demosaic or sharpening is modelled.
    """
    if panel_layout not in PANEL_LAYOUTS or camera_layout not in (
        "unknown", "bayer", "full-grid"
    ):
        raise ValueError("Unsupported focus sampling layout")
    angle = np.deg2rad(float(edge_angle_deg))
    normal = np.asarray([-np.sin(angle), np.cos(angle)])
    jacobian = np.asarray(jacobian_display_px_per_camera_px, dtype=np.float64)
    if (
        jacobian.shape != (2, 2)
        or not np.all(np.isfinite(jacobian))
        or not np.all(np.isfinite(normal))
        or abs(float(np.linalg.det(jacobian))) < 1e-12
    ):
        raise ValueError("Invalid local camera/display sampling geometry")
    camera_normal = jacobian.T @ normal

    def square(vector: np.ndarray) -> float:
        return float(0.5 / np.max(np.abs(vector)))

    def checkerboard(vector: np.ndarray) -> float:
        return float(0.5 / np.sum(np.abs(vector)))

    panels = {
        layout: {
            "green": square(normal),
            "red_blue": (
                checkerboard(normal) if layout == "diamond-pentile" else square(normal)
            ),
        }
        for layout in (PANEL_LAYOUTS[1:] if panel_layout == "unknown" else (panel_layout,))
    }
    cameras = {
        layout: {
            "green": checkerboard(camera_normal) if layout == "bayer" else square(camera_normal),
            "red_blue": square(camera_normal) * (0.5 if layout == "bayer" else 1.0),
        }
        for layout in (("bayer", "full-grid") if camera_layout == "unknown" else (camera_layout,))
    }
    combinations = []
    for panel_name, panel in panels.items():
        for camera_name, camera in cameras.items():
            green = min(panel["green"], camera["green"])
            red_blue = min(panel["red_blue"], camera["red_blue"])
            combinations.append({
                "panel_layout": panel_name,
                "camera_layout": camera_name,
                "ideal_luminance_mtf50": green,
                "ideal_red_blue_mtf50": red_blue,
            })
    return {
        "model": "independent_channel_ideal_low_pass_v1",
        "unit": "cycles_per_display_pixel",
        "edge_normal_display_xy": normal.tolist(),
        "frequency_vector_camera_per_display": camera_normal.tolist(),
        "panel_layout": panel_layout,
        "camera_layout": camera_layout,
        "panel": panels,
        "camera": cameras,
        "combined": combinations,
        "measured_esfr_frequency_max": 0.5,
    }


def focus_theory_lines(edges: Sequence[Mapping[str, Any]], camera_source: str) -> list[str]:
    """Compact focus-panel summary; require fresh geometry for all four edges."""
    rows = [edge.get("sampling_theory") for edge in edges]
    if len(rows) != 4 or not all(rows):
        reason = next((edge.get("sampling_theory_error") for edge in edges
                       if edge.get("sampling_theory_error")), "need geometry at all 4 edges")
        return ["Theoretical MTF50 unavailable: {}".format(reason)]

    def values(group: str, layout: str, channel: str) -> float:
        return min(row[group][layout][channel] for row in rows)

    lines = [
        "Theory: ideal low-pass MTF50 (cy/dpx)",
        "Min of 4 edges; G=green, R/B=red/blue",
        "No aperture/ISP model; actual may differ.",
        "Assumes 1 dpx = 1 physical panel pixel.",
    ]
    panel_unknown = rows[0]["panel_layout"] == "unknown"
    for layout in rows[0]["panel"]:
        name = "Diamond PenTile" if layout == "diamond-pentile" else "RGB stripe"
        lines.append("Panel {}{}: G {:.3f}, R/B {:.3f}".format(
            name, "?" if panel_unknown else "", values("panel", layout, "green"),
            values("panel", layout, "red_blue"),
        ))
    for layout in rows[0]["camera"]:
        name = "Bayer" if layout == "bayer" else "full grid"
        lines.append("Camera {}{}: G {:.3f}, R/B {:.3f}".format(
            name, "?" if rows[0]["camera_layout"] == "unknown" else "",
            values("camera", layout, "green"), values("camera", layout, "red_blue"),
        ))
    # Report the range of per-scenario four-edge minima, not a mix of scenarios.
    def combined_range(key: str) -> str:
        minima = [min(row["combined"][i][key] for row in rows)
                  for i in range(len(rows[0]["combined"]))]
        lo, hi = min(minima), max(minima)
        return "{:.3f}".format(lo) if abs(hi - lo) < 0.0005 else "{:.3f}-{:.3f}".format(lo, hi)

    lines.append("Combined Y: {}; R/B: {}".format(
        combined_range("ideal_luminance_mtf50"), combined_range("ideal_red_blue_mtf50")))
    if panel_unknown or rows[0]["camera_layout"] == "unknown":
        lines.append("? = unknown layout; named alternatives")
    lines.extend([
        "Camera layout source: {}".format(camera_source),
        "Reference only; measured range <=0.500.",
    ])
    return lines
