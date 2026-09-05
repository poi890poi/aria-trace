"""Render a phone-readable benchmark summary from report.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _text(image, value, xy, scale=0.62, color=(220, 226, 235), thickness=1):
    cv2.putText(
        image,
        str(value),
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def render(report_path: Path, output_path: Path) -> None:
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    reference = report["reference"]["method"]
    ordered = sorted(
        report["aggregate"].items(),
        key=lambda pair: (
            pair[0] != reference,
            pair[1]["worst_session_latency_p95_ms"],
        ),
    )
    height = 410 + len(ordered) * 150
    image = np.full((height, 1080, 3), (23, 27, 35), np.uint8)
    _text(image, "Whole-scene shift - full 30 FPS replay", (42, 62), 0.95, (245, 246, 250), 2)
    _text(image, "Run 17 development + Run 18 independent reverse traversal", (42, 105), 0.58, (170, 185, 205))
    _text(image, "Error = disagreement from accurate KLT; not physical ground truth", (42, 139), 0.56, (170, 185, 205))
    _text(image, "PASS needs >=95% fresh <=33.3 ms, error P95 <=0.10 deg, worst <=0.50 deg", (42, 173), 0.52, (170, 185, 205))
    y = 215
    for name, item in ordered:
        supported = item["meets_30fps_core"] and item["meets_reference_consistency"]
        is_reference = name == reference
        border = (210, 160, 70) if is_reference else (75, 185, 105) if supported else (70, 90, 205)
        cv2.rectangle(image, (30, y), (1050, y + 126), (32, 39, 51), thickness=-1)
        cv2.rectangle(image, (30, y), (1050, y + 126), border, thickness=3)
        label = name + ("  [REFERENCE]" if is_reference else "")
        _text(image, label, (52, y + 34), 0.57, (238, 241, 246), 1)
        _text(
            image,
            "fresh <=33ms {:5.1f}%    latency P95 {:6.2f} ms".format(
                item["worst_session_fresh_within_33_3ms_coverage"] * 100.0,
                item["worst_session_latency_p95_ms"],
            ),
            (52, y + 72),
            0.55,
        )
        _text(
            image,
            "error P95 {:6.3f} deg    worst {:6.3f} deg    {}".format(
                item["worst_session_reference_error_p95_deg"],
                item["worst_reference_error_deg"],
                "REFERENCE" if is_reference else "PASS" if supported else "REJECT",
            ),
            (52, y + 106),
            0.55,
            (120, 220, 145) if supported else (120, 185, 245) if is_reference else (110, 130, 245),
            2,
        )
        y += 150
    _text(image, "Decision", (42, y + 34), 0.72, (245, 246, 250), 2)
    _text(image, "Route mode: bypass unused scene KLT.", (42, y + 73), 0.62, (120, 220, 145), 2)
    _text(image, "Free roam: retain accurate KLT; no faster candidate survived holdout.", (42, y + 110), 0.58, (220, 205, 125), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError("Could not write {}".format(output_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.report, args.output)


if __name__ == "__main__":
    main()
