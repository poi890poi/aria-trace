"""Compare known-pivot inputs with the pre-fix revision; no capture/device claim.

Run from the repository root with python -m benchmarks.cursor_pose.calibration_cold_circle.
"""

import hashlib
import json
import subprocess
import time

import cv2
import numpy as np

from rig_runtime.services.calibration.minimap.calibration import _cursor_temporal_center
from tests.test_cursor_cold_circle import rotating_cursor_frames


BASELINE = "b9be825e0f3bbb3494674e13d4f4e5d1f2803514"


def main():
    source = subprocess.check_output([
        "git", "show", BASELINE + ":rig_runtime/services/calibration/minimap/calibration.py"
    ]).decode("utf-8")
    namespace = {"__name__": "calibration_baseline"}
    exec(compile(source, "calibration_baseline", "exec"), namespace)
    old = namespace["_cursor_temporal_center"]
    cases = []
    for shape in ("triangle", "dot"):
        for name, angles in (
            ("balanced", np.arange(0, 360, 30)),
            ("unequal_dwell", np.r_[np.zeros(35), [25, 65, 110, 175, 235, 295]]),
            ("incomplete_disc", [0, 25, 55, 85, 125, 160, 205]),
            ("short_arc_limit", [0, 20, 45, 65, 90]),
        ):
            cases.append((name, shape, (62.3, 58.7), rotating_cursor_frames(angles, shape=shape)))
        for pivot, size, angles in (
            ((66.6, 61.2), .85, [17, 44, 76, 113, 159, 207, 253]),
            ((59.8, 65.4), 1.3, [83, 115, 154, 191, 226, 271, 316]),
        ):
            cases.append(("fresh_holdout", shape, pivot, rotating_cursor_frames(angles, pivot, shape, size, 8307, .6)))
    cases.append(("static", "triangle", None, rotating_cursor_frames([0] * 10, shape="triangle")))
    flicker = np.stack([np.full((128, 128, 3), 30 + index, np.uint8) for index in range(10)])
    cases.append(("uniform_flicker", "none", None, flicker))
    translation = np.full((12, 128, 128, 3), 30, np.uint8)
    for index, frame in enumerate(translation):
        cv2.circle(frame, (58 + index, 62), 4, (25, 30, 235), -1)
    cases.append(("translation_only", "dot", None, translation))
    for name, shape, pivot, frames in cases:
        row = {"case": name, "shape": shape, "pivot": pivot, "input_sha256": hashlib.sha256(frames.tobytes()).hexdigest(), "baseline_revision": BASELINE}
        for label, fit in (("baseline", old), ("candidate", _cursor_temporal_center)):
            start = time.perf_counter()
            try:
                result = fit(frames, np.array([64., 64.]), 12)["metrics"]
                row[label] = {"status": "fresh", "xy": [result["x"], result["y"]], "error_px": float(np.linalg.norm(np.array([result["x"], result["y"]]) - pivot)) if pivot else None, "metrics": result}
            except RuntimeError as error:
                row[label] = {"status": "unavailable", "reason": str(error)}
            row[label]["fit_ms"] = 1000 * (time.perf_counter() - start)
        print(json.dumps(row))


if __name__ == "__main__":
    main()
