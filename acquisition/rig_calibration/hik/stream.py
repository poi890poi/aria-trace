"""Configure a HIK camera from a saved calibration and stream rectified frames."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import cv2

from acquisition.rig_calibration.hik.camera import Camera, HikCamera
from acquisition.rig_calibration.hik.driver import HikMvsCameraAdapter, RectifiedHikCamera


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Stream the calibrated phone-display ROI from HIK MVS.")
    value.add_argument("calibration", type=Path, help="hik_camera_calibration.json")
    value.add_argument("--mvs-python-path")
    value.add_argument("--gui", action="store_true", help="Show live rectified frames; Q/Esc closes")
    return value


def open_camera(calibration: Path, mvs_python_path: Optional[str] = None) -> RectifiedHikCamera:
    """Public UVC-like constructor for application code (read/release/isOpened)."""

    adapter = HikMvsCameraAdapter(sdk_python_path=mvs_python_path)
    return RectifiedHikCamera(calibration, adapter=adapter).open()


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    camera = open_camera(arguments.calibration, arguments.mvs_python_path)
    try:
        if not arguments.gui:
            print("Rectified stream configured. Use open_camera(...) from Python, or pass --gui.")
            return 0
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError("HIK camera returned no rectified frame")
            cv2.imshow("HIK rectified phone display", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                return 0
    finally:
        camera.release()
        if arguments.gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
