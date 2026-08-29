"""Configure a HIK camera from a saved calibration and stream rectified frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import cv2

from acquisition.rig_calibration.hik.camera import Camera, HikCamera
from acquisition.rig_calibration.hik.driver import HikMvsCameraAdapter, RectifiedHikCamera
from acquisition.rig_calibration.hik.phone import AdbPhoneSession


class PhoneDisplayPowerSession:
    """Manage only Android panel power; never mutate settings or launch an app."""

    def __init__(
        self,
        serial: str,
        adb_executable: str = "adb",
        phone: Optional[AdbPhoneSession] = None,
    ) -> None:
        self.phone = phone or AdbPhoneSession(serial, adb_executable=adb_executable)
        self._opened = False
        self.last_error: Optional[str] = None

    def open(self) -> "PhoneDisplayPowerSession":
        if self._opened:
            return self
        self._opened = True
        try:
            self.phone.shell("input", "keyevent", "KEYCODE_WAKEUP")
        except Exception as exc:
            self.last_error = str(exc)
        return self

    def close(self) -> None:
        if not self._opened:
            return
        try:
            self.phone.shell("input", "keyevent", "KEYCODE_SLEEP")
        except Exception as exc:
            self.last_error = str(exc)
        finally:
            self._opened = False

    def __enter__(self) -> "PhoneDisplayPowerSession":
        return self.open()

    def __exit__(self, *_exc) -> None:
        self.close()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Stream the calibrated phone-display ROI from HIK MVS.")
    value.add_argument("calibration", type=Path, help="hik_camera_calibration.json")
    value.add_argument("--mvs-python-path")
    value.add_argument("--gui", action="store_true", help="Show live rectified frames; Q/Esc closes")
    value.add_argument(
        "--no-rectify",
        action="store_true",
        help="Show the hardware-ROI camera frame without remap/warp for minimum latency",
    )
    value.add_argument(
        "--manage-phone-display",
        action="store_true",
        help="Wake the calibrated phone display for the GUI session and sleep it on exit",
    )
    value.add_argument("--adb", default="adb", help="ADB executable used only for display power")
    return value


def open_camera(
    calibration: Path,
    mvs_python_path: Optional[str] = None,
    rectify: bool = True,
) -> RectifiedHikCamera:
    """Public UVC-like constructor for application code (read/release/isOpened)."""

    adapter = HikMvsCameraAdapter(sdk_python_path=mvs_python_path)
    return RectifiedHikCamera(
        calibration, adapter=adapter, rectify=rectify
    ).open()


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.manage_phone_display and not arguments.gui:
        parser().error("--manage-phone-display requires --gui")
    display = None
    camera = None
    window = "HIK rectified phone display"
    try:
        if arguments.manage_phone_display:
            configuration = json.loads(
                arguments.calibration.read_text(encoding="utf-8")
            )
            serial = str(configuration.get("phone", {}).get("serial", "")).strip()
            if serial:
                try:
                    display = PhoneDisplayPowerSession(
                        serial, adb_executable=arguments.adb
                    )
                    print(
                        "Turning on Android display {} (best effort).".format(serial),
                        flush=True,
                    )
                    display.open()
                    if display.last_error:
                        print(
                            "Display wake warning: {}. Camera startup continues."
                            .format(display.last_error),
                            flush=True,
                        )
                except Exception as exc:
                    display = None
                    print(
                        "Display wake warning: {}. Camera startup continues."
                        .format(exc),
                        flush=True,
                    )
        camera = open_camera(
            arguments.calibration,
            arguments.mvs_python_path,
            rectify=not arguments.no_rectify,
        )
        if not arguments.gui:
            print("Rectified stream configured. Use open_camera(...) from Python, or pass --gui.")
            return 0
        print(
            "Live {} stream opened. Press Q/Esc or close the window to exit."
            .format("rectified" if not arguments.no_rectify else "hardware-ROI"),
            flush=True,
        )
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    return 0
                continue
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                return 0
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                return 0
    finally:
        try:
            if camera is not None:
                camera.release()
        finally:
            if arguments.gui:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass
            if display is not None:
                print("Turning off Android display.", flush=True)
                display.close()


if __name__ == "__main__":
    raise SystemExit(main())
