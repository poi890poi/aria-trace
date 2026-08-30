"""Configure a HIK camera from a saved calibration and stream rectified frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import cv2

from .camera import Camera, HikCamera
from .driver import HikMvsCameraAdapter, RectifiedHikCamera
from .game_camera import ProfiledHikGameCamera
from .phone import AdbPhoneSession


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
    value.add_argument(
        "--diagnostic-calibration-override",
        type=Path,
        help="explicit rig JSON for a diagnostic run; production uses the registry",
    )
    value.add_argument(
        "--diagnostic-rig-game-profile-override",
        type=Path,
        help="explicit immutable rig-game profile for a diagnostic run",
    )
    value.add_argument(
        "--mode",
        choices=("minimap", "full", "dual"),
        default="full",
        help=(
            "adapter stream mode; the registry resolves all required profiles"
        ),
    )
    value.add_argument("--mvs-python-path")
    value.add_argument("--profile-root", type=Path)
    value.add_argument("--game-id")
    value.add_argument("--camera-id")
    value.add_argument("--phone-serial")
    value.add_argument("--color-order", choices=("RGB", "BGR"), default="BGR")
    value.add_argument(
        "--color-policy",
        choices=("auto", "rig_locked", "game_matched", "unadjusted"),
        default="auto",
    )
    value.add_argument("--minimap-margin-px", type=int, default=6)
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
    diagnostic_calibration_override: Optional[Path] = None,
    mvs_python_path: Optional[str] = None,
    rectify: bool = True,
    diagnostic_rig_game_profile_override: Optional[Path] = None,
    mode: str = "full",
    profile_root: Optional[Path] = None,
    game_id: Optional[str] = None,
    camera_id: Optional[str] = None,
    phone_serial: Optional[str] = None,
    color_order: str = "BGR",
    color_policy: str = "auto",
    minimap_margin_px: int = 6,
):
    """Public UVC-like constructor for application code (read/release/isOpened)."""

    if diagnostic_calibration_override is None:
        if diagnostic_rig_game_profile_override is not None:
            raise ValueError(
                "A diagnostic rig-game profile requires a diagnostic rig calibration"
            )
        return HikCamera(
            config={
                "profile_root": profile_root,
                "game_id": game_id,
                "camera_id": camera_id,
                "phone_id": phone_serial,
                "mode": mode,
                "rectify": rectify,
                "color_order": color_order,
                "color_policy": color_policy,
                "minimap_margin_px": minimap_margin_px,
                "mvs_python_path": mvs_python_path,
            }
        ).open()
    if diagnostic_rig_game_profile_override is None and mode != "full":
        raise ValueError(
            "Diagnostic --mode {} requires --diagnostic-rig-game-profile-override"
            .format(mode)
        )
    adapter = HikMvsCameraAdapter(sdk_python_path=mvs_python_path)
    if diagnostic_rig_game_profile_override is not None:
        return ProfiledHikGameCamera(
            diagnostic_calibration_override,
            diagnostic_rig_game_profile_override,
            mode=mode,
            rectify_minimap=rectify,
            adapter=adapter,
        ).open()
    return RectifiedHikCamera(
        diagnostic_calibration_override, adapter=adapter, rectify=rectify
    ).open()


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.manage_phone_display and not arguments.gui:
        parser().error("--manage-phone-display requires --gui")
    if (
        arguments.diagnostic_calibration_override is not None
        and arguments.diagnostic_rig_game_profile_override is None
        and arguments.mode != "full"
    ):
        parser().error(
            "Diagnostic --mode {} requires "
            "--diagnostic-rig-game-profile-override".format(arguments.mode)
        )
    if (
        arguments.diagnostic_calibration_override is None
        and arguments.diagnostic_rig_game_profile_override is not None
    ):
        parser().error(
            "--diagnostic-rig-game-profile-override requires "
            "--diagnostic-calibration-override"
        )
    display = None
    camera = None
    windows = []
    try:
        if arguments.diagnostic_calibration_override is None:
            camera = HikCamera(
                config={
                    "profile_root": arguments.profile_root,
                    "game_id": arguments.game_id,
                    "camera_id": arguments.camera_id,
                    "phone_id": arguments.phone_serial,
                    "mode": arguments.mode,
                    "rectify": not arguments.no_rectify,
                    "color_order": arguments.color_order,
                    "color_policy": arguments.color_policy,
                    "minimap_margin_px": arguments.minimap_margin_px,
                    "mvs_python_path": arguments.mvs_python_path,
                }
            )
            configuration = camera.calibration
        else:
            print(
                "Warning: diagnostic calibration override bypasses automatic "
                "profile selection.",
                flush=True,
            )
            configuration = json.loads(
                arguments.diagnostic_calibration_override.read_text(encoding="utf-8")
            )
        if arguments.manage_phone_display:
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
        if camera is None:
            camera = open_camera(
                arguments.diagnostic_calibration_override,
                arguments.mvs_python_path,
                rectify=not arguments.no_rectify,
                diagnostic_rig_game_profile_override=(
                    arguments.diagnostic_rig_game_profile_override
                ),
                mode=arguments.mode,
                color_order=arguments.color_order,
                color_policy=arguments.color_policy,
                minimap_margin_px=arguments.minimap_margin_px,
            )
        elif not camera.is_open:
            camera.open()
        if not arguments.gui:
            print("Rectified stream configured. Use open_camera(...) from Python, or pass --gui.")
            return 0
        profiled = bool(
            arguments.diagnostic_rig_game_profile_override is not None
            or getattr(camera, "config", {}).get("minimap_calibration")
        )
        stream_names = (
            ["full", "minimap"]
            if profiled and arguments.mode == "dual"
            else [arguments.mode if profiled else "full"]
        )
        windows = ["HIK {}".format(name) for name in stream_names]
        print(
            "Live {} stream opened. Press Q/Esc or close a window to exit."
            .format(arguments.mode if profiled else "rectified phone"),
            flush=True,
        )
        for window in windows:
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        while True:
            if profiled and arguments.mode == "dual":
                if hasattr(camera, "get_frames"):
                    displayed = camera.get_frames()
                else:
                    frame_set = camera.read_streams()
                    displayed = frame_set.streams
            else:
                ok, frame = camera.read()
                displayed = {stream_names[0]: frame} if ok and frame is not None else {}
            for name, frame in displayed.items():
                cv2.imshow("HIK {}".format(name), frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                return 0
            if any(
                cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
                for window in windows
            ):
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
