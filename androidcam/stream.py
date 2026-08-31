"""Standalone GUI for an Android front camera streamed through scrcpy."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import cv2

from .driver import AndroidCamera, find_adb


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Stream an Android camera over ADB")
    value.add_argument("--serial", help="ADB serial of the camera phone")
    value.add_argument("--target-serial", help="optional facing display device")
    value.add_argument("--camera-id")
    value.add_argument(
        "--camera-facing", choices=("front", "back", "external"), default="front"
    )
    value.add_argument("--width", type=int, default=1280)
    value.add_argument("--height", type=int, default=720)
    value.add_argument("--fps", type=int, default=30)
    value.add_argument("--bit-rate", type=int, default=12_000_000)
    value.add_argument("--adb", type=Path)
    value.add_argument("--scrcpy-server", type=Path)
    value.add_argument("--ffmpeg", type=Path)
    value.add_argument("--title", default="Android front camera")
    value.add_argument("--no-overlay", action="store_true")
    value.add_argument("--headless", action="store_true")
    value.add_argument("--frames", type=int, default=0)
    value.add_argument(
        "--leave-displays-on",
        action="store_true",
        help="do not send KEYCODE_SLEEP to source and target on exit",
    )
    return value


def _opencv_gui_backend() -> Optional[str]:
    for line in cv2.getBuildInformation().splitlines():
        if line.strip().startswith("GUI:"):
            backend = line.split(":", 1)[1].strip()
            return backend if backend.upper() != "NONE" else None
    return None


def _overlay(frame, text):
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        output,
        text,
        (10, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 235, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def _sleep_display(adb: Path, serial: Optional[str]) -> None:
    if not serial:
        return
    try:
        result = subprocess.run(
            [str(adb), "-s", str(serial), "shell", "input", "keyevent", "223"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        if result.returncode == 0:
            print("Turned off Android display {}.".format(serial), flush=True)
        else:
            print(
                "Warning: Android display {} did not accept KEYCODE_SLEEP."
                .format(serial),
                file=sys.stderr,
                flush=True,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(
            "Warning: could not turn off Android display {}: {}".format(serial, exc),
            file=sys.stderr,
            flush=True,
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.headless and arguments.frames <= 0:
        parser().error("--headless requires --frames greater than zero")
    if arguments.frames < 0:
        parser().error("--frames cannot be negative")
    if not arguments.headless and _opencv_gui_backend() is None:
        print(
            "Android camera GUI is unavailable: this Python loaded headless OpenCV.",
            file=sys.stderr,
            flush=True,
        )
        return 2

    adb = find_adb(arguments.adb)
    camera = AndroidCamera(
        arguments.serial,
        camera_id=arguments.camera_id,
        camera_facing=arguments.camera_facing,
        width_px=arguments.width,
        height_px=arguments.height,
        fps=arguments.fps,
        bit_rate=arguments.bit_rate,
        adb=adb,
        scrcpy_server=arguments.scrcpy_server,
        ffmpeg=arguments.ffmpeg,
    )
    frame_count = 0
    fps_estimate = 0.0
    previous_time = None
    source_serial = arguments.serial
    try:
        camera.open()
        source_serial = camera.serial
        print(json.dumps(camera.effective_configuration, indent=2), flush=True)
        if not arguments.headless:
            cv2.namedWindow(arguments.title, cv2.WINDOW_NORMAL)
        while True:
            sample = camera.read_sample()
            frame_count += 1
            now = time.perf_counter()
            if previous_time is not None and now > previous_time:
                instantaneous = 1.0 / (now - previous_time)
                fps_estimate = (
                    instantaneous
                    if fps_estimate <= 0.0
                    else 0.9 * fps_estimate + 0.1 * instantaneous
                )
            previous_time = now
            if not arguments.headless:
                image = sample.image
                if not arguments.no_overlay:
                    image = _overlay(
                        image,
                        "{}x{}  {:.1f} fps  transport {:.1f} ms  dropped {}".format(
                            image.shape[1],
                            image.shape[0],
                            fps_estimate,
                            sample.metadata["transport_latency_ms"],
                            sample.metadata["dropped_before"],
                        ),
                    )
                cv2.imshow(arguments.title, image)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    break
                if cv2.getWindowProperty(arguments.title, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if arguments.frames and frame_count >= arguments.frames:
                break
        return 0
    except (RuntimeError, OSError) as exc:
        print("Android camera stream failed: {}".format(exc), file=sys.stderr, flush=True)
        return 1
    except cv2.error as exc:
        print("Android camera GUI failed: {}".format(exc), file=sys.stderr, flush=True)
        return 2
    finally:
        camera.close()
        if not arguments.headless:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        if not arguments.leave_displays_on:
            _sleep_display(adb, source_serial)
            if arguments.target_serial != source_serial:
                _sleep_display(adb, arguments.target_serial)


if __name__ == "__main__":
    raise SystemExit(main())
