"""Command-line entry point for recording acquisition sessions."""

import argparse
from datetime import datetime
from pathlib import Path
import shutil

from aria_trace.evidence.features import OnlineSiftRecorder
from aria_trace.workflows.recording import AcquisitionRecorder
from aria_trace.adapters.sources import (
    AdbClockMapper,
    AdbGetEventSource,
    AdbScreenshotFrameSource,
    OpenCvCameraFrameSource,
    SyntheticInputSource,
    VideoFileFrameSource,
)
from aria_trace.adapters.android.capture import (
    AndroidRoiFrameSource,
    ScrcpyCaptureHub,
    find_scrcpy_server,
    parse_android_roi,
    push_session_archive_to_device,
)
from aria_trace.adapters.android.phone import probe_android_capture_surface
from aria_trace.adapters.windows import (
    WindowsKeyboardMouseSource,
    WindowsRawKeyboardMouseSource,
    WindowsWindowFrameSource,
)


def parse_assignment(value: str, default_id: str):
    if "=" in value:
        stream_id, assigned = value.split("=", 1)
        return stream_id, assigned
    return default_id, value


def default_adb():
    """Resolve ADB from PATH; callers may still supply --adb explicitly."""
    located = shutil.which("adb")
    return Path(located) if located else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Record synchronized frames and raw inputs")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video", action="append", default=[], metavar="[ID=]PATH")
    parser.add_argument("--fast-video", action="store_true", help="Do not pace video-file sources")
    parser.add_argument("--camera", action="append", default=[], metavar="[ID=]INDEX")
    parser.add_argument("--window", help="Visible Windows game-window title or unique substring")
    parser.add_argument("--window-fps", type=float, default=30.0)
    parser.add_argument("--exact-window-title", action="store_true")
    pc_input = parser.add_mutually_exclusive_group()
    pc_input.add_argument(
        "--pc-raw-input",
        action="store_true",
        help="Record raw keyboard transitions and relative mouse input for the selected window",
    )
    pc_input.add_argument(
        "--pc-input",
        action="store_true",
        help="Record legacy keyboard/cursor state (no raw relative mouse motion)",
    )
    parser.add_argument("--pc-input-rate", type=float, default=125.0)
    parser.add_argument("--camera-width", type=int)
    parser.add_argument("--camera-height", type=int)
    parser.add_argument("--camera-fps", type=float)
    parser.add_argument(
        "--adb-screenshot",
        action="store_true",
        help=(
            "capture the full Android raster through ADB; this stream uses "
            "OpenCV MJPEG and does not require external FFmpeg"
        ),
    )
    parser.add_argument("--adb", type=Path, default=default_adb())
    parser.add_argument("--serial")
    parser.add_argument("--adb-fps", type=float, default=2.0)
    parser.add_argument("--getevent", action="store_true")
    parser.add_argument(
        "--android-roi",
        action="append",
        default=[],
        metavar="ID=X,Y,W,H[,OUT_W,OUT_H[,CRF]]",
        help="Continuous Android display stream; repeat for synchronized ROIs (0 width/height means to edge)",
    )
    parser.add_argument("--scrcpy-server", type=Path)
    parser.add_argument("--android-bit-rate", type=int, default=16_000_000)
    parser.add_argument("--android-max-fps", type=float, default=60.0)
    parser.add_argument(
        "--no-android-input",
        action="store_true",
        help="Disable raw getevent capture that is enabled with --android-roi",
    )
    parser.add_argument(
        "--android-save-phone",
        nargs="?",
        const="/sdcard/AriaTrace",
        metavar="DIR",
        help="After capture, save the synchronized session ZIP on the phone and print its path",
    )
    parser.add_argument("--synthetic-input", action="store_true")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--queue-size", type=int, default=4096)
    parser.add_argument(
        "--video-encoding", choices=("h264", "mjpeg"), default="h264",
        help=(
            "default encoding for ordinary streams; ADB screenshot sources "
            "declare MJPEG so they do not require external FFmpeg"
        ),
    )
    parser.add_argument("--video-fps", type=float, default=30.0, help="Container playback rate")
    parser.add_argument("--video-crf", type=int, default=20, help="H.264 quality (lower is larger)")
    parser.add_argument("--video-preset", default="veryfast", help="libx264 speed preset")
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        help="FFmpeg executable for H.264 streams; unused by ADB screenshots",
    )
    parser.add_argument(
        "--online-features", choices=("none", "sift"), default="none",
        help="Persist features extracted from raw frames before video encoding",
    )
    parser.add_argument("--feature-rate", type=float, default=1.0)
    parser.add_argument("--feature-max-count", type=int, default=4096)
    parser.add_argument(
        "--feature-stream", action="append", default=[],
        help="Limit online extraction to a stream ID; repeat for multiple streams",
    )
    parser.add_argument(
        "--feature-lossless-frames", action="store_true",
        help="Also retain PNG source frames for each sampled feature observation",
    )
    parser.add_argument("--portal-id", help="Known teleport portal for this session")
    parser.add_argument("--route-id", help="Planned route identifier for this session")
    args = parser.parse_args()
    if (args.adb_screenshot or args.getevent or args.android_roi) and args.adb is None:
        parser.error("ADB is required; install it on PATH or pass --adb PATH")

    android_surface = None
    if args.adb_screenshot or args.android_roi:
        try:
            args.serial, android_surface = probe_android_capture_surface(
                str(args.adb), args.serial
            )
        except RuntimeError as exc:
            parser.error(str(exc))

    frame_sources = []
    for index, specification in enumerate(args.video):
        stream_id, path = parse_assignment(specification, "video{}".format(index))
        frame_sources.append(
            VideoFileFrameSource(Path(path), stream_id, realtime=not args.fast_video)
        )
    if args.window:
        frame_sources.append(
            WindowsWindowFrameSource(
                args.window,
                stream_id="main",
                fps=args.window_fps,
                exact_title=args.exact_window_title,
            )
        )
    for index, specification in enumerate(args.camera):
        stream_id, device = parse_assignment(specification, "uvc{}".format(index))
        frame_sources.append(
            OpenCvCameraFrameSource(
                int(device), stream_id, args.camera_width, args.camera_height, args.camera_fps
            )
        )
    if args.adb_screenshot:
        frame_sources.append(
            AdbScreenshotFrameSource(
                args.adb,
                args.serial,
                "adb",
                args.adb_fps,
                image_space_context=android_surface,
            )
        )
    android_clock = None
    android_hub = None
    android_specs = []
    if args.android_roi:
        try:
            android_specs = [parse_android_roi(value) for value in args.android_roi]
            if len({spec.stream_id for spec in android_specs}) != len(android_specs):
                raise ValueError("Android ROI stream IDs must be unique")
            server = find_scrcpy_server(args.scrcpy_server)
        except (ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        android_clock = AdbClockMapper(args.adb, args.serial)
        android_hub = ScrcpyCaptureHub(
            args.adb,
            server,
            serial=args.serial,
            ffmpeg=args.ffmpeg,
            clock=android_clock,
            bit_rate=args.android_bit_rate,
            max_fps=args.android_max_fps,
        )
        frame_sources.extend(
            AndroidRoiFrameSource(
                android_hub, spec, image_space_context=android_surface
            )
            for spec in android_specs
        )
    if not frame_sources:
        parser.error("Specify at least one --window, --video, --camera, or --adb-screenshot source")

    input_sources = []
    if args.pc_raw_input:
        if not args.window:
            parser.error("--pc-raw-input requires --window")
        input_sources.append(
            WindowsRawKeyboardMouseSource(
                args.window,
                exact_title=args.exact_window_title,
            )
        )
    if args.pc_input:
        if not args.window:
            parser.error("--pc-input requires --window")
        input_sources.append(
            WindowsKeyboardMouseSource(
                args.window,
                poll_hz=args.pc_input_rate,
                exact_title=args.exact_window_title,
            )
        )
    if args.getevent or (args.android_roi and not args.no_android_input):
        input_sources.append(
            AdbGetEventSource(args.adb, args.serial, clock=android_clock)
        )
    if args.synthetic_input:
        input_sources.append(SyntheticInputSource())

    output = args.output
    if output is None:
        output = Path("sessions") / datetime.now().strftime("%Y%m%d_%H%M%S")
    frame_processors = []
    if args.online_features == "sift":
        frame_processors.append(
            OnlineSiftRecorder(
                rate_hz=args.feature_rate,
                max_features=args.feature_max_count,
                streams=args.feature_stream or None,
                save_lossless_frames=args.feature_lossless_frames,
            )
        )
    recorder = AcquisitionRecorder(
        output,
        frame_sources,
        input_sources,
        queue_size=args.queue_size,
        video_encoding=args.video_encoding,
        video_fps=args.video_fps,
        video_crf=args.video_crf,
        video_preset=args.video_preset,
        ffmpeg=args.ffmpeg,
        frame_processors=frame_processors,
        session_context={"portal_id": args.portal_id, "route_id": args.route_id},
        video_stream_options={
            spec.stream_id: {"crf": spec.crf}
            for spec in android_specs
            if spec.crf is not None
        },
    )
    manifest = recorder.run(args.duration)
    print("Session: {}".format(output.resolve()))
    print("Status: {}".format(manifest["status"]))
    print("Frames: {}".format(manifest.get("frame_counts", {})))
    print("Inputs: {}".format(manifest.get("input_counts", {})))
    if args.android_save_phone:
        phone_path = push_session_archive_to_device(
            args.adb, args.serial, output, args.android_save_phone
        )
        print("Phone session: {}".format(phone_path))


if __name__ == "__main__":
    main()
