"""Command-line entry point for recording acquisition sessions."""

import argparse
from datetime import datetime
from pathlib import Path
import shutil

from .recorder import AcquisitionRecorder
from .features import OnlineSiftRecorder
from .sources import (
    AdbGetEventSource,
    AdbScreenshotFrameSource,
    OpenCvCameraFrameSource,
    SyntheticInputSource,
    VideoFileFrameSource,
)


def parse_assignment(value: str, default_id: str):
    if "=" in value:
        stream_id, assigned = value.split("=", 1)
        return stream_id, assigned
    return default_id, value


def default_adb() -> Path:
    located = shutil.which("adb")
    if located:
        return Path(located)
    return Path(r"E:\Android\Sdk\platform-tools\adb.exe")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record synchronized frames and raw inputs")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--video", action="append", default=[], metavar="[ID=]PATH")
    parser.add_argument("--fast-video", action="store_true", help="Do not pace video-file sources")
    parser.add_argument("--camera", action="append", default=[], metavar="[ID=]INDEX")
    parser.add_argument("--camera-width", type=int)
    parser.add_argument("--camera-height", type=int)
    parser.add_argument("--camera-fps", type=float)
    parser.add_argument("--adb-screenshot", action="store_true")
    parser.add_argument("--adb", type=Path, default=default_adb())
    parser.add_argument("--serial")
    parser.add_argument("--adb-fps", type=float, default=2.0)
    parser.add_argument("--getevent", action="store_true")
    parser.add_argument("--synthetic-input", action="store_true")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--queue-size", type=int, default=256)
    parser.add_argument(
        "--video-encoding", choices=("h264", "mjpeg"), default="h264",
        help="H.264 is compact; MJPEG is a large compatibility fallback",
    )
    parser.add_argument("--video-fps", type=float, default=30.0, help="Container playback rate")
    parser.add_argument("--video-crf", type=int, default=20, help="H.264 quality (lower is larger)")
    parser.add_argument("--video-preset", default="veryfast", help="libx264 speed preset")
    parser.add_argument("--ffmpeg", type=Path, help="Path to ffmpeg executable")
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

    frame_sources = []
    for index, specification in enumerate(args.video):
        stream_id, path = parse_assignment(specification, "video{}".format(index))
        frame_sources.append(
            VideoFileFrameSource(Path(path), stream_id, realtime=not args.fast_video)
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
            AdbScreenshotFrameSource(args.adb, args.serial, "adb", args.adb_fps)
        )
    if not frame_sources:
        parser.error("Specify at least one --video, --camera, or --adb-screenshot source")

    input_sources = []
    if args.getevent:
        input_sources.append(AdbGetEventSource(args.adb, args.serial))
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
    )
    manifest = recorder.run(args.duration)
    print("Session: {}".format(output.resolve()))
    print("Status: {}".format(manifest["status"]))
    print("Frames: {}".format(manifest.get("frame_counts", {})))
    print("Inputs: {}".format(manifest.get("input_counts", {})))


if __name__ == "__main__":
    main()
