"""Replaceable compressed-video writers for acquisition sessions."""

import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import cv2


def find_ffmpeg(explicit: Optional[Path] = None) -> Path:
    """Find FFmpeg without making the rest of acquisition depend on its location."""
    if explicit is not None:
        candidate = Path(explicit)
        if candidate.is_file():
            return candidate
        raise RuntimeError("FFmpeg was not found at {}".format(candidate))
    located = shutil.which("ffmpeg")
    if located:
        return Path(located)

    raise RuntimeError(
        "FFmpeg is required for H.264 recording. Install it, pass --ffmpeg, "
        "or explicitly select --video-encoding mjpeg."
    )


class OpenCvMjpegSink:
    extension = ".avi"

    def __init__(self, path: Path, shape: Tuple[int, int], fps: float) -> None:
        self.path = Path(path)
        self.shape = shape
        self.writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*"MJPG"), fps, shape
        )
        if not self.writer.isOpened():
            raise RuntimeError("Cannot open MJPEG writer: {}".format(self.path))

    def write(self, image) -> None:
        self.writer.write(image)

    def close(self) -> None:
        self.writer.release()

    def describe(self) -> dict:
        return {"encoding": "mjpeg", "container": "avi"}


class FfmpegH264Sink:
    extension = ".mkv"

    def __init__(
        self,
        path: Path,
        shape: Tuple[int, int],
        fps: float,
        ffmpeg: Path,
        crf: int,
        preset: str,
    ) -> None:
        self.path = Path(path)
        self.shape = shape
        width, height = shape
        if width % 2 or height % 2:
            raise RuntimeError(
                "H.264 yuv420p requires even frame dimensions, got {}x{}".format(width, height)
            )
        self.log_path = self.path.with_suffix(".ffmpeg.log")
        self._log = self.log_path.open("wb")
        command = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            "{}x{}".format(width, height),
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-g",
            str(max(1, int(round(fps * 2.0)))),
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(self.path),
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._log,
            creationflags=creationflags,
        )
        self.ffmpeg = Path(ffmpeg)
        self.crf = crf
        self.preset = preset

    def write(self, image) -> None:
        if image.shape[1::-1] != self.shape or image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError("Unexpected image shape for {}".format(self.path.name))
        if self.process.poll() is not None:
            raise RuntimeError(
                "FFmpeg exited early with code {}; see {}".format(
                    self.process.returncode, self.log_path
                )
            )
        try:
            self.process.stdin.write(image.tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("FFmpeg write failed; see {}: {}".format(self.log_path, exc))

    def close(self) -> None:
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            return_code = self.process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            return_code = self.process.returncode
        self._log.close()
        if return_code != 0:
            raise RuntimeError(
                "FFmpeg failed with code {}; see {}".format(return_code, self.log_path)
            )

    def describe(self) -> dict:
        return {
            "encoding": "h264",
            "container": "matroska",
            "encoder": "libx264",
            "crf": self.crf,
            "preset": self.preset,
            "ffmpeg": str(self.ffmpeg),
        }


def create_video_sink(
    path_without_suffix: Path,
    shape: Tuple[int, int],
    encoding: str,
    fps: float,
    ffmpeg: Optional[Path],
    crf: int,
    preset: str,
):
    if encoding == "mjpeg":
        path = path_without_suffix.with_suffix(OpenCvMjpegSink.extension)
        return OpenCvMjpegSink(path, shape, fps)
    if encoding == "h264":
        path = path_without_suffix.with_suffix(FfmpegH264Sink.extension)
        return FfmpegH264Sink(path, shape, fps, find_ffmpeg(ffmpeg), crf, preset)
    raise ValueError("Unsupported video encoding: {}".format(encoding))
