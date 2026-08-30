"""Reusable time- and space-synchronized calibrated-rig capture bundle."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Mapping, Optional

from aria_trace.adapters.android.capture import (
    AndroidRoiFrameSource,
    AndroidRoiSpec,
    ScrcpyCaptureHub,
    find_scrcpy_server,
)
from aria_trace.services.calibration.rig.dual_source_spaces import write_dual_source_space_yaml
from aria_trace.services.calibration.rig.cross_source import (
    GameCrossSourceEvidenceRecorder,
    orient_hik_source_from_first_adb_frame,
)
from aria_trace.adapters.hik.capture import CalibratedHikFrameSource
from aria_trace.services.calibration.rig.hik.phone import AdbPhoneSession
from aria_trace.adapters.sources import AdbClockMapper, AdbGetEventSource


@dataclass
class RecordingSourceBundle:
    """Everything one recorder lifetime needs from a capture adapter."""

    frame_sources: List[object]
    input_sources: List[object] = field(default_factory=list)
    frame_processors: List[object] = field(default_factory=list)
    primary_stream_id: str = "main"
    required_stream_ids: List[str] = field(default_factory=lambda: ["main"])
    session_context: dict = field(default_factory=dict)
    _finalizer: Optional[Callable[[Path, Mapping[str, object]], None]] = None

    def finalize(self, session_path: Path, manifest: Mapping[str, object]) -> None:
        if self._finalizer is not None:
            self._finalizer(Path(session_path), manifest)


def single_source_recording_bundle(frame_source, input_source=None) -> RecordingSourceBundle:
    """Adapt a legacy single frame source to the recorder bundle contract."""

    stream_id = str(frame_source.stream_id)
    return RecordingSourceBundle(
        frame_sources=[frame_source],
        input_sources=[input_source] if input_source is not None else [],
        primary_stream_id=stream_id,
        required_stream_ids=[stream_id],
    )


def _resolve_calibration_file(value: Path) -> Path:
    path = Path(value)
    if path.is_dir():
        path = path / "hik_camera_calibration.json"
    if not path.is_file():
        raise RuntimeError("Rig calibration does not exist: {}".format(path))
    return path


def _phone_surface(adb: Path, serial: str) -> dict:
    metrics = AdbPhoneSession(serial, adb_executable=str(adb)).metrics()
    quarter_turns = int(metrics.orientation_quarter_turns)
    return {
        "quarter_turns_clockwise_from_natural": quarter_turns,
        "degrees_clockwise_from_natural": quarter_turns * 90,
        "logical_size_px": list(map(int, metrics.screen_size_px)),
        "natural_size_px": list(map(int, metrics.natural_screen_size_px)),
        "source": "adb_surface_orientation_at_capture",
    }


def _write_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(manifest), indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


def build_calibrated_rig_recording_bundle(
    calibration_file: Path,
    *,
    adb: Path,
    scrcpy_server: Optional[Path] = None,
    ffmpeg: Optional[Path] = None,
    input_adapter: str = "none",
    input_source_id: str = "android-input",
    bit_rate: int = 16_000_000,
    max_fps: float = 60.0,
) -> RecordingSourceBundle:
    """Build the established ADB + rig-normalized HIK acquisition contract.

    Android timestamps are mapped to the host monotonic clock. HIK receive
    timestamps already use that host clock, while the raw device counter stays
    in frame metadata. The saved rig geometry and first cross-source image pair
    establish the session's spatial mapping and discrete output orientation.
    """

    calibration_file = _resolve_calibration_file(calibration_file)
    calibration = json.loads(calibration_file.read_text(encoding="utf-8"))
    phone_serial = str((calibration.get("phone") or {}).get("serial") or "").strip()
    camera_id = str((calibration.get("camera") or {}).get("device_id") or "").strip()
    if not phone_serial:
        raise RuntimeError("Rig calibration does not identify its Android phone")
    if not camera_id:
        raise RuntimeError("Rig calibration does not identify its HIK camera")
    if input_adapter not in ("adb_getevent", "none"):
        raise ValueError(
            "Calibrated rig capture supports Android getevent or no input capture"
        )

    adb = Path(adb)
    clock = AdbClockMapper(adb, phone_serial)
    server = find_scrcpy_server(scrcpy_server)
    hub = ScrcpyCaptureHub(
        adb,
        server,
        serial=phone_serial,
        ffmpeg=ffmpeg,
        clock=clock,
        bit_rate=int(bit_rate),
        max_fps=float(max_fps),
    )
    android = AndroidRoiFrameSource(
        hub,
        AndroidRoiSpec("android_phone", 0, 0, 0, 0),
    )
    hik = CalibratedHikFrameSource(
        calibration_file,
        "hik_phone",
        rectify=True,
        output_quarter_turns_clockwise=0,
    )
    surface = _phone_surface(adb, phone_serial)
    try:
        orientation_match, orientation_images = orient_hik_source_from_first_adb_frame(
            android,
            hik,
            calibration_file,
            android_reported_quarter_turns=surface[
                "quarter_turns_clockwise_from_natural"
            ],
        )
        selected_surface_turns = int(
            orientation_match[
                "selected_adb_surface_quarter_turns_clockwise_from_phone_natural"
            ]
        )
        aligned_surface = {
            **surface,
            "quarter_turns_clockwise_from_natural": selected_surface_turns,
            "degrees_clockwise_from_natural": selected_surface_turns * 90,
            "source": "first_game_adb_and_hik_image_evidence",
            "android_reported_quarter_turns_clockwise_from_natural": surface[
                "quarter_turns_clockwise_from_natural"
            ],
            "orientation_evidence": "cross_source_check/orientation_match/summary.json",
        }
        processor = GameCrossSourceEvidenceRecorder(
            calibration_file,
            aligned_surface,
            orientation_match=orientation_match,
            orientation_evidence_images=orientation_images,
        )
        input_source = (
            AdbGetEventSource(
                adb,
                serial=phone_serial,
                source_id=input_source_id,
                clock=clock,
            )
            if input_adapter == "adb_getevent"
            else None
        )
    except Exception:
        hik.stop()
        android.stop()
        raise

    def finalize(session_path: Path, manifest: Mapping[str, object]) -> None:
        counts = manifest.get("frame_counts") or {}
        missing = [
            stream_id
            for stream_id in ("android_phone", "hik_phone")
            if int(counts.get(stream_id, 0)) <= 0
        ]
        if missing:
            raise RuntimeError(
                "Calibrated rig recording is missing required frame streams: {}".format(
                    ", ".join(missing)
                )
            )
        write_dual_source_space_yaml(
            session_path,
            calibration_file,
            aligned_surface,
            manifest,
        )
        mutable_manifest = dict(manifest)
        mutable_manifest["coordinate_spaces"] = "coordinate_spaces.yaml"
        if isinstance(manifest, dict):
            manifest["coordinate_spaces"] = "coordinate_spaces.yaml"
        _write_manifest(session_path / "manifest.json", mutable_manifest)

    return RecordingSourceBundle(
        frame_sources=[android, hik],
        input_sources=[input_source] if input_source is not None else [],
        frame_processors=[processor],
        primary_stream_id="hik_phone",
        required_stream_ids=["android_phone", "hik_phone"],
        session_context={
            "capture_source_kind": "calibrated_rig_dual",
            "primary_stream_id": "hik_phone",
            "reference_stream_id": "android_phone",
            "image_sources": ["android_scrcpy", "hik_mvs_rig_rectified"],
            "rig_capture": {
                "calibration": str(calibration_file.resolve()),
                "camera_id": camera_id,
                "phone_serial": phone_serial,
                "timestamp_domain": "host_perf_counter_ns",
                "coordinate_spaces": "coordinate_spaces.yaml",
            },
            "phone_surface_orientation": aligned_surface,
        },
        _finalizer=finalize,
    )
