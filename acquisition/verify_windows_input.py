"""Run a bounded Raw Input receiver check without starting a gameplay take."""

import argparse
import ctypes
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .models import FramePacket
from .recorder import AcquisitionRecorder
from .session import SessionReader
from .windows import WindowsRawKeyboardMouseSource


class _VerificationFrameSource:
    stream_id = "verification"

    def __init__(self) -> None:
        self.running = False
        self.next_frame_ns = 0

    def start(self) -> None:
        self.running = True
        self.next_frame_ns = time.perf_counter_ns()

    def read(self):
        if not self.running:
            return None
        remaining = self.next_frame_ns - time.perf_counter_ns()
        if remaining > 0:
            time.sleep(remaining / 1.0e9)
        self.next_frame_ns += 33_333_333
        now = time.perf_counter_ns()
        return FramePacket(
            self.stream_id,
            np.zeros((32, 48, 3), dtype=np.uint8),
            now,
            now,
        )

    def stop(self) -> None:
        self.running = False

    def describe(self):
        return {"type": "verification-frame-source", "stream_id": self.stream_id}


def _inject_key(virtual_key: int) -> None:
    """Inject an otherwise-unused key as a best-effort automated smoke event."""
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.keybd_event.argtypes = [
        ctypes.c_ubyte,
        ctypes.c_ubyte,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    user32.keybd_event(int(virtual_key), 0, 0, 0)
    user32.keybd_event(int(virtual_key), 0, 0x0002, 0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the active-lifetime Windows Raw Input path"
    )
    parser.add_argument("--window", required=True, help="Exact visible window title")
    parser.add_argument("--wait", type=float, default=3.0)
    parser.add_argument("--settle", type=float, default=1.5)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--inject-f24",
        action="store_true",
        help="Inject F24 as a best-effort event; some systems expose only hardware events",
    )
    args = parser.parse_args()
    if os.name != "nt":
        raise RuntimeError("This verifier requires Windows")

    output = args.output or (
        Path("artifacts")
        / "workbench"
        / "input_verification"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    source = WindowsRawKeyboardMouseSource(args.window, exact_title=True)
    source.disable_foreground_filter()
    safety_stop = threading.Event()
    safety_timer = threading.Timer(max(0.5, args.wait), safety_stop.set)

    def sources_started() -> None:
        if args.inject_f24:
            # F23 simulates the queue/switch residue that must be discarded;
            # F24 is the first eligible input and must become session time zero.
            threading.Timer(0.1, _inject_key, args=(0x86,)).start()
            threading.Timer(
                max(0.2, args.settle + 0.1), _inject_key, args=(0x87,)
            ).start()
        safety_timer.start()

    manifest = AcquisitionRecorder(
        output,
        [_VerificationFrameSource()],
        [source],
        video_encoding="mjpeg",
        session_context={"purpose": "windows_raw_input_verification"},
    ).run(
        duration_s=0.25,
        external_stop=safety_stop,
        on_sources_started=sources_started,
        start_on_input=True,
        input_start_delay_s=args.settle,
        input_start_predicate=lambda packet: packet.kind == "pc_raw_keyboard",
    )
    safety_timer.cancel()
    reader = SessionReader(output)
    first_input_time = (
        reader.inputs[0]["session_time_ns"] if reader.inputs else None
    )
    first_virtual_key = (
        reader.inputs[0]["payload"].get("virtual_key") if reader.inputs else None
    )
    diagnostics = source.describe()["raw_input_diagnostics"]
    passed = bool(
        (manifest.get("recording_start") or {}).get("started")
        and first_input_time == 0
        and diagnostics["packets_accepted"] > 0
        and diagnostics["packets_rejected_foreground"] == 0
        and (not args.inject_f24 or first_virtual_key == 0x87)
    )

    result = {
        "passed": passed,
        "output": str(output.resolve()),
        "manifest_status": manifest["status"],
        "recording_start": manifest.get("recording_start"),
        "first_input_session_time_ns": first_input_time,
        "first_input_virtual_key": first_virtual_key,
        "input_count": len(reader.inputs),
        "frame_count": len(reader.frames_by_stream.get("verification", [])),
        "diagnostics": diagnostics,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
