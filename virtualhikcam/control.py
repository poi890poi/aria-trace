"""Operator control for persistent virtual-camera displacement state."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import urlencode

from aria_trace.adapters.rig.devices import CameraConfiguration

from .driver import VirtualHikCameraAdapter


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect, set, randomize, or reset the persistent artificial pose "
            "of one virtual HIK camera model. The source camera is opened only "
            "for the operation and is always released afterward."
        )
    )
    parser.add_argument(
        "action", choices=("show", "set", "randomize", "reset", "benchmark")
    )
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--camera-id", help="complete virtual-hik:// camera model URI")
    identity.add_argument(
        "--camera-source-serial",
        help="Android serial; avoids shell-sensitive ampersands in a complete URI",
    )
    parser.add_argument("--camera-source-id", default="1")
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--bit-rate", type=int, default=12000000)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--x-px", type=float, default=0.0)
    parser.add_argument("--y-px", type=float, default=0.0)
    parser.add_argument("--rotation-deg", type=float, default=0.0)
    parser.add_argument("--max-x-px", type=float)
    parser.add_argument("--max-y-px", type=float)
    parser.add_argument("--max-rotation-deg", type=float, default=2.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--benchmark-frames", type=int, default=30)
    return parser


def _percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * float(fraction)
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.camera_id is None:
        arguments.camera_id = "virtual-hik://{}/camera/{}?{}".format(
            arguments.camera_source_serial,
            arguments.camera_source_id,
            urlencode(
                [
                    ("width", arguments.width),
                    ("height", arguments.height),
                    ("fps", arguments.fps),
                    ("zoom", arguments.zoom),
                    ("bit_rate", arguments.bit_rate),
                ]
            ),
        )
    adapter = VirtualHikCameraAdapter(state_root=arguments.state_root)
    configuration = CameraConfiguration(
        arguments.camera_id,
        arguments.width,
        arguments.height,
        arguments.fps,
        "virtual_hik",
    )
    try:
        adapter.open(configuration)
        if arguments.action == "set":
            value = adapter.set_simulated_displacement(
                arguments.x_px, arguments.y_px, arguments.rotation_deg
            )
        elif arguments.action == "randomize":
            value = adapter.randomize_simulated_displacement(
                max_x_px=arguments.max_x_px,
                max_y_px=arguments.max_y_px,
                max_rotation_deg=arguments.max_rotation_deg,
                seed=arguments.seed,
            )
        elif arguments.action == "reset":
            value = adapter.reset_simulated_displacement()
        elif arguments.action == "show":
            value = adapter.simulated_displacement()
        else:
            if arguments.benchmark_frames < 5:
                raise ValueError("Stream benchmark requires at least five frames")
            samples = [adapter.read() for _ in range(arguments.benchmark_frames)]
            measured = samples[3:]
            processing = [sample.metadata["virtual_processing_ms"] for sample in measured]
            remap = [sample.metadata["displacement_remap_ms"] for sample in measured]
            value = adapter.simulated_displacement()
            benchmark = {
                "frame_count": len(measured),
                "effective_roi_xywh": list(adapter.active_roi),
                "virtual_processing_ms_p50": statistics.median(processing),
                "virtual_processing_ms_p95": _percentile(processing, 0.95),
                "displacement_remap_ms_p50": statistics.median(remap),
                "displacement_remap_ms_p95": _percentile(remap, 0.95),
                "map_generation_values": sorted(
                    set(
                        int(sample.metadata["displacement_map_generation"])
                        for sample in measured
                    )
                ),
            }
        print(
            json.dumps(
                {
                    "camera_model_id": adapter.metadata["device_id"],
                    "state_file": adapter.metadata["state_file"],
                    "state_generation": int(adapter._state["state_generation"]),
                    "simulated_displacement": dict(value),
                    "displacement_map_build_ms": adapter._displacement_map_build_ms,
                    "stream_benchmark": (
                        benchmark if arguments.action == "benchmark" else None
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
