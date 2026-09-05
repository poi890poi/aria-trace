"""Build a replay-only sparse reference from an atlas transition analysis.

This artifact is a calibration-derived proxy, not independent ground truth.  It
may initialize a replay once, but future samples are evaluation-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def build(atlas_path: Path, session_path: Path, output_path: Path) -> dict:
    atlas_path = Path(atlas_path).resolve()
    session_path = Path(session_path).resolve()
    output_path = Path(output_path).resolve()
    atlas = json.loads((atlas_path / "map_atlas.json").read_text(encoding="utf-8"))
    session = json.loads((session_path / "manifest.json").read_text(encoding="utf-8"))
    model = atlas.get("transition_model") or {}
    samples = list(((model.get("analysis") or {}).get("samples") or ()))
    if len(samples) < 2:
        raise ValueError("Atlas transition analysis needs at least two samples")
    states = []
    distances = [0.0]
    for index, sample in enumerate(samples):
        likelihoods = sample.get("likelihoods") or {}
        if not likelihoods:
            raise ValueError("Transition sample has no mode likelihoods")
        mode_id = max(likelihoods, key=lambda key: float(likelihoods[key]))
        xy = [float(value) for value in sample["canonical_xy"]]
        if states:
            distances.append(
                distances[-1]
                + float(
                    np.linalg.norm(
                        np.asarray(xy) - np.asarray(states[-1]["canonical_xy"])
                    )
                )
            )
        states.append(
            {
                "state_index": index,
                "frame_index": int(sample["frame_index"]),
                "session_time_ns": int(sample["session_time_ns"]),
                "canonical_xy": xy,
                "mode_id": str(mode_id),
                "route_distance_px": distances[-1],
            }
        )
    intervals = np.diff([row["session_time_ns"] for row in states]).astype(np.float64)
    reference_rate_hz = 1.0e9 / float(np.median(intervals))
    output_path.mkdir(parents=True, exist_ok=True)
    state_file = "reference_states.jsonl"
    transition_file = "transitions.jsonl"
    descriptor_file = "unused_descriptors.npz"
    _write_jsonl(output_path / state_file, states)
    _write_jsonl(output_path / transition_file, [])
    np.savez_compressed(
        output_path / descriptor_file,
        descriptors=np.zeros((len(states), 1), dtype=np.float32),
    )
    manifest = {
        "schema_version": "1.0",
        "package_type": "benchmark_sparse_reference",
        "reference_role": "offline-transition-calibration-proxy",
        "independent_ground_truth": False,
        "future_samples_feed_tracker": False,
        "known_start_may_feed_tracker_once": True,
        "route_id": "transition-proxy:{}".format(session["session_id"]),
        "atlas_id": atlas.get("atlas_id"),
        "coordinate_space_id": atlas.get("coordinate_space_id"),
        "source_session": {
            "session_id": session["session_id"],
            "path": str(session_path),
        },
        "reference_rate_hz": reference_rate_hz,
        "corridor_radius_px": 35.0,
        "state_count": len(states),
        "files": {
            "states": state_file,
            "transitions": transition_file,
            "descriptors": descriptor_file,
        },
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atlas", type=Path)
    parser.add_argument("session", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.atlas, args.session, args.output), indent=2))


if __name__ == "__main__":
    main()
