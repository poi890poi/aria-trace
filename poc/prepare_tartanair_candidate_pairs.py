"""Prepare an oracle-retrieval pair list for a feature-matcher upper bound."""

import argparse
import json
from pathlib import Path

import numpy as np


def load_positions(data_root: Path, names):
    cache = {}
    positions = []
    for name in names:
        trajectory = Path(name).parts[0]
        if trajectory not in cache:
            cache[trajectory] = np.loadtxt(str(data_root / trajectory / "pose_lcam_front.txt"))
        frame_index = int(Path(name).stem.split("_")[0])
        positions.append(cache[trajectory][frame_index, :3])
    return np.stack(positions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--map-list", type=Path, required=True)
    parser.add_argument("--query-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-neighbors", type=int, default=10)
    args = parser.parse_args()

    map_names = args.map_list.read_text(encoding="utf-8").splitlines()
    query_names = args.query_list.read_text(encoding="utf-8").splitlines()
    map_positions = load_positions(args.data_root, map_names)
    query_positions = load_positions(args.data_root, query_names)
    pairs = set()

    # Enough temporal baselines to triangulate the known-pose map.
    for gap in (1, 2, 4, 8):
        for index in range(len(map_names) - gap):
            pairs.add((map_names[index], map_names[index + gap]))

    # Oracle position retrieval isolates local matching from image retrieval.
    neighbor_count = min(args.query_neighbors, len(map_names))
    for query_name, query_position in zip(query_names, query_positions):
        distances = np.linalg.norm(map_positions - query_position, axis=1)
        for index in np.argsort(distances)[:neighbor_count]:
            pairs.add((map_names[int(index)], query_name))

    ordered = sorted(pairs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join("{} {}".format(first, second) for first, second in ordered) + "\n")
    summary = {
        "map_images": len(map_names),
        "query_images": len(query_names),
        "query_neighbors": neighbor_count,
        "candidate_pairs": len(ordered),
        "uses_ground_truth_retrieval": True,
    }
    (args.output.with_suffix(".json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
