"""Create a deterministic map/query split for a COLMAP image sequence."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-stride", type=int, default=2)
    args = parser.parse_args()

    names = sorted(path.name for path in args.images.glob("*.jpg"))
    if not names:
        raise RuntimeError("No JPG images found")
    map_names = names[:: args.map_stride]
    map_set = set(map_names)
    query_names = [name for name in names if name not in map_set]

    args.output.mkdir(parents=True, exist_ok=True)
    # Keep generated lists byte-stable across Windows and Unix environments.
    with (args.output / "map_images.txt").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(map_names) + "\n")
    with (args.output / "query_images.txt").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(query_names) + "\n")
    summary = {
        "total_images": len(names),
        "map_stride": args.map_stride,
        "map_images": len(map_names),
        "query_images": len(query_names),
    }
    (args.output / "split.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
