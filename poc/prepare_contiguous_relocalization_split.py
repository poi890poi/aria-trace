"""Split a sequence into an earlier map traversal and later query traversal."""

import argparse
import json
from pathlib import Path


def write_lines(path: Path, names) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(names) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-fraction", type=float, default=0.5)
    args = parser.parse_args()
    if not 0.0 < args.map_fraction < 1.0:
        raise ValueError("--map-fraction must be between zero and one")

    names = sorted(path.name for path in args.images.glob("*.jpg"))
    if len(names) < 2:
        raise RuntimeError("At least two JPG images are required")
    split_index = max(1, min(len(names) - 1, int(round(len(names) * args.map_fraction))))
    map_names = names[:split_index]
    query_names = names[split_index:]

    args.output.mkdir(parents=True, exist_ok=True)
    write_lines(args.output / "map_images.txt", map_names)
    write_lines(args.output / "query_images.txt", query_names)
    summary = {
        "split_type": "contiguous",
        "total_images": len(names),
        "map_fraction": args.map_fraction,
        "map_images": len(map_names),
        "query_images": len(query_names),
    }
    (args.output / "split.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
