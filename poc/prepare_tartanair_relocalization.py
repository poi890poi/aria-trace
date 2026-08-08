"""Create COLMAP image lists for two TartanAir trajectories."""

import argparse
import json
from pathlib import Path


def relative_names(root: Path, trajectory: str):
    image_dir = root / trajectory / "image_lcam_front"
    return [path.relative_to(root).as_posix() for path in sorted(image_dir.glob("*.png"))]


def write_lines(path: Path, names) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(names) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--map-trajectory", default="P000")
    parser.add_argument("--query-trajectory", default="P005")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    map_names = relative_names(args.data_root, args.map_trajectory)
    query_names = relative_names(args.data_root, args.query_trajectory)
    if not map_names or not query_names:
        raise RuntimeError("Both trajectories must contain front-camera PNG images")
    args.output.mkdir(parents=True, exist_ok=True)
    write_lines(args.output / "map_images.txt", map_names)
    write_lines(args.output / "query_images.txt", query_names)
    write_lines(args.output / "all_images.txt", map_names + query_names)
    summary = {
        "dataset": "TartanAir-V2/ArchVizTinyHouseDay/Data_easy",
        "map_trajectory": args.map_trajectory,
        "query_trajectory": args.query_trajectory,
        "map_images": len(map_names),
        "query_images": len(query_names),
    }
    (args.output / "split.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
