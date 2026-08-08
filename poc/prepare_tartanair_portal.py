"""Create repeated-arrival splits around a synthetic TartanAir portal."""

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .evaluate_relocalization import quaternion_to_matrix
except ImportError:
    from evaluate_relocalization import quaternion_to_matrix


def contiguous_episodes(indices):
    episodes = []
    for index in indices:
        if not episodes or index != episodes[-1][-1] + 1:
            episodes.append([index])
        else:
            episodes[-1].append(index)
    return episodes


def select(root: Path, trajectory: str, center, radius: float):
    poses = np.loadtxt(str(root / trajectory / "pose_lcam_front.txt"))
    distances = np.linalg.norm(poses[:, :3] - center, axis=1)
    indices = np.where(distances <= radius)[0].tolist()
    names = [
        "{}/image_lcam_front/{:06d}_lcam_front.png".format(trajectory, index)
        for index in indices
    ]
    headings = []
    for index in indices:
        qx, qy, qz, qw = poses[index, 3:]
        forward = quaternion_to_matrix(qw, qx, qy, qz)[:, 0]
        headings.append(float(np.degrees(np.arctan2(forward[2], forward[0]))))
    return names, indices, distances, headings


def write_lines(path: Path, values) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-trajectories", nargs="+", default=["P000", "P002"])
    parser.add_argument("--query-trajectories", nargs="+", default=["P003", "P004", "P005", "P006"])
    parser.add_argument("--center", nargs=3, type=float, default=[0.8, 0.46, -0.85])
    parser.add_argument("--radius", type=float, default=0.75)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Refusing to overwrite nonempty output: {}".format(args.output))
    args.output.mkdir(parents=True, exist_ok=True)
    center = np.asarray(args.center, dtype=float)

    split = {
        "dataset": "TartanAir-V2/ArchVizTinyHouseDay/Data_easy",
        "interpretation": "synthetic repeated portal arrivals",
        "portal_center_m": center.tolist(),
        "portal_radius_m": args.radius,
        "map_trajectories": args.map_trajectories,
        "query_trajectories": args.query_trajectories,
        "trajectories": {},
    }
    map_names = []
    query_names = []
    for role, trajectories in (("map", args.map_trajectories), ("query", args.query_trajectories)):
        for trajectory in trajectories:
            names, indices, distances, headings = select(
                args.data_root, trajectory, center, args.radius
            )
            if not names:
                raise RuntimeError("{} has no frames inside the portal".format(trajectory))
            if role == "map":
                map_names.extend(names)
            else:
                query_names.extend(names)
            split["trajectories"][trajectory] = {
                "role": role,
                "frames": len(names),
                "indices": indices,
                "episodes": contiguous_episodes(indices),
                "nearest_center_distance_m": float(np.min(distances)),
                "heading_deg_min": min(headings),
                "heading_deg_max": max(headings),
            }
    write_lines(args.output / "map_images.txt", map_names)
    write_lines(args.output / "query_images.txt", query_names)
    write_lines(args.output / "all_images.txt", map_names + query_names)
    split["map_images"] = len(map_names)
    split["query_images"] = len(query_names)
    (args.output / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(json.dumps(split, indent=2))


if __name__ == "__main__":
    main()
