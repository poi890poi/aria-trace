"""Evaluate multi-frame initialization around a known portal prior."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from .evaluate_relocalization import (
        estimate_similarity,
        load_colmap_images,
        rotation_error_deg,
    )
    from .evaluate_tartanair_relocalization import load_ground_truth
except ImportError:
    from evaluate_relocalization import estimate_similarity, load_colmap_images, rotation_error_deg
    from evaluate_tartanair_relocalization import load_ground_truth


def frame_name(trajectory: str, index: int) -> str:
    return "{}/image_lcam_front/{:06d}_lcam_front.png".format(trajectory, index)


def first_consistent_window(
    indices,
    estimated,
    portal_center,
    portal_limit_m: float,
    window_size: int = 3,
    max_step_m: float = 0.5,
    max_rotation_step_deg: float = 15.0,
):
    if len(indices) < window_size:
        return None
    for offset in range(len(indices) - window_size + 1):
        window = indices[offset : offset + window_size]
        if any(index not in estimated for index in window):
            continue
        poses = [estimated[index] for index in window]
        if any(np.linalg.norm(center - portal_center) > portal_limit_m for _, center in poses):
            continue
        if any(
            np.linalg.norm(poses[i][1] - poses[i - 1][1]) > max_step_m
            for i in range(1, len(poses))
        ):
            continue
        if any(
            rotation_error_deg(poses[i][0], poses[i - 1][0]) > max_rotation_step_deg
            for i in range(1, len(poses))
        ):
            continue
        return offset, window
    return None


def evaluate(data_root: Path, experiment: Path) -> dict:
    split = json.loads((experiment / "split.json").read_text(encoding="utf-8"))
    map_names = (experiment / "map_images.txt").read_text(encoding="utf-8").splitlines()
    query_names = (experiment / "query_images.txt").read_text(encoding="utf-8").splitlines()
    registered = load_colmap_images(experiment / "registered_text" / "images.txt")
    ground_truth = load_ground_truth(data_root, map_names + query_names)

    common_map = [name for name in map_names if name in registered]
    scale, world_rotation, translation = estimate_similarity(
        np.stack([registered[name][1] for name in common_map]),
        np.stack([ground_truth[name][1] for name in common_map]),
    )
    aligned = {}
    for name, (rotation, center) in registered.items():
        aligned[name] = (
            rotation @ world_rotation.T,
            scale * (world_rotation @ center) + translation,
        )

    with (experiment / "evaluation" / "errors.csv").open(encoding="utf-8") as stream:
        error_rows = {row["name"]: row for row in csv.DictReader(stream) if row["split"] == "query"}

    center = np.asarray(split["portal_center_m"], dtype=float)
    portal_limit = float(split["portal_radius_m"]) + 0.1
    episodes = []
    trajectory_summaries = {}
    for trajectory in split["query_trajectories"]:
        metadata = split["trajectories"][trajectory]
        names = [frame_name(trajectory, index) for index in metadata["indices"]]
        valid_names = [
            name for name in names
            if error_rows[name]["registered"] == "True"
            and float(error_rows[name]["position_error_m"]) <= 0.25
            and float(error_rows[name]["rotation_error_deg"]) <= 5.0
        ]
        eligible = 0
        accepted = 0
        false_accepted = 0
        for episode_number, indices in enumerate(metadata["episodes"]):
            if len(indices) < 3:
                episodes.append(
                    {
                        "trajectory": trajectory,
                        "episode": episode_number,
                        "frames": len(indices),
                        "eligible": False,
                        "accepted": False,
                        "reason": "fewer_than_three_frames",
                    }
                )
                continue
            eligible += 1
            estimated = {
                index: aligned[frame_name(trajectory, index)]
                for index in indices
                if frame_name(trajectory, index) in aligned
            }
            result = first_consistent_window(indices, estimated, center, portal_limit)
            item = {
                "trajectory": trajectory,
                "episode": episode_number,
                "frames": len(indices),
                "eligible": True,
                "registered_frames": len(estimated),
                "accepted": result is not None,
            }
            if result is not None:
                offset, window = result
                accepted += 1
                item["accepted_frame_indices"] = window
                item["frames_observed_before_acceptance"] = offset + len(window)
                is_valid = all(frame_name(trajectory, index) in valid_names for index in window)
                item["accepted_pose_valid"] = is_valid
                if not is_valid:
                    false_accepted += 1
            episodes.append(item)
        trajectory_summaries[trajectory] = {
            "frames": len(names),
            "registered_frames": sum(error_rows[name]["registered"] == "True" for name in names),
            "valid_frames": len(valid_names),
            "eligible_episodes": eligible,
            "accepted_episodes": accepted,
            "false_accepted_episodes": false_accepted,
        }

    eligible_episodes = [item for item in episodes if item["eligible"]]
    accepted_episodes = [item for item in eligible_episodes if item["accepted"]]
    valid_frames = sum(value["valid_frames"] for value in trajectory_summaries.values())
    summary = {
        "map_trajectories": split["map_trajectories"],
        "query_trajectories": split["query_trajectories"],
        "portal_center_m": split["portal_center_m"],
        "portal_radius_m": split["portal_radius_m"],
        "query_frames": len(query_names),
        "valid_query_frames_0.25m_5deg": valid_frames,
        "valid_query_frame_rate": valid_frames / len(query_names),
        "eligible_arrival_episodes": len(eligible_episodes),
        "accepted_arrival_episodes": len(accepted_episodes),
        "arrival_acceptance_rate": len(accepted_episodes) / len(eligible_episodes),
        "false_accepted_arrival_episodes": sum(
            not item.get("accepted_pose_valid", True) for item in accepted_episodes
        ),
        "confirmation_rule": {
            "consecutive_poses": 3,
            "portal_prior_limit_m": portal_limit,
            "max_position_step_m": 0.5,
            "max_rotation_step_deg": 15.0,
        },
        "trajectories": trajectory_summaries,
        "episodes": episodes,
    }
    output = experiment / "portal_evaluation.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.data_root, args.experiment), indent=2))


if __name__ == "__main__":
    main()
