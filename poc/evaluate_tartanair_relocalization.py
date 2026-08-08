"""Evaluate a COLMAP map/query reconstruction against TartanAir metric poses."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from .evaluate_relocalization import (
        estimate_similarity,
        load_colmap_images,
        quaternion_to_matrix,
        rotation_error_deg,
        summarize,
        view_direction_error_deg,
    )
except ImportError:  # Direct script execution.
    from evaluate_relocalization import (
        estimate_similarity,
        load_colmap_images,
        quaternion_to_matrix,
        rotation_error_deg,
        summarize,
        view_direction_error_deg,
    )


# Convert OpenCV camera coordinates (right, down, forward) to TartanAir's
# camera NED convention (forward, right, down).
OPENCV_TO_TARTAN_CAMERA = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
)


def load_ground_truth(data_root: Path, names):
    trajectories = {}
    result = {}
    for name in names:
        parts = Path(name).parts
        trajectory = parts[0]
        if trajectory not in trajectories:
            trajectories[trajectory] = np.loadtxt(
                str(data_root / trajectory / "pose_lcam_front.txt")
            )
        index = int(Path(name).stem.split("_")[0])
        values = trajectories[trajectory][index]
        center = values[:3]
        qx, qy, qz, qw = values[3:]
        camera_to_world_tartan = quaternion_to_matrix(qw, qx, qy, qz)
        camera_to_world_opencv = camera_to_world_tartan @ OPENCV_TO_TARTAN_CAMERA
        result[name] = (camera_to_world_opencv.T, center)
    return result


def distance_coverage(rows):
    edges = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, float("inf")]
    result = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = [row for row in rows if lower <= row["nearest_map_distance_m"] < upper]
        registered = [row for row in selected if row["registered"]]
        valid = [
            row
            for row in registered
            if row["position_error_m"] <= 0.25 and row["rotation_error_deg"] <= 5.0
        ]
        item = {
            "min_m": lower,
            "max_m": None if np.isinf(upper) else upper,
            "queries": len(selected),
            "registered": len(registered),
            "registration_rate": len(registered) / len(selected) if selected else None,
            "valid_0.25m_5deg": len(valid),
            "valid_rate": len(valid) / len(selected) if selected else None,
        }
        if registered:
            item["position_error_m_median"] = float(
                np.median([row["position_error_m"] for row in registered])
            )
            item["rotation_error_deg_median"] = float(
                np.median([row["rotation_error_deg"] for row in registered])
            )
        result.append(item)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--registered-images", type=Path, required=True)
    parser.add_argument("--map-list", type=Path, required=True)
    parser.add_argument("--query-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    registered = load_colmap_images(args.registered_images)
    map_names = args.map_list.read_text(encoding="utf-8").splitlines()
    query_names = args.query_list.read_text(encoding="utf-8").splitlines()
    ground_truth = load_ground_truth(args.data_root, map_names + query_names)
    common_map = [name for name in map_names if name in registered]
    registered_queries = [name for name in query_names if name in registered]
    if len(common_map) < 3:
        raise RuntimeError("At least three mapped images are required for alignment")

    source_centers = np.stack([registered[name][1] for name in common_map])
    target_centers = np.stack([ground_truth[name][1] for name in common_map])
    scale, world_rotation, translation = estimate_similarity(source_centers, target_centers)
    map_gt_centers = np.stack([ground_truth[name][1] for name in common_map])

    rows = []
    for split, names in (("map", map_names), ("query", query_names)):
        for name in names:
            gt_rotation, gt_center = ground_truth[name]
            nearest_distance = float(np.min(np.linalg.norm(map_gt_centers - gt_center, axis=1)))
            row = {
                "name": name,
                "split": split,
                "registered": name in registered,
                "nearest_map_distance_m": nearest_distance,
                "position_error_m": "",
                "rotation_error_deg": "",
                "view_direction_error_deg": "",
            }
            if name in registered:
                source_rotation, source_center = registered[name]
                aligned_center = scale * (world_rotation @ source_center) + translation
                aligned_rotation = source_rotation @ world_rotation.T
                row.update(
                    {
                        "position_error_m": float(np.linalg.norm(aligned_center - gt_center)),
                        "rotation_error_deg": rotation_error_deg(aligned_rotation, gt_rotation),
                        "view_direction_error_deg": view_direction_error_deg(
                            aligned_rotation, gt_rotation
                        ),
                    }
                )
            rows.append(row)

    map_rows = [row for row in rows if row["split"] == "map" and row["registered"]]
    query_rows = [row for row in rows if row["split"] == "query"]
    registered_query_rows = [row for row in query_rows if row["registered"]]
    valid_query_rows = [
        row
        for row in registered_query_rows
        if row["position_error_m"] <= 0.25 and row["rotation_error_deg"] <= 5.0
    ]
    summary = {
        "map_images_requested": len(map_names),
        "map_images_aligned": len(common_map),
        "query_images_requested": len(query_names),
        "query_images_registered": len(registered_queries),
        "query_registration_rate": len(registered_queries) / len(query_names),
        "query_valid_0.25m_5deg": len(valid_query_rows),
        "query_valid_rate": len(valid_query_rows) / len(query_names),
        "registered_false_pose_count": len(registered_query_rows) - len(valid_query_rows),
        "alignment_scale_to_meters": scale,
        "query_coverage_by_nearest_map_distance_m": distance_coverage(query_rows),
    }
    for split, split_rows in (("map", map_rows), ("query", registered_query_rows)):
        if not split_rows:
            continue
        summary.update(
            summarize((row["position_error_m"] for row in split_rows), split + "_position_error_m")
        )
        summary[split + "_position_rmse_m"] = float(
            np.sqrt(np.mean([row["position_error_m"] ** 2 for row in split_rows]))
        )
        summary.update(
            summarize((row["rotation_error_deg"] for row in split_rows), split + "_rotation_error_deg")
        )
        summary.update(
            summarize(
                (row["view_direction_error_deg"] for row in split_rows),
                split + "_view_direction_error_deg",
            )
        )
    if valid_query_rows:
        summary.update(
            summarize(
                (row["position_error_m"] for row in valid_query_rows),
                "valid_query_position_error_m",
            )
        )
        summary.update(
            summarize(
                (row["rotation_error_deg"] for row in valid_query_rows),
                "valid_query_rotation_error_deg",
            )
        )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "errors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
