"""Evaluate a held-out COLMAP registration against a reference reconstruction.

The two reconstructions have unrelated similarity gauges.  We estimate that
gauge using map images only, then report errors for the held-out query images.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


def quaternion_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(rotation: np.ndarray):
    """Return a normalized scalar-first quaternion for a rotation matrix."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.array([qw, qx, qy, qz], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def load_colmap_images(path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Return image name -> (world-to-camera rotation, camera center)."""
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    poses: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for line in lines:
        fields = line.split()
        # Pose records have IMAGE_ID through CAMERA_ID followed by NAME.
        # Observation records contain repeated X/Y/POINT3D_ID triplets.
        if len(fields) != 10:
            continue
        rotation = quaternion_to_matrix(*(float(value) for value in fields[1:5]))
        translation = np.array([float(value) for value in fields[5:8]])
        poses[fields[9]] = (rotation, -rotation.T @ translation)
    return poses


def estimate_similarity(source: np.ndarray, target: np.ndarray):
    """Estimate target = scale * rotation @ source + translation (Umeyama)."""
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u) * np.linalg.det(vt))
    rotation = u @ correction @ vt
    variance = np.sum(source_centered * source_centered) / len(source)
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def rotation_error_deg(estimated: np.ndarray, reference: np.ndarray) -> float:
    relative = estimated @ reference.T
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def view_direction_error_deg(estimated: np.ndarray, reference: np.ndarray) -> float:
    optical_axis = np.array([0.0, 0.0, 1.0])
    estimated_view = estimated.T @ optical_axis
    reference_view = reference.T @ optical_axis
    cosine = np.clip(np.dot(estimated_view, reference_view), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def summarize(values: Iterable[float], prefix: str) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        prefix + "_mean": float(np.mean(array)),
        prefix + "_median": float(np.median(array)),
        prefix + "_p95": float(np.percentile(array, 95)),
        prefix + "_max": float(np.max(array)),
    }


def coverage_by_distance(query_rows):
    edges = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, float("inf")]
    result = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        rows = [
            row
            for row in query_rows
            if lower <= row["nearest_map_distance_steps"] < upper
        ]
        result.append(
            {
                "min_steps": lower,
                "max_steps": None if np.isinf(upper) else upper,
                "queries": len(rows),
                "registered": sum(bool(row["registered"]) for row in rows),
                "registration_rate": (
                    sum(bool(row["registered"]) for row in rows) / len(rows) if rows else None
                ),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-images", type=Path, required=True)
    parser.add_argument("--registered-images", type=Path, required=True)
    parser.add_argument("--map-list", type=Path, required=True)
    parser.add_argument("--query-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = load_colmap_images(args.reference_images)
    registered = load_colmap_images(args.registered_images)
    map_names = args.map_list.read_text(encoding="utf-8").splitlines()
    query_names = args.query_list.read_text(encoding="utf-8").splitlines()
    common_map = [name for name in map_names if name in reference and name in registered]
    reference_queries = [name for name in query_names if name in reference]
    registered_queries = [name for name in reference_queries if name in registered]
    if len(common_map) < 3:
        raise RuntimeError("At least three common map images are required for gauge alignment")

    source = np.stack([registered[name][1] for name in common_map])
    target = np.stack([reference[name][1] for name in common_map])
    scale, world_rotation, translation = estimate_similarity(source, target)

    ordered_reference_names = sorted(reference)
    ordered_reference_centers = np.stack([reference[name][1] for name in ordered_reference_names])
    median_reference_step = float(
        np.median(np.linalg.norm(np.diff(ordered_reference_centers, axis=0), axis=1))
    )
    reference_map_centers = np.stack([reference[name][1] for name in common_map])

    rows = []
    for split, names in (("map", common_map), ("query", reference_queries)):
        for name in names:
            nearest_map_distance = float(
                np.min(np.linalg.norm(reference_map_centers - reference[name][1], axis=1))
            )
            if name not in registered:
                rows.append(
                    {
                        "name": name,
                        "split": split,
                        "registered": False,
                        "nearest_map_distance": nearest_map_distance,
                        "nearest_map_distance_steps": nearest_map_distance / median_reference_step,
                        "position_error": "",
                        "rotation_error_deg": "",
                        "view_direction_error_deg": "",
                    }
                )
                continue
            source_rotation, source_center = registered[name]
            reference_rotation, reference_center = reference[name]
            aligned_center = scale * (world_rotation @ source_center) + translation
            aligned_rotation = source_rotation @ world_rotation.T
            rows.append(
                {
                    "name": name,
                    "split": split,
                    "registered": True,
                    "nearest_map_distance": nearest_map_distance,
                    "nearest_map_distance_steps": nearest_map_distance / median_reference_step,
                    "position_error": float(np.linalg.norm(aligned_center - reference_center)),
                    "rotation_error_deg": rotation_error_deg(aligned_rotation, reference_rotation),
                    "view_direction_error_deg": view_direction_error_deg(
                        aligned_rotation, reference_rotation
                    ),
                }
            )

    reference_query_centers = np.stack([reference[name][1] for name in reference_queries])
    path_length = float(np.linalg.norm(np.diff(reference_query_centers, axis=0), axis=1).sum())
    query_rows = [row for row in rows if row["split"] == "query"]
    registered_query_rows = [row for row in query_rows if row["registered"]]
    map_rows = [row for row in rows if row["split"] == "map"]
    summary = {
        "map_images_requested": len(map_names),
        "map_images_aligned": len(common_map),
        "query_images_requested": len(query_names),
        "query_images_registered": len(registered_queries),
        "query_registration_rate": len(registered_queries) / len(query_names),
        "alignment_scale_to_reference": scale,
        "reference_query_path_length": path_length,
        "median_reference_frame_step": median_reference_step,
        "query_coverage_by_nearest_map_distance_steps": coverage_by_distance(query_rows),
    }
    for split, split_rows in (("map", map_rows), ("query", registered_query_rows)):
        summary.update(summarize((row["position_error"] for row in split_rows), split + "_position_error"))
        summary[split + "_position_rmse"] = float(
            np.sqrt(np.mean([row["position_error"] ** 2 for row in split_rows]))
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
    summary["query_position_rmse_fraction_of_path"] = (
        summary["query_position_rmse"] / path_length if path_length else None
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
