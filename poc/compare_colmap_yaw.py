"""Compare an online yaw CSV with an offline COLMAP rotation reference."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def quaternion_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def load_colmap_images(path: Path) -> List[Tuple[int, np.ndarray]]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    poses: List[Tuple[int, np.ndarray]] = []
    for index in range(0, len(lines), 2):
        fields = lines[index].split()
        quaternion = [float(value) for value in fields[1:5]]
        timestamp_ns = int(Path(fields[9]).stem)
        poses.append((timestamp_ns, quaternion_to_matrix(*quaternion)))
    return sorted(poses, key=lambda item: item[0])


def save_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--online-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    poses = load_colmap_images(args.images)
    with args.online_csv.open(encoding="utf-8") as stream:
        online = {int(row["frame"]): row for row in csv.DictReader(stream)}

    reference_yaw = 0.0
    rows: List[Dict[str, object]] = []
    previous_rotation = poses[0][1]
    previous_online_yaw = 0.0
    for index, (timestamp_ns, rotation) in enumerate(poses):
        frame = int(round(timestamp_ns * args.fps / 1e9))
        online_yaw = float(online[frame]["estimated_yaw_deg"])
        if index == 0:
            reference_delta = 0.0
            online_delta = 0.0
        else:
            relative = rotation @ previous_rotation.T
            reference_delta = float(np.degrees(np.arctan2(relative[0, 2], relative[2, 2])))
            reference_yaw += reference_delta
            online_delta = online_yaw - previous_online_yaw
        rows.append(
            {
                "sparse_index": index,
                "video_frame": frame,
                "time_sec": timestamp_ns / 1e9,
                "reference_delta_deg": reference_delta,
                "online_delta_deg": online_delta,
                "reference_yaw_deg": reference_yaw,
                "online_yaw_deg": online_yaw,
                "yaw_error_deg": online_yaw - reference_yaw,
            }
        )
        previous_rotation = rotation
        previous_online_yaw = online_yaw

    save_csv(args.output / "comparison.csv", rows)
    reference = np.array([row["reference_yaw_deg"] for row in rows], dtype=float)
    estimated = np.array([row["online_yaw_deg"] for row in rows], dtype=float)
    reference_delta = np.diff(reference)
    estimated_delta = np.diff(estimated)
    error = estimated - reference
    delta_error = estimated_delta - reference_delta
    gain = float(np.dot(reference_delta, estimated_delta) / np.dot(reference_delta, reference_delta))
    summary = {
        "reference_frames": len(rows),
        "yaw_mae_deg": float(np.mean(np.abs(error))),
        "yaw_rmse_deg": float(np.sqrt(np.mean(error * error))),
        "yaw_max_abs_error_deg": float(np.max(np.abs(error))),
        "final_error_deg": float(error[-1]),
        "delta_mae_deg": float(np.mean(np.abs(delta_error))),
        "delta_rmse_deg": float(np.sqrt(np.mean(delta_error * delta_error))),
        "delta_p95_abs_error_deg": float(np.percentile(np.abs(delta_error), 95)),
        "delta_correlation": float(np.corrcoef(reference_delta, estimated_delta)[0, 1]),
        "delta_gain_online_over_reference": gain,
    }
    (args.output / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    times = [row["time_sec"] for row in rows]
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(times, reference, label="COLMAP offline reference")
    axes[0].plot(times, estimated, label="online estimator")
    axes[0].set_ylabel("accumulated view yaw (deg)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(times, error, color="tab:red")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("online - reference (deg)")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output / "yaw_comparison.png", dpi=150)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
