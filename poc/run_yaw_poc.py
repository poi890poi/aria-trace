"""Run the yaw-only POC on controlled rotations or a real video."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from yaw_estimation import (
    KltAngularYawEstimator,
    KltEssentialYawEstimator,
    camera_matrix,
    yaw_rotation,
)


def make_estimator(args: argparse.Namespace, matrix: np.ndarray):
    if args.backend == "essential":
        return KltEssentialYawEstimator(matrix)
    return KltAngularYawEstimator(matrix, use_essential_gate=not args.no_geometry_gate)


def save_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_synthetic(args: argparse.Namespace) -> Dict[str, float]:
    source = cv2.imread(str(args.image))
    if source is None:
        raise FileNotFoundError(args.image)
    height, width = source.shape[:2]
    k = camera_matrix(width, height, args.focal_ratio)
    estimator = make_estimator(args, k)

    phase = np.linspace(0.0, 2.0 * np.pi, args.frames)
    truth = args.amplitude * np.sin(phase) + 0.25 * args.amplitude * np.sin(3.0 * phase)
    rows: List[Dict[str, object]] = []
    for index, angle in enumerate(truth):
        homography = k @ yaw_rotation(float(angle)) @ np.linalg.inv(k)
        frame = cv2.warpPerspective(source, homography, (width, height))
        estimate = estimator.update(frame)
        if index == 0:
            continue
        rows.append(
            {
                "frame": index,
                "truth_yaw_deg": float(angle - truth[0]),
                "estimated_yaw_deg": estimate.total_deg,
                "truth_delta_deg": float(angle - truth[index - 1]),
                "estimated_delta_deg": estimate.delta_deg,
                "tracks": estimate.tracks,
                "inliers": estimate.inliers,
                "confidence": estimate.confidence,
                "elapsed_ms": estimate.elapsed_ms,
                "status": estimate.status,
            }
        )

    output = args.output
    save_csv(output / "synthetic.csv", rows)
    truth_yaw = np.array([row["truth_yaw_deg"] for row in rows], dtype=float)
    estimated = np.array([row["estimated_yaw_deg"] for row in rows], dtype=float)
    delta_truth = np.array([row["truth_delta_deg"] for row in rows], dtype=float)
    delta_est = np.array([row["estimated_delta_deg"] for row in rows], dtype=float)
    elapsed = np.array([row["elapsed_ms"] for row in rows], dtype=float)
    failures = sum(row["status"] != "ok" for row in rows)
    summary = {
        "frames": len(rows),
        "yaw_mae_deg": float(np.mean(np.abs(estimated - truth_yaw))),
        "yaw_rmse_deg": float(np.sqrt(np.mean((estimated - truth_yaw) ** 2))),
        "delta_mae_deg": float(np.mean(np.abs(delta_est - delta_truth))),
        "failures": failures,
        "mean_elapsed_ms": float(np.mean(elapsed)),
        "p95_elapsed_ms": float(np.percentile(elapsed, 95)),
    }
    (output / "synthetic_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    plt.figure(figsize=(9, 4))
    plt.plot(truth_yaw, label="known yaw")
    plt.plot(estimated, label="estimated yaw", linewidth=1.2)
    plt.xlabel("frame")
    plt.ylabel("degrees")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "synthetic_yaw.png", dpi=150)
    plt.close()
    return summary


def run_video(args: argparse.Namespace) -> Dict[str, float]:
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise FileNotFoundError(args.video)
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    estimator = make_estimator(args, camera_matrix(width, height, args.focal_ratio))

    rows: List[Dict[str, object]] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok or (args.max_frames and frame_index >= args.max_frames):
            break
        if frame_index % args.frame_step != 0:
            frame_index += 1
            continue
        estimate = estimator.update(frame)
        rows.append(
            {
                "frame": frame_index,
                "time_sec": frame_index / fps,
                "estimated_yaw_deg": estimate.total_deg,
                "estimated_delta_deg": estimate.delta_deg,
                "tracks": estimate.tracks,
                "inliers": estimate.inliers,
                "confidence": estimate.confidence,
                "elapsed_ms": estimate.elapsed_ms,
                "status": estimate.status,
            }
        )
        frame_index += 1
    capture.release()

    output = args.output
    save_csv(output / "video.csv", rows)
    elapsed = np.array([row["elapsed_ms"] for row in rows[1:]], dtype=float)
    summary = {
        "frames": len(rows),
        "video_frames": frame_index,
        "frame_step": args.frame_step,
        "duration_sec": frame_index / fps,
        "final_yaw_deg": float(rows[-1]["estimated_yaw_deg"]),
        "failures": sum(row["status"] not in ("ok", "initializing") for row in rows),
        "mean_elapsed_ms": float(np.mean(elapsed)),
        "p95_elapsed_ms": float(np.percentile(elapsed, 95)),
    }
    (output / "video_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plt.figure(figsize=(9, 4))
    plt.plot([row["time_sec"] for row in rows], [row["estimated_yaw_deg"] for row in rows])
    plt.xlabel("time (s)")
    plt.ylabel("relative yaw (degrees)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output / "video_yaw.png", dpi=150)
    plt.close()
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/yaw_poc"))
    parser.add_argument("--focal-ratio", type=float, default=0.9)
    parser.add_argument("--backend", choices=("angular", "essential"), default="angular")
    parser.add_argument("--no-geometry-gate", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("synthetic")
    synthetic.add_argument("--image", type=Path, required=True)
    synthetic.add_argument("--frames", type=int, default=181)
    synthetic.add_argument("--amplitude", type=float, default=12.0)
    synthetic.set_defaults(run=run_synthetic)

    video = subparsers.add_parser("video")
    video.add_argument("--video", type=Path, required=True)
    video.add_argument("--max-frames", type=int, default=0)
    video.add_argument("--frame-step", type=int, default=1)
    video.set_defaults(run=run_video)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(json.dumps(args.run(args), indent=2))


if __name__ == "__main__":
    main()
