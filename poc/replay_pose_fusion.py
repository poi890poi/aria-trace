"""Replay real relocalization hypotheses through the minimal fusion gate.

TartanAir supplies metric ground truth and the PnP hypotheses are actual outputs
from the completed COLMAP experiments.  Because no game controls or minimap are
available, local-motion and coarse-prior measurements are simulated from ground
truth with explicit noise and bias.  They are inputs to the gate, never scoring
labels for individual hypotheses.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from .evaluate_relocalization import estimate_similarity, load_colmap_images
    from .evaluate_tartanair_relocalization import load_ground_truth
    from .pose_fusion import (
        FusionConfig,
        Pose2D,
        PoseFusionGate,
        angle_difference_deg,
        wrap_angle_deg,
    )
except ImportError:  # Direct script execution.
    from evaluate_relocalization import estimate_similarity, load_colmap_images
    from evaluate_tartanair_relocalization import load_ground_truth
    from pose_fusion import (
        FusionConfig,
        Pose2D,
        PoseFusionGate,
        angle_difference_deg,
        wrap_angle_deg,
    )


def rotation_to_pose(center: np.ndarray, world_to_camera: np.ndarray) -> Pose2D:
    forward_world = world_to_camera.T @ np.array([0.0, 0.0, 1.0])
    yaw_deg = math.degrees(math.atan2(forward_world[1], forward_world[0]))
    return Pose2D(float(center[0]), float(center[1]), wrap_angle_deg(yaw_deg))


def load_pose_series(
    data_root: Path, map_list: Path, query_list: Path, registered_images: Path
) -> Tuple[List[str], List[Pose2D], Dict[str, Pose2D]]:
    map_names = map_list.read_text(encoding="utf-8").splitlines()
    query_names = query_list.read_text(encoding="utf-8").splitlines()
    ground_truth = load_ground_truth(data_root, map_names + query_names)
    registered = load_colmap_images(registered_images)
    common_map = [name for name in map_names if name in registered]
    if len(common_map) < 3:
        raise RuntimeError("At least three registered map images are required")

    source_centers = np.stack([registered[name][1] for name in common_map])
    target_centers = np.stack([ground_truth[name][1] for name in common_map])
    scale, world_rotation, translation = estimate_similarity(source_centers, target_centers)

    truth = [rotation_to_pose(ground_truth[name][1], ground_truth[name][0]) for name in query_names]
    hypotheses = {}
    for name in query_names:
        if name not in registered:
            continue
        source_rotation, source_center = registered[name]
        aligned_center = scale * (world_rotation @ source_center) + translation
        aligned_rotation = source_rotation @ world_rotation.T
        hypotheses[name] = rotation_to_pose(aligned_center, aligned_rotation)
    return query_names, truth, hypotheses


def load_validity_labels(path: Path) -> Dict[str, bool]:
    labels = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["split"] != "query" or row["registered"] != "True":
                continue
            labels[row["name"]] = (
                float(row["position_error_m"]) <= 0.25
                and float(row["rotation_error_deg"]) <= 5.0
            )
    return labels


def local_delta(previous: Pose2D, current: Pose2D) -> Tuple[float, float]:
    dx = current.x - previous.x
    dy = current.y - previous.y
    heading = math.radians(previous.yaw_deg)
    return (
        math.cos(heading) * dx + math.sin(heading) * dy,
        -math.sin(heading) * dx + math.cos(heading) * dy,
    )


def percentile(values, quantile: float) -> Optional[float]:
    return float(np.percentile(values, quantile)) if values else None


def run_trial(
    names: List[str],
    truth: List[Pose2D],
    hypotheses: Dict[str, Pose2D],
    validity: Dict[str, bool],
    seed: int,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    rng = np.random.RandomState(seed)
    fusion = PoseFusionGate()
    fusion.initialize(truth[0])  # KnownStartInitializer in the MVP.
    naive = PoseFusionGate(
        FusionConfig(
            prediction_position_floor_m=1.0e9,
            prediction_position_limit_m=1.0e9,
            coarse_position_limit_m=1.0e9,
            prediction_yaw_floor_deg=1.0e9,
            prediction_yaw_limit_deg=1.0e9,
            coarse_yaw_limit_deg=1.0e9,
            position_correction_weight=1.0,
            yaw_correction_weight=1.0,
        )
    )
    naive.initialize(truth[0])

    # Trial-level biases represent an imperfect calibrated game-motion model.
    motion_scale = 1.06 + rng.normal(0.0, 0.01)
    yaw_bias_deg = 0.06 + rng.normal(0.0, 0.015)
    rows = []
    accepted_valid = accepted_false = rejected_valid = rejected_false = 0
    correction_jumps = []
    position_errors = []
    yaw_errors = []
    naive_position_errors = []
    naive_yaw_errors = []
    naive_correction_jumps = []
    mode_counts = {"TRACK": 0, "CAUTIOUS": 0, "RELOCALIZE": 0, "STOP": 0}
    recovery_frames = []
    high_error_since = None

    for index, (name, gt_pose) in enumerate(zip(names, truth)):
        if index > 0:
            motion = local_delta(truth[index - 1], gt_pose)
            noisy_motion = (
                motion_scale * motion[0] + rng.normal(0.0, 0.004),
                motion_scale * motion[1] + rng.normal(0.0, 0.004),
            )
            true_delta_yaw = angle_difference_deg(gt_pose.yaw_deg, truth[index - 1].yaw_deg)
            measured_delta_yaw = true_delta_yaw + yaw_bias_deg + rng.normal(0.0, 0.30)
            fusion.predict(noisy_motion, measured_delta_yaw)
            naive.predict(noisy_motion, measured_delta_yaw)

        pre_position_error = math.hypot(
            fusion.state.pose.x - gt_pose.x, fusion.state.pose.y - gt_pose.y
        )
        pre_yaw_error = abs(angle_difference_deg(fusion.state.pose.yaw_deg, gt_pose.yaw_deg))
        if high_error_since is None and (pre_position_error > 0.25 or pre_yaw_error > 5.0):
            high_error_since = index

        decision = None
        hypothesis = hypotheses.get(name)
        if hypothesis is not None:
            coarse_prior = Pose2D(
                gt_pose.x + rng.normal(0.0, 0.28),
                gt_pose.y + rng.normal(0.0, 0.28),
                wrap_angle_deg(gt_pose.yaw_deg + rng.normal(0.0, 10.0)),
            )
            decision = fusion.consider_absolute(hypothesis, coarse_prior)
            naive_decision = naive.consider_absolute(hypothesis, coarse_prior)
            naive_correction_jumps.append(naive_decision.applied_position_change_m)
            is_valid = validity[name]
            if decision.accepted:
                correction_jumps.append(decision.applied_position_change_m)
                if is_valid:
                    accepted_valid += 1
                else:
                    accepted_false += 1
            elif is_valid:
                rejected_valid += 1
            else:
                rejected_false += 1

        position_error = math.hypot(
            fusion.state.pose.x - gt_pose.x, fusion.state.pose.y - gt_pose.y
        )
        yaw_error = abs(angle_difference_deg(fusion.state.pose.yaw_deg, gt_pose.yaw_deg))
        naive_position_error = math.hypot(
            naive.state.pose.x - gt_pose.x, naive.state.pose.y - gt_pose.y
        )
        naive_yaw_error = abs(angle_difference_deg(naive.state.pose.yaw_deg, gt_pose.yaw_deg))
        if high_error_since is not None and position_error <= 0.25 and yaw_error <= 5.0:
            recovery_frames.append(index - high_error_since)
            high_error_since = None
        position_errors.append(position_error)
        yaw_errors.append(yaw_error)
        naive_position_errors.append(naive_position_error)
        naive_yaw_errors.append(naive_yaw_error)
        mode_counts[fusion.state.mode] += 1
        rows.append(
            {
                "index": index,
                "name": name,
                "gt_x": gt_pose.x,
                "gt_y": gt_pose.y,
                "gt_yaw_deg": gt_pose.yaw_deg,
                "estimate_x": fusion.state.pose.x,
                "estimate_y": fusion.state.pose.y,
                "estimate_yaw_deg": fusion.state.pose.yaw_deg,
                "position_error_m": position_error,
                "yaw_error_deg": yaw_error,
                "naive_position_error_m": naive_position_error,
                "naive_yaw_error_deg": naive_yaw_error,
                "position_sigma_m": fusion.state.position_sigma_m,
                "yaw_sigma_deg": fusion.state.yaw_sigma_deg,
                "mode": fusion.state.mode,
                "pnp_available": hypothesis is not None,
                "pnp_valid": validity.get(name, ""),
                "pnp_accepted": decision.accepted if decision else "",
                "pnp_reason": decision.reason if decision else "",
                "correction_jump_m": decision.applied_position_change_m if decision else "",
            }
        )

    summary = {
        "seed": seed,
        "frames": len(names),
        "raw_valid_hypotheses": accepted_valid + rejected_valid,
        "raw_false_hypotheses": accepted_false + rejected_false,
        "accepted_valid": accepted_valid,
        "accepted_false": accepted_false,
        "rejected_valid": rejected_valid,
        "rejected_false": rejected_false,
        "position_error_median": float(np.median(position_errors)),
        "position_error_p95": percentile(position_errors, 95),
        "position_error_max": max(position_errors),
        "yaw_error_deg_median": float(np.median(yaw_errors)),
        "yaw_error_deg_p95": percentile(yaw_errors, 95),
        "yaw_error_deg_max": max(yaw_errors),
        "naive_position_error_median": float(np.median(naive_position_errors)),
        "naive_position_error_p95": percentile(naive_position_errors, 95),
        "naive_position_error_max": max(naive_position_errors),
        "naive_yaw_error_deg_median": float(np.median(naive_yaw_errors)),
        "naive_yaw_error_deg_p95": percentile(naive_yaw_errors, 95),
        "naive_yaw_error_deg_max": max(naive_yaw_errors),
        "naive_correction_jump_m_max": (
            max(naive_correction_jumps) if naive_correction_jumps else None
        ),
        "correction_jump_m_p95": percentile(correction_jumps, 95),
        "correction_jump_m_max": max(correction_jumps) if correction_jumps else None,
        "recovery_frames_median": percentile(recovery_frames, 50),
        "recovery_frames_max": max(recovery_frames) if recovery_frames else None,
        "unrecovered_at_end": high_error_since is not None,
        "mode_fraction": {
            mode: count / len(names) for mode, count in mode_counts.items()
        },
    }
    return summary, rows


def aggregate_trials(trials: List[Dict[str, object]]) -> Dict[str, object]:
    total_valid = sum(int(trial["raw_valid_hypotheses"]) for trial in trials)
    total_false = sum(int(trial["raw_false_hypotheses"]) for trial in trials)
    accepted_valid = sum(int(trial["accepted_valid"]) for trial in trials)
    accepted_false = sum(int(trial["accepted_false"]) for trial in trials)
    return {
        "trials": len(trials),
        "raw_valid_hypotheses_per_trial": trials[0]["raw_valid_hypotheses"],
        "raw_false_hypotheses_per_trial": trials[0]["raw_false_hypotheses"],
        "valid_accept_rate": accepted_valid / total_valid if total_valid else None,
        "false_accept_rate": accepted_false / total_false if total_false else None,
        "accepted_false_total": accepted_false,
        "accepted_false_worst_trial": max(int(trial["accepted_false"]) for trial in trials),
        "position_error_median_mean": float(
            np.mean([trial["position_error_median"] for trial in trials])
        ),
        "position_error_p95_mean": float(
            np.mean([trial["position_error_p95"] for trial in trials])
        ),
        "yaw_error_deg_median_mean": float(
            np.mean([trial["yaw_error_deg_median"] for trial in trials])
        ),
        "yaw_error_deg_p95_mean": float(
            np.mean([trial["yaw_error_deg_p95"] for trial in trials])
        ),
        "naive_position_error_p95_mean": float(
            np.mean([trial["naive_position_error_p95"] for trial in trials])
        ),
        "naive_yaw_error_deg_p95_mean": float(
            np.mean([trial["naive_yaw_error_deg_p95"] for trial in trials])
        ),
        "naive_correction_jump_m_max_worst_trial": max(
            float(trial["naive_correction_jump_m_max"] or 0.0) for trial in trials
        ),
        "unrecovered_trial_count": sum(bool(trial["unrecovered_at_end"]) for trial in trials),
        "first_trial": trials[0],
    }


def default_cases(root: Path):
    data_root = root / "data/tartanair2/ArchVizTinyHouseDay/Data_easy"
    p003 = root / "artifacts/relocalization_tartanair_tinyhouse_p000_p003"
    p005 = root / "artifacts/relocalization_tartanair_tinyhouse_p000_p005"
    return [
        {
            "name": "p003_balanced",
            "data_root": data_root,
            "map_list": p003 / "map_images.txt",
            "query_list": p003 / "query_images.txt",
            "registered_images": p003 / "registered_text/images.txt",
            "evaluation_csv": p003 / "evaluation/errors.csv",
        },
        {
            "name": "p005_reverse_view_aliked",
            "data_root": data_root,
            "map_list": p005 / "map_images.txt",
            "query_list": p005 / "query_images.txt",
            "registered_images": p005 / "registered_gt_map_aliked_text/images.txt",
            "evaluation_csv": p005 / "evaluation_gt_map_aliked/errors.csv",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("artifacts/fusion_replay"))
    parser.add_argument("--trials", type=int, default=100)
    args = parser.parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be positive")

    output = args.output if args.output.is_absolute() else args.root / args.output
    output.mkdir(parents=True, exist_ok=True)
    all_results = {
        "experiment": "minimal planar predictor plus independently gated PnP correction",
        "measured_inputs": "TartanAir ground truth and actual COLMAP PnP hypotheses",
        "simulated_inputs": {
            "known_start": True,
            "control_motion": "ground-truth local increments with 6% scale bias and noise",
            "relative_heading": "ground-truth increments with per-trial bias and 0.30 degree noise",
            "coarse_prior": "ground truth plus 0.28 m/axis and 10 degree Gaussian noise",
        },
        "cases": {},
    }
    for case in default_cases(args.root):
        names, truth, hypotheses = load_pose_series(
            case["data_root"], case["map_list"], case["query_list"], case["registered_images"]
        )
        validity = load_validity_labels(case["evaluation_csv"])
        trials = []
        first_rows = None
        for seed in range(args.trials):
            trial, rows = run_trial(names, truth, hypotheses, validity, seed)
            trials.append(trial)
            if first_rows is None:
                first_rows = rows
        result = aggregate_trials(trials)
        all_results["cases"][case["name"]] = result
        with (output / (case["name"] + "_seed0.csv")).open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(first_rows[0]))
            writer.writeheader()
            writer.writerows(first_rows)

    (output / "summary.json").write_text(
        json.dumps(all_results, indent=2), encoding="utf-8"
    )
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
