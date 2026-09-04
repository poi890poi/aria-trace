"""Extract timestamped control, KLT scene-turn, and cursor-pose evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from benchmarks.temporal_turns import evaluate_reversal_response
from rig_runtime.adapters.filesystem.session import SessionReader
from rig_runtime.services.calibration.cursor.pose import CursorPoseEstimator
from rig_runtime.services.vision import KltAngularYawEstimator, camera_matrix


GENSHIN_IMPACT_EXCLUDED_RECTS = (
    (0.0, 0.0, 0.24, 0.30),
    (0.72, 0.0, 1.0, 0.28),
    (0.0, 0.76, 1.0, 1.0),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def _sample_indexes(records: list[dict], sample_hz: float) -> list[int]:
    interval_ns = int(round(1.0e9 / sample_hz))
    output = []
    next_time = int(records[0]["session_time_ns"])
    for ordinal, row in enumerate(records):
        timestamp = int(row["session_time_ns"])
        if timestamp >= next_time:
            output.append(ordinal)
            next_time += interval_ns
            while next_time <= timestamp:
                next_time += interval_ns
    return output


def _mouse_signal(inputs: list[dict], sample_times_ns: list[int]) -> list[dict]:
    mouse = [
        row
        for row in inputs
        if row.get("kind") == "pc_raw_mouse"
        and (row.get("payload") or {}).get("delta_x") is not None
    ]
    output = []
    cursor = 0
    previous = sample_times_ns[0]
    for timestamp in sample_times_ns[1:]:
        total = 0.0
        count = 0
        while cursor < len(mouse) and int(mouse[cursor]["session_time_ns"]) <= timestamp:
            event_time = int(mouse[cursor]["session_time_ns"])
            if event_time > previous:
                total += float((mouse[cursor].get("payload") or {}).get("delta_x") or 0.0)
                count += 1
            cursor += 1
        elapsed_s = max((timestamp - previous) / 1.0e9, 1.0e-6)
        output.append(
            {
                "session_time_ns": int(timestamp),
                "value": total / elapsed_s,
                "event_count": count,
                "units": "raw_mouse_counts_per_second",
                "role": "control_intent_not_heading_truth",
            }
        )
        previous = timestamp
    return output


def extract(
    session_path: Path,
    output_path: Path,
    *,
    sample_hz: float = 15.0,
    cursor_calibration: Path | None = None,
    focal_ratio: float = 0.9,
    maximum_scene_width: int = 960,
    excluded_rects=(),
    pose_method: str = "angular_projection_ncc_parabolic",
    gaussian_fit_method: str = "cascade",
    validation_policy: str = "minimal",
    candidates: dict[str, Path] | None = None,
) -> dict:
    reader = SessionReader(session_path)
    records = list(reader.frames_by_stream.get("main") or ())
    if len(records) < 10:
        raise RuntimeError("Session has too few main-stream frames")
    selected = _sample_indexes(records, sample_hz)
    selected_set = set(selected)
    pose_estimator = (
        CursorPoseEstimator(
            cursor_calibration,
            gaussian_fit_method=gaussian_fit_method,
            validation_policy=validation_policy,
            pose_method=pose_method,
        )
        if cursor_calibration is not None
        else None
    )
    capture = cv2.VideoCapture(str(reader.video_path("main")))
    scene_rows = []
    pose_angles = []
    estimator = None
    previous_time_ns = None
    previous_pose = None
    previous_pose_time_ns = None
    try:
        ordinal = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if ordinal not in selected_set:
                ordinal += 1
                continue
            record = records[ordinal]
            timestamp = int(record["session_time_ns"])
            height, width = frame.shape[:2]
            scale = min(1.0, maximum_scene_width / float(width))
            scene = (
                cv2.resize(
                    frame,
                    (int(round(width * scale)), int(round(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
                if scale < 1.0
                else frame
            )
            if estimator is None:
                scene_height, scene_width = scene.shape[:2]
                estimator = KltAngularYawEstimator(
                    camera_matrix(scene_width, scene_height, focal_ratio),
                    max_corners=800,
                    min_tracks=20,
                    use_essential_gate=True,
                    excluded_rects=excluded_rects,
                )
            scene_result = estimator.update(scene)
            elapsed_s = (
                (timestamp - previous_time_ns) / 1.0e9
                if previous_time_ns is not None
                else None
            )
            scene_rows.append(
                {
                    "frame_index": int(record["frame_index"]),
                    "session_time_ns": timestamp,
                    "value": (
                        float(scene_result.delta_deg) / max(elapsed_s, 1.0e-6)
                        if elapsed_s is not None and scene_result.status == "ok"
                        else None
                    ),
                    "confidence": float(scene_result.confidence),
                    "tracks": int(scene_result.tracks),
                    "inliers": int(scene_result.inliers),
                    "status": scene_result.status,
                    "units": "screen_scene_yaw_degrees_per_second",
                    "role": "observed_camera_motion_not_player_heading_truth",
                }
            )
            if pose_estimator is not None:
                pose = pose_estimator.estimate(
                    frame,
                    frame_index=int(record["frame_index"]),
                    session_time_ns=timestamp,
                )
                angle = pose.get("angle_screen_deg")
                delta = None
                if (
                    angle is not None
                    and previous_pose is not None
                    and previous_pose_time_ns is not None
                ):
                    delta = ((float(angle) - previous_pose + 180.0) % 360.0) - 180.0
                    pose_elapsed_s = max(
                        (timestamp - previous_pose_time_ns) / 1.0e9, 1.0e-6
                    )
                else:
                    pose_elapsed_s = None
                pose_angles.append(
                    {
                        "frame_index": int(record["frame_index"]),
                        "session_time_ns": timestamp,
                        "value": (
                            delta / pose_elapsed_s if delta is not None else None
                        ),
                        "angle_screen_deg": float(angle) if angle is not None else None,
                        "confidence": float(pose.get("confidence") or 0.0),
                        "detected": bool(pose.get("detected")),
                        "units": "cursor_screen_degrees_per_second",
                        "role": "candidate_player_heading",
                    }
                )
                if angle is not None:
                    previous_pose = float(angle)
                    previous_pose_time_ns = timestamp
            previous_time_ns = timestamp
            ordinal += 1
    finally:
        capture.release()
    sample_times = [int(records[index]["session_time_ns"]) for index in selected]
    input_rows = _mouse_signal(reader.inputs, sample_times)
    valid_scene = [row for row in scene_rows if row["value"] is not None]
    result = {
        "schema_version": "1.0",
        "method": "timestamp_aligned_raw_input_klt_scene_cursor_turn_evidence",
        "session": str(Path(session_path).resolve()),
        "sample_hz": float(sample_hz),
        "sample_count": len(selected),
        "parameters": {
            "sample_hz": float(sample_hz),
            "focal_ratio": float(focal_ratio),
            "maximum_scene_width": int(maximum_scene_width),
            "excluded_rects": [list(row) for row in excluded_rects],
            "pose_method": pose_method if pose_estimator is not None else None,
            "gaussian_fit_method": (
                gaussian_fit_method if pose_estimator is not None else None
            ),
            "validation_policy": validation_policy if pose_estimator is not None else None,
        },
        "source_files": {
            "manifest": _sha256(Path(session_path) / "manifest.json"),
            "frames": _sha256(Path(session_path) / "frames.jsonl"),
            "inputs": _sha256(Path(session_path) / "inputs.jsonl"),
            "video": _sha256(reader.video_path("main")),
        },
        "implementation_files": {
            "extractor": _sha256(Path(__file__)),
            "turn_metrics": _sha256(Path(__file__).with_name("temporal_turns.py")),
        },
        "evidence_contract": {
            "input": "control_intent_not_truth",
            "scene_klt": "observed_camera_motion_not_player_heading_truth",
            "cursor_pose": "candidate_player_heading",
            "same_frame_index_assumed": False,
            "sign_and_lag_fitted": True,
            "wrong_pose_requires_input_and_scene_corroboration": True,
        },
        "input_to_scene": evaluate_reversal_response(
            input_rows, valid_scene, alignment_sample_hz=sample_hz
        ),
    }
    if pose_angles:
        valid_pose = [row for row in pose_angles if row["value"] is not None]
        result["input_to_pose"] = evaluate_reversal_response(
            input_rows, valid_pose, alignment_sample_hz=sample_hz
        )
        result["scene_to_pose"] = evaluate_reversal_response(
            valid_scene, valid_pose, alignment_sample_hz=sample_hz
        )
        input_pose_sign = result["input_to_pose"]["alignment"]["sign"]
        input_scene_sign = result["input_to_scene"]["alignment"]["sign"]
        scene_pose_sign = result["scene_to_pose"]["alignment"]["sign"]
        result["pose_direction_consistency"] = {
            "consistent_composition": input_pose_sign == input_scene_sign * scene_pose_sign,
            "verdict": (
                "sign_consistent_after_lag_fit"
                if input_pose_sign == input_scene_sign * scene_pose_sign
                else "inconclusive_or_coordinate_convention_defect"
            ),
        }
        result["source_files"]["cursor_calibration"] = _sha256(
            Path(cursor_calibration)
            if Path(cursor_calibration).is_file()
            else Path(cursor_calibration) / "calibration.json"
        )
    result["candidate_turn_response"] = {}
    for name, path in sorted((candidates or {}).items()):
        rows = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        result["candidate_turn_response"][name] = {
            "input": evaluate_reversal_response(
                input_rows, rows, alignment_sample_hz=sample_hz
            ),
            "scene_klt": evaluate_reversal_response(
                valid_scene, rows, alignment_sample_hz=sample_hz
            ),
            "source": {"path": str(Path(path).resolve()), "sha256": _sha256(path)},
        }
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path / "input_turn_signal.jsonl", input_rows)
    _write_jsonl(output_path / "scene_klt_turn_signal.jsonl", scene_rows)
    if pose_angles:
        _write_jsonl(output_path / "cursor_pose_turn_signal.jsonl", pose_angles)
    (output_path / "turn_evidence.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    lines = [
        "TURN RESPONSE EVIDENCE",
        "",
        "Input = control intent; KLT = observed camera motion; pose = candidate player heading.",
        "Signals are aligned by timestamp with fitted lag and sign, never by equal frame index.",
        "Pose is called wrong only when independent evidence corroborates the verdict.",
        "",
    ]
    for key, label in (
        ("input_to_scene", "Raw input -> KLT scene"),
        ("input_to_pose", "Raw input -> cursor pose"),
        ("scene_to_pose", "KLT scene -> cursor pose"),
    ):
        if key not in result:
            continue
        item = result[key]
        alignment = item["alignment"]
        onset = item["onset_lag_ms"]
        wrong = item["wrong_direction_duration_ms"]
        settling = item["settling_time_ms"]
        lines.extend(
            [
                label,
                "Sign / fitted lag / correlation: {} / {:.1f} ms / {:.3f}".format(
                    alignment["sign_relation"],
                    alignment["lag_ms"],
                    alignment["correlation"],
                ),
                "Sharp reversals responded: {} / {}".format(
                    item["responded_reversal_count"], item["sharp_reversal_count"]
                ),
                "Onset lag median / P95: {} / {} ms".format(
                    "n/a" if onset["median"] is None else "{:.1f}".format(onset["median"]),
                    "n/a" if onset["p95"] is None else "{:.1f}".format(onset["p95"]),
                ),
                "Wrong-direction duration median / P95: {} / {} ms".format(
                    "n/a" if wrong["median"] is None else "{:.1f}".format(wrong["median"]),
                    "n/a" if wrong["p95"] is None else "{:.1f}".format(wrong["p95"]),
                ),
                "Settling time median / P95: {} / {} ms".format(
                    "n/a" if settling["median"] is None else "{:.1f}".format(settling["median"]),
                    "n/a" if settling["p95"] is None else "{:.1f}".format(settling["p95"]),
                ),
                "",
            ]
        )
    if result.get("pose_direction_consistency"):
        lines.append(
            "Pose direction verdict: {}".format(
                result["pose_direction_consistency"]["verdict"]
            )
        )
    for name, candidate in result["candidate_turn_response"].items():
        lines.extend(["", "Candidate: {}".format(name)])
        for evidence_name in ("input", "scene_klt"):
            response = candidate[evidence_name]
            alignment = response["alignment"]
            onset = response["onset_lag_ms"]
            lines.append(
                "{}: sign {}, lag {:.1f} ms, corr {:.3f}, reversals {}/{}, onset P95 {} ms".format(
                    evidence_name,
                    alignment["sign_relation"],
                    alignment["lag_ms"],
                    alignment["correlation"],
                    response["responded_reversal_count"],
                    response["sharp_reversal_count"],
                    "n/a" if onset["p95"] is None else "{:.1f}".format(onset["p95"]),
                )
            )
    (output_path / "REPORT.txt").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-hz", type=float, default=15.0)
    parser.add_argument("--cursor-calibration", type=Path)
    parser.add_argument("--focal-ratio", type=float, default=0.9)
    parser.add_argument("--maximum-scene-width", type=int, default=960)
    parser.add_argument(
        "--excluded-rect",
        action="append",
        default=[],
        help="UI exclusion as x0,y0,x1,y1 fractions; may be repeated",
    )
    parser.add_argument(
        "--pose-method",
        choices=CursorPoseEstimator.POSE_METHODS,
        default="angular_projection_ncc_parabolic",
    )
    parser.add_argument(
        "--gaussian-fit-method",
        choices=CursorPoseEstimator.GAUSSIAN_FIT_METHODS,
        default="cascade",
    )
    parser.add_argument(
        "--validation-policy",
        choices=CursorPoseEstimator.VALIDATION_POLICIES,
        default="minimal",
    )
    parser.add_argument("--candidate", action="append", default=[])
    args = parser.parse_args()
    excluded_rects = []
    for value in args.excluded_rect:
        fields = tuple(float(item) for item in value.split(","))
        if len(fields) != 4:
            parser.error("--excluded-rect needs x0,y0,x1,y1")
        excluded_rects.append(fields)
    candidates = {}
    for value in args.candidate:
        if "=" not in value:
            parser.error("--candidate needs NAME=SIGNAL.jsonl")
        name, path = value.split("=", 1)
        candidates[name] = Path(path)
    extract(
        args.session,
        args.output,
        sample_hz=args.sample_hz,
        cursor_calibration=args.cursor_calibration,
        focal_ratio=args.focal_ratio,
        maximum_scene_width=args.maximum_scene_width,
        excluded_rects=excluded_rects,
        pose_method=args.pose_method,
        gaussian_fit_method=args.gaussian_fit_method,
        validation_policy=args.validation_policy,
        candidates=candidates,
    )


if __name__ == "__main__":
    main()
