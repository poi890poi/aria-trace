"""Causal recorded-video benchmark for route-assisted map tracking.

This module is deliberately outside the production live tracker.  It compares
small route-tracing mechanisms without changing workbench behavior or artifact
schemas.  A demonstrated route may propose a bounded map search, but it never
supplies the reported pose or the post-run compliance score.
"""

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from acquisition.live_tracker import MinimapExtractor, _gradient
from acquisition.map_layers import LayeredGlobalLocalizer
from acquisition.session import SessionReader
from replay.route_similarity import route_similarity_report
from replay.route_tracking import RouteTrackingPackage, describe_minimap


VARIANTS = (
    "route_descriptor",
    "route_refine_top1",
    "route_refine_top3",
    "continuous_local",
    "continuous_gated",
)


@dataclass
class TraceResult:
    valid: bool
    x: Optional[float]
    y: Optional[float]
    score: float
    margin: float
    source: str
    route_state_index: Optional[int] = None
    mode_id: Optional[str] = None
    measurement_accepted: bool = True


def _percentiles(values) -> dict:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not len(array):
        return {"mean": None, "median": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _loss_metrics(valid_flags) -> dict:
    episodes = []
    current = 0
    for valid in valid_flags:
        if valid:
            if current:
                episodes.append(current)
                current = 0
        else:
            current += 1
    if current:
        episodes.append(current)
    return {
        "tracked_fraction": float(np.mean(valid_flags)) if valid_flags else 0.0,
        "loss_episode_count": len(episodes),
        "longest_loss_frames": max(episodes, default=0),
        "lost_frame_count": int(sum(episodes)),
    }


class CausalRouteTracer:
    """One-frame-in, one-pose-out experimental route tracer.

    The class has no clock from the demonstration and does not consume route
    heading, motion vectors, or progress.  Route descriptors are only visual
    search proposals.  Every refined pose comes from the current mini-map's
    correlation against the map atlas.
    """

    def __init__(
        self,
        package: RouteTrackingPackage,
        atlas: LayeredGlobalLocalizer,
        variant: str,
        *,
        score_min: float = 0.55,
        recovery_radius_px: float = 55.0,
        local_radius_px: float = 18.0,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError("Unknown route tracer variant: {}".format(variant))
        self.package = package
        self.atlas = atlas
        self.variant = variant
        self.score_min = float(score_min)
        self.recovery_radius_px = float(recovery_radius_px)
        self.local_radius_px = float(local_radius_px)
        self.previous_xy = None
        self.previous_time_ns = None
        motion = package.manifest.get("motion_envelope") or {}
        speed = (motion.get("speed_px_s") or {}).get("p99")
        self.continuity_speed_limit_px_s = max(120.0, 4.0 * float(speed or 0.0))

    def _route_candidates(self, descriptor, top_k: int):
        # No previous_state_index: the replay may pause, reverse, deviate, or
        # travel at a completely different rate from the demonstration.
        return self.package.candidates(descriptor, top_k=top_k)

    def _refine_centers(self, gradient, mask, centers, radius_px, source):
        hypotheses = []
        for center, route_state_index in centers:
            for mode_id, localizer in self.atlas.localizers.items():
                match = self.atlas._observe_one_mode(
                    localizer, gradient, mask, center, radius_px
                )
                if not match.get("valid"):
                    continue
                offset = match["best_offset_canonical_xy"]
                hypotheses.append(
                    {
                        "x": float(center[0]) + float(offset[0]),
                        "y": float(center[1]) + float(offset[1]),
                        "score": float(match["score"]),
                        "mode_id": str(mode_id),
                        "route_state_index": route_state_index,
                    }
                )
        if not hypotheses:
            return TraceResult(False, None, None, 0.0, 0.0, source)
        hypotheses.sort(key=lambda item: item["score"], reverse=True)
        best = hypotheses[0]
        distinct_scores = [
            item["score"]
            for item in hypotheses[1:]
            if math.hypot(item["x"] - best["x"], item["y"] - best["y"])
            >= 8.0
        ]
        second = max(distinct_scores, default=0.0)
        valid = bool(best["score"] >= self.score_min)
        return TraceResult(
            valid,
            best["x"] if valid else None,
            best["y"] if valid else None,
            best["score"],
            best["score"] - second,
            source,
            best["route_state_index"],
            best["mode_id"],
        )

    def _route_refine(self, observation, gradient, mask, top_k: int):
        descriptor = describe_minimap(observation, mask)
        candidates = self._route_candidates(descriptor, top_k)
        centers = [
            (item["state"]["canonical_xy"], int(item["state_index"]))
            for item in candidates
        ]
        return self._refine_centers(
            gradient, mask, centers, self.recovery_radius_px, "route_recovery"
        )

    def track(self, observation, mask, session_time_ns=None) -> TraceResult:
        if self.variant == "route_descriptor":
            candidates = self._route_candidates(
                describe_minimap(observation, mask), top_k=2
            )
            if not candidates:
                return TraceResult(False, None, None, 0.0, 0.0, "route_descriptor")
            best = candidates[0]
            second = candidates[1]["score"] if len(candidates) > 1 else -1.0
            margin = float(best["score"] - second)
            valid = bool(best["score"] >= 0.25 and margin >= 0.015)
            state = best["state"]
            result = TraceResult(
                valid,
                float(state["canonical_xy"][0]) if valid else None,
                float(state["canonical_xy"][1]) if valid else None,
                float(best["score"]),
                margin,
                "route_descriptor",
                int(best["state_index"]),
                str(state["mode_id"]),
            )
        else:
            gradient = _gradient(observation)
            if self.variant == "route_refine_top1":
                result = self._route_refine(observation, gradient, mask, 1)
            elif self.variant == "route_refine_top3":
                result = self._route_refine(observation, gradient, mask, 3)
            else:
                result = TraceResult(False, None, None, 0.0, 0.0, "local")
                if self.previous_xy is not None:
                    result = self._refine_centers(
                        gradient,
                        mask,
                        [(self.previous_xy, None)],
                        self.local_radius_px,
                        "local",
                    )
                if not result.valid:
                    result = self._route_refine(observation, gradient, mask, 3)
        if (
            self.variant == "continuous_gated"
            and result.valid
            and self.previous_xy is not None
            and self.previous_time_ns is not None
            and session_time_ns is not None
        ):
            elapsed_s = max(
                0.0, (int(session_time_ns) - int(self.previous_time_ns)) / 1.0e9
            )
            maximum_step = max(6.0, self.continuity_speed_limit_px_s * elapsed_s)
            step = math.hypot(
                float(result.x) - self.previous_xy[0],
                float(result.y) - self.previous_xy[1],
            )
            if step > maximum_step:
                result = TraceResult(
                    True,
                    self.previous_xy[0],
                    self.previous_xy[1],
                    result.score,
                    result.margin,
                    "continuity_hold",
                    result.route_state_index,
                    result.mode_id,
                    measurement_accepted=False,
                )
        if result.valid:
            if result.measurement_accepted:
                self.previous_xy = (float(result.x), float(result.y))
            if session_time_ns is not None:
                self.previous_time_ns = int(session_time_ns)
        return result


def _reference_errors(rows, reference_package, session_id):
    if reference_package is None:
        return None
    source = reference_package.manifest.get("source_session") or {}
    if str(source.get("session_id")) != str(session_id):
        return None
    states = reference_package.states
    times = np.asarray([int(item["session_time_ns"]) for item in states], np.float64)
    xs = np.asarray([float(item["canonical_xy"][0]) for item in states])
    ys = np.asarray([float(item["canonical_xy"][1]) for item in states])
    errors = []
    for row in rows:
        if not row["valid"]:
            continue
        timestamp = float(row["session_time_ns"])
        if timestamp < times[0] or timestamp > times[-1]:
            continue
        reference = np.asarray(
            [np.interp(timestamp, times, xs), np.interp(timestamp, times, ys)]
        )
        errors.append(
            float(np.linalg.norm(np.asarray([row["x"], row["y"]]) - reference))
        )
    if not errors:
        return None
    values = np.asarray(errors, dtype=np.float64)
    return {
        "role": "offline-sparse-map-localization-reference",
        "sample_count": len(errors),
        "rmse_px": float(math.sqrt(float(np.mean(values * values)))),
        "median_px": float(np.median(values)),
        "p95_px": float(np.percentile(values, 95)),
        "max_px": float(np.max(values)),
    }


def _supported_interval_metrics(rows, reference_package, session_id, demonstrated):
    if reference_package is None:
        return None
    source = reference_package.manifest.get("source_session") or {}
    if str(source.get("session_id")) != str(session_id):
        return None
    states = reference_package.states
    if not states:
        return None
    start_ns = int(states[0]["session_time_ns"])
    end_ns = int(states[-1]["session_time_ns"])
    selected = [
        row for row in rows if start_ns <= int(row["session_time_ns"]) <= end_ns
    ]
    valid = [row for row in selected if row["valid"]]
    points = [[row["x"], row["y"]] for row in valid]
    return {
        "role": "post-run-map-supported-interval-review",
        "feeds_tracker": False,
        "start_session_time_ns": start_ns,
        "end_session_time_ns": end_ns,
        "frame_count": len(selected),
        "continuity": _loss_metrics([row["valid"] for row in selected]),
        "visual_measurement_continuity": _loss_metrics(
            [row["measurement_accepted"] for row in selected]
        ),
        "algorithm_latency_ms": _percentiles(
            [row["algorithm_elapsed_ms"] for row in selected]
        ),
        "route_compliance": route_similarity_report(
            points,
            demonstrated,
            float(reference_package.manifest.get("corridor_radius_px") or 35.0),
        ),
    }


def benchmark_session(
    session_path: Path,
    package_path: Path,
    atlas_path: Path,
    minimap_config: dict,
    minimap_calibration: dict,
    variant: str,
    *,
    score_min: float = 0.55,
    reference_package_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> dict:
    reader = SessionReader(Path(session_path))
    records = reader.frames_by_stream["main"]
    package = RouteTrackingPackage(Path(package_path))
    reference_package = (
        RouteTrackingPackage(Path(reference_package_path))
        if reference_package_path
        else None
    )
    extractor = MinimapExtractor(minimap_config["crop_xywh"], minimap_calibration)
    atlas = LayeredGlobalLocalizer(Path(atlas_path))
    tracer = CausalRouteTracer(package, atlas, variant, score_min=score_min)
    capture = cv2.VideoCapture(str(reader.video_path("main")))
    if not capture.isOpened():
        atlas.close()
        raise RuntimeError("Could not open video: {}".format(reader.video_path("main")))
    rows = []
    decode_times = []
    try:
        for record in records:
            decode_started = time.perf_counter_ns()
            ok, frame = capture.read()
            decode_times.append((time.perf_counter_ns() - decode_started) / 1.0e6)
            if not ok:
                raise RuntimeError(
                    "Video ended before frame {}".format(record["frame_index"])
                )
            observation, mask = extractor.extract(frame)
            started = time.perf_counter_ns()
            result = tracer.track(
                observation, mask, session_time_ns=record["session_time_ns"]
            )
            elapsed_ms = (time.perf_counter_ns() - started) / 1.0e6
            rows.append(
                {
                    "frame_index": int(record["frame_index"]),
                    "session_time_ns": int(record["session_time_ns"]),
                    "valid": bool(result.valid),
                    "measurement_accepted": bool(
                        result.valid and result.measurement_accepted
                    ),
                    "x": result.x,
                    "y": result.y,
                    "score": float(result.score),
                    "margin": float(result.margin),
                    "source": result.source,
                    "route_state_index": result.route_state_index,
                    "mode_id": result.mode_id,
                    "algorithm_elapsed_ms": elapsed_ms,
                }
            )
    finally:
        capture.release()
        atlas.close()

    valid_rows = [row for row in rows if row["valid"]]
    points = [[row["x"], row["y"]] for row in valid_rows]
    demonstrated = [state["canonical_xy"] for state in package.states]
    adjacent_jumps = []
    for first, second in zip(rows, rows[1:]):
        if first["valid"] and second["valid"]:
            adjacent_jumps.append(
                math.hypot(second["x"] - first["x"], second["y"] - first["y"])
            )
    algorithm_times = [row["algorithm_elapsed_ms"] for row in rows]
    source_counts = {}
    for row in rows:
        source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1
    report = {
        "schema_version": "1.0",
        "experiment": "causal-route-tracer-poc",
        "variant": variant,
        "causal_constraints": {
            "demo_timing_used": False,
            "demo_motion_vector_used": False,
            "demo_heading_used": False,
            "route_pose_used": variant == "route_descriptor",
            "route_role": (
                "pose baseline only"
                if variant == "route_descriptor"
                else "visual bounded-search proposal only"
            ),
        },
        "session": {
            "path": str(Path(session_path)),
            "session_id": reader.manifest.get("session_id"),
            "frame_count": len(rows),
        },
        "route_package": str(Path(package_path)),
        "atlas": str(Path(atlas_path)),
        "parameters": {
            "score_min": float(score_min),
            "recovery_radius_px": tracer.recovery_radius_px,
            "local_radius_px": tracer.local_radius_px,
            "continuity_speed_limit_px_s": tracer.continuity_speed_limit_px_s,
        },
        "algorithm_latency_ms": _percentiles(algorithm_times),
        "algorithm_throughput_fps": (
            1000.0 / float(np.mean(algorithm_times)) if algorithm_times else 0.0
        ),
        "decode_latency_ms": _percentiles(decode_times),
        "continuity": _loss_metrics([row["valid"] for row in rows]),
        "visual_measurement_continuity": _loss_metrics(
            [row["measurement_accepted"] for row in rows]
        ),
        "source_frame_counts": source_counts,
        "adjacent_pose_jump_px": _percentiles(adjacent_jumps),
        "route_compliance": route_similarity_report(
            points,
            demonstrated,
            float(package.manifest.get("corridor_radius_px") or 35.0),
        ),
        "reference_pose_error": _reference_errors(
            rows, reference_package, reader.manifest.get("session_id")
        ),
        "map_supported_interval": _supported_interval_metrics(
            rows,
            reference_package,
            reader.manifest.get("session_id"),
            demonstrated,
        ),
        "acceptance": {
            "target_algorithm_fps": 30.0,
            "target_frame_budget_ms": 1000.0 / 30.0,
            "meets_mean_30fps_budget": bool(
                algorithm_times and float(np.mean(algorithm_times)) <= 1000.0 / 30.0
            ),
            "meets_p95_30fps_budget": bool(
                algorithm_times
                and float(np.percentile(algorithm_times, 95)) <= 1000.0 / 30.0
            ),
        },
        "rows_file": "telemetry.jsonl" if output_path else None,
    }
    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / "telemetry.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        (output_path / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return report


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--route-package", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--minimap-calibration", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--score-min", type=float, default=0.55)
    parser.add_argument("--reference-package", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    calibration = json.loads(args.minimap_calibration.read_text(encoding="utf-8"))
    report = benchmark_session(
        args.session,
        args.route_package,
        args.atlas,
        profile["minimap_calibration"],
        calibration,
        args.variant,
        score_min=args.score_min,
        reference_package_path=args.reference_package,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
