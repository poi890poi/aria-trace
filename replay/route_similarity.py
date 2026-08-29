"""Post-run route similarity reporting with no tracker feedback path."""

import json
import math
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "1.0"


def _point_to_polyline_distances(points, polyline):
    points = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    polyline = np.asarray(polyline, dtype=np.float64).reshape((-1, 2))
    if len(points) == 0 or len(polyline) < 2:
        return np.empty(0, dtype=np.float64)
    starts = polyline[:-1]
    vectors = polyline[1:] - starts
    denominators = np.sum(vectors * vectors, axis=1)
    output = []
    for offset in range(0, len(points), 256):
        chunk = points[offset : offset + 256]
        relative = chunk[:, None, :] - starts[None, :, :]
        fractions = np.sum(relative * vectors[None, :, :], axis=2)
        fractions /= np.maximum(denominators[None, :], 1.0e-12)
        fractions = np.clip(fractions, 0.0, 1.0)
        projections = starts[None, :, :] + fractions[:, :, None] * vectors[None, :, :]
        distances = np.linalg.norm(chunk[:, None, :] - projections, axis=2)
        output.extend(np.min(distances, axis=1).tolist())
    return np.asarray(output, dtype=np.float64)


def _spatially_sample(points, minimum_step_px=1.0):
    selected = []
    for point in np.asarray(points, dtype=np.float64).reshape((-1, 2)):
        if not selected or np.linalg.norm(point - selected[-1]) >= minimum_step_px:
            selected.append(point)
    return np.asarray(selected, dtype=np.float64).reshape((-1, 2))


def route_similarity_report(live_points, demonstrated_points, corridor_radius_px):
    """Compare two paths spatially, without time alignment or pose correction."""

    live = _spatially_sample(live_points)
    demonstrated = _spatially_sample(demonstrated_points)
    base = {
        "schema_version": SCHEMA_VERSION,
        "role": "post-run-review-only",
        "feeds_tracker": False,
        "alignment": "none",
        "time_synchronization": "none",
        "metric": "live-point-to-demonstrated-polyline-distance",
        "units": "canonical-map-px",
        "live_sample_count": int(len(live)),
        "demonstrated_sample_count": int(len(demonstrated)),
    }
    if len(live) < 2 or len(demonstrated) < 2:
        base.update(
            {
                "status": "unavailable",
                "reason": "need-at-least-two-spatial-samples-per-path",
            }
        )
        return base
    cross_track = _point_to_polyline_distances(live, demonstrated)
    reverse = _point_to_polyline_distances(demonstrated, live)
    corridor = max(1.0, float(corridor_radius_px))
    base.update(
        {
            "status": "complete",
            "cross_track_rmse_px": float(
                math.sqrt(float(np.mean(cross_track * cross_track)))
            ),
            "cross_track_median_px": float(np.median(cross_track)),
            "cross_track_p95_px": float(np.percentile(cross_track, 95)),
            "cross_track_max_px": float(np.max(cross_track)),
            "demonstrated_route_coverage_fraction": float(
                np.mean(reverse <= corridor)
            ),
            "coverage_radius_px": corridor,
            "interpretation": (
                "Lower RMSE means the independently estimated live path stayed "
                "closer to the demonstrated path. Coverage is reported separately."
            ),
        }
    )
    return base


def write_live_route_similarity(run_path: Path, package) -> dict:
    """Write the review-only report after tracking has fully stopped."""

    run_path = Path(run_path)
    live_points = []
    telemetry_path = run_path / "telemetry.jsonl"
    if telemetry_path.is_file():
        with telemetry_path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                pose = row.get("pose") or {}
                if pose.get("x") is not None and pose.get("y") is not None:
                    live_points.append([float(pose["x"]), float(pose["y"])])
    demonstrated = [state["canonical_xy"] for state in package.states]
    report = route_similarity_report(
        live_points,
        demonstrated,
        float(package.manifest.get("corridor_radius_px") or 35.0),
    )
    report.update(
        {
            "route_id": package.manifest.get("route_id"),
            "route_package_id": package.manifest.get("package_id"),
            "coordinate_space_id": package.manifest.get("coordinate_space_id"),
        }
    )
    report_path = run_path / "route_similarity.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(report_path)
    manifest_path = run_path / "live_tracking.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("files", {})["route_similarity"] = report_path.name
    manifest["route_similarity"] = report
    manifest_temporary = manifest_path.with_suffix(".json.tmp")
    manifest_temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest_temporary.replace(manifest_path)
    return report
