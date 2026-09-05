"""Benchmark rigid placement and hard, non-averaging full-map seam methods."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from acquisition.session import SessionReader
from aria_trace.services.mapping.stitching import _estimate_rigid_translation


METHODS = (
    "current_feather_owner",
    "pose_graph_feather_owner",
    "pose_graph_dp_color_grad",
    "pose_graph_graphcut_color",
    "pose_graph_graphcut_color_grad",
)


@dataclass(frozen=True)
class Placement:
    origin_xy: np.ndarray
    base_xy: tuple[int, int]
    fraction_xy: np.ndarray
    roi_wh: tuple[int, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], text=True
        ).strip()
    )


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


def _load_frames(source_path: Path, indices: list[int], crop_xywh) -> list[np.ndarray]:
    reader = SessionReader(source_path)
    capture = cv2.VideoCapture(str(reader.video_path("main")))
    wanted = set(indices)
    found = {}
    frame_index = 0
    x, y, width, height = map(int, crop_xywh)
    try:
        while wanted:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in wanted:
                found[frame_index] = frame[y : y + height, x : x + width].copy()
                wanted.remove(frame_index)
            frame_index += 1
    finally:
        capture.release()
    if wanted:
        raise RuntimeError("Could not decode selected source frames: {}".format(sorted(wanted)))
    return [found[index] for index in indices]


def _baseline_positions(stitch: dict) -> np.ndarray:
    selected = stitch["selected_frame_indices"]
    registrations = stitch["registrations"]
    positions_by_frame = {int(selected[0]): np.zeros(2, np.float64)}
    last_position = positions_by_frame[int(selected[0])]
    for registration in registrations:
        from_frame = int(registration["from_frame"])
        to_frame = int(registration["to_frame"])
        reference_position = positions_by_frame.get(from_frame)
        if reference_position is None:
            raise RuntimeError(
                "Registration references an unknown source frame: {}".format(from_frame)
            )
        if registration["accepted"]:
            position = reference_position - np.asarray(
                registration["content_shift_xy_px"], dtype=np.float64
            )
        else:
            position = last_position.copy()
        positions_by_frame[to_frame] = position
        last_position = position
    missing = [index for index in selected if int(index) not in positions_by_frame]
    if missing:
        raise RuntimeError("No recorded position for selected frames: {}".format(missing))
    return np.asarray([positions_by_frame[int(index)] for index in selected])


def _candidate_loop_pairs(positions: np.ndarray, viewport_wh, per_frame=3):
    width, height = viewport_wh
    pairs = set()
    for first in range(len(positions)):
        candidates = []
        for second in range(first + 12, len(positions)):
            delta = np.abs(positions[second] - positions[first])
            if delta[0] >= width * 0.72 or delta[1] >= height * 0.72:
                continue
            overlap_fraction = (1.0 - delta[0] / width) * (1.0 - delta[1] / height)
            if overlap_fraction < 0.32:
                continue
            candidates.append((float(np.linalg.norm(delta)), second))
        for _, second in sorted(candidates)[:per_frame]:
            pairs.add((first, second))
    return sorted(pairs)


def _discover_loop_edges(frames, positions, maximum_edges=900):
    height, width = frames[0].shape[:2]
    pairs = _candidate_loop_pairs(positions, (width, height))
    if len(pairs) > maximum_edges:
        sample = np.linspace(0, len(pairs) - 1, maximum_edges).round().astype(int)
        pairs = [pairs[index] for index in np.unique(sample)]
    edges = []
    for ordinal, (first, second) in enumerate(pairs):
        rigid = _estimate_rigid_translation(frames[first], frames[second])
        shift = np.asarray(rigid["shift_xy_px"], dtype=np.float64)
        expected = positions[first] - positions[second]
        gate_error = float(np.linalg.norm(shift - expected))
        accepted = bool(
            rigid["response"] >= 0.10
            and rigid["spatially_coherent"]
            and gate_error <= 18.0
        )
        if accepted:
            edges.append(
                {
                    "first": first,
                    "second": second,
                    "delta_xy": (-shift).tolist(),
                    "response": float(rigid["response"]),
                    "discovery_gate_error_px": gate_error,
                    "held_out": ((first * 1009 + second * 9176 + ordinal) % 5 == 0),
                }
            )
    return edges


def _sequential_edges(stitch: dict):
    rows = []
    local_by_frame = {
        int(frame_index): local_index
        for local_index, frame_index in enumerate(stitch["selected_frame_indices"])
    }
    for registration in stitch["registrations"]:
        first = local_by_frame.get(int(registration["from_frame"]))
        second = local_by_frame.get(int(registration["to_frame"]))
        if first is None or second is None or not registration["accepted"]:
            continue
        rows.append(
            {
                "first": first,
                "second": second,
                "delta_xy": (-np.asarray(registration["content_shift_xy_px"])).tolist(),
                "response": float(registration["response"]),
                "held_out": False,
            }
        )
    return rows


def _fit_pose_graph(count: int, edges: list[dict], iterations=5) -> np.ndarray:
    training = [edge for edge in edges if not edge.get("held_out")]
    weights = np.asarray([max(edge["response"], 0.05) for edge in training])
    positions = np.zeros((count, 2), np.float64)
    for _ in range(iterations):
        rows = len(training) + 1
        matrix = np.zeros((rows, count), np.float64)
        target = np.zeros((rows, 2), np.float64)
        for row, (edge, weight) in enumerate(zip(training, weights)):
            scale = float(np.sqrt(weight))
            matrix[row, edge["first"]] = -scale
            matrix[row, edge["second"]] = scale
            target[row] = np.asarray(edge["delta_xy"]) * scale
        matrix[-1, 0] = 10.0
        positions = np.linalg.lstsq(matrix, target, rcond=None)[0]
        residuals = np.asarray(
            [
                np.linalg.norm(
                    positions[edge["second"]]
                    - positions[edge["first"]]
                    - np.asarray(edge["delta_xy"])
                )
                for edge in training
            ]
        )
        huber = np.minimum(1.0, 2.0 / np.maximum(residuals, 1e-6))
        weights = np.asarray(
            [max(edge["response"], 0.05) for edge in training]
        ) * huber
    return positions


def _residual_summary(positions: np.ndarray, edges: list[dict], held_out: bool):
    values = []
    for edge in edges:
        if bool(edge.get("held_out")) != held_out:
            continue
        residual = positions[edge["second"]] - positions[edge["first"]]
        values.append(float(np.linalg.norm(residual - np.asarray(edge["delta_xy"]))))
    if not values:
        return {"count": 0, "median_px": None, "p95_px": None, "worst_px": None}
    return {
        "count": len(values),
        "median_px": float(np.median(values)),
        "p95_px": float(np.percentile(values, 95)),
        "worst_px": float(np.max(values)),
    }


def _placements(positions: np.ndarray, viewport_wh):
    width, height = viewport_wh
    minimum = np.floor(positions.min(axis=0)).astype(int)
    maximum = np.ceil(positions.max(axis=0)).astype(int)
    canvas_wh = (
        int(maximum[0] - minimum[0] + width),
        int(maximum[1] - minimum[1] + height),
    )
    rows = []
    for position in positions:
        origin = position - minimum
        base = np.floor(origin).astype(int)
        fraction = origin - base
        roi_wh = (
            min(width + 1, canvas_wh[0] - int(base[0])),
            min(height + 1, canvas_wh[1] - int(base[1])),
        )
        rows.append(
            Placement(
                origin_xy=origin,
                base_xy=(int(base[0]), int(base[1])),
                fraction_xy=fraction,
                roi_wh=roi_wh,
            )
        )
    return rows, canvas_wh


def _warp(image: np.ndarray, placement: Placement, interpolation=cv2.INTER_LINEAR):
    transform = np.float32(
        [
            [1.0, 0.0, float(placement.fraction_xy[0])],
            [0.0, 1.0, float(placement.fraction_xy[1])],
        ]
    )
    return cv2.warpAffine(
        image,
        transform,
        placement.roi_wh,
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
    )


def _seam_masks(frames, placements, method, seam_scale):
    if method.endswith("feather_owner"):
        return None, 0.0, 0
    finder_name = method.removeprefix("pose_graph_")
    height, width = frames[0].shape[:2]
    small_wh = (
        max(32, int(round(width * seam_scale))),
        max(32, int(round(height * seam_scale))),
    )
    images = [
        cv2.resize(frame, small_wh, interpolation=cv2.INTER_AREA).astype(np.float32)
        for frame in frames
    ]
    corners = [
        (
            int(round(placement.origin_xy[0] * seam_scale)),
            int(round(placement.origin_xy[1] * seam_scale)),
        )
        for placement in placements
    ]
    if finder_name == "voronoi":
        finder = cv2.detail_VoronoiSeamFinder()
    elif finder_name == "dp_color_grad":
        finder = cv2.detail_DpSeamFinder("COLOR_GRAD")
    elif finder_name == "graphcut_color":
        finder = cv2.detail_GraphCutSeamFinder("COST_COLOR")
    elif finder_name == "graphcut_color_grad":
        finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
    else:
        raise ValueError("Unknown seam method: {}".format(method))
    started = time.perf_counter()
    canvas_width = max(corner[0] + small_wh[0] for corner in corners)
    canvas_height = max(corner[1] + small_wh[1] for corner in corners)
    owner = np.full((canvas_height, canvas_width), -1, np.int32)
    composite = np.zeros((canvas_height, canvas_width, 3), np.float32)
    failures = 0
    for index, (image, corner) in enumerate(zip(images, corners)):
        ox, oy = corner
        roi_width = min(small_wh[0], canvas_width - ox)
        roi_height = min(small_wh[1], canvas_height - oy)
        owner_roi = owner[oy : oy + roi_height, ox : ox + roi_width]
        composite_roi = composite[oy : oy + roi_height, ox : ox + roi_width]
        incoming = image[:roi_height, :roi_width]
        old_mask = (owner_roi >= 0).astype(np.uint8) * 255
        new_mask = np.full((roi_height, roi_width), 255, np.uint8)
        if not np.any(old_mask):
            replace = new_mask > 0
        else:
            # Give both sources exclusive support. Cropping the old mosaic to
            # exactly the incoming tile leaves no old-only terminal and causes
            # every OpenCV finder to degenerate to the same first-owner mask.
            padding = 12
            ux0 = max(0, ox - padding)
            uy0 = max(0, oy - padding)
            ux1 = min(canvas_width, ox + roi_width + padding)
            uy1 = min(canvas_height, oy + roi_height + padding)
            old_image = composite[uy0:uy1, ux0:ux1].copy()
            old_union_mask = (owner[uy0:uy1, ux0:ux1] >= 0).astype(np.uint8) * 255
            try:
                result = finder.find(
                    [old_image, incoming],
                    [(ux0, uy0), (ox, oy)],
                    [old_union_mask, new_mask],
                )
                old_result = result[0].get() if hasattr(result[0], "get") else np.asarray(result[0])
                new_result = result[1].get() if hasattr(result[1], "get") else np.asarray(result[1])
                dx = ox - ux0
                dy = oy - uy0
                old_at_incoming = old_result[
                    dy : dy + roi_height, dx : dx + roi_width
                ]
                replace = (new_result[:roi_height, :roi_width] > 0) & (
                    old_at_incoming == 0
                )
                replace |= (owner_roi < 0) & (new_result > 0)
            except cv2.error:
                # Keep the prior owner in overlap and use the new frame only for
                # uncovered pixels. The failure count makes this explicit rather
                # than silently presenting the fallback as the tested method.
                failures += 1
                replace = owner_roi < 0
        composite_roi[replace] = incoming[replace]
        owner_roi[replace] = index
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    materialized = []
    for index, corner in enumerate(corners):
        ox, oy = corner
        array = np.zeros((small_wh[1], small_wh[0]), np.uint8)
        roi_width = min(small_wh[0], canvas_width - ox)
        roi_height = min(small_wh[1], canvas_height - oy)
        array[:roi_height, :roi_width] = (
            owner[oy : oy + roi_height, ox : ox + roi_width] == index
        ).astype(np.uint8) * 255
        materialized.append(
            cv2.resize(array, (width, height), interpolation=cv2.INTER_NEAREST)
        )
    return materialized, elapsed_ms, failures


def _compose(frames, positions, method, seam_scale):
    height, width = frames[0].shape[:2]
    placements, canvas_wh = _placements(positions, (width, height))
    masks, seam_plan_ms, seam_finder_failures = _seam_masks(
        frames, placements, method, seam_scale
    )
    mosaic = np.zeros((canvas_wh[1], canvas_wh[0], 3), np.uint8)
    owner = np.full((canvas_wh[1], canvas_wh[0]), -1, np.int32)
    best = np.zeros((canvas_wh[1], canvas_wh[0]), np.float32)
    feature = np.zeros((canvas_wh[1], canvas_wh[0]), np.uint8)
    feather = np.maximum(cv2.createHanningWindow((width, height), cv2.CV_32F), 0.08)
    started = time.perf_counter()
    for index, (frame, placement) in enumerate(zip(frames, placements)):
        warped = _warp(frame, placement)
        score_source = feather
        if masks is not None:
            score_source = feather * (masks[index] > 0).astype(np.float32)
        warped_score = _warp(score_source, placement)
        ox, oy = placement.base_xy
        roi_width, roi_height = placement.roi_wh
        target_score = best[oy : oy + roi_height, ox : ox + roi_width]
        replace = warped_score > target_score
        mosaic_roi = mosaic[oy : oy + roi_height, ox : ox + roi_width]
        owner_roi = owner[oy : oy + roi_height, ox : ox + roi_width]
        mosaic_roi[replace] = warped[replace]
        owner_roi[replace] = index
        target_score[replace] = warped_score[replace]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 120)
        protected = cv2.dilate(edges, np.ones((9, 9), np.uint8))
        warped_protected = _warp(protected, placement, cv2.INTER_NEAREST)
        feature_roi = feature[oy : oy + roi_height, ox : ox + roi_width]
        np.maximum(feature_roi, warped_protected, out=feature_roi)

    # Low-resolution seam planning can leave isolated unowned pixels. Fill only
    # those from the current deterministic owner; never blend source pixels.
    if masks is not None and np.any(owner < 0):
        fallback, fallback_owner, _, _, _ = _compose(
            frames, positions, "pose_graph_feather_owner", seam_scale
        )
        missing = (owner < 0) & (fallback_owner >= 0)
        mosaic[missing] = fallback[missing]
        owner[missing] = fallback_owner[missing]
    compose_ms = (time.perf_counter() - started) * 1000.0
    return mosaic, owner, feature, seam_plan_ms + compose_ms, seam_finder_failures


def _seam_mask(owner: np.ndarray):
    valid = owner >= 0
    seam = np.zeros(owner.shape, np.uint8)
    horizontal = valid[:, 1:] & valid[:, :-1] & (owner[:, 1:] != owner[:, :-1])
    vertical = valid[1:, :] & valid[:-1, :] & (owner[1:, :] != owner[:-1, :])
    seam[:, 1:][horizontal] = 255
    seam[1:, :][vertical] = 255
    return seam


def _metrics(mosaic, owner, feature, elapsed_ms, seam_finder_failures):
    valid = owner >= 0
    seam = _seam_mask(owner)
    seam_bool = seam > 0
    gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    jump = np.zeros(gray.shape, np.float32)
    jump[:, 1:] = np.maximum(
        jump[:, 1:], np.abs(gray[:, 1:].astype(np.float32) - gray[:, :-1])
    )
    jump[1:, :] = np.maximum(
        jump[1:, :], np.abs(gray[1:, :].astype(np.float32) - gray[:-1, :])
    )
    seam_jumps = jump[seam_bool]
    valid_gray = gray[valid]
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F)[valid].var()) if np.any(valid) else 0.0
    return {
        "covered_pixels": int(np.count_nonzero(valid)),
        "coverage_fraction": float(np.count_nonzero(valid) / valid.size),
        "unowned_in_bounding_canvas": int(np.count_nonzero(~valid)),
        "mixed_source_pixels": 0,
        "seam_length_px": int(np.count_nonzero(seam_bool)),
        "protected_feature_seam_fraction": float(
            np.count_nonzero(seam_bool & (feature > 0)) / max(np.count_nonzero(seam_bool), 1)
        ),
        "seam_jump_mean_luma": float(np.mean(seam_jumps)) if seam_jumps.size else 0.0,
        "seam_jump_p95_luma": float(np.percentile(seam_jumps, 95)) if seam_jumps.size else 0.0,
        "seam_jump_worst_luma": float(np.max(seam_jumps)) if seam_jumps.size else 0.0,
        "valid_luma_mean": float(np.mean(valid_gray)) if valid_gray.size else 0.0,
        "output_laplacian_variance": sharpness,
        "offline_runtime_ms": float(elapsed_ms),
        "seam_finder_failures": int(seam_finder_failures),
    }, seam


def _write_candidate(output_path: Path, name, mosaic, seam, feature, metrics):
    target = output_path / name
    target.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target / "mosaic.png"), mosaic)
    overlay = mosaic.copy()
    overlay[feature > 0] = (
        overlay[feature > 0].astype(np.float32) * 0.72
        + np.asarray([80, 60, 0], np.float32) * 0.28
    ).astype(np.uint8)
    thick = cv2.dilate(seam, np.ones((5, 5), np.uint8)) > 0
    overlay[thick] = (30, 30, 255)
    cv2.imwrite(str(target / "seam_overlay.png"), overlay)
    scale = min(1.0, 1400.0 / mosaic.shape[1])
    preview = cv2.resize(
        mosaic, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
    )
    cv2.imwrite(str(target / "preview.jpg"), preview, [cv2.IMWRITE_JPEG_QUALITY, 92])
    _atomic_json(target / "metrics.json", metrics)


def _comparison_image(output_path: Path, rows: list[dict]):
    cards = []
    for row in rows:
        preview = cv2.imread(str(output_path / row["method"] / "preview.jpg"))
        preview = cv2.resize(preview, (700, 620), interpolation=cv2.INTER_AREA)
        card = np.full((735, 700, 3), 22, np.uint8)
        card[:620] = preview
        metrics = row["metrics"]
        cv2.putText(card, row["method"], (18, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
        text = "jump p95 {:.1f} | feature {:.1%} | seam {:,} | {:.0f} ms".format(
            metrics["seam_jump_p95_luma"],
            metrics["protected_feature_seam_fraction"],
            metrics["seam_length_px"],
            metrics["offline_runtime_ms"],
        )
        cv2.putText(card, text, (18, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 220, 245), 1, cv2.LINE_AA)
        cards.append(card)
    width = 1400
    height = ((len(cards) + 1) // 2) * 735
    canvas = np.full((height, width, 3), 12, np.uint8)
    for index, card in enumerate(cards):
        y = (index // 2) * 735
        x = (index % 2) * 700
        canvas[y : y + 735, x : x + 700] = card
    cv2.imwrite(str(output_path / "comparison.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])


def run(artifact_path: Path, output_path: Path, seam_scale: float) -> dict:
    artifact_path = artifact_path.resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    stitch = json.loads(artifact_path.read_text(encoding="utf-8"))
    source_path = Path(stitch["provenance"]["source_session_path"])
    selected = [int(value) for value in stitch["selected_frame_indices"]]
    frames = _load_frames(source_path, selected, stitch["viewport_crop_xywh"])
    baseline = _baseline_positions(stitch)
    loops = _discover_loop_edges(frames, baseline)
    sequential = _sequential_edges(stitch)
    optimized = _fit_pose_graph(len(frames), sequential + loops)
    heldout_before = _residual_summary(baseline, loops, held_out=True)
    heldout_after = _residual_summary(optimized, loops, held_out=True)
    rows = []
    for method in METHODS:
        positions = baseline if method == "current_feather_owner" else optimized
        mosaic, owner, feature, elapsed_ms, seam_finder_failures = _compose(
            frames, positions, method, seam_scale
        )
        metrics, seam = _metrics(
            mosaic, owner, feature, elapsed_ms, seam_finder_failures
        )
        row = {"method": method, "metrics": metrics}
        rows.append(row)
        _write_candidate(output_path, method, mosaic, seam, feature, metrics)
    _comparison_image(output_path, rows)
    result = {
        "schema_version": "1.0",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact_path": str(artifact_path),
        "artifact_sha256": _sha256(artifact_path),
        "source_session_path": str(source_path.resolve()),
        "source_video_sha256": _sha256(SessionReader(source_path).video_path("main")),
        "selected_frame_count": len(frames),
        "seam_scale": seam_scale,
        "git_revision": _git_revision(),
        "tracked_worktree_dirty": _git_dirty(),
        "environment": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "platform": platform.platform(),
        },
        "placement": {
            "loop_edges_total": len(loops),
            "loop_edges_training": sum(not edge["held_out"] for edge in loops),
            "loop_edges_held_out": sum(edge["held_out"] for edge in loops),
            "baseline_heldout_residual": heldout_before,
            "pose_graph_heldout_residual": heldout_after,
        },
        "methods": rows,
        "interpretation_limits": [
            "Held-out loop edges are independent of fitting but candidate discovery uses baseline footprint overlap.",
            "Protected-feature evidence covers road/street edges and other stable line detail; it is not a semantic road classifier.",
            "Seam scores compare composition on one recorded map session and require visual review before landing.",
        ],
    }
    _atomic_json(output_path / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seam-scale", type=float, default=0.25)
    args = parser.parse_args()
    result = run(args.artifact, args.output, args.seam_scale)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
