"""Post-run loss evaluation against a sparse inferred reference, never runtime input."""

import hashlib
import json
from pathlib import Path

import numpy as np


def calibrate_loss_tolerances(reference_paths, *, exclude_reference=None):
    """Estimate the reference resolution envelope without candidate outputs.

    Leave the evaluated recording out. Use reference-only leave-one-out temporal
    residuals plus the relative quantization bound for two raster matches.
    """
    excluded = Path(exclude_reference).resolve() if exclude_reference is not None else None
    modes, inputs = {}, []
    atlas_ids = set()
    for path in sorted({Path(p).resolve() for p in reference_paths}):
        if path == excluded:
            continue
        manifest = json.loads((path/"manifest.json").read_text())
        atlas_ids.add(manifest.get("atlas_id"))
        payload = (path/"route_states.jsonl").read_bytes()
        inputs.append({"path": str(path), "states_sha256": hashlib.sha256(payload).hexdigest(),
                       "manifest_sha256": hashlib.sha256((path/"manifest.json").read_bytes()).hexdigest()})
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        gap = 1.5e9/manifest["reference_rate_hz"]
        for row in rows:
            data = modes.setdefault(row["mode_id"], {"scales": [], "residuals": []})
            if row.get("map_scale", 0) > 0:
                data["scales"].append(row["map_scale"])
        for before, middle, after in zip(rows, rows[1:], rows[2:]):
            if not before["mode_id"] == middle["mode_id"] == after["mode_id"]:
                continue
            first = middle["session_time_ns"]-before["session_time_ns"]
            second = after["session_time_ns"]-middle["session_time_ns"]
            if min(first, second) <= 0 or max(first, second) > gap:
                continue
            fraction = first/(first+second)
            estimate = np.array(before["canonical_xy"])*(1-fraction)+np.array(after["canonical_xy"])*fraction
            modes[middle["mode_id"]]["residuals"].append(float(np.linalg.norm(estimate-middle["canonical_xy"])))
    if len(atlas_ids) > 1:
        raise ValueError("Loss calibration references must share one atlas")
    calibrated = {}
    for mode, values in modes.items():
        residuals, scales = values["residuals"], values["scales"]
        # Two integer-grid matches may differ by one pixel on both axes.
        quantization = float(np.sqrt(2)*np.median(scales)) if scales else None
        residual_p99 = float(np.percentile(residuals, 99)) if len(residuals) >= 10 else None
        calibrated[mode] = {"reference_residual_count": len(residuals),
                            "reference_temporal_residual_p99_px": residual_p99,
                            "relative_raster_quantization_px": quantization,
                            "error_limit_px": max(quantization, residual_p99) if quantization is not None and residual_p99 is not None else None}
    return {"method": "leave-session-out-reference-resolution-v1", "modes": calibrated,
            "inputs": inputs, "excluded_reference": str(excluded) if excluded else None,
            "interpretation": "proxy distinguishability envelope, not a validated gameplay tolerance"}


def evaluate_tracking_loss(rows, *, start_ns, end_ns, error_limit_px=10.0,
                           recovery_s=0.5, max_sample_gap_s=0.75, calibration=None):
    """Measure elapsed time from observed failure to sustained verified recovery.

    Unknown intervals cannot establish loss or recovery. They remain inside an
    already open episode, explicitly counted as uncertainty about its duration.
    Durations use recorded source timestamps, not worker/publication wall time.
    """
    if (error_limit_px is not None and error_limit_px <= 0) or recovery_s <= 0 or max_sample_gap_s <= 0:
        raise ValueError("Loss tolerances and durations must be positive")
    recovery_ns = round(recovery_s * 1e9)
    gap_ns = round(max_sample_gap_s * 1e9)
    timeline = []
    for row in rows:
        reasons = []
        error = row.get("reference_error_px")
        reference_mode = row.get("reference_mode")
        limit = error_limit_px if calibration is None else (calibration["modes"].get(reference_mode) or {}).get("error_limit_px")
        if not row.get("pose"):
            reasons.append("unavailable-xy")
        else:
            if reference_mode is not None and row.get("active_map_mode_id") != reference_mode:
                reasons.append("wrong-map-layer")
            if error is not None and limit is not None and error > limit:
                reasons.append("position-error")
        state = "lost" if reasons else "tracked" if error is not None and reference_mode is not None and limit is not None else "unknown"
        row["tracking_loss_state"] = state
        row["tracking_loss_reasons"] = reasons
        row["tracking_loss_error_limit_px"] = limit
        timeline.append((row["session_time_ns"], state, reasons,
                         bool(row.get("xy_measurement_fresh_accepted"))))
    if any(b[0] <= a[0] for a, b in zip(timeline, timeline[1:])):
        raise ValueError("Loss evaluation requires strictly chronological samples")
    if not timeline or timeline[0][0] > start_ns:
        timeline.insert(0, (start_ns, "lost", ["unavailable-xy"], False))
    # Large holes in processed telemetry also cannot verify continued tracking.
    expanded = []
    for i, item in enumerate(timeline):
        expanded.append(item)
        following = timeline[i+1][0] if i+1 < len(timeline) else end_ns
        if following - item[0] > gap_ns:
            expanded.append((item[0]+gap_ns, "unknown", [], False))

    episodes = []
    active = None
    candidate = None
    acquired = False
    first_acquired_ns = None
    unknown_ns = 0
    for i, (timestamp, state, reasons, fresh) in enumerate(expanded):
        following = expanded[i+1][0] if i+1 < len(expanded) else end_ns
        interval_ns = max(0, following-timestamp)
        if state == "unknown":
            unknown_ns += interval_ns
        if state == "lost":
            if active is None:
                active = {"start_ns": timestamp, "kind": "tracking" if acquired else "acquisition",
                          "reasons": set(), "unknown_ns": 0}
            active["reasons"].update(reasons)
        if state == "tracked" and fresh:
            if candidate is None:
                candidate = timestamp
            if timestamp-candidate >= recovery_ns:
                if active is not None:
                    episodes.append(_finish(active, candidate, timestamp, False))
                    active = None
                if not acquired:
                    first_acquired_ns = candidate
                acquired = True
        else:
            candidate = None
        if active is not None and state == "unknown":
            active["unknown_ns"] += interval_ns
    if active is not None:
        episodes.append(_finish(active, end_ns, None, True))
    observable = any(item[1] != "unknown" for item in expanded)
    return {
        "method": "reference-tracking-loss-v1",
        "reference_role": "slow-inferred-atlas-proxy-not-external-truth",
        "error_limit_px": error_limit_px,
        "calibration": calibration,
        "recovery_confirmation_s": recovery_s,
        "max_sample_gap_s": max_sample_gap_s,
        "time_basis": "recorded-source-time; sampled evidence, excludes publication delay",
        "includes_initial_acquisition": True,
        "episode_count": len(episodes),
        "episodes": episodes,
        "longest_lost_s": max((e["seconds"] for e in episodes), default=0.0) if observable else None,
        "longest_post_acquisition_lost_s": max((e["seconds"] for e in episodes if e["kind"] == "tracking"), default=0.0) if acquired else None,
        "first_verified_acquisition_s": (first_acquired_ns-start_ns)/1e9 if first_acquired_ns is not None else None,
        "unrecovered_at_end": active is not None,
        "unknown_s": unknown_ns/1e9,
        "duration_s": (end_ns-start_ns)/1e9,
        "all_intervals_observable": unknown_ns == 0,
    }


def _finish(active, end_ns, confirmed_ns, censored):
    return {"start_ns": active["start_ns"], "end_ns": end_ns,
            "seconds": (end_ns-active["start_ns"])/1e9,
            "kind": active["kind"], "reasons": sorted(active["reasons"]),
            "recovery_confirmed_ns": confirmed_ns,
            "unrecovered_at_end": censored,
            "unknown_s": active["unknown_ns"]/1e9,
            "duration_interpretation": "time-to-verified-recovery-includes-unknown" if active["unknown_ns"] else "sampled-observed-loss-to-verified-recovery"}
