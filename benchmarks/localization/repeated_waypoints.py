"""Automatic repeated complete-state waypoint discovery and scoring.

This is an offline evaluation utility.  Waypoint membership must be derived from
an independent post-run reference and must never be supplied to a live estimator.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from typing import Iterable, Mapping, Optional

import numpy as np


def circular_delta_deg(first: float, second: float) -> float:
    """Return the signed shortest angular difference ``first - second``."""

    return ((float(first) - float(second) + 180.0) % 360.0) - 180.0


def _summary(values) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {
            "sample_count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "sample_count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def normalize_state_rows(rows: Iterable[Mapping]) -> list[dict]:
    """Normalize route-state, tracker-telemetry, or generic state records."""

    output = []
    for ordinal, source in enumerate(rows):
        row = dict(source)
        point = row.get("canonical_xy")
        x = row.get("x") if point is None else point[0]
        y = row.get("y") if point is None else point[1]
        heading = None
        for name in (
            "heading_deg",
            "cursor_heading_deg",
            "yaw_deg",
            "route_heading_deg",
        ):
            if row.get(name) is not None:
                heading = float(row[name]) % 360.0
                break
        frame_index = row.get("source_frame_index")
        if frame_index is None:
            frame_index = row.get("frame_index")
        timestamp = row.get("session_time_ns")
        valid = bool(row.get("valid", True)) and x is not None and y is not None
        if valid and not np.isfinite([float(x), float(y)]).all():
            valid = False
        output.append(
            {
                "ordinal": ordinal,
                "frame_index": int(frame_index) if frame_index is not None else None,
                "session_time_ns": int(timestamp) if timestamp is not None else ordinal,
                "x": float(x) if x is not None else None,
                "y": float(y) if y is not None else None,
                "heading_deg": heading,
                "mode_id": str(row.get("mode_id") or "default"),
                "valid": valid,
                "fresh": bool(row.get("measurement_accepted", valid)),
                "latency_ms": (
                    float(row["localization_core_elapsed_ms"])
                    if row.get("localization_core_elapsed_ms") is not None
                    else None
                ),
            }
        )
    output.sort(key=lambda item: (item["session_time_ns"], item["ordinal"]))
    return output


def _visit_representatives(
    indexes,
    times_s,
    distances,
    *,
    visit_merge_s: float,
    minimum_recurrence_s: float,
) -> list[int]:
    visits = []
    for index in sorted(indexes, key=lambda item: times_s[item]):
        if not visits or times_s[index] - times_s[visits[-1][-1]] > visit_merge_s:
            visits.append([index])
        else:
            visits[-1].append(index)
    representatives = [min(visit, key=lambda item: distances[item]) for visit in visits]
    retained = []
    for index in representatives:
        if not retained or times_s[index] - times_s[retained[-1]] >= minimum_recurrence_s:
            retained.append(index)
        elif distances[index] < distances[retained[-1]]:
            retained[-1] = index
    return retained


def discover_repeated_waypoints(
    reference_rows: Iterable[Mapping],
    *,
    spatial_radius_px: float = 5.0,
    heading_radius_deg: float = 45.0,
    waypoint_spacing_px: float = 10.0,
    waypoint_spacing_deg: float = 35.0,
    visit_merge_s: float = 1.5,
    minimum_recurrence_s: float = 8.0,
    minimum_visits: int = 3,
    maximum_waypoints: int = 128,
    require_heading: bool = True,
) -> dict:
    """Find repeatedly visited nearest-state neighborhoods without lap cuts.

    A visit is one continuous passage through a neighborhood.  Opposite travel
    directions remain different waypoints when heading is available.  Sampling
    time, speed, path shape, and the number of observations per visit may differ.
    """

    if spatial_radius_px <= 0 or heading_radius_deg <= 0:
        raise ValueError("Waypoint radii must be positive")
    if minimum_recurrence_s <= visit_merge_s:
        raise ValueError("Recurrence separation must exceed visit merge time")
    if minimum_visits < 2:
        raise ValueError("Repeated waypoints need at least two visits")
    rows = [item for item in normalize_state_rows(reference_rows) if item["valid"]]
    if require_heading and any(item["heading_deg"] is None for item in rows):
        raise ValueError("Complete-state grouping requires heading for every reference row")
    if len(rows) < minimum_visits:
        return {"groups": [], "reference_sample_count": len(rows)}

    xy = np.asarray([[item["x"], item["y"]] for item in rows], dtype=np.float64)
    times_s = np.asarray(
        [item["session_time_ns"] / 1.0e9 for item in rows], dtype=np.float64
    )
    headings = np.asarray(
        [item["heading_deg"] if item["heading_deg"] is not None else 0.0 for item in rows],
        dtype=np.float64,
    )
    modes = np.asarray([item["mode_id"] for item in rows], dtype=object)
    proposals = []
    for anchor in range(len(rows)):
        position_distance = np.linalg.norm(xy - xy[anchor], axis=1)
        if require_heading:
            heading_distance = np.abs(
                (headings - headings[anchor] + 180.0) % 360.0 - 180.0
            )
        else:
            heading_distance = np.zeros(len(rows), dtype=np.float64)
        eligible = np.flatnonzero(
            (position_distance <= spatial_radius_px)
            & (heading_distance <= heading_radius_deg)
            & (modes == modes[anchor])
        )
        normalized_distance = np.hypot(
            position_distance / spatial_radius_px,
            heading_distance / heading_radius_deg if require_heading else 0.0,
        )
        members = _visit_representatives(
            eligible,
            times_s,
            normalized_distance,
            visit_merge_s=visit_merge_s,
            minimum_recurrence_s=minimum_recurrence_s,
        )
        if len(members) < minimum_visits:
            continue
        proposals.append(
            {
                "anchor": anchor,
                "members": members,
                "visit_count": len(members),
                "median_normalized_distance": float(
                    np.median(normalized_distance[members])
                ),
            }
        )

    proposals.sort(
        key=lambda item: (
            -item["visit_count"],
            item["median_normalized_distance"],
            rows[item["anchor"]]["session_time_ns"],
        )
    )
    selected = []
    assigned_members = set()
    for proposal in proposals:
        anchor = proposal["anchor"]
        duplicate = False
        for existing in selected:
            other = existing["anchor"]
            if np.linalg.norm(xy[anchor] - xy[other]) > waypoint_spacing_px:
                continue
            if require_heading and abs(
                circular_delta_deg(headings[anchor], headings[other])
            ) > waypoint_spacing_deg:
                continue
            if modes[anchor] == modes[other]:
                duplicate = True
                break
        if duplicate:
            continue
        # Do not count one reference observation in more than one subgroup;
        # overlapping neighborhoods would otherwise overweight easy locations.
        if assigned_members.intersection(proposal["members"]):
            continue
        selected.append(proposal)
        assigned_members.update(proposal["members"])
        if len(selected) >= maximum_waypoints:
            break

    groups = []
    for group_index, proposal in enumerate(selected):
        anchor = proposal["anchor"]
        members = proposal["members"]
        member_xy = xy[members]
        position_center = np.median(member_xy, axis=0)
        position_deviation = np.linalg.norm(member_xy - position_center, axis=1)
        heading_deviation = (
            [abs(circular_delta_deg(headings[index], headings[anchor])) for index in members]
            if require_heading
            else []
        )
        groups.append(
            {
                "group_id": "waypoint-{:03d}".format(group_index + 1),
                "mode_id": rows[anchor]["mode_id"],
                "anchor_xy": xy[anchor].tolist(),
                "anchor_heading_deg": (
                    float(headings[anchor]) if require_heading else None
                ),
                "visit_count": len(members),
                "reference_position_spread_px": _summary(position_deviation),
                "reference_heading_spread_deg": _summary(heading_deviation),
                "members": [
                    {
                        "frame_index": rows[index]["frame_index"],
                        "session_time_ns": rows[index]["session_time_ns"],
                        "reference_xy": xy[index].tolist(),
                        "reference_heading_deg": (
                            float(headings[index]) if require_heading else None
                        ),
                    }
                    for index in members
                ],
            }
        )
    return {
        "schema_version": "1.0",
        "method": "greedy_recurrent_complete_state_neighborhoods",
        "reference_sample_count": len(rows),
        "group_count": len(groups),
        "grouped_visit_count": int(sum(item["visit_count"] for item in groups)),
        "group_members_are_unique": True,
        "parameters": {
            "spatial_radius_px": float(spatial_radius_px),
            "heading_radius_deg": float(heading_radius_deg),
            "waypoint_spacing_px": float(waypoint_spacing_px),
            "waypoint_spacing_deg": float(waypoint_spacing_deg),
            "visit_merge_s": float(visit_merge_s),
            "minimum_recurrence_s": float(minimum_recurrence_s),
            "minimum_visits": int(minimum_visits),
            "maximum_waypoints": int(maximum_waypoints),
            "require_heading": bool(require_heading),
        },
        "groups": groups,
    }


def _candidate_lookup(rows: Iterable[Mapping]) -> tuple[dict, list, list]:
    normalized = normalize_state_rows(rows)
    by_frame = {
        item["frame_index"]: item
        for item in normalized
        if item["frame_index"] is not None
    }
    by_time = {
        item["session_time_ns"]: item
        for item in normalized
        if item["session_time_ns"] is not None
    }
    times = sorted(by_time)
    return by_frame, times, [by_time[item] for item in times]


def _nearest_time(times, rows, target_ns: int, tolerance_ns: int):
    insertion = bisect_left(times, target_ns)
    candidates = []
    if insertion < len(times):
        candidates.append((abs(times[insertion] - target_ns), rows[insertion]))
    if insertion:
        candidates.append((abs(times[insertion - 1] - target_ns), rows[insertion - 1]))
    if not candidates:
        return None, None
    distance, row = min(candidates, key=lambda item: item[0])
    return (row, distance) if distance <= tolerance_ns else (None, None)


def evaluate_candidate_repeatability(
    waypoint_result: Mapping,
    candidate_rows: Iterable[Mapping],
    *,
    maximum_time_delta_ms: float = 20.0,
) -> dict:
    """Score candidate error and error dispersion at repeated waypoints."""

    by_frame, candidate_times, candidate_time_rows = _candidate_lookup(candidate_rows)
    tolerance_ns = int(round(maximum_time_delta_ms * 1.0e6))
    position_errors = []
    heading_errors = []
    position_residuals = []
    heading_residuals = []
    latency = []
    time_alignment_delta_ms = []
    expected = 0
    available = 0
    fresh = 0
    group_results = []
    for group in waypoint_result.get("groups") or ():
        observations = []
        for member in group["members"]:
            expected += 1
            candidate = None
            time_delta_ns = None
            if member.get("frame_index") is not None:
                candidate = by_frame.get(int(member["frame_index"]))
            if candidate is None:
                candidate, time_delta_ns = _nearest_time(
                    candidate_times,
                    candidate_time_rows,
                    int(member["session_time_ns"]),
                    tolerance_ns,
                )
            if candidate is None or not candidate["valid"]:
                continue
            available += 1
            fresh += int(candidate["fresh"])
            if time_delta_ns is not None:
                time_alignment_delta_ms.append(time_delta_ns / 1.0e6)
            reference_xy = np.asarray(member["reference_xy"], dtype=np.float64)
            candidate_xy = np.asarray([candidate["x"], candidate["y"]], dtype=np.float64)
            position_error_vector = candidate_xy - reference_xy
            position_errors.append(float(np.linalg.norm(position_error_vector)))
            if candidate["latency_ms"] is not None:
                latency.append(candidate["latency_ms"])
            heading_error = None
            reference_heading = member.get("reference_heading_deg")
            if reference_heading is not None and candidate["heading_deg"] is not None:
                heading_error = circular_delta_deg(
                    candidate["heading_deg"], reference_heading
                )
                heading_errors.append(abs(heading_error))
            observations.append((position_error_vector, heading_error))
        if len(observations) >= 2:
            vectors = np.asarray([item[0] for item in observations], dtype=np.float64)
            bias = np.median(vectors, axis=0)
            residuals = np.linalg.norm(vectors - bias, axis=1)
            position_residuals.extend(float(item) for item in residuals)
            angular = [item[1] for item in observations if item[1] is not None]
            if len(angular) >= 2:
                radians = np.deg2rad(angular)
                center = math.degrees(
                    math.atan2(float(np.mean(np.sin(radians))), float(np.mean(np.cos(radians))))
                )
                heading_residuals.extend(
                    abs(circular_delta_deg(value, center)) for value in angular
                )
        group_results.append(
            {
                "group_id": group["group_id"],
                "expected_visits": len(group["members"]),
                "available_visits": len(observations),
            }
        )
    return {
        "expected_waypoint_observations": expected,
        "available_waypoint_observations": available,
        "availability_rate": available / float(expected) if expected else 0.0,
        "fresh_rate": fresh / float(expected) if expected else 0.0,
        "reference_position_error_px": _summary(position_errors),
        "reference_heading_error_deg": _summary(heading_errors),
        "position_repeatability_residual_px": _summary(position_residuals),
        "heading_repeatability_residual_deg": _summary(heading_residuals),
        "latency_ms": _summary(latency),
        "time_alignment_delta_ms": _summary(time_alignment_delta_ms),
        "maximum_time_delta_ms": float(maximum_time_delta_ms),
        "groups": group_results,
        "metric_note": (
            "Repeatability is dispersion of candidate-minus-reference error. "
            "Reference displacement between naturally different passes is subtracted."
        ),
    }
