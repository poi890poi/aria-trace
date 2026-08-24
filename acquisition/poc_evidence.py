"""Structured progress index for guided proof-of-concept captures."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .annotations import AnnotationStore
from .session import input_capture_health


READY_MARKERS = {
    "route": ("route_start", "route_complete"),
    "capture": ("capture_start", "capture_complete"),
}


def _session_status(manifest: dict, marker_kinds: Iterable[str]) -> str:
    kinds = set(marker_kinds)
    if "route_failed" in kinds or "capture_failed" in kinds:
        return "failed"
    input_health = input_capture_health(manifest)
    if manifest.get("status") == "complete" and not input_health["healthy"]:
        return "failed"
    expected = (
        READY_MARKERS["route"]
        if manifest.get("context", {}).get("capture_kind") == "route"
        else READY_MARKERS["capture"]
    )
    if all(kind in kinds for kind in expected):
        return "ready"
    if "take_start" in kinds and "take_end" in kinds:
        return "captured_needs_confirmation"
    if manifest.get("status") == "recording":
        return "recording"
    return "incomplete"


def _indexed_session(session_root: Path, manifest_path: Path) -> dict:
    session_path = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    annotations = AnnotationStore(session_path).list()
    marker_kinds = [item["kind"] for item in annotations]
    context = manifest.get("context") or {}
    input_health = input_capture_health(manifest)
    try:
        relative_path = str(session_path.relative_to(session_root))
    except ValueError:
        relative_path = str(session_path)
    superseded = any(".previous-" in part for part in session_path.parts)
    return {
        "session_id": manifest.get("session_id"),
        "relative_path": relative_path,
        "superseded": superseded,
        "status": _session_status(manifest, marker_kinds),
        "manifest_status": manifest.get("status"),
        "created_utc": manifest.get("created_utc"),
        "finished_utc": manifest.get("finished_utc"),
        "duration_ns": int(manifest.get("duration_ns") or 0),
        "frame_counts": dict(manifest.get("frame_counts") or {}),
        "input_counts": dict(manifest.get("input_counts") or {}),
        "input_capture": input_health,
        "dropped_frames": dict(manifest.get("dropped_frames") or {}),
        "markers": marker_kinds,
        "experiment_id": context.get("experiment_id"),
        "game_profile_id": context.get("game_profile_id"),
        "run_index": context.get("run_index"),
        "capture_kind": context.get("capture_kind"),
        "capture_id": context.get("capture_id"),
        "workflow_stage_id": context.get("workflow_stage_id"),
        "profile_draft_updated_utc": (
            (context.get("game_profile_draft") or {}).get("updated_utc")
        ),
    }


def build_poc_evidence_index(
    session_root: Path,
    game_profile: dict,
    profile_draft: dict = None,
) -> dict:
    """Index recorded evidence without interpreting map or movement content."""
    session_root = Path(session_root)
    workflow = list(game_profile.get("poc_workflow") or [])
    sessions = []
    errors = []
    for manifest_path in sorted(session_root.glob("**/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            context = manifest.get("context") or {}
            if context.get("game_profile_id") == game_profile.get("profile_id"):
                sessions.append(_indexed_session(session_root, manifest_path))
        except Exception as exc:
            errors.append(
                {
                    "relative_manifest_path": str(
                        manifest_path.relative_to(session_root)
                    ),
                    "error": "{}: {}".format(type(exc).__name__, exc),
                }
            )

    by_stage = {}
    unassigned = []
    known_stage_ids = {item.get("stage_id") for item in workflow}
    for session in sessions:
        stage_id = session.get("workflow_stage_id")
        if stage_id in known_stage_ids:
            by_stage.setdefault(stage_id, []).append(session)
        else:
            unassigned.append(session)

    stages = []
    for order, stage in enumerate(workflow, 1):
        stage_sessions = by_stage.get(stage.get("stage_id"), [])
        current_sessions = [
            item for item in stage_sessions if not item.get("superseded")
        ]
        status_counts = {
            status: sum(1 for item in current_sessions if item["status"] == status)
            for status in (
                "ready",
                "captured_needs_confirmation",
                "recording",
                "failed",
                "incomplete",
            )
        }
        required = int(stage.get("target_runs") or 1)
        if status_counts["ready"] >= required:
            status = "ready"
        elif status_counts["captured_needs_confirmation"]:
            status = "captured_needs_confirmation"
        elif status_counts["recording"]:
            status = "recording"
        elif current_sessions:
            status = "needs_capture"
        else:
            status = "missing"
        stages.append(
            {
                "order": order,
                "stage_id": stage.get("stage_id"),
                "display_name": stage.get("display_name"),
                "capture_kind": stage.get("capture_kind"),
                "capture_id": stage.get("capture_id"),
                "required_captures": required,
                "ready_captures": status_counts["ready"],
                "status": status,
                "status_counts": status_counts,
                "superseded_session_count": len(stage_sessions)
                - len(current_sessions),
                "sessions": stage_sessions,
            }
        )

    return {
        "schema_version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "game_profile_id": game_profile.get("profile_id"),
        "profile_draft": profile_draft,
        "complete": bool(stages) and all(item["status"] == "ready" for item in stages),
        "stages": stages,
        "unassigned_sessions": unassigned,
        "index_errors": errors,
    }
