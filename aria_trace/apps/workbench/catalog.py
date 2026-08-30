"""Filesystem-backed session catalog for the Workbench application."""

import json
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional

from acquisition.annotations import AnnotationStore
from acquisition.session import SessionReader, input_capture_health


def session_primary_stream_id(reader) -> str:
    """Resolve the recorded stream used by game-analysis workflows."""
    context = reader.manifest.get("context") or {}
    requested = str(context.get("primary_stream_id") or "").strip()
    if requested and requested in reader.frames_by_stream:
        return requested
    for candidate in ("main", "android_phone", "hik_phone"):
        if candidate in reader.frames_by_stream:
            return candidate
    if reader.frames_by_stream:
        return next(iter(reader.frames_by_stream))
    return requested or "main"


def archive_existing(path: Path) -> None:
    """Move an existing generated path aside without destroying it."""
    path = Path(path)
    if not path.exists():
        return
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = path.with_name(path.name + ".previous-" + suffix)
    counter = 1
    while candidate.exists():
        candidate = path.with_name(
            path.name + ".previous-{}-{}".format(suffix, counter)
        )
        counter += 1
    path.rename(candidate)


class SessionCatalog:
    """Validate, resolve, and summarize sessions below one explicit root."""

    def __init__(self, session_root: Path, labels: Iterable[dict], metadata_filename: str):
        self.session_root = Path(session_root)
        self.labels = tuple(dict(item) for item in labels)
        self.metadata_filename = str(metadata_filename)
        self._summary_cache = {}

    def key(self, path: Path) -> str:
        return Path(path).resolve().relative_to(self.session_root.resolve()).as_posix()

    def resolve(self, session_key: str, require_manifest: bool = True) -> Path:
        pure = PurePosixPath(str(session_key or ""))
        if (
            pure.is_absolute()
            or len(pure.parts) != 2
            or any(part in ("", ".", "..") for part in pure.parts)
            or not re.fullmatch(r"run_\d+", pure.parts[1])
        ):
            raise ValueError("Invalid session identifier")
        root = self.session_root.resolve()
        path = (root / Path(*pure.parts)).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            raise ValueError("Session must stay inside the session root")
        if require_manifest and not (path / "manifest.json").is_file():
            raise ValueError("Unknown recorded session")
        return path

    def metadata(self, path: Path) -> dict:
        metadata_path = Path(path) / self.metadata_filename
        if not metadata_path.is_file():
            return {}
        try:
            value = json.loads(metadata_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def label_definition(self, label: str) -> dict:
        for item in self.labels:
            if item["value"] == label:
                return dict(item)
        raise ValueError("Unknown session label")

    def invalidate(self, path: Path) -> None:
        self._summary_cache.pop(str(Path(path).resolve()), None)

    def invalidate(self, path: Path) -> None:
        self._summary_cache.pop(str(Path(path).resolve()), None)

    def describe(self, path: Path) -> dict:
        path = Path(path)
        signature = []
        for name in (
            "manifest.json",
            "annotations.jsonl",
            self.metadata_filename,
            "frames.jsonl",
            "inputs.jsonl",
        ):
            item_path = path / name
            if item_path.is_file():
                stat = item_path.stat()
                signature.append((name, stat.st_mtime_ns, stat.st_size))
        cache_key = str(path.resolve())
        cached = self._summary_cache.get(cache_key)
        if cached and cached["signature"] == signature:
            return dict(cached["value"])
        reader = SessionReader(path)
        annotations = AnnotationStore(path).list()
        kinds = [item["kind"] for item in annotations]
        input_health = input_capture_health(reader.manifest, reader.inputs)
        metadata = self.metadata(path)
        context = reader.manifest.get("context") or {}
        primary_stream_id = session_primary_stream_id(reader)
        if "route_failed" in kinds or "capture_failed" in kinds:
            status = "failed"
        elif not input_health["healthy"]:
            status = "failed"
        elif (
            "route_start" in kinds and "route_complete" in kinds
        ) or (
            "capture_start" in kinds and "capture_complete" in kinds
        ):
            status = "ready"
        elif "take_start" in kinds and "take_end" in kinds:
            status = "recorded"
        else:
            status = "incomplete"
        label = metadata.get("label")
        if label is None:
            label = context.get("segment_label") or ""
        value = {
            "session_key": self.key(path),
            "session_id": reader.manifest.get("session_id"),
            "experiment_id": context.get("experiment_id") or path.parent.name,
            "run_index": context.get("run_index"),
            "game_profile_id": context.get("game_profile_id"),
            "window_title": (
                (reader.manifest.get("frame_sources") or [{}])[0].get(
                    "matched_window_title"
                )
                or (reader.manifest.get("frame_sources") or [{}])[0].get(
                    "window_title_query"
                )
            ),
            "created_utc": reader.manifest.get("created_utc"),
            "finished_utc": reader.manifest.get("finished_utc"),
            "duration_s": reader.manifest.get("duration_ns", 0) / 1.0e9,
            "frames": len(reader.frames_by_stream.get(primary_stream_id, [])),
            "input_events": len(reader.inputs),
            "dropped_frames": reader.manifest.get("dropped_frames", {}).get(
                primary_stream_id, 0
            ),
            "status": status,
            "label": label,
            "markers": kinds,
            "input_capture": input_health,
        }
        self._summary_cache[cache_key] = {
            "signature": signature,
            "value": dict(value),
        }
        return value

    def list(self, active: Optional[dict] = None) -> list:
        values = []
        active_path = Path(active["path"]).resolve() if active is not None else None
        for manifest_path in self.session_root.glob("*/run_*/manifest.json"):
            path = manifest_path.parent
            if not re.fullmatch(r"run_\d+", path.name) or path.resolve() == active_path:
                continue
            try:
                values.append(self.describe(path))
            except Exception as exc:
                values.append(
                    {
                        "session_key": self.key(path),
                        "status": "invalid",
                        "label": "",
                        "error": "{}: {}".format(type(exc).__name__, exc),
                    }
                )
        values.sort(
            key=lambda item: item.get("finished_utc")
            or item.get("created_utc")
            or item.get("session_key", ""),
            reverse=True,
        )
        if active is not None:
            values.insert(
                0,
                {
                    "session_key": self.key(active_path),
                    "experiment_id": active_path.parent.name,
                    "run_index": active["run_index"],
                    "game_profile_id": active.get("game_profile_id"),
                    "created_utc": None,
                    "duration_s": None,
                    "frames": None,
                    "input_events": active.get("recorded_input_events", 0),
                    "dropped_frames": None,
                    "status": active["phase"],
                    "label": "",
                },
            )
        return values
