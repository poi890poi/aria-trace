"""Append-only human annotations for recorded sessions."""

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


MARKER_KINDS = ("teleport_start", "world_ready", "route_start", "note")


class AnnotationStore:
    def __init__(self, session_path: Path) -> None:
        self.session_path = Path(session_path)
        self.path = self.session_path / "annotations.jsonl"
        self._lock = threading.Lock()

    def _records(self) -> List[dict]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "Invalid annotation JSON at line {}: {}".format(line_number, exc)
                    )
        return records

    def list(self) -> List[dict]:
        with self._lock:
            active: Dict[str, dict] = {}
            for record in self._records():
                annotation_id = record.get("annotation_id")
                if record.get("operation") == "delete":
                    active.pop(annotation_id, None)
                elif record.get("operation") == "add":
                    active[annotation_id] = record
            return sorted(
                active.values(),
                key=lambda item: (item["session_time_ns"], item["created_utc"]),
            )

    def add(
        self,
        kind: str,
        session_time_ns: int,
        stream_id: str,
        frame_index: int,
        portal_id: Optional[str] = None,
        route_id: Optional[str] = None,
        note: Optional[str] = None,
    ) -> dict:
        if kind not in MARKER_KINDS:
            raise ValueError("Unsupported annotation kind: {}".format(kind))
        if session_time_ns < 0 or frame_index < 0 or not stream_id:
            raise ValueError("Annotation frame and time must identify a recorded frame")
        record = {
            "schema_version": "1.0",
            "operation": "add",
            "annotation_id": str(uuid.uuid4()),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "session_time_ns": int(session_time_ns),
            "stream_id": stream_id,
            "frame_index": int(frame_index),
            "portal_id": portal_id or None,
            "route_id": route_id or None,
            "note": note or None,
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", buffering=1) as stream:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        return record

    def delete(self, annotation_id: str) -> dict:
        with self._lock:
            active_ids = {
                record["annotation_id"]
                for record in self._resolve_records_without_lock()
            }
            if annotation_id not in active_ids:
                raise KeyError("Unknown annotation: {}".format(annotation_id))
            record = {
                "schema_version": "1.0",
                "operation": "delete",
                "annotation_id": annotation_id,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            }
            with self.path.open("a", encoding="utf-8", buffering=1) as stream:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            return record

    def _resolve_records_without_lock(self) -> List[dict]:
        active = {}
        for record in self._records():
            annotation_id = record.get("annotation_id")
            if record.get("operation") == "delete":
                active.pop(annotation_id, None)
            elif record.get("operation") == "add":
                active[annotation_id] = record
        return list(active.values())
