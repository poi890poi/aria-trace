"""Versioned calibration profiles scoped by rig, game, and image source."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


PROFILE_SCHEMA_VERSION = "1.0"


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip()).strip("-.")
    if not cleaned:
        raise ValueError("Calibration profile identifiers cannot be empty")
    return cleaned


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


@dataclass(frozen=True)
class CalibrationProfileKey:
    """A calibration reuse boundary, not merely a game configuration."""

    rig_id: str
    game_id: str
    image_source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "rig_id", _safe_id(self.rig_id))
        object.__setattr__(self, "game_id", _safe_id(self.game_id))
        object.__setattr__(self, "image_source", _safe_id(self.image_source))

    @property
    def profile_id(self) -> str:
        return "{}--{}--{}".format(self.rig_id, self.game_id, self.image_source)

    def as_dict(self) -> dict:
        return {
            "rig_id": self.rig_id,
            "game_id": self.game_id,
            "image_source": self.image_source,
            "profile_id": self.profile_id,
        }


class CalibrationProfileStore:
    """Publish immutable revisions and one atomic pointer to the current one."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def profile_directory(self, key: CalibrationProfileKey) -> Path:
        return self.root / key.rig_id / key.game_id / key.image_source

    def create_revision_directory(
        self, key: CalibrationProfileKey, revision_id: Optional[str] = None
    ) -> Path:
        revision = _safe_id(
            revision_id
            or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        )
        path = self.profile_directory(key) / "revisions" / revision
        path.mkdir(parents=True, exist_ok=False)
        return path

    def publish(
        self,
        key: CalibrationProfileKey,
        revision_directory: Path,
        artifacts: Mapping[str, Any],
        *,
        session_path: Path,
        rig_calibration_path: Optional[Path] = None,
        reuse_parent_revision: Optional[str] = None,
        notes: Optional[Mapping[str, Any]] = None,
    ) -> dict:
        revision_directory = Path(revision_directory).resolve()
        expected_parent = (self.profile_directory(key) / "revisions").resolve()
        if expected_parent not in revision_directory.parents:
            raise ValueError("Revision directory is outside its calibration profile")
        manifest = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "status": "review_required",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "profile": key.as_dict(),
            "revision_id": revision_directory.name,
            "session_path": str(Path(session_path).resolve()),
            "rig_calibration_path": (
                str(Path(rig_calibration_path).resolve())
                if rig_calibration_path is not None
                else None
            ),
            "reuse": {
                "scope": "same rig_id + game_id + image_source only",
                "small_rig_shift_supported": True,
                "parent_revision_id": reuse_parent_revision,
                "rule": (
                    "A small physical shift creates a new geometry revision. "
                    "Source-specific observations are never silently reused across sources."
                ),
            },
            "artifacts": dict(artifacts),
            "notes": dict(notes or {}),
        }
        _atomic_json(revision_directory / "profile_revision.json", manifest)
        pointer = {
            "schema_version": PROFILE_SCHEMA_VERSION,
            "profile": key.as_dict(),
            "current_revision_id": revision_directory.name,
            "current_revision": str(revision_directory),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(self.profile_directory(key) / "current.json", pointer)
        return manifest

    def current(self, key: CalibrationProfileKey) -> Optional[dict]:
        path = self.profile_directory(key) / "current.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
