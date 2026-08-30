"""Obsolete compatibility surface for pre-registry path-pointer profiles.

Production code must use :mod:`acquisition.profile_registry`. The legacy store
classes remain importable only so callers receive a precise migration error;
they can no longer publish or resolve mutable ``current.json`` pointers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


OBSOLETE_MESSAGE = (
    "Path-based calibration profile stores are obsolete. Use "
    "acquisition.profile_registry.ProfileRegistry and active immutable revisions."
)


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip()).strip("-.")
    if not cleaned:
        raise ValueError("Calibration profile identifiers cannot be empty")
    return cleaned


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
    """Retired mutable-current store; construction always fails."""

    def __init__(self, root: Path) -> None:
        raise RuntimeError(OBSOLETE_MESSAGE)


@dataclass(frozen=True)
class ScopedProfileKey:
    """A decoupled phone-game or rig-game profile identity."""

    scope_kind: str
    owner_id: str
    game_id: str

    def __post_init__(self) -> None:
        kind = _safe_id(self.scope_kind)
        if kind not in ("phone_game", "rig_game"):
            raise ValueError("scope_kind must be phone_game or rig_game")
        object.__setattr__(self, "scope_kind", kind)
        object.__setattr__(self, "owner_id", _safe_id(self.owner_id))
        object.__setattr__(self, "game_id", _safe_id(self.game_id))

    @property
    def profile_id(self) -> str:
        return "{}--{}--{}".format(
            self.scope_kind, self.owner_id, self.game_id
        )

    def as_dict(self) -> dict:
        return {
            "scope_kind": self.scope_kind,
            "owner_id": self.owner_id,
            "game_id": self.game_id,
            "profile_id": self.profile_id,
        }


class ScopedCalibrationProfileStore:
    """Retired scoped mutable-current store; construction always fails."""

    def __init__(self, root: Path) -> None:
        raise RuntimeError(OBSOLETE_MESSAGE)
