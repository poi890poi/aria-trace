"""Coordinate-space references shared by pixels and geometric results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SpaceRef:
    space_id: str
    kind: str = "unspecified"
    transform_refs: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.space_id.strip():
            raise ValueError("Space ID is required; use 'unknown' explicitly")
        if not self.kind.strip():
            raise ValueError("Space kind is required")
        if any(not value.strip() for value in self.transform_refs):
            raise ValueError("Transform references must be non-empty")
