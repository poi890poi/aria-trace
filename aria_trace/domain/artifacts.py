"""Storage-independent references to evidence and persisted results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _validate_sha256(value: Optional[str]) -> None:
    if value is None:
        return
    if len(value) != 64 or any(character not in "0123456789abcdefABCDEF" for character in value):
        raise ValueError("SHA-256 must contain 64 hexadecimal characters")


@dataclass(frozen=True)
class ArtifactRef:
    artifact_id: str
    kind: str
    media_type: str
    locator: Optional[str] = None
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("Artifact ID is required")
        if not self.kind.strip():
            raise ValueError("Artifact kind is required")
        if not self.media_type.strip():
            raise ValueError("Artifact media type is required")
        _validate_sha256(self.sha256)
        if self.size_bytes is not None and self.size_bytes < 0:
            raise ValueError("Artifact size must be non-negative")
