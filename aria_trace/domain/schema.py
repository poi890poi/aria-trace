"""Payload schema identity without a serialization-library dependency."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchemaRef:
    """Stable identity of a component payload contract."""

    schema_id: str
    version: str

    def __post_init__(self) -> None:
        if not self.schema_id.strip():
            raise ValueError("Schema ID is required")
        if not self.version.strip():
            raise ValueError("Schema version is required")
