"""Device-neutral image-frame payload."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ImageFrame:
    stream_id: str
    image: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)
    dropped_before: int = 0

    def __post_init__(self) -> None:
        if not self.stream_id.strip():
            raise ValueError("Frame stream ID is required")
        if self.image is None:
            raise ValueError("Frame image is required")
        if self.dropped_before < 0:
            raise ValueError("Dropped-frame count must be non-negative")
