"""Explicit timestamp and clock-domain values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TimePoint:
    value_ns: int
    clock_id: str

    def __post_init__(self) -> None:
        if self.value_ns < 0:
            raise ValueError("Timestamp must be non-negative")
        if not self.clock_id.strip():
            raise ValueError("Clock ID is required; use 'unknown' explicitly")


@dataclass(frozen=True)
class TimingInfo:
    """Known stages of one observation without concealing unknown delay."""

    source: Optional[TimePoint] = None
    capture: Optional[TimePoint] = None
    receive: Optional[TimePoint] = None
    publication: Optional[TimePoint] = None

    @staticmethod
    def difference_ns(later: Optional[TimePoint], earlier: Optional[TimePoint]) -> Optional[int]:
        if later is None or earlier is None:
            return None
        if later.clock_id != earlier.clock_id:
            return None
        return later.value_ns - earlier.value_ns

    @property
    def capture_to_receive_ns(self) -> Optional[int]:
        return self.difference_ns(self.receive, self.capture)

    @property
    def capture_to_publication_ns(self) -> Optional[int]:
        return self.difference_ns(self.publication, self.capture)
