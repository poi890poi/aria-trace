"""Explainable component quality measurements and decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Tuple


class CheckStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    NOT_EVALUATED = "not_evaluated"


class Decision(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class QualityMetric:
    name: str
    value: Any
    unit: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Quality metric name is required")


@dataclass(frozen=True)
class QualityCheck:
    name: str
    status: CheckStatus
    reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Quality check name is required")


@dataclass(frozen=True)
class QualityInfo:
    decision: Decision = Decision.UNKNOWN
    metrics: Tuple[QualityMetric, ...] = ()
    checks: Tuple[QualityCheck, ...] = ()
    warnings: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        metric_names = [item.name for item in self.metrics]
        check_names = [item.name for item in self.checks]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("Quality metric names must be unique")
        if len(check_names) != len(set(check_names)):
            raise ValueError("Quality check names must be unique")
        if any(not warning.strip() for warning in self.warnings):
            raise ValueError("Quality warnings must be non-empty")
