"""Structured failures that preserve ownership and causal meaning."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Optional


class FailureKind(str, Enum):
    INVALID_INPUT = "invalid_input"
    INCOMPATIBLE_CAPABILITY = "incompatible_capability"
    UNAVAILABLE = "unavailable"
    CANCELED = "canceled"
    REJECTED = "rejected"
    INTERNAL = "internal"


class ComponentError(RuntimeError):
    def __init__(
        self,
        component_id: str,
        code: str,
        kind: FailureKind,
        message: str,
        *,
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if not component_id.strip() or not code.strip():
            raise ValueError("Component ID and error code are required")
        super().__init__(message)
        self.component_id = component_id
        self.code = code
        self.kind = kind
        self.details = dict(details or {})

    def to_dict(self) -> dict:
        return {
            "component_id": self.component_id,
            "code": self.code,
            "kind": self.kind.value,
            "message": str(self),
            "details": dict(self.details),
        }
