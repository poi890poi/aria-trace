"""Traceable lifecycle records for component accountability."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Tuple

from .envelope import DiagnosticValue
from .provenance import ConfigurationRef, ProducerRef
from .timing import TimePoint


class InvocationStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELED = "canceled"
    REJECTED = "rejected"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in {
            InvocationStatus.COMPLETED,
            InvocationStatus.CANCELED,
            InvocationStatus.REJECTED,
            InvocationStatus.FAILED,
        }


@dataclass(frozen=True)
class FailureRecord:
    code: str
    kind: str
    message: str
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.kind.strip() or not self.message.strip():
            raise ValueError("Failure code, kind, and message are required")


@dataclass(frozen=True)
class ComponentInvocation:
    invocation_id: str
    trace_id: str
    run_id: str
    producer: ProducerRef
    status: InvocationStatus
    started: TimePoint
    configuration: Optional[ConfigurationRef] = None
    finished: Optional[TimePoint] = None
    input_envelope_ids: Tuple[str, ...] = ()
    output_envelope_ids: Tuple[str, ...] = ()
    diagnostics: Tuple[DiagnosticValue, ...] = ()
    failure: Optional[FailureRecord] = None

    def __post_init__(self) -> None:
        if not self.invocation_id.strip():
            raise ValueError("Invocation ID is required")
        if not self.trace_id.strip() or not self.run_id.strip():
            raise ValueError("Trace ID and run ID are required")
        if any(not value.strip() for value in self.input_envelope_ids):
            raise ValueError("Input envelope IDs must be non-empty")
        if any(not value.strip() for value in self.output_envelope_ids):
            raise ValueError("Output envelope IDs must be non-empty")
        if self.status.terminal != (self.finished is not None):
            raise ValueError("Only terminal invocations have a finished timestamp")
        if self.finished is not None:
            if self.finished.clock_id != self.started.clock_id:
                raise ValueError("Invocation start and finish must use one clock")
            if self.finished.value_ns < self.started.value_ns:
                raise ValueError("Invocation finish precedes start")
        if self.status == InvocationStatus.FAILED and self.failure is None:
            raise ValueError("Failed invocation requires a failure record")
        if self.failure is not None and self.status not in {
            InvocationStatus.FAILED,
            InvocationStatus.REJECTED,
            InvocationStatus.CANCELED,
        }:
            raise ValueError("Failure record is incompatible with invocation status")
        names = [item.name for item in self.diagnostics]
        if len(names) != len(set(names)):
            raise ValueError("Invocation diagnostic names must be unique")
