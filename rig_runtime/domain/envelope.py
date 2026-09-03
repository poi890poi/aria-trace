"""Universal immutable metadata envelope for component values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Tuple, TypeVar

from .artifacts import ArtifactRef
from .provenance import ProvenanceInfo
from .quality import QualityInfo
from .schema import SchemaRef
from .spaces import SpaceRef
from .timing import TimingInfo

T = TypeVar("T")


@dataclass(frozen=True)
class EnvelopeIdentity:
    envelope_id: str
    trace_id: str
    parent_ids: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.envelope_id.strip():
            raise ValueError("Envelope ID is required")
        if not self.trace_id.strip():
            raise ValueError("Trace ID is required")
        if any(not value.strip() for value in self.parent_ids):
            raise ValueError("Parent envelope IDs must be non-empty")


@dataclass(frozen=True)
class DiagnosticValue:
    name: str
    value: Any

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Diagnostic name is required")


@dataclass(frozen=True)
class DataEnvelope(Generic[T]):
    schema: SchemaRef
    value: T
    identity: EnvelopeIdentity
    timing: TimingInfo
    space: SpaceRef
    provenance: ProvenanceInfo
    quality: QualityInfo = QualityInfo()
    diagnostics: Tuple[DiagnosticValue, ...] = ()
    artifacts: Tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        names = [item.name for item in self.diagnostics]
        if len(names) != len(set(names)):
            raise ValueError("Diagnostic names must be unique")
