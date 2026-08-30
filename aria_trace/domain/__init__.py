"""Universal values exchanged across AriaTrace component boundaries."""

from .artifacts import ArtifactRef
from .envelope import DataEnvelope, DiagnosticValue, EnvelopeIdentity
from .errors import ComponentError, FailureKind
from .frames import ImageFrame
from .provenance import (
    ConfigurationRef,
    ProducerRef,
    ProvenanceInfo,
    SourceRef,
)
from .quality import CheckStatus, Decision, QualityCheck, QualityInfo, QualityMetric
from .schema import SchemaRef
from .spaces import SpaceRef
from .timing import TimePoint, TimingInfo

__all__ = [
    "ArtifactRef",
    "CheckStatus",
    "ComponentError",
    "ConfigurationRef",
    "DataEnvelope",
    "Decision",
    "DiagnosticValue",
    "EnvelopeIdentity",
    "FailureKind",
    "ImageFrame",
    "ProducerRef",
    "ProvenanceInfo",
    "QualityCheck",
    "QualityInfo",
    "QualityMetric",
    "SchemaRef",
    "SourceRef",
    "SpaceRef",
    "TimePoint",
    "TimingInfo",
]
