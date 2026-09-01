"""Universal values exchanged across AriaTrace component boundaries."""

from .artifacts import ArtifactRef
from .behaviors import TeleportBehaviorSample
from .envelope import DataEnvelope, DiagnosticValue, EnvelopeIdentity
from .errors import ComponentError, FailureKind
from .execution import (
    ComponentInvocation,
    FailureRecord,
    InvocationStatus,
)
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
from .spatial import (
    PIXEL_SPACE_KIND,
    SPATIAL_SCHEMA_VERSION,
    bind_geometry,
    normalize_legacy_geometry,
    raster_space,
    require_same_space,
    require_spatial_geometry,
    transform_circle_similarity,
    transform_point,
    validate_raster_space,
)
from .timing import TimePoint, TimingInfo

__all__ = [
    "ArtifactRef",
    "CheckStatus",
    "ComponentError",
    "ComponentInvocation",
    "ConfigurationRef",
    "DataEnvelope",
    "Decision",
    "DiagnosticValue",
    "EnvelopeIdentity",
    "FailureKind",
    "FailureRecord",
    "ImageFrame",
    "InvocationStatus",
    "ProducerRef",
    "ProvenanceInfo",
    "QualityCheck",
    "QualityInfo",
    "QualityMetric",
    "SchemaRef",
    "SourceRef",
    "SpaceRef",
    "PIXEL_SPACE_KIND",
    "SPATIAL_SCHEMA_VERSION",
    "bind_geometry",
    "normalize_legacy_geometry",
    "raster_space",
    "require_same_space",
    "require_spatial_geometry",
    "transform_circle_similarity",
    "transform_point",
    "validate_raster_space",
    "TimePoint",
    "TimingInfo",
    "TeleportBehaviorSample",
]
