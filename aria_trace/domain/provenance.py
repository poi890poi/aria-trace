"""Producer, configuration, and source identity for component results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .artifacts import _validate_sha256


@dataclass(frozen=True)
class ProducerRef:
    component_id: str
    component_version: str
    algorithm_id: Optional[str] = None
    git_revision: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.component_id.strip():
            raise ValueError("Producer component ID is required")
        if not self.component_version.strip():
            raise ValueError("Producer component version is required")


@dataclass(frozen=True)
class ConfigurationRef:
    configuration_id: str
    sha256: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.configuration_id.strip():
            raise ValueError("Configuration ID is required")
        _validate_sha256(self.sha256)


@dataclass(frozen=True)
class SourceRef:
    source_id: str
    kind: str
    sha256: Optional[str] = None
    locator: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("Source ID is required")
        if not self.kind.strip():
            raise ValueError("Source kind is required")
        _validate_sha256(self.sha256)


@dataclass(frozen=True)
class ProvenanceInfo:
    producer: ProducerRef
    configuration: Optional[ConfigurationRef] = None
    sources: Tuple[SourceRef, ...] = ()
    dependency_versions: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        names = [name for name, _ in self.dependency_versions]
        if any(not name.strip() for name in names):
            raise ValueError("Dependency names must be non-empty")
        if len(names) != len(set(names)):
            raise ValueError("Dependency names must be unique")
