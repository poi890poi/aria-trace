"""Universal port families without hardware or framework dependencies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Mapping, Optional, TypeVar

from aria_trace.domain import ArtifactRef, DataEnvelope

I = TypeVar("I")
O = TypeVar("O")
T = TypeVar("T")
Q = TypeVar("Q")


class CancellationToken(ABC):
    @abstractmethod
    def is_cancelled(self) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class ComponentContext:
    trace_id: str
    run_id: str
    configuration: Mapping[str, Any]
    cancellation: Optional[CancellationToken] = None

    def __post_init__(self) -> None:
        if not self.trace_id.strip() or not self.run_id.strip():
            raise ValueError("Trace ID and run ID are required")

    def raise_if_cancelled(self) -> None:
        if self.cancellation is not None and self.cancellation.is_cancelled():
            from aria_trace.domain import ComponentError, FailureKind

            raise ComponentError(
                "workflow",
                "cancelled",
                FailureKind.CANCELED,
                "Component execution was cancelled",
            )


class Source(ABC, Generic[T]):
    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self) -> DataEnvelope[T]:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError


class Transform(ABC, Generic[I, O]):
    @abstractmethod
    def process(self, value: DataEnvelope[I], context: ComponentContext) -> DataEnvelope[O]:
        raise NotImplementedError


class Estimator(ABC, Generic[I, O]):
    @abstractmethod
    def update(self, value: DataEnvelope[I], context: ComponentContext) -> DataEnvelope[O]:
        raise NotImplementedError


class Sink(ABC, Generic[T]):
    @abstractmethod
    def write(self, value: DataEnvelope[T]) -> ArtifactRef:
        raise NotImplementedError


class Repository(ABC, Generic[Q, T]):
    @abstractmethod
    def resolve(self, query: Q) -> Optional[T]:
        raise NotImplementedError

    @abstractmethod
    def save(self, value: T) -> T:
        raise NotImplementedError
