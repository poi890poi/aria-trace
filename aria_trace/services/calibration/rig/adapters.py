"""Minimal integration interfaces used by the calibration core."""

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

from .contracts import ControlEvent, FrameSample, SignalObservation


class FrameStream(ABC):
    """Provide timestamped images without exposing a concrete camera driver."""

    @abstractmethod
    def read(self) -> Optional[FrameSample]:
        raise NotImplementedError


class TargetPresenter(ABC):
    """Present a calibration target through any external display adapter."""

    @abstractmethod
    def present(self, target_id: str, payload: Mapping[str, Any]) -> ControlEvent:
        raise NotImplementedError


class AlternatingStimulus(ABC):
    """Request one state of a timestamped alternating calibration signal."""

    @abstractmethod
    def set_state(self, state: str, token: str) -> ControlEvent:
        raise NotImplementedError


class SignalObserver(ABC):
    """Convert a frame into state probabilities for latency measurement."""

    @abstractmethod
    def observe(self, packet: FrameSample) -> SignalObservation:
        raise NotImplementedError
