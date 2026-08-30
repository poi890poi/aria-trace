"""Stable, dependency-light contracts for the AriaTrace system."""

from .domain.envelope import DataEnvelope, DiagnosticValue, EnvelopeIdentity

__all__ = ["DataEnvelope", "DiagnosticValue", "EnvelopeIdentity"]
