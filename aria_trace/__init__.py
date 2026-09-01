"""Stable, dependency-light contracts for IRIS and its host integrations."""

from .domain.envelope import DataEnvelope, DiagnosticValue, EnvelopeIdentity

__all__ = ["DataEnvelope", "DiagnosticValue", "EnvelopeIdentity"]
