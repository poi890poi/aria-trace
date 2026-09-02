"""Causal cursor-pose benchmark contracts and reporting helpers."""

from .stateful import (
    FALLBACK_FACTORIES,
    Measurement,
    apply_fallback_strategy,
    summarize_e2e_rows,
)

__all__ = (
    "FALLBACK_FACTORIES",
    "Measurement",
    "apply_fallback_strategy",
    "summarize_e2e_rows",
)

