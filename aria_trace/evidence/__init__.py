"""Serialization helpers for independently inspectable component evidence."""

from .json_codec import canonical_json_bytes, to_primitive

__all__ = ["canonical_json_bytes", "to_primitive"]
