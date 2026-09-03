"""Strict deterministic JSON conversion for contract values."""

from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Mapping


def to_primitive(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return to_primitive(value.value)
    if is_dataclass(value):
        return {
            item.name: to_primitive(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Evidence mapping keys must be strings")
            result[key] = to_primitive(item)
        return result
    if isinstance(value, (tuple, list)):
        return [to_primitive(item) for item in value]
    raise TypeError("Unsupported evidence value: {}".format(type(value).__name__))


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        to_primitive(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
