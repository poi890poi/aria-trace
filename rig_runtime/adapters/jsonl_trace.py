"""Append-only filesystem adapter for component invocation evidence."""

from __future__ import annotations

import threading
from pathlib import Path

from rig_runtime.domain import ComponentInvocation
from rig_runtime.evidence import canonical_json_bytes
from rig_runtime.ports import InvocationSink


class JsonlInvocationSink(InvocationSink):
    def __init__(self, path: Path, *, create_parent: bool = False) -> None:
        self.path = Path(path)
        if create_parent:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.parent.is_dir():
            raise ValueError("Invocation trace parent directory does not exist")
        self._lock = threading.Lock()

    def record(self, value: ComponentInvocation) -> None:
        line = canonical_json_bytes(value) + b"\n"
        with self._lock:
            with self.path.open("ab") as stream:
                stream.write(line)
                stream.flush()
