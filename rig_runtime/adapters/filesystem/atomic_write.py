"""Atomic filesystem publication with bounded retries for Windows file locks."""
from __future__ import annotations

import os
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Callable, TypeVar


_T = TypeVar("_T")
_RETRY_DELAYS = (0.05, 0.10, 0.20, 0.40, 0.80)
_WINDOWS_ACCESS_ERRORS = {5, 32, 33}  # access denied, sharing, byte-range lock


def retry_windows_access(operation: Callable[[], _T]) -> _T:
    """Retry only Windows access/locking errors, then preserve the OS error.

    Access denied can also be permanent. Six attempts, with 1.55 seconds of
    total backoff, bound that case without changing permissions or deleting
    the destination. Other filesystem errors fail immediately.
    """
    for attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            return operation()
        except OSError as exc:
            if getattr(exc, "winerror", None) not in _WINDOWS_ACCESS_ERRORS:
                raise
            if attempt == len(_RETRY_DELAYS):
                exc.add_note("IRIS file update still failed after 6 attempts (1.55 seconds of backoff).")
                raise
            time.sleep(_RETRY_DELAYS[attempt])
    raise AssertionError("unreachable")


def replace_with_retry(source: Path, destination: Path) -> None:
    """Keep a completed file/directory intact while retrying its atomic rename."""
    retry_windows_access(lambda: os.replace(str(source), str(destination)))


def atomic_write_text(path: Path, text: str) -> None:
    """Publish UTF-8 text from a private sibling file, never a shared .tmp name."""
    path = Path(path)

    def write_once():
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(prefix=".{}-".format(path.name), suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(text)
            os.replace(str(temporary), str(path))
        finally:
            # A failed cleanup must not hide the original publication error.
            with suppress(OSError):
                temporary.unlink(missing_ok=True)

    retry_windows_access(write_once)
