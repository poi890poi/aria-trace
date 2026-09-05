"""Capacity-one frame pump for low-latency live processing."""

import threading
import time
from typing import Generic, Optional, TypeVar


T = TypeVar("T")


class LatestFramePump(Generic[T]):
    """Read independently and expose only the newest undelivered source value."""

    def __init__(self, source, observer=None) -> None:
        self.source = source
        self.observer = observer
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = None
        self._latest = None
        self._latest_sequence = 0
        self._delivered_sequence = 0
        self._error = None
        self._observer_error = None
        self.dropped_before_processing = 0

    def start(self, timeout_s: float = 5.0) -> None:
        if self._thread is not None:
            raise RuntimeError("Frame pump is already started")
        self._thread = threading.Thread(
            target=self._run,
            name="acquisition-live-capture",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout_s):
            raise RuntimeError("Frame source did not start in time")
        if self._error is not None:
            raise RuntimeError("Frame source failed to start: {}".format(self._error))

    def _run(self) -> None:
        try:
            self.source.start()
            self._ready.set()
            while not self._stop.is_set():
                packet = self.source.read()
                if packet is None:
                    self._stop.wait(0.001)
                    continue
                with self._condition:
                    if self._latest_sequence > self._delivered_sequence:
                        self.dropped_before_processing += 1
                    self._latest = packet
                    self._latest_sequence += 1
                    self._condition.notify_all()
                if self.observer is not None:
                    try:
                        self.observer(packet)
                    except Exception as exc:
                        # Auxiliary consumers (recording, metrics, previews) must
                        # never terminate the latency-critical capture pump.
                        self._observer_error = "{}: {}".format(
                            type(exc).__name__, exc
                        )
        except Exception as exc:
            with self._condition:
                self._error = "{}: {}".format(type(exc).__name__, exc)
                self._ready.set()
                self._condition.notify_all()
        finally:
            self._ready.set()
            with self._condition:
                self._condition.notify_all()

    def read_latest(self, timeout_s: float = 0.25) -> Optional[T]:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._condition:
            while (
                self._latest_sequence <= self._delivered_sequence
                and self._error is None
                and not self._stop.is_set()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)
            if self._error is not None:
                raise RuntimeError("Frame capture failed: {}".format(self._error))
            if self._latest_sequence <= self._delivered_sequence:
                return None
            self._delivered_sequence = self._latest_sequence
            return self._latest

    @property
    def observer_error(self) -> Optional[str]:
        return self._observer_error

    def stop(self) -> None:
        self._stop.set()
        try:
            self.source.stop()
        finally:
            with self._condition:
                self._condition.notify_all()
            if self._thread is not None and self._thread is not threading.current_thread():
                self._thread.join(timeout=2.0)
