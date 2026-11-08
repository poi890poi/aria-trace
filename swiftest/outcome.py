"""Intent outcomes that remain serializable after failures and teardown."""

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from functools import wraps
import inspect
import math
import time


def json_value(value, depth=0):
    if depth > 8:
        return "<nested value>"
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, dict):
        return {str(k): json_value(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v, depth + 1) for v in value]
    if is_dataclass(value):
        return json_value(asdict(value), depth + 1)
    if hasattr(value, "shape"):
        return {"type": type(value).__name__, "shape": list(value.shape)}
    return {"type": type(value).__name__}


def tracked(method):
    signature = inspect.signature(method)

    @wraps(method)
    def call(self, *args, **kwargs):
        details = dict(signature.bind(self, *args, **kwargs).arguments)
        details.pop("self", None)
        # Text entry can contain credentials; record its intent without its payload.
        if method.__name__ == "text":
            details = {"value": "<text omitted>"}
        with self.intent(method.__name__, **details):
            result = method(self, *args, **kwargs)
            self._intent_result = json_value(result)
            self._intent_confirmation = (
                "postcondition" if method.__name__ == "touch" and kwargs.get("expect") is not None
                and result is not False else
                "dispatch" if method.__name__ in ("touch", "swipe", "key", "text", "open")
                and result is not False else "observation"
            )
            return result
    return call


class Outcomes:
    def reset_outcome(self):
        self._intent_depth = 0
        self._intent_result = None
        self._intent_confirmation = None
        self._outcome_status = "not_started"
        self.last_valid_result = self.failed_intent = None
        self._failure_exception = None

    @contextmanager
    def intent(self, name, **details):
        """Group actions under a human-readable intent; errors still propagate."""
        outer = self._intent_depth == 0
        started = time.monotonic()
        if outer:
            self._intent_result = None
            self._intent_confirmation = "completed"
            self._outcome_status = "running"
        self._intent_depth += 1
        try:
            yield
        except BaseException as exc:
            if outer:
                self.record_failure(name, exc, **details)
            raise
        else:
            if outer:
                self.last_valid_result = dict(
                    intent=str(name), details=json_value(details), result=self._intent_result,
                    confirmation=self._intent_confirmation,
                    elapsed_s=round(time.monotonic() - started, 6),
                )
                self.failed_intent = None
                self._failure_exception = None
                self._outcome_status = "succeeded"
        finally:
            self._intent_depth -= 1

    def checkpoint(self, value, name="checkpoint"):
        """Explicitly report a script-defined valid result, including partial progress."""
        if self._intent_depth:
            self._intent_result = json_value(value)
            self._intent_confirmation = "checkpoint"
            self.last_valid_result = dict(intent=str(name), details={},
                                          result=self._intent_result,
                                          confirmation="checkpoint", elapsed_s=0.0)
        else:
            with self.intent(name):
                self._intent_result = json_value(value)
                self._intent_confirmation = "checkpoint"
        return value

    def record_failure(self, name, exc, **details):
        self._failure_exception = exc
        self.failed_intent = dict(intent=str(name), details=json_value(details),
                                  error=str(exc), error_type=type(exc).__name__)
        self._outcome_status = "failed"

    def complete_outcome(self, value=None, name="script result"):
        if value is not None:
            self.checkpoint(value, name)
        self._outcome_status = "succeeded"
        self.failed_intent = None
        self._failure_exception = None
        return self.outcome()

    def outcome(self):
        return json_value(dict(status=self._outcome_status,
                               last_valid_result=self.last_valid_result,
                               failed_intent=self.failed_intent))
