"""Coalesce retention after accepted calibration publications finish."""

from contextvars import ContextVar
from functools import wraps
import logging

from rig_runtime.adapters.filesystem.profile_retention import purge_profiles


_pending = ContextVar("calibration_profile_cleanup", default=None)
_logger = logging.getLogger(__name__)


def _cleanup(registry):
    try:
        report = purge_profiles(registry, dry_run=False)
    except Exception as exc:
        # Maintenance must never turn an already activated calibration into a
        # failure. The retention journal and next successful run permit retry.
        _logger.warning("Automatic profile cleanup deferred for %s: %s: %s",
                        registry.root, type(exc).__name__, exc)
        return
    _logger.info("Automatic profile cleanup for %s: %d revisions, %d evidence entries removed",
                 registry.root, len(report["removed_revisions"]), len(report["removed_evidence"]))
    for message in report["warnings"]:
        _logger.warning("Automatic profile cleanup for %s: %s", registry.root, message)


def request_profile_cleanup(registry):
    """Call only after successful publication and activation of a calibration."""
    pending = _pending.get()
    if pending is None:
        _cleanup(registry)
    else:
        pending[str(registry.root)] = registry


def calibration_cleanup_boundary(function):
    """Defer nested component requests until the outer calibration has exited.

    A failed run with no accepted publications requests nothing. Accepted
    components still get maintenance if a later independent component fails.
    Do not attach this boundary to per-frame resolution or raw registry writes.
    """
    @wraps(function)
    def wrapped(*args, **kwargs):
        if _pending.get() is not None:
            return function(*args, **kwargs)
        pending = {}
        token = _pending.set(pending)
        try:
            return function(*args, **kwargs)
        finally:
            _pending.reset(token)
            for registry in pending.values():
                _cleanup(registry)
    return wrapped
