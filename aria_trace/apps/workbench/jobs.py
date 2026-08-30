"""Observable background-job state machine for the Workbench application."""

import threading
import time
from datetime import datetime, timezone
from typing import Callable, Mapping


ACTIVE_JOB_STATES = frozenset(("queued", "running"))
TERMINAL_JOB_STATES = frozenset(("complete", "failed", "canceled"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AnalysisJobManager:
    """Own background execution and its serializable lifecycle state."""

    def __init__(self, lock=None, thread_factory=threading.Thread) -> None:
        self._lock = lock or threading.RLock()
        self._thread_factory = thread_factory
        self._jobs = {}

    def snapshot(self) -> dict:
        with self._lock:
            return {key: dict(value) for key, value in self._jobs.items()}

    def has_active_job(self) -> bool:
        with self._lock:
            return any(
                value.get("status") in ACTIVE_JOB_STATES
                for value in self._jobs.values()
            )

    def queue(
        self,
        kind: str,
        request: Mapping,
        runner: Callable[[dict, Callable[[str], None]], object],
        preflight: Callable[[], None] = lambda: None,
    ) -> dict:
        """Queue one job after atomically validating application preconditions."""
        with self._lock:
            preflight()
            if self.has_active_job():
                raise RuntimeError("Wait for the active analysis task to finish")
            job_id = "{}-{}".format(kind, time.time_ns())
            serializable_request = {
                key: item
                for key, item in dict(request).items()
                if isinstance(item, (str, int, float, bool)) or item is None
            }
            now = _utc_now()
            self._jobs[kind] = {
                "job_id": job_id,
                "kind": kind,
                "status": "queued",
                "queued_utc": now,
                "started_utc": None,
                "finished_utc": None,
                "request": serializable_request,
                "error": None,
                "message": "Waiting for the background worker",
                "updated_utc": now,
            }

        def mutate(**changes) -> bool:
            with self._lock:
                job = self._jobs.get(kind)
                if not job or job.get("job_id") != job_id:
                    return False
                job.update(changes)
                return True

        def report(message: str) -> None:
            mutate(message=str(message), updated_utc=_utc_now())

        def work() -> None:
            started = _utc_now()
            if not mutate(
                status="running",
                started_utc=started,
                message="Starting analysis",
                updated_utc=started,
            ):
                return
            try:
                runner(dict(request), report)
            except Exception as exc:
                finished = _utc_now()
                mutate(
                    status="failed",
                    error="{}: {}".format(type(exc).__name__, exc),
                    message="Analysis failed",
                    finished_utc=finished,
                    updated_utc=finished,
                )
            else:
                finished = _utc_now()
                mutate(
                    status="complete",
                    message="Analysis complete",
                    finished_utc=finished,
                    updated_utc=finished,
                )

        self._thread_factory(
            target=work,
            name="aria-trace-analysis-{}".format(kind),
            daemon=True,
        ).start()
        return dict(self._jobs[kind])
