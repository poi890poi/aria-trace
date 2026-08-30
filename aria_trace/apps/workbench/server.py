"""HTTP runtime concerns for the acquisition Workbench.

This module owns listener behavior and instance discovery.  It deliberately has
no dependency on acquisition devices, tracking algorithms, or Workbench state.
"""

import json
import sys
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


WORKBENCH_SERVICE = "aria-trace-workbench"


def is_client_disconnect(exc: BaseException) -> bool:
    """Return whether a request ended because the HTTP client went away."""
    return isinstance(
        exc,
        (BrokenPipeError, ConnectionAbortedError, ConnectionResetError),
    )


class WorkbenchHttpServer(ThreadingHTTPServer):
    """Threaded HTTP server that treats browser disconnects as routine."""

    daemon_threads = True

    def handle_error(self, request, client_address) -> None:
        if is_client_disconnect(sys.exc_info()[1]):
            return
        super().handle_error(request, client_address)


def connect_host(host: str) -> str:
    """Return a loopback address suitable for connecting to a bound host."""
    return "127.0.0.1" if host in ("", "0.0.0.0", "::") else host


def discover_workbench_instance(host: str, port: int, timeout: float = 0.75):
    """Identify a Workbench already listening on host/port, including old builds."""
    base = "http://{}:{}".format(connect_host(host), int(port))
    try:
        with urlopen(base + "/api/instance", timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
        if value.get("service") == WORKBENCH_SERVICE:
            return value
    except HTTPError as exc:
        if exc.code != 404:
            return None
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return None

    # Workbench builds predating /api/instance can still be distinguished from
    # unrelated services so a duplicate launch produces useful guidance. Check
    # the static shell before the more expensive full-state descriptor.
    try:
        with urlopen(base + "/", timeout=timeout) as response:
            body = response.read(16384).decode("utf-8", errors="replace")
        if "<title>AriaTrace Recorder</title>" in body:
            return {
                "service": WORKBENCH_SERVICE,
                "legacy": True,
                "url": base + "/",
            }
    except (OSError, URLError, ValueError):
        return None

    try:
        with urlopen(base + "/api/state", timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
        if "session_labels" in value and "sources" in value:
            return {
                "service": WORKBENCH_SERVICE,
                "legacy": True,
                "url": base + "/",
            }
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return None
    return None


def occupied_port_message(host: str, port: int, existing) -> str:
    """Describe why a Workbench listener cannot claim an occupied endpoint."""
    endpoint = "http://{}:{}/".format(connect_host(host), int(port))
    if not existing:
        return (
            "Cannot start the Workbench at {} because the address is already in "
            "use by another process. This command did not replace or stop it."
        ).format(endpoint)
    if existing.get("legacy"):
        return (
            "An older AriaTrace Workbench is already running at {}. Stop it with "
            "Ctrl+C in the terminal that started it, then launch this version. "
            "This command did not replace or stop it."
        ).format(endpoint)
    details = ["PID {}".format(existing.get("process_id", "unknown"))]
    if existing.get("started_utc"):
        details.append("started {}".format(existing["started_utc"]))
    if existing.get("session_root"):
        details.append("sessions {}".format(existing["session_root"]))
    return (
        "AriaTrace Workbench instance {instance_id} is already running at {url} "
        "({details}). Stop it with Ctrl+C in its owning terminal if you intend to "
        "restart it. This command did not replace or stop it."
    ).format(
        instance_id=existing.get("instance_id", "unknown"),
        url=existing.get("url") or endpoint,
        details=", ".join(details),
    )
