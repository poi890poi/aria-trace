"""HTTP request/response translation for the Workbench application."""

import json
import math
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def _strict_json_value(value):
    """Return browser-compatible JSON data without NaN or infinity tokens."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _strict_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_json_value(item) for item in value]
    return value


def make_handler(state):
    static_path = (
        Path(__file__).resolve().parents[3]
        / "acquisition"
        / "static"
        / "recorder.html"
    )

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, content_type: str, body: bytes) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                # Browsers routinely cancel obsolete polling and image requests.
                # The response no longer has a recipient, so there is nothing to
                # retry or report as a Workbench failure.
                self.close_connection = True

        def _json(self, status: int, value: object) -> None:
            self._send(
                status,
                "application/json",
                json.dumps(
                    _strict_json_value(value), allow_nan=False
                ).encode("utf-8"),
            )

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > 65536:
                raise ValueError("Invalid request size")
            return (
                json.loads(self.rfile.read(length).decode("utf-8"))
                if length
                else {}
            )

        def do_GET(self):
            try:
                parsed = urlparse(self.path)
                path = parsed.path
                if path == "/":
                    self._send(
                        200,
                        "text/html; charset=utf-8",
                        static_path.read_bytes(),
                    )
                elif path == "/api/state":
                    self._json(200, state.descriptor())
                elif path == "/api/instance":
                    self._json(200, state.instance_descriptor())
                elif path == "/api/android/devices":
                    self._json(200, state.android_devices())
                elif path == "/api/capture-sources":
                    self._json(200, state.capture_source_inventory())
                elif path == "/api/hud":
                    self._json(200, state.hud_descriptor())
                elif path == "/api/minimap-calibration/image":
                    query = parse_qs(parsed.query)
                    body = state.minimap_calibration_image(query.get("game_id", [""])[0], query.get("calibration_id", [""])[0], query.get("name", [""])[0])
                    self._send(200, "image/png", body)
                elif path == "/api/map-stitch/image":
                    query = parse_qs(parsed.query)
                    body = state.map_stitch_image(
                        query.get("game_id", [""])[0],
                        query.get("stitch_id", [""])[0],
                        query.get("name", [""])[0],
                    )
                    self._send(200, "image/png", body)
                elif path == "/api/scene-yaw/image":
                    query = parse_qs(parsed.query)
                    body = state.scene_yaw_image(
                        query.get("game_id", [""])[0],
                        query.get("calibration_id", [""])[0],
                        query.get("name", [""])[0],
                    )
                    self._send(200, "image/png", body)
                elif path == "/api/map-atlas/image":
                    query = parse_qs(parsed.query)
                    body = state.map_atlas_image(
                        query.get("game_id", [""])[0],
                        query.get("atlas_id", [""])[0],
                        query.get("name", [""])[0],
                    )
                    self._send(200, "image/png", body)
                elif path == "/api/teleport-analysis/image":
                    query = parse_qs(parsed.query)
                    body = state.teleport_behavior_image(
                        query.get("game_id", [""])[0],
                        query.get("behavior_id", [""])[0],
                        query.get("name", [""])[0],
                    )
                    self._send(200, "image/png", body)
                elif path == "/api/tracker/overlay":
                    query = parse_qs(parsed.query)
                    self._send(
                        200,
                        "image/png",
                        state.live_tracker_overlay_image(
                            compact=query.get("compact", [""])[0] == "1"
                        ),
                    )
                elif path == "/api/tracker/minimap-route-overlay":
                    self._send(
                        200,
                        "image/png",
                        state.live_tracker_minimap_route_overlay_image(),
                    )
                elif path == "/api/live-tracking/image":
                    query = parse_qs(parsed.query)
                    content_type, body = state.live_tracking_image(
                        query.get("game_id", [""])[0],
                        query.get("tracking_id", [""])[0],
                        query.get("fix_id", [""])[0],
                        query.get("name", [""])[0],
                    )
                    self._send(200, content_type, body)
                else:
                    self._send(404, "text/plain; charset=utf-8", b"Not found")
            except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
                self._send(400, "text/plain; charset=utf-8", str(exc).encode("utf-8"))
            except Exception as exc:
                self._send(
                    500,
                    "text/plain; charset=utf-8",
                    str(exc).encode("utf-8"),
                )

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                value = self._body()
                if path == "/api/arm":
                    result = state.arm(value)
                elif path == "/api/disarm":
                    result = state.disarm()
                elif path == "/api/session/start":
                    result = state.start_session(value)
                elif path == "/api/android/straight-forward":
                    result = state.run_android_straight_forward(value)
                elif path == "/api/session/label":
                    result = state.label_session(
                        str(value.get("session_key") or ""),
                        str(value.get("label") or ""),
                    )
                elif path == "/api/session/delete":
                    result = state.delete_session(
                        str(value.get("session_key") or "")
                    )
                elif path == "/api/session/open-folder":
                    result = state.open_session_folder(
                        str(value.get("session_key") or "")
                    )
                elif path == "/api/hud/toggle":
                    result = state.set_hud_enabled(bool(value.get("enabled")))
                elif path == "/api/take/queue":
                    result = state.queue_next_take()
                elif path == "/api/take/cancel":
                    result = state.cancel_active_take()
                elif path == "/api/take/confirm":
                    result = state.confirm_take(int(value["run_index"]))
                elif path == "/api/compile":
                    result = state.compile_and_evaluate()
                elif path == "/api/profile/draft":
                    result = state.save_profile_draft(value)
                elif path == "/api/minimap-calibration/run":
                    result = state.queue_minimap_calibration(value)
                elif path == "/api/pose-verification/run":
                    result = state.queue_pose_verification(value)
                elif path == "/api/map-stitch/run":
                    result = state.queue_map_stitch(value)
                elif path == "/api/map-atlas/run":
                    result = state.queue_map_atlas(value)
                elif path == "/api/route-tracking/compile":
                    result = state.queue_route_tracking_compile(value)
                elif path == "/api/scene-yaw/run":
                    result = state.queue_scene_yaw_calibration(value)
                elif path == "/api/teleport-analysis/run":
                    result = state.queue_teleport_analysis(value)
                elif path == "/api/tracker/start":
                    result = state.start_live_tracker(value)
                elif path == "/api/tracker/stop":
                    result = state.stop_live_tracker()
                else:
                    self._send(404, "text/plain; charset=utf-8", b"Not found")
                    return
                self._json(200, result)
            except (ValueError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
                self._send(
                    400,
                    "text/plain; charset=utf-8",
                    str(exc).encode("utf-8"),
                )
            except Exception as exc:
                self._send(
                    500,
                    "text/plain; charset=utf-8",
                    str(exc).encode("utf-8"),
                )

        def log_message(self, format, *args):
            return

    return Handler
