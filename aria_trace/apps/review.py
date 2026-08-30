"""Local web reviewer for synchronized acquisition sessions."""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

from aria_trace.adapters.filesystem.annotations import AnnotationStore
from aria_trace.adapters.filesystem.session import SessionReader


class ReviewState:
    def __init__(self, session: Path) -> None:
        self.reader = SessionReader(session)
        self.annotations = AnnotationStore(session)
        self._captures = {}
        self._lock = threading.Lock()

    def descriptor(self) -> dict:
        summary = self.reader.summary()
        summary["stream_order"] = list(self.reader.frames_by_stream)
        summary["frame_counts"] = {
            stream: len(frames) for stream, frames in self.reader.frames_by_stream.items()
        }
        summary["annotations"] = self.annotations.list()
        return summary

    def frame_info(self, stream_id: str, index: int) -> dict:
        frames = self.reader.frames_by_stream.get(stream_id)
        if frames is None or index < 0 or index >= len(frames):
            raise IndexError("Unknown frame {}:{}".format(stream_id, index))
        frame = frames[index]
        return {
            "frame": frame,
            "nearby_inputs": self.reader.nearby_inputs(frame["session_time_ns"]),
            "online_features": self.reader.online_features_for_frame(stream_id, index),
            "annotations": [
                item for item in self.annotations.list()
                if item["stream_id"] == stream_id and item["frame_index"] == index
            ],
        }

    def add_annotation(self, value: dict) -> dict:
        stream_id = value.get("stream_id", "")
        index = int(value.get("frame_index", -1))
        frames = self.reader.frames_by_stream.get(stream_id)
        if frames is None or index < 0 or index >= len(frames):
            raise ValueError("Unknown frame {}:{}".format(stream_id, index))
        frame = frames[index]
        context = self.reader.manifest.get("context", {})
        return self.annotations.add(
            value.get("kind", ""),
            frame["session_time_ns"],
            stream_id,
            index,
            value.get("portal_id") or context.get("portal_id"),
            value.get("route_id") or context.get("route_id"),
            value.get("note"),
        )

    def frame_jpeg(self, stream_id: str, index: int) -> bytes:
        frames = self.reader.frames_by_stream.get(stream_id)
        if frames is None or index < 0 or index >= len(frames):
            raise IndexError("Unknown frame {}:{}".format(stream_id, index))
        with self._lock:
            capture = self._captures.get(stream_id)
            if capture is None:
                capture = cv2.VideoCapture(str(self.reader.video_path(stream_id)))
                if not capture.isOpened():
                    raise RuntimeError("Cannot open video for {}".format(stream_id))
                self._captures[stream_id] = capture
            capture.set(cv2.CAP_PROP_POS_FRAMES, frames[index]["frame_index"])
            ok, image = capture.read()
            if not ok:
                raise RuntimeError("Cannot decode frame {}:{}".format(stream_id, index))
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError("Cannot encode review frame")
            return encoded.tobytes()

    def close(self) -> None:
        for capture in self._captures.values():
            capture.release()
        self._captures.clear()


def make_handler(state: ReviewState):
    static_path = Path(__file__).resolve().parent / "static" / "review.html"

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    self._send(200, "text/html; charset=utf-8", static_path.read_bytes())
                elif parsed.path == "/api/session":
                    body = json.dumps(state.descriptor()).encode("utf-8")
                    self._send(200, "application/json", body)
                elif parsed.path == "/api/frame-info":
                    stream = query.get("stream", [""])[0]
                    index = int(query.get("index", ["0"])[0])
                    body = json.dumps(state.frame_info(stream, index)).encode("utf-8")
                    self._send(200, "application/json", body)
                elif parsed.path == "/api/frame.jpg":
                    stream = query.get("stream", [""])[0]
                    index = int(query.get("index", ["0"])[0])
                    self._send(200, "image/jpeg", state.frame_jpeg(stream, index))
                elif parsed.path == "/api/annotations":
                    body = json.dumps(state.annotations.list()).encode("utf-8")
                    self._send(200, "application/json", body)
                else:
                    self._send(404, "text/plain; charset=utf-8", b"Not found")
            except (IndexError, ValueError) as exc:
                self._send(400, "text/plain; charset=utf-8", str(exc).encode("utf-8"))
            except Exception as exc:
                self._send(500, "text/plain; charset=utf-8", str(exc).encode("utf-8"))

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/api/annotations":
                self._send(404, "text/plain; charset=utf-8", b"Not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 65536:
                    raise ValueError("Invalid request size")
                value = json.loads(self.rfile.read(length).decode("utf-8"))
                annotation = state.add_annotation(value)
                self._send(201, "application/json", json.dumps(annotation).encode("utf-8"))
            except (ValueError, json.JSONDecodeError) as exc:
                self._send(400, "text/plain; charset=utf-8", str(exc).encode("utf-8"))
            except Exception as exc:
                self._send(500, "text/plain; charset=utf-8", str(exc).encode("utf-8"))

        def do_DELETE(self):
            parsed = urlparse(self.path)
            if parsed.path != "/api/annotations":
                self._send(404, "text/plain; charset=utf-8", b"Not found")
                return
            try:
                annotation_id = parse_qs(parsed.query).get("id", [""])[0]
                if not annotation_id:
                    raise ValueError("Annotation ID is required")
                deleted = state.annotations.delete(annotation_id)
                self._send(200, "application/json", json.dumps(deleted).encode("utf-8"))
            except (ValueError, KeyError) as exc:
                self._send(400, "text/plain; charset=utf-8", str(exc).encode("utf-8"))
            except Exception as exc:
                self._send(500, "text/plain; charset=utf-8", str(exc).encode("utf-8"))

        def log_message(self, format, *args):
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    state = ReviewState(args.session)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print("Reviewing {}".format(args.session.resolve()))
    print("Open http://{}:{}/".format(args.host, server.server_address[1]))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        state.close()


if __name__ == "__main__":
    main()
