"""Replaceable phone target presenter and the built-in LAN implementation."""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

import cv2
import numpy as np

from aria_trace.services.calibration.rig.geometry import CharucoLayout, generate_charuco_target


@dataclass(frozen=True)
class Presentation:
    token: str
    mode: str
    issued_time_ns: int
    revision: int


class PhoneTargetAdapter(ABC):
    """Customization point for a browser, native phone app, ADB, or HDMI target."""

    adapter_id = "custom"

    @abstractmethod
    def start(self, layout: CharucoLayout) -> str:
        """Start presenting and return operator-facing connection information."""

    @abstractmethod
    def present_charuco(self) -> Presentation:
        pass

    @abstractmethod
    def present_image(self, image: np.ndarray, label: str) -> Presentation:
        pass

    @abstractmethod
    def present_signal(self, state: str, token: str) -> Presentation:
        pass

    @abstractmethod
    def telemetry(self) -> Mapping[str, Any]:
        """Return presentation telemetry using host-monotonic receive times.

        Controlled image capture expects an ``acknowledgements`` sequence whose
        entries identify ``revision``, set ``painted`` true, and include
        ``server_receive_time_ns`` after the target has reached the display.
        """

    @abstractmethod
    def stop(self) -> None:
        pass


_PHONE_PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport"
content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover">
<title>IRIS calibration target</title><style>
html,body{margin:0;width:100%;height:100%;overflow:hidden;background:#111;color:#fff;
font:16px system-ui,sans-serif}canvas{position:fixed;inset:0;width:100vw;height:100vh}
#gate{position:fixed;z-index:2;inset:0;display:grid;place-items:center;background:#151515}
button{font:600 20px system-ui;padding:18px 28px;border:0;border-radius:10px}
#note{position:fixed;z-index:3;left:10px;bottom:10px;background:#000b;padding:6px 9px}
</style></head><body><canvas id="target"></canvas><div id="gate"><button id="go">
Enter fullscreen calibration</button></div><div id="note">Waiting for target</div><script>
const canvas=document.querySelector('#target'),ctx=canvas.getContext('2d');
const note=document.querySelector('#note'),gate=document.querySelector('#gate');
const autoStart=new URLSearchParams(location.search).get('autostart')==='1';
let revision=-1,lastMode='',image=null;
function size(){const d=window.devicePixelRatio||1; canvas.width=Math.round(innerWidth*d);
 canvas.height=Math.round(innerHeight*d); report();}
function report(){fetch('/telemetry',{method:'POST',headers:{'content-type':'application/json'},
 body:JSON.stringify({inner_width:innerWidth,inner_height:innerHeight,
 pixel_ratio:devicePixelRatio||1,canvas_width:canvas.width,canvas_height:canvas.height,
 screen_width:screen.width,screen_height:screen.height,
 screen_orientation_type:screen.orientation?.type||'',screen_orientation_angle:screen.orientation?.angle||0,
 window_orientation:window.orientation||0,fullscreen:!!document.fullscreenElement,
 visual_viewport_width:visualViewport?.width||innerWidth,
 visual_viewport_height:visualViewport?.height||innerHeight,
 user_agent:navigator.userAgent})}).catch(()=>{});}
function drawImage(){ctx.setTransform(1,0,0,1,0,0);ctx.clearRect(0,0,canvas.width,canvas.height);
 ctx.drawImage(image,0,0,canvas.width,canvas.height);}
function draw(mode){if(mode==='black'||mode==='white'){ctx.fillStyle=mode;ctx.fillRect(0,0,canvas.width,canvas.height);}
 else if(image){ctx.imageSmoothingEnabled=false;drawImage();}}
function afterPaint(s){requestAnimationFrame(()=>requestAnimationFrame(()=>fetch('/ack',
 {method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(
 {revision:s.revision,token:s.token,client_time_ms:performance.now(),painted:true,
 canvas_width:canvas.width,canvas_height:canvas.height,
 screen_orientation_type:screen.orientation?.type||'',screen_orientation_angle:screen.orientation?.angle||0,
 fullscreen:!!document.fullscreenElement,
 image_natural_width:image?.naturalWidth||0,image_natural_height:image?.naturalHeight||0})}).catch(()=>{})));}
async function poll(){try{const s=await (await fetch('/state.json',{cache:'no-store'})).json();
 note.textContent=s.label+' · '+canvas.width+'×'+canvas.height;
 if(s.revision!==revision){revision=s.revision;lastMode=s.mode;
  if(s.mode==='image'){const next=new Image();next.onload=()=>{image=next;draw('image');afterPaint(s)};
   next.src='/image.png?v='+revision;}else{draw(s.mode);afterPaint(s);}
 }else draw(lastMode);}catch(e){note.textContent='PC target service disconnected';}
 setTimeout(poll,40);}
if(autoStart){note.textContent='ADB will activate fullscreen calibration';}
document.querySelector('#go').onclick=async()=>{try{await document.documentElement.requestFullscreen();}catch(e){}
 gate.style.display='none';note.style.display='none';size();}; addEventListener('resize',size);
document.addEventListener('fullscreenchange',size);size();poll();
</script></body></html>"""


class _TargetState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.mode = "image"
        self.label = "ChArUco screen atlas"
        self.token = "charuco"
        self.revision = 0
        self.issued_time_ns = 0
        self.image_png = b""
        self.telemetry_value: dict[str, Any] = {}
        self.acknowledgements: list[dict[str, Any]] = []

    def describe(self) -> dict[str, Any]:
        with self.lock:
            return {
                "mode": self.mode,
                "label": self.label,
                "token": self.token,
                "revision": self.revision,
                "issued_time_ns": self.issued_time_ns,
            }


def _handler(state: _TargetState):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, content_type: str, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/":
                self._send(200, "text/html; charset=utf-8", _PHONE_PAGE.encode("utf-8"))
                return
            if path == "/state.json":
                body = json.dumps(state.describe(), separators=(",", ":")).encode("utf-8")
                self._send(200, "application/json", body)
                return
            if path == "/image.png":
                with state.lock:
                    body = state.image_png
                self._send(200, "image/png", body)
                return
            self._send(404, "text/plain; charset=utf-8", b"Not found")

        def do_POST(self) -> None:
            if self.path not in ("/telemetry", "/ack"):
                self._send(404, "text/plain; charset=utf-8", b"Not found")
                return
            try:
                length = min(int(self.headers.get("Content-Length", "0")), 65536)
                value = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("JSON object required")
                value["server_receive_time_ns"] = time.monotonic_ns()
                with state.lock:
                    if self.path == "/telemetry":
                        state.telemetry_value = value
                    else:
                        state.acknowledgements.append(value)
                        del state.acknowledgements[:-256]
                self._send(204, "text/plain", b"")
            except Exception as exc:
                self._send(400, "text/plain; charset=utf-8", str(exc).encode("utf-8"))

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


class LocalPhoneTargetServer(PhoneTargetAdapter):
    """A narrowly scoped HTTP target service started only by the operator."""

    adapter_id = "local_http"

    def __init__(
        self,
        bind_host: str = "0.0.0.0",
        port: int = 0,
        advertised_host: Optional[str] = None,
    ) -> None:
        self.bind_host = str(bind_host)
        self.port = int(port)
        self.advertised_host = advertised_host
        self._state = _TargetState()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._layout: Optional[CharucoLayout] = None
        self._charuco: Optional[np.ndarray] = None

    @staticmethod
    def _encode_png(image: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("Cannot encode phone target")
        return encoded.tobytes()

    def _set(
        self, mode: str, label: str, token: str, image: Optional[np.ndarray] = None
    ) -> Presentation:
        issued = time.monotonic_ns()
        with self._state.lock:
            if image is not None:
                self._state.image_png = self._encode_png(image)
            self._state.mode = mode
            self._state.label = label
            self._state.token = token
            self._state.revision += 1
            self._state.issued_time_ns = issued
            revision = self._state.revision
        return Presentation(token, mode, issued, revision)

    def _host_for_operator(self) -> str:
        if self.advertised_host:
            return self.advertised_host
        if self.bind_host not in ("", "0.0.0.0", "::"):
            return self.bind_host
        try:
            addresses = socket.gethostbyname_ex(socket.gethostname())[2]
            return next(address for address in addresses if not address.startswith("127."))
        except Exception:
            return "127.0.0.1"

    @property
    def bound_port(self) -> int:
        """Return the actual listening port after ``start``."""

        if self._server is None:
            raise RuntimeError("Phone target service is not running")
        return int(self._server.server_address[1])

    def start(self, layout: CharucoLayout) -> str:
        if self._server is not None:
            raise RuntimeError("Phone target service is already running")
        self._layout = layout
        self._charuco = generate_charuco_target(layout)
        self._set("image", "ChArUco screen atlas", "charuco", self._charuco)
        server = ThreadingHTTPServer((self.bind_host, self.port), _handler(self._state))
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever,
            name="iris-rig-phone-target",
            daemon=True,
        )
        self._thread.start()
        port = int(server.server_address[1])
        return "http://{}:{}/".format(self._host_for_operator(), port)

    def configure_layout(self, layout: CharucoLayout) -> Presentation:
        """Replace the target raster while retaining the HTTP/ack session."""

        if self._server is None:
            raise RuntimeError("Phone target service is not running")
        self._layout = layout
        self._charuco = generate_charuco_target(layout)
        return self._set(
            "image", "ChArUco screen atlas", "charuco", self._charuco
        )

    def present_charuco(self) -> Presentation:
        if self._charuco is None:
            raise RuntimeError("Phone target service is not running")
        return self._set(
            "image", "ChArUco screen atlas", "charuco", self._charuco
        )

    def present_image(self, image: np.ndarray, label: str) -> Presentation:
        if self._server is None:
            raise RuntimeError("Phone target service is not running")
        return self._set("image", str(label), str(label), image)

    def present_signal(self, state: str, token: str) -> Presentation:
        normalized = str(state).strip().lower()
        if normalized not in ("black", "white"):
            raise ValueError("Signal state must be black or white")
        if self._server is None:
            raise RuntimeError("Phone target service is not running")
        return self._set(normalized, "Latency signal: {}".format(normalized), token)

    def telemetry(self) -> Mapping[str, Any]:
        with self._state.lock:
            return {
                "browser": dict(self._state.telemetry_value),
                "acknowledgements": list(self._state.acknowledgements),
            }

    def stop(self) -> None:
        server, self._server = self._server, None
        thread, self._thread = self._thread, None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._layout = None
        self._charuco = None


class NativeImmersivePhoneTarget(LocalPhoneTargetServer):
    """Host half of the native immersive Android SurfaceView presenter."""

    adapter_id = "android_native_surface"
    minimum_version_code = 2
    package_name = "io.iris.phonetarget"
    component_name = (
        "io.iris.phonetarget/"
        "io.iris.phonetarget.PhoneTargetActivity"
    )

    def __init__(
        self,
        bind_host: str = "127.0.0.1",
        port: int = 0,
        advertised_host: Optional[str] = "127.0.0.1",
        apk_path: Optional[Path] = None,
    ) -> None:
        super().__init__(bind_host, port, advertised_host)
        self.apk_path = Path(apk_path).resolve() if apk_path else None

    @staticmethod
    def default_apk_candidates() -> List[Path]:
        configured = os.environ.get("IRIS_PHONE_TARGET_APK")
        repository = Path(__file__).resolve().parents[3]
        candidates = []
        if configured:
            candidates.append(Path(configured))
        candidates.extend(
            [
                repository
                / "android"
                / "phone-target"
                / "iris-phone-target.apk",
                repository
                / "artifacts"
                / "android-phone-target"
                / "iris-phone-target.apk",
                repository / "phone-target" / "iris-phone-target.apk",
                Path.cwd() / "phone-target" / "iris-phone-target.apk",
            ]
        )
        module = Path(__file__).resolve()
        candidates.extend(
            parent / "phone-target" / "iris-phone-target.apk"
            for parent in module.parents
        )
        executable = Path(sys.executable).resolve()
        candidates.extend(
            parent / "phone-target" / "iris-phone-target.apk"
            for parent in executable.parents
        )
        return candidates

    def resolved_apk_path(self) -> Optional[Path]:
        candidates = (
            [self.apk_path] if self.apk_path is not None else self.default_apk_candidates()
        )
        return next(
            (path.resolve() for path in candidates if path is not None and path.is_file()),
            None,
        )

    def configure_surface_size(self, surface_size_px: Sequence[int]) -> Presentation:
        """Publish a target raster exactly matching the native SurfaceView."""

        if self._server is None or self._layout is None:
            raise RuntimeError("Phone target service is not running")
        width, height = map(int, surface_size_px)
        if min(width, height) <= 0:
            raise ValueError("Native surface size must be positive")
        nominal = generate_charuco_target(self._layout)
        self._charuco = cv2.resize(
            nominal, (width, height), interpolation=cv2.INTER_NEAREST
        )
        return self._set(
            "image", "ChArUco screen atlas", "charuco", self._charuco
        )

    def activate_phone(
        self,
        phone: Any,
        screen_size_px: Sequence[int],
        rotation_quarter_turns: int = 0,
    ) -> None:
        """Launch the native target after the host server has bound its port."""

        phone.wake_and_hold_native_target(
            self.bound_port,
            screen_size_px,
            rotation_quarter_turns,
            package_name=self.package_name,
            component_name=self.component_name,
            apk_path=self.resolved_apk_path(),
            minimum_version_code=self.minimum_version_code,
        )
