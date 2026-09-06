import hashlib
import io
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace

from aria_trace.apps.workbench.api import make_handler
from aria_trace.apps.workbench.catalog import SessionCatalog
from aria_trace.apps.workbench.server import WorkbenchHttpServer
from aria_trace.apps.workbench.session_bundles import SessionBundles, inspect_bundle, safe_relative


class SessionBundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.state = self.make_state(self.root / "sessions")
        self.bundles = SessionBundles(self.state)

    def make_state(self, root):
        root.mkdir()
        return SimpleNamespace(session_root=root, _lock=threading.RLock(), _active=None,
                               _session_catalog=SessionCatalog(root, [], "session_metadata.json"))

    def session(self, number):
        path = self.state.session_root / "recordings-game" / ("run_%02d" % number)
        path.mkdir(parents=True)
        (path / "images").mkdir()
        files = {
            "manifest.json": json.dumps({"schema_version": "1.0", "status": "complete", "session_id": "original-%d" % number,
                                         "videos": {"main": "video_main.mkv"}, "context": {"run_index": number}}).encode(),
            "frames.jsonl": (json.dumps({"stream_id": "main", "frame_index": 0, "session_time_ns": 123, "storage": {"kind": "video", "path": "video_main.mkv"}}) + "\n" +
                             json.dumps({"stream_id": "second", "frame_index": 0, "session_time_ns": 130, "storage": {"kind": "image_series", "path": "images/frame.png"}}) + "\n").encode(),
            "inputs.jsonl": b'{"session_time_ns":125,"kind":"mouse","dx":1}\n',
            "video_main.mkv": bytes(range(256)) * 512,
            "images/frame.png": b"image payload",
            "annotations.jsonl": b'{"kind":"take_end","session_time_ns":900}\n',
            "session_metadata.json": b'{"label":"route_repeatability"}',
        }
        for name, content in files.items():
            (path / name).write_bytes(content)
        return path, files

    def upload(self, path, bundles=None):
        bundles = bundles or self.bundles
        with path.open("rb") as stream:
            return bundles.upload(stream, path.stat().st_size)

    def rewrite(self, path, change):
        with zipfile.ZipFile(path) as source:
            entries = {name: source.read(name) for name in source.namelist()}
        change(entries)
        out = self.root / "modified.zip"
        with zipfile.ZipFile(out, "w") as archive:
            for name, value in entries.items():
                archive.writestr(name, value)
        return out

    def test_all_export_selected_import_collision_preserves_every_byte(self):
        first, files = self.session(1)
        self.session(2)
        export = self.bundles.export()
        preview = self.upload(self.bundles.download(export["token"]))
        self.assertEqual(2, len(preview["sessions"]))
        result = self.bundles.import_sessions(preview["token"], ["recordings-game/run_01"])
        self.assertEqual([{"source_key": "recordings-game/run_01", "session_key": "recordings-game/run_03"}], result["imported"])
        for name, content in files.items():
            self.assertEqual(content, (self.state.session_root / "recordings-game/run_03" / name).read_bytes())
            self.assertEqual(content, (first / name).read_bytes())
        self.assertFalse((self.state.session_root / "recordings-game/run_04").exists())
        with self.assertRaises(ValueError):
            self.bundles.import_sessions(preview["token"])

    def test_selected_export_all_import_into_other_root(self):
        self.session(1)
        _, files = self.session(2)
        export = self.bundles.export(["recordings-game/run_02"])
        other = SessionBundles(self.make_state(self.root / "other"))
        preview = self.upload(self.bundles.download(export["token"]), other)
        result = other.import_sessions(preview["token"])
        self.assertEqual("recordings-game/run_02", result["imported"][0]["session_key"])
        for name, content in files.items():
            self.assertEqual(content, (other.state.session_root / "recordings-game/run_02" / name).read_bytes())
        self.assertFalse((other.state.session_root / "recordings-game/run_01").exists())

    def test_active_session_excluded_from_all_and_refused_when_selected(self):
        self.session(1)
        active, _ = self.session(2)
        self.state._active = {"path": str(active)}
        result = self.bundles.export()
        self.assertEqual(1, result["session_count"])
        with self.assertRaisesRegex(ValueError, "active"):
            self.bundles.export(["recordings-game/run_02"])

    def test_corrupt_selected_session_publishes_nothing(self):
        self.session(1)
        self.session(2)
        exported = self.bundles.download(self.bundles.export()["token"])
        changed = self.rewrite(exported, lambda entries: entries.__setitem__("sessions/recordings-game/run_02/video_main.mkv", b"X" * len(entries["sessions/recordings-game/run_02/video_main.mkv"])))
        uploaded = self.upload(changed)
        with self.assertRaisesRegex(ValueError, "Checksum"):
            self.bundles.import_sessions(uploaded["token"])
        self.assertEqual(2, len(list(self.state.session_root.glob("*/run_*/manifest.json"))))

    def test_unsafe_paths_unlisted_files_and_duplicate_paths_rejected(self):
        for value in ("../outside", "/abs", "C:/outside", "x\\y", "x/NUL.txt", "x/a.", "x/a:b", "x//a", "x/.. /a"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                safe_relative(value)
        self.session(1)
        exported = self.bundles.download(self.bundles.export()["token"])
        for name in ("../outside", "unexpected.txt", "BUNDLE.JSON"):
            changed = self.rewrite(exported, lambda entries: entries.__setitem__(name, b"bad"))
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.upload(changed)

    def test_embedded_external_video_path_refused_even_with_valid_checksum(self):
        self.session(1)
        exported = self.bundles.download(self.bundles.export()["token"])
        def change(entries):
            name = "sessions/recordings-game/run_01/manifest.json"
            manifest = json.loads(entries[name]); manifest["videos"]["main"] = "../outside.mkv"
            entries[name] = json.dumps(manifest).encode()
            index = json.loads(entries["bundle.json"])
            record = next(f for f in index["sessions"][0]["files"] if f["path"] == "manifest.json")
            record.update(bytes=len(entries[name]), sha256=hashlib.sha256(entries[name]).hexdigest())
            entries["bundle.json"] = json.dumps(index).encode()
        uploaded = self.upload(self.rewrite(exported, change))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.bundles.import_sessions(uploaded["token"])
        self.assertEqual(1, len(list(self.state.session_root.glob("*/run_*/manifest.json"))))

    def test_interrupted_upload_and_bad_zip_leave_no_staging(self):
        for payload, length in ((b"abc", 8), (b"not a zip", 9)):
            with self.assertRaises(ValueError):
                self.bundles.upload(io.BytesIO(payload), length)
        self.assertEqual([], list(self.bundles.root.iterdir()))

    def test_http_export_download_preview_import(self):
        self.session(1)
        server = WorkbenchHttpServer(("127.0.0.1", 0), make_handler(self.state))
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        base = "http://127.0.0.1:%d" % server.server_address[1]
        def post(path, data, content_type="application/json"):
            payload = json.dumps(data).encode() if content_type == "application/json" else data
            with urllib.request.urlopen(urllib.request.Request(base + path, data=payload, headers={"Content-Type": content_type}), timeout=10) as response:
                return json.load(response)
        exported = post("/api/sessions/export", {"session_keys": ["recordings-game/run_01"]})
        with urllib.request.urlopen(base + exported["download_url"], timeout=10) as response:
            self.assertIn("attachment", response.headers["Content-Disposition"])
            archive = response.read()
        preview = post("/api/sessions/upload", archive, "application/zip")
        self.assertEqual(1, len(preview["sessions"]))
        imported = post("/api/sessions/import", {"token": preview["token"]})
        self.assertEqual("recordings-game/run_02", imported["imported"][0]["session_key"])


if __name__ == "__main__":
    unittest.main()
