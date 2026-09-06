"""Portable, checksummed session bundles; no derived artifacts or profile state."""

import hashlib
import json
import re
import shutil
import stat
import threading
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath


MAX_ARCHIVE_BYTES = 32 * 1024**3
MAX_EXPANDED_BYTES = 64 * 1024**3
MAX_FILES = 200_000
MAX_INDEX_BYTES = 32 * 1024**2
CHUNK = 1024 * 1024


def safe_relative(value):
    """Use the same portable interpretation on Windows and POSIX."""
    if not isinstance(value, str) or not value or len(value) > 1024:
        raise ValueError("Invalid bundle path")
    parts = value.split("/")
    for part in parts:
        if (part in ("", ".", "..") or part.endswith((" ", "."))
                or any(ord(c) < 32 or c in '\\:<>"|?*' for c in part)
                or re.fullmatch(r"(?i:con|prn|aux|nul|com[0-9]|lpt[0-9])", part.split(".")[0])):
            raise ValueError("Unsafe bundle path: " + value)
    return PurePosixPath(value)


def session_key(value):
    path = safe_relative(value)
    if len(path.parts) != 2 or path.parts[0].startswith(".") or not re.fullmatch(r"run_\d+", path.parts[1]):
        raise ValueError("Invalid session key")
    return path


def checked_manifest(directory):
    """Validate data paths used by SessionReader before making imports visible."""
    if (directory / "manifest.json").stat().st_size > 8 * CHUNK:
        raise ValueError("Session manifest is too large")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0" or manifest.get("status") != "complete":
        raise ValueError("Only complete version 1.0 sessions can be transferred")
    if not manifest.get("session_id"):
        raise ValueError("Session identity is missing")
    def check_file(name):
        safe_relative(name)
        if not (directory / name).is_file():
            raise ValueError("Session data file is missing: " + name)
    for name in (manifest.get("videos") or {}).values():
        check_file(name)
    for name in ("frames.jsonl", "inputs.jsonl"):
        check_file(name)
        with (directory / name).open(encoding="utf-8") as stream:
            while True:
                line = stream.readline(CHUNK + 1)
                if not line:
                    break
                if len(line) > CHUNK:
                    raise ValueError("Session record is too large")
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Invalid session record")
                if name == "frames.jsonl":
                    for relative in (row.get("image_file"), (row.get("storage") or {}).get("path")):
                        if relative:
                            check_file(relative)
    return manifest


def inspect_bundle(path):
    """Validate the index and central directory; payload hashes are checked on import."""
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_FILES or sum(i.file_size for i in infos) > MAX_EXPANDED_BYTES:
                raise ValueError("Bundle exceeds the file count or expanded-size limit")
            by_name = {}
            folded = set()
            for info in infos:
                safe_relative(info.filename)
                kind = stat.S_IFMT(info.external_attr >> 16)
                if info.is_dir() or kind not in (0, stat.S_IFREG) or info.flag_bits & 1:
                    raise ValueError("Bundle contains a non-regular or encrypted file")
                if info.filename.casefold() in folded:
                    raise ValueError("Bundle contains duplicate paths")
                folded.add(info.filename.casefold())
                by_name[info.filename] = info
            if "bundle.json" not in by_name or by_name["bundle.json"].file_size > MAX_INDEX_BYTES:
                raise ValueError("Missing or oversized bundle index")
            index = json.loads(archive.read("bundle.json"))
            if not isinstance(index, dict) or index.get("format") != "aria-trace-sessions" or index.get("version") != 1:
                raise ValueError("Unsupported session bundle")
            sessions = index.get("sessions")
            if not isinstance(sessions, list) or not sessions:
                raise ValueError("Bundle has no sessions")
            expected = {"bundle.json"}
            keys = set()
            for item in sessions:
                key = item["session_key"]
                session_key(key)
                if key.casefold() in keys:
                    raise ValueError("Duplicate session key")
                keys.add(key.casefold())
                names = set()
                for record in item["files"]:
                    relative = record["path"]
                    safe_relative(relative)
                    full = "sessions/" + key + "/" + relative
                    if full in expected or full not in by_name or by_name[full].file_size != record["bytes"]:
                        raise ValueError("Bundle file index does not match its contents")
                    if not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
                        raise ValueError("Invalid bundle checksum")
                    expected.add(full)
                    names.add(relative)
                if not {"manifest.json", "frames.jsonl", "inputs.jsonl"}.issubset(names):
                    raise ValueError("Session is missing required files")
            if expected != set(by_name):
                raise ValueError("Bundle contains unlisted files")
            return index
    except (zipfile.BadZipFile, KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid session bundle: " + str(exc)) from exc


class SessionBundles:
    """One transfer lock isolates large I/O from recorder and catalog locks.

    Staged downloads/uploads expire after 24 hours, including across restarts.
    Import validates every selected session before publishing under the catalog
    lock. Original files and IDs are retained when a directory is renumbered.
    """

    def __init__(self, state):
        self.state = state
        self.root = state.session_root / ".transfers"
        self.lock = threading.RLock()

    def _new(self):
        self.root.mkdir(parents=True, exist_ok=True)
        for entry in self.root.iterdir():
            if re.fullmatch(r"[0-9a-f]{32}", entry.name) and not entry.is_symlink() and entry.is_dir() and time.time() - entry.stat().st_mtime > 86400:
                shutil.rmtree(entry)
        token = uuid.uuid4().hex
        directory = self.root / token
        directory.mkdir()
        return token, directory

    def _directory(self, token):
        if not isinstance(token, str) or not re.fullmatch(r"[0-9a-f]{32}", token):
            raise ValueError("Invalid transfer token")
        directory = self.root / token
        if directory.is_symlink() or not directory.is_dir() or time.time() - directory.stat().st_mtime > 86400:
            raise ValueError("Transfer expired; select the file again")
        return directory

    def export(self, selected=None):
        with self.lock:
            with self.state._lock:
                active = self.state._active
                active_path = Path(active["path"]).resolve() if active else None
                if selected is None:
                    paths = [p.parent for p in self.state.session_root.glob("*/run_*/manifest.json")
                             if p.parent.resolve() != active_path]
                    complete = []
                    for path in paths:
                        try:
                            manifest_path = path / "manifest.json"
                            if manifest_path.stat().st_size <= 8 * CHUNK and json.loads(manifest_path.read_text(encoding="utf-8")).get("status") == "complete":
                                complete.append(path)
                        except (OSError, ValueError):
                            continue
                    paths = complete
                else:
                    if not isinstance(selected, list) or not selected:
                        raise ValueError("Select at least one session")
                    paths = [self.state._session_catalog.resolve(str(session_key(k))) for k in dict.fromkeys(selected)]
                if not paths:
                    raise ValueError("No complete sessions to export")
                if any(p.resolve() == active_path for p in paths):
                    raise ValueError("Wait for the active recording to finish")
            token, directory = self._new()
            try:
                index = {"format": "aria-trace-sessions", "version": 1, "sessions": []}
                with zipfile.ZipFile(directory / "export.zip", "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                    for path in sorted(paths):
                        key = str(session_key(self.state._session_catalog.key(path)))
                        if path.is_symlink() or path.parent.is_symlink():
                            raise ValueError("Linked session directories cannot be exported")
                        manifest = checked_manifest(path)
                        item = {"session_key": key, "session_id": manifest["session_id"], "files": []}
                        before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in path.rglob("*") if p.is_file()}
                        for file in sorted(before):
                            if file.is_symlink() or not file.resolve().is_relative_to(path.resolve()):
                                raise ValueError("Linked session files cannot be exported")
                            relative = file.relative_to(path).as_posix()
                            safe_relative(relative)
                            digest = hashlib.sha256()
                            size = 0
                            with file.open("rb") as source, archive.open("sessions/" + key + "/" + relative, "w", force_zip64=True) as target:
                                for block in iter(lambda: source.read(CHUNK), b""):
                                    digest.update(block)
                                    size += len(block)
                                    target.write(block)
                            item["files"].append({"path": relative, "bytes": size, "sha256": digest.hexdigest()})
                        after = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in path.rglob("*") if p.is_file()}
                        if before != after:
                            raise ValueError("Session changed during export; retry when it is idle")
                        index["sessions"].append(item)
                    archive.writestr("bundle.json", json.dumps(index, ensure_ascii=False))
                inspect_bundle(directory / "export.zip")
                if (directory / "export.zip").stat().st_size > MAX_ARCHIVE_BYTES:
                    raise ValueError("Export exceeds 32 GiB; select fewer sessions")
                return {"token": token, "session_count": len(index["sessions"]), "download_url": "/api/sessions/download?token=" + token}
            except Exception:
                shutil.rmtree(directory)
                raise

    def upload(self, stream, length):
        if not 0 < length <= MAX_ARCHIVE_BYTES:
            raise ValueError("Select a ZIP session bundle up to 32 GiB")
        with self.lock:
            token, directory = self._new()
            try:
                with (directory / "upload.zip").open("wb") as output:
                    remaining = length
                    while remaining:
                        block = stream.read(min(CHUNK, remaining))
                        if not block:
                            raise ValueError("Upload was interrupted")
                        output.write(block)
                        remaining -= len(block)
                index = inspect_bundle(directory / "upload.zip")
                return {"token": token, "sessions": [{"session_key": s["session_key"], "session_id": s["session_id"], "bytes": sum(f["bytes"] for f in s["files"])} for s in index["sessions"]]}
            except Exception:
                shutil.rmtree(directory)
                raise

    def download(self, token):
        path = self._directory(token) / "export.zip"
        if not path.is_file():
            raise ValueError("Download not found")
        return path

    def discard(self, token):
        with self.lock:
            shutil.rmtree(self._directory(token))
        return {"discarded": True}

    def import_sessions(self, token, selected=None):
        with self.lock:
            directory = self._directory(token)
            index = inspect_bundle(directory / "upload.zip")
            all_keys = {s["session_key"] for s in index["sessions"]}
            if selected is not None and (not isinstance(selected, list) or not selected or not set(selected).issubset(all_keys)):
                raise ValueError("Select sessions from this bundle")
            chosen = [s for s in index["sessions"] if selected is None or s["session_key"] in selected]
            staging = directory / ("stage-" + uuid.uuid4().hex)
            staging.mkdir()
            published = []
            try:
                with zipfile.ZipFile(directory / "upload.zip") as archive:
                    for item in chosen:
                        base = staging / item["session_key"]
                        for record in item["files"]:
                            target = base / record["path"]
                            target.parent.mkdir(parents=True, exist_ok=True)
                            digest = hashlib.sha256()
                            with archive.open("sessions/" + item["session_key"] + "/" + record["path"]) as source, target.open("xb") as output:
                                for block in iter(lambda: source.read(CHUNK), b""):
                                    digest.update(block)
                                    output.write(block)
                            if digest.hexdigest() != record["sha256"]:
                                raise ValueError("Checksum mismatch: " + record["path"])
                        manifest = checked_manifest(base)
                        if manifest["session_id"] != item["session_id"]:
                            raise ValueError("Session identity does not match the bundle")
                with self.state._lock:
                    try:
                        for item in chosen:
                            source = staging / item["session_key"]
                            target = self.state._session_catalog.resolve(item["session_key"], require_manifest=False)
                            target.parent.mkdir(parents=True, exist_ok=True)
                            if target.exists():
                                numbers = [int(p.name[4:]) for p in target.parent.iterdir() if re.fullmatch(r"run_\d+", p.name)]
                                target = target.parent / ("run_%02d" % (max(numbers, default=0) + 1))
                            source.rename(target)
                            published.append((source, target, item["session_key"]))
                        for _, target, _ in published:
                            self.state._session_catalog.invalidate(target)
                    except Exception:
                        for source, target, _ in reversed(published):
                            target.rename(source)
                        raise
                result = {"imported": [{"source_key": key, "session_key": self.state._session_catalog.key(target)} for _, target, key in published]}
            finally:
                shutil.rmtree(staging)
            shutil.rmtree(directory)
            return result
