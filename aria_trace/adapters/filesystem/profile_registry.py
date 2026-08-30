"""Production profile context, immutable revisions, and automatic resolution.

The registry owns *selection*. Calibration algorithms continue to own their
payloads, while camera adapters continue to own frame acquisition. Profiles
are resolved once before an adapter opens; no registry access occurs per frame.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence

from .commented_yaml import write_commented_yaml


SCHEMA_VERSION = "2.0"
PROFILE_KINDS = ("rig", "phone_game", "rig_game")
REVISION_STATES = ("review_required", "accepted")
ADAPTER_MODES = ("full", "minimap", "dual")
NORMALIZATION_MODES = ("auto", "dense_remap", "homography", "none")
COLOR_ORDERS = ("RGB", "BGR")
COLOR_POLICIES = ("auto", "rig_locked", "game_matched", "unadjusted")
ROI_POLICIES = ("auto", "full_phone", "minimap_only")

PROFILE_HEADER = """# AriaTrace production profile revision.
#
# Revisions are immutable. The SQLite registry is the activation authority;
# this commented YAML is the human-readable payload and provenance record."""

PROFILE_COMMENTS = {
    "identity": "Exact ownership and display-compatibility dimensions.",
    "context": "Observed facts used when this revision was created.",
    "dependencies": "Exact immutable revisions consumed by this revision.",
    "payload": "Runtime configuration owned by this profile kind.",
    "runtime_files": "Files copied into this immutable revision directory.",
    "provenance": "Source artifacts and producer information; not selection keys.",
}


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip()).strip("-.")
    if not cleaned:
        raise ValueError("Profile identifiers cannot be empty")
    return cleaned


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hash_json(value: Any, length: int = 16) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:length]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


def default_profile_root(explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        return Path(explicit).resolve()
    configured = os.environ.get("ARIA_PROFILE_ROOT")
    if configured:
        return Path(configured).resolve()
    return (Path.cwd() / "profiles").resolve()


def _normalized_display(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    source = dict(value or {})
    normalized = {}
    for name in (
        "natural_panel_px",
        "logical_frame_px",
        "game_viewport_xywh",
        "insets_px",
    ):
        item = source.get(name)
        if item is not None:
            normalized[name] = [int(part) for part in item]
    for name in ("rotation_quarter_turns", "density_dpi"):
        if source.get(name) is not None:
            normalized[name] = int(source[name])
    if source.get("refresh_hz") is not None:
        # Millihertz precision avoids signatures changing from float noise.
        normalized["refresh_millihz"] = int(round(float(source["refresh_hz"]) * 1000.0))
    elif source.get("refresh_millihz") is not None:
        normalized["refresh_millihz"] = int(source["refresh_millihz"])
    for name in ("ui_layout_id", "content_layout_id"):
        if source.get(name) is not None:
            normalized[name] = str(source[name])
    return normalized


@dataclass(frozen=True)
class ProfileContext:
    """Observed identity and compatibility facts, never adapter behavior."""

    game_id: Optional[str] = None
    platform: str = "android"
    package: Optional[str] = None
    game_version: Optional[str] = None
    camera_adapter: str = "hik_mvs"
    camera_id: Optional[str] = None
    phone_id: Optional[str] = None
    phone_model: Optional[str] = None
    panel_display: Mapping[str, Any] = field(default_factory=dict)
    game_display: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.game_id is not None:
            object.__setattr__(self, "game_id", _safe_id(self.game_id))
        if self.camera_id is not None:
            object.__setattr__(self, "camera_id", _safe_id(self.camera_id))
        if self.phone_id is not None:
            object.__setattr__(self, "phone_id", _safe_id(self.phone_id))
        object.__setattr__(self, "platform", _safe_id(self.platform))
        object.__setattr__(self, "camera_adapter", _safe_id(self.camera_adapter))
        object.__setattr__(self, "panel_display", _normalized_display(self.panel_display))
        object.__setattr__(self, "game_display", _normalized_display(self.game_display))

    @property
    def panel_signature(self) -> Optional[str]:
        return _hash_json(self.panel_display) if self.panel_display else None

    @property
    def game_display_signature(self) -> Optional[str]:
        return _hash_json(self.game_display) if self.game_display else None

    @property
    def combined_display_signature(self) -> Optional[str]:
        if not self.panel_display and not self.game_display:
            return None
        return _hash_json(
            {"panel": self.panel_display, "game": self.game_display}
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "game": {
                "id": self.game_id,
                "platform": self.platform,
                "package": self.package,
                "version": self.game_version,
            },
            "devices": {
                "camera": {
                    "adapter": self.camera_adapter,
                    "id": self.camera_id,
                },
                "phone": {"id": self.phone_id, "model": self.phone_model},
            },
            "display": {
                "panel": dict(self.panel_display),
                "game": dict(self.game_display),
                "panel_signature": self.panel_signature,
                "game_signature": self.game_display_signature,
                "combined_signature": self.combined_display_signature,
            },
        }


@dataclass(frozen=True)
class AdapterRequest:
    """Requested runtime product; options do not create profile identities."""

    purpose: str = "application"
    mode: str = "full"
    normalization: str = "auto"
    color_order: str = "RGB"
    color_policy: str = "auto"
    roi_policy: str = "auto"
    minimap_margin_px: int = 6
    frame_rate_policy: str = "calibrated"
    frame_rate: Optional[float] = None

    def __post_init__(self) -> None:
        if self.mode not in ADAPTER_MODES:
            raise ValueError("Adapter mode must be full, minimap, or dual")
        if self.normalization not in NORMALIZATION_MODES:
            raise ValueError("Unsupported normalization mode")
        order = str(self.color_order).upper()
        if order not in COLOR_ORDERS:
            raise ValueError("Color order must be RGB or BGR")
        object.__setattr__(self, "color_order", order)
        if self.color_policy not in COLOR_POLICIES:
            raise ValueError("Unsupported color policy")
        if self.roi_policy not in ROI_POLICIES:
            raise ValueError("Unsupported ROI policy")
        if int(self.minimap_margin_px) < 0:
            raise ValueError("Mini-map margin cannot be negative")
        if self.frame_rate_policy not in ("calibrated", "exact"):
            raise ValueError("Frame-rate policy must be calibrated or exact")
        if self.frame_rate_policy == "exact" and self.frame_rate is None:
            raise ValueError("Exact frame-rate policy requires frame_rate")

    @property
    def requires_game_profile(self) -> bool:
        return self.mode in ("minimap", "dual") or self.color_policy == "game_matched"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "purpose": self.purpose,
            "mode": self.mode,
            "normalization": self.normalization,
            "color_order": self.color_order,
            "color_policy": self.color_policy,
            "roi_policy": self.roi_policy,
            "minimap_margin_px": int(self.minimap_margin_px),
            "frame_rate": {
                "policy": self.frame_rate_policy,
                "value": self.frame_rate,
            },
        }


@dataclass(frozen=True)
class ProfileKey:
    kind: str
    owner_id: str
    game_id: str
    panel_signature: str
    game_signature: str

    @classmethod
    def from_context(cls, kind: str, context: ProfileContext) -> "ProfileKey":
        if kind not in PROFILE_KINDS:
            raise ValueError("Unsupported profile kind: {}".format(kind))
        if kind == "rig":
            if not context.camera_id or not context.phone_id or not context.panel_signature:
                raise ValueError("Rig profiles require camera, phone, and panel display")
            return cls(
                kind,
                "{}--{}".format(context.camera_id, context.phone_id),
                "_",
                context.panel_signature,
                "_",
            )
        if kind == "phone_game":
            if not context.phone_id or not context.game_id or not context.game_display_signature:
                raise ValueError("Phone-game profiles require phone, game, and game display")
            return cls(
                kind,
                context.phone_id,
                context.game_id,
                "_",
                context.game_display_signature,
            )
        if (
            not context.camera_id
            or not context.phone_id
            or not context.game_id
            or not context.panel_signature
            or not context.game_display_signature
        ):
            raise ValueError(
                "Rig-game profiles require camera, phone, game, panel, and game display"
            )
        return cls(
            kind,
            "{}--{}".format(context.camera_id, context.phone_id),
            context.game_id,
            context.panel_signature,
            context.game_display_signature,
        )

    @property
    def variant_id(self) -> str:
        return _hash_json(
            {
                "kind": self.kind,
                "owner": self.owner_id,
                "game": self.game_id,
                "panel": self.panel_signature,
                "game_display": self.game_signature,
            }
        )

    def as_dict(self) -> Dict[str, str]:
        return {
            "kind": self.kind,
            "owner_id": self.owner_id,
            "game_id": self.game_id,
            "panel_signature": self.panel_signature,
            "game_display_signature": self.game_signature,
            "variant_id": self.variant_id,
        }


class ProfileResolutionError(RuntimeError):
    pass


class ProfileRegistry:
    """SQLite activation index plus immutable human-readable revision bundles."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = default_profile_root(root)
        self.registry_directory = self.root / ".registry"
        self.database = self.registry_directory / "profiles.sqlite3"
        self.registry_directory.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(str(self.database), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS revisions (
                    revision_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    game_id TEXT NOT NULL,
                    panel_signature TEXT NOT NULL,
                    game_signature TEXT NOT NULL,
                    variant_id TEXT NOT NULL,
                    review_state TEXT NOT NULL,
                    created_utc TEXT NOT NULL,
                    relative_directory TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    camera_id TEXT,
                    phone_id TEXT,
                    context_json TEXT NOT NULL,
                    dependencies_json TEXT NOT NULL,
                    UNIQUE(kind, owner_id, game_id, panel_signature, game_signature, content_sha256)
                );
                CREATE INDEX IF NOT EXISTS revisions_context
                    ON revisions(kind, camera_id, phone_id, game_id,
                                 panel_signature, game_signature);
                CREATE TABLE IF NOT EXISTS active_profiles (
                    kind TEXT NOT NULL,
                    owner_id TEXT NOT NULL,
                    game_id TEXT NOT NULL,
                    panel_signature TEXT NOT NULL,
                    game_signature TEXT NOT NULL,
                    revision_id TEXT NOT NULL REFERENCES revisions(revision_id),
                    activated_utc TEXT NOT NULL,
                    PRIMARY KEY(kind, owner_id, game_id, panel_signature, game_signature)
                );
                """
            )

    def _profile_directory(self, key: ProfileKey) -> Path:
        game = key.game_id if key.game_id != "_" else "_"
        return self.root / key.kind / key.owner_id / game / key.variant_id

    def _revision_row(self, revision_id: str) -> sqlite3.Row:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM revisions WHERE revision_id=?", (revision_id,)
            ).fetchone()
        if row is None:
            raise KeyError("Unknown profile revision: {}".format(revision_id))
        return row

    def revision(self, revision_id: str) -> Dict[str, Any]:
        row = self._revision_row(revision_id)
        directory = self.root / row["relative_directory"]
        manifest = json.loads((directory / "profile.json").read_text(encoding="utf-8"))
        manifest["revision_directory"] = str(directory.resolve())
        manifest["registry_review_state"] = row["review_state"]
        return manifest

    def publish(
        self,
        kind: str,
        context: ProfileContext,
        payload: Mapping[str, Any],
        *,
        runtime_files: Optional[Mapping[str, Path]] = None,
        dependencies: Optional[Mapping[str, str]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
        review_state: str = "review_required",
        activate: bool = False,
    ) -> Dict[str, Any]:
        if review_state not in REVISION_STATES:
            raise ValueError("Unsupported profile review state")
        key = ProfileKey.from_context(kind, context)
        sources = {str(name): Path(path).resolve() for name, path in (runtime_files or {}).items()}
        for name, path in sources.items():
            if not path.is_file():
                raise FileNotFoundError("Runtime profile file {} does not exist: {}".format(name, path))
        file_hashes = {name: _hash_file(path) for name, path in sorted(sources.items())}
        content_sha256 = hashlib.sha256(
            _canonical_json(
                {
                    "kind": kind,
                    "identity": key.as_dict(),
                    "payload": payload,
                    "files": file_hashes,
                    "dependencies": dependencies or {},
                }
            ).encode("utf-8")
        ).hexdigest()

        with self._connect() as connection:
            existing = connection.execute(
                """SELECT revision_id FROM revisions
                   WHERE kind=? AND owner_id=? AND game_id=?
                     AND panel_signature=? AND game_signature=?
                     AND content_sha256=?""",
                (
                    key.kind,
                    key.owner_id,
                    key.game_id,
                    key.panel_signature,
                    key.game_signature,
                    content_sha256,
                ),
            ).fetchone()
        if existing is not None:
            if activate:
                self.activate(existing["revision_id"])
            result = self.revision(existing["revision_id"])
            result["publication"] = "unchanged_existing_revision"
            return result

        created = datetime.now(timezone.utc)
        revision_id = "{}-{}-{}".format(
            kind.replace("_", "-"),
            created.strftime("%Y%m%dT%H%M%S%fZ"),
            content_sha256[:10],
        )
        profile_directory = self._profile_directory(key)
        revisions_directory = profile_directory / "revisions"
        revisions_directory.mkdir(parents=True, exist_ok=True)
        final_directory = revisions_directory / revision_id
        temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=str(revisions_directory)))
        copied_files = {}
        try:
            files_directory = temporary / "files"
            for logical_name, source in sources.items():
                target_name = _safe_id(logical_name) + source.suffix.lower()
                target = files_directory / target_name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(source), str(target))
                copied_files[logical_name] = {
                    "path": "files/{}".format(target_name),
                    "sha256": file_hashes[logical_name],
                    "size_bytes": target.stat().st_size,
                }
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "revision_id": revision_id,
                "review_state": review_state,
                "created_utc": created.isoformat(),
                "identity": key.as_dict(),
                "context": context.as_dict(),
                "dependencies": dict(dependencies or {}),
                "payload": dict(payload),
                "runtime_files": copied_files,
                "content_sha256": content_sha256,
                "provenance": dict(provenance or {}),
            }
            _atomic_json(temporary / "profile.json", manifest)
            write_commented_yaml(
                temporary / "profile.yaml",
                manifest,
                header=PROFILE_HEADER,
                section_comments=PROFILE_COMMENTS,
            )
            os.replace(str(temporary), str(final_directory))
        except Exception:
            shutil.rmtree(str(temporary), ignore_errors=True)
            raise

        relative_directory = str(final_directory.relative_to(self.root))
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """INSERT INTO revisions
                       (revision_id, kind, owner_id, game_id, panel_signature,
                        game_signature, variant_id, review_state, created_utc,
                        relative_directory, content_sha256, camera_id, phone_id,
                        context_json, dependencies_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        revision_id,
                        key.kind,
                        key.owner_id,
                        key.game_id,
                        key.panel_signature,
                        key.game_signature,
                        key.variant_id,
                        review_state,
                        created.isoformat(),
                        relative_directory,
                        content_sha256,
                        context.camera_id,
                        context.phone_id,
                        _canonical_json(context.as_dict()),
                        _canonical_json(dict(dependencies or {})),
                    ),
                )
        except Exception:
            shutil.rmtree(str(final_directory), ignore_errors=True)
            raise
        if activate:
            self.activate(revision_id)
        result = self.revision(revision_id)
        result["publication"] = "new_revision"
        return result

    def activate(
        self, revision_id: str, *, expected_current: Optional[str] = None
    ) -> Dict[str, Any]:
        row = self._revision_row(revision_id)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                """SELECT revision_id FROM active_profiles
                   WHERE kind=? AND owner_id=? AND game_id=?
                     AND panel_signature=? AND game_signature=?""",
                (
                    row["kind"], row["owner_id"], row["game_id"],
                    row["panel_signature"], row["game_signature"],
                ),
            ).fetchone()
            current_id = current["revision_id"] if current is not None else None
            if expected_current is not None and current_id != expected_current:
                raise RuntimeError(
                    "Profile activation conflict: expected {}, found {}".format(
                        expected_current, current_id
                    )
                )
            connection.execute(
                """INSERT INTO active_profiles
                   (kind, owner_id, game_id, panel_signature, game_signature,
                    revision_id, activated_utc)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(kind, owner_id, game_id, panel_signature, game_signature)
                   DO UPDATE SET revision_id=excluded.revision_id,
                                 activated_utc=excluded.activated_utc""",
                (
                    row["kind"], row["owner_id"], row["game_id"],
                    row["panel_signature"], row["game_signature"],
                    revision_id, now,
                ),
            )
            connection.execute(
                "UPDATE revisions SET review_state='accepted' WHERE revision_id=?",
                (revision_id,),
            )
        pointer = {
            "schema_version": SCHEMA_VERSION,
            "registry_authority": str(self.database.resolve()),
            "active_revision_id": revision_id,
            "activated_utc": now,
            "identity": {
                "kind": row["kind"], "owner_id": row["owner_id"],
                "game_id": row["game_id"],
                "panel_signature": row["panel_signature"],
                "game_display_signature": row["game_signature"],
            },
        }
        profile_directory = self.root / Path(row["relative_directory"]).parents[1]
        _atomic_json(profile_directory / "active.json", pointer)
        write_commented_yaml(
            profile_directory / "active.yaml",
            pointer,
            header="# Human-readable mirror; the SQLite registry is authoritative.",
            section_comments={"identity": "Display-specific active profile key."},
        )
        return self.revision(revision_id)

    def _active_rows(
        self,
        kind: str,
        context: ProfileContext,
    ) -> Sequence[sqlite3.Row]:
        clauses = ["r.kind=?"]
        values = [kind]
        for column, value in (
            ("r.camera_id", context.camera_id),
            ("r.phone_id", context.phone_id),
            ("r.game_id", context.game_id if kind != "rig" else None),
            (
                "r.panel_signature",
                context.panel_signature if kind in ("rig", "rig_game") else None,
            ),
            (
                "r.game_signature",
                context.game_display_signature
                if kind in ("phone_game", "rig_game")
                else None,
            ),
        ):
            if value is not None:
                clauses.append("{}=?".format(column))
                values.append(value)
        query = """SELECT r.* FROM active_profiles a
                   JOIN revisions r ON r.revision_id=a.revision_id
                   WHERE {} ORDER BY r.created_utc DESC""".format(" AND ".join(clauses))
        with self._connect() as connection:
            return connection.execute(query, values).fetchall()

    def resolve(self, kind: str, context: ProfileContext) -> Dict[str, Any]:
        rows = list(self._active_rows(kind, context))
        if not rows:
            raise ProfileResolutionError(
                "No active {} profile matches camera={!r}, phone={!r}, game={!r}, "
                "panel={!r}, game_display={!r}".format(
                    kind, context.camera_id, context.phone_id, context.game_id,
                    context.panel_signature, context.game_display_signature,
                )
            )
        if len(rows) != 1:
            raise ProfileResolutionError(
                "{} active {} profiles match incomplete context; provide phone/display "
                "identity or an explicit revision: {}".format(
                    len(rows), kind, ", ".join(row["revision_id"] for row in rows)
                )
            )
        return self.revision(rows[0]["revision_id"])

    @staticmethod
    def runtime_file(profile: Mapping[str, Any], logical_name: str) -> Path:
        entry = (profile.get("runtime_files") or {}).get(logical_name)
        if entry is None:
            raise KeyError("Profile has no runtime file {!r}".format(logical_name))
        return Path(profile["revision_directory"]) / str(entry["path"])

    def resolve_adapter(
        self, context: ProfileContext, request: AdapterRequest
    ) -> Dict[str, Any]:
        rig_game = phone_game = None
        if request.requires_game_profile:
            if not context.game_id:
                raise ProfileResolutionError(
                    "Adapter mode {} requires a game_id".format(request.mode)
                )
            rig_game = self.resolve("rig_game", context)
            dependencies = rig_game.get("dependencies") or {}
            try:
                rig = self.revision(str(dependencies["rig"]))
                phone_game = self.revision(str(dependencies["phone_game"]))
            except KeyError as exc:
                raise ProfileResolutionError(
                    "Rig-game profile has incomplete dependencies: {}".format(exc)
                )
        else:
            rig = self.resolve("rig", context)
        calibration_path = self.runtime_file(rig, "hik_camera_calibration")
        rig_game_path = (
            Path(rig_game["revision_directory"]) / "profile.json"
            if rig_game is not None
            else None
        )
        normalization = request.normalization
        result = {
            "schema_version": SCHEMA_VERSION,
            "resolved_utc": datetime.now(timezone.utc).isoformat(),
            "context": context.as_dict(),
            "request": request.as_dict(),
            "profiles": {
                "rig": rig["revision_id"],
                "phone_game": phone_game["revision_id"] if phone_game else None,
                "rig_game": rig_game["revision_id"] if rig_game else None,
            },
            "paths": {
                "rig_calibration": str(calibration_path.resolve()),
                "rig_game_profile": str(rig_game_path.resolve()) if rig_game_path else None,
            },
            "adapter_plan": {
                "mode": request.mode,
                "normalization": normalization,
                "rectify": normalization != "none",
                "color_order": request.color_order,
                "color_policy": request.color_policy,
                "roi_policy": request.roi_policy,
                "minimap_margin_px": int(request.minimap_margin_px),
                "registry_reads_per_frame": 0,
                "phone_operations": "none",
            },
        }
        return result


def context_from_rig_calibration(document: Mapping[str, Any]) -> ProfileContext:
    camera = document["camera"]
    phone = document["phone"]
    natural = phone.get("natural_screen_size_px") or phone.get("screen_size_px")
    logical = phone.get("screen_size_px") or natural
    return ProfileContext(
        camera_adapter=str(camera.get("adapter_id") or "hik_mvs"),
        camera_id=str(camera["device_id"]),
        phone_id=str(phone["serial"]),
        phone_model=phone.get("model"),
        panel_display={
            "natural_panel_px": natural,
            "logical_frame_px": logical,
            "refresh_hz": phone.get("refresh_hz"),
            "density_dpi": phone.get("logical_density_dpi") or phone.get("density_dpi"),
        },
    )


__all__ = [
    "AdapterRequest",
    "ProfileContext",
    "ProfileKey",
    "ProfileRegistry",
    "ProfileResolutionError",
    "context_from_rig_calibration",
    "default_profile_root",
]
