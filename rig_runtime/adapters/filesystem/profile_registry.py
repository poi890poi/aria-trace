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
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence

from .commented_yaml import write_commented_yaml


SCHEMA_VERSION = "2.3"
PROFILE_KINDS = (
    "rig",
    "game_model",
    "phone_game",
    "phone_game_color",
    "rig_game",
    "rig_game_color",
    "rig_game_orientation",
)
REVISION_STATES = ("review_required", "accepted")
ADAPTER_MODES = ("full", "minimap", "dual")
NORMALIZATION_MODES = ("auto", "dense_remap", "homography", "none")
COLOR_ORDERS = ("RGB", "BGR")
COLOR_POLICIES = ("auto", "rig_locked", "game_matched", "unadjusted")
ROI_POLICIES = ("auto", "full_phone", "minimap_only")
MASK_POLICIES = ("none", "minimap_circle")

PROFILE_HEADER = """# IRIS production profile revision.
#
# Revisions are immutable. The SQLite registry is the activation authority;
# this commented YAML is the human-readable payload and provenance record."""

PROFILE_COMMENTS = {
    "identity": "Portable ownership and display-compatibility dimensions.",
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
    configured = os.environ.get("IRIS_PROFILE_ROOT")
    if configured:
        # ``set IRIS_PROFILE_ROOT="C:\\path with spaces"`` retains the quote
        # characters in cmd.exe. Accept that common spelling so an external
        # application does not silently open a new registry under its cwd.
        configured = os.path.expandvars(str(configured).strip())
        if (
            len(configured) >= 2
            and configured[0] == configured[-1]
            and configured[0] in ("'", '"')
        ):
            configured = configured[1:-1].strip()
        if not configured:
            raise ValueError("IRIS_PROFILE_ROOT cannot be empty")
        configured_path = Path(configured).expanduser()
        if not configured_path.is_absolute():
            raise ValueError(
                "IRIS_PROFILE_ROOT must be an absolute path so profile selection "
                "does not depend on the calling application's working directory: "
                "{}".format(configured)
            )
        return configured_path.resolve()
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

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProfileContext":
        """Read a stored context while keeping device identity as provenance."""

        game = dict(value.get("game") or {})
        devices = dict(value.get("devices") or {})
        camera = dict(devices.get("camera") or {})
        phone = dict(devices.get("phone") or {})
        display = dict(value.get("display") or {})
        return cls(
            game_id=game.get("id"),
            platform=str(game.get("platform") or "android"),
            package=game.get("package"),
            game_version=game.get("version"),
            camera_adapter=str(camera.get("adapter") or "hik_mvs"),
            camera_id=camera.get("id"),
            phone_id=phone.get("id"),
            phone_model=phone.get("model"),
            panel_display=display.get("panel") or {},
            game_display=display.get("game") or {},
        )


@dataclass(frozen=True)
class AdapterRequest:
    """Requested runtime product; options do not create profile identities."""

    purpose: str = "application"
    mode: str = "full"
    normalization: str = "auto"
    color_order: str = "RGB"
    color_policy: str = "auto"
    roi_policy: str = "auto"
    mask_policy: str = "none"
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
        if self.mask_policy not in MASK_POLICIES:
            raise ValueError("Unsupported mask policy")
        if self.mask_policy != "none" and self.mode not in ("minimap", "dual"):
            raise ValueError("Mini-map masking requires minimap or dual mode")
        if self.mask_policy != "none" and self.normalization == "none":
            raise ValueError("Mini-map masking requires rectification")
        if int(self.minimap_margin_px) < 0:
            raise ValueError("Mini-map margin cannot be negative")
        if self.frame_rate_policy not in ("calibrated", "exact"):
            raise ValueError("Frame-rate policy must be calibrated or exact")
        if self.frame_rate_policy == "exact" and self.frame_rate is None:
            raise ValueError("Exact frame-rate policy requires frame_rate")

    @property
    def requires_game_profile(self) -> bool:
        """Whether this request needs any game-scoped calibration."""

        return self.requires_minimap_profile or self.requires_game_color

    @property
    def requires_minimap_profile(self) -> bool:
        return self.mode in ("minimap", "dual")

    @property
    def requires_game_color(self) -> bool:
        return self.color_policy == "game_matched"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "purpose": self.purpose,
            "mode": self.mode,
            "normalization": self.normalization,
            "color_order": self.color_order,
            "color_policy": self.color_policy,
            "roi_policy": self.roi_policy,
            "mask_policy": self.mask_policy,
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
            if not context.camera_id or not context.panel_signature:
                raise ValueError("Rig profiles require camera and panel display")
            return cls(
                kind,
                context.camera_id,
                "_",
                context.panel_signature,
                "_",
            )
        if kind == "game_model":
            if not context.game_id:
                raise ValueError("Game-model profiles require a game_id")
            return cls(kind, context.platform, context.game_id, "_", "_")
        if kind in ("phone_game", "phone_game_color"):
            if (
                not context.game_id
                or not context.panel_signature
                or not context.game_display_signature
            ):
                raise ValueError(
                    "Phone-game profiles require platform, game, panel, and game display"
                )
            return cls(
                kind,
                context.platform,
                context.game_id,
                context.panel_signature,
                context.game_display_signature,
            )
        if (
            not context.camera_id
            or not context.game_id
            or not context.panel_signature
            or not context.game_display_signature
        ):
            raise ValueError(
                "Rig-game profiles require camera, game, panel, and game display"
            )
        return cls(
            kind,
            context.camera_id,
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


def _profile_compatibility(
    kind: str,
    requested: ProfileContext,
    stored: ProfileContext,
) -> Dict[str, Any]:
    """Describe compatibility without treating source handset identity as ownership."""

    matched = []
    mismatches = []
    messages = []
    notes = []

    def compare(field_name: str, requested_value: Any, stored_value: Any) -> None:
        if requested_value is None or requested_value == {}:
            return
        if stored_value == requested_value:
            matched.append(field_name)
            return
        mismatch = {
            "field": field_name,
            "requested": requested_value,
            "profile": stored_value,
        }
        mismatches.append(mismatch)
        messages.append(
            "{} mismatch: requested {!r}, profile has {!r}".format(
                field_name, requested_value, stored_value
            )
        )

    compare("platform", requested.platform, stored.platform)
    if kind in ("rig", "rig_game", "rig_game_color", "rig_game_orientation"):
        compare("camera_id", requested.camera_id, stored.camera_id)
        compare("panel_display", requested.panel_display, stored.panel_display)
    if kind in (
        "game_model", "phone_game", "phone_game_color", "rig_game",
        "rig_game_color", "rig_game_orientation",
    ):
        compare("game_id", requested.game_id, stored.game_id)
        compare("game_package", requested.package, stored.package)
        compare("game_version", requested.game_version, stored.game_version)
        if kind != "game_model":
            if kind in ("phone_game", "phone_game_color"):
                compare("panel_display", requested.panel_display, stored.panel_display)
            compare("game_display", requested.game_display, stored.game_display)

    if requested.phone_id and stored.phone_id and requested.phone_id != stored.phone_id:
        notes.append(
            "Source phone differs (requested {!r}, profile {!r}); phone identity is "
            "provenance only".format(requested.phone_id, stored.phone_id)
        )
    if (
        requested.phone_model
        and stored.phone_model
        and requested.phone_model != stored.phone_model
    ):
        notes.append(
            "Source phone model differs (requested {!r}, profile {!r}); panel and "
            "game geometry determine portability".format(
                requested.phone_model, stored.phone_model
            )
        )
    return {
        "status": "compatible" if not mismatches else "incompatible_override",
        "warnings": messages,
        "mismatches": mismatches,
        "matched_fields": matched,
        "provenance_notes": notes,
        "policy": "advisory_for_explicit_revision",
    }


def _profile_match_rank(
    kind: str,
    requested: ProfileContext,
    stored: ProfileContext,
) -> Dict[str, Any]:
    """Rank compatible active variants from physical/static to fluid facts.

    Camera identity and requested game identity are filtered before ranking.
    Missing caller facts are neutral: they never masquerade as a match.  The
    ordered score is deliberately lexicographic so a later software or runtime
    fact cannot outweigh an earlier physical display fact.
    """

    rig_scoped = kind in (
        "rig", "rig_game", "rig_game_color", "rig_game_orientation"
    )
    game_scoped = kind != "rig"
    fields = []

    def add(name: str, requested_value: Any, stored_value: Any) -> None:
        if requested_value is None or requested_value == {}:
            fields.append({"field": name, "state": "caller_unspecified", "score": 1})
        elif stored_value is None or stored_value == {}:
            fields.append({"field": name, "state": "profile_unspecified", "score": 1})
        elif requested_value == stored_value:
            fields.append({"field": name, "state": "exact", "score": 2})
        else:
            fields.append({"field": name, "state": "mismatch", "score": 0})

    if rig_scoped:
        add("camera_id", requested.camera_id, stored.camera_id)
        add("camera_adapter", requested.camera_adapter, stored.camera_adapter)
    if kind != "game_model":
        for name in ("natural_panel_px", "density_dpi", "refresh_millihz"):
            add(
                "panel.{}".format(name),
                requested.panel_display.get(name),
                stored.panel_display.get(name),
            )
    add("platform", requested.platform, stored.platform)
    if game_scoped:
        add("game_id", requested.game_id, stored.game_id)
        add("game_package", requested.package, stored.package)
        if kind != "game_model":
            for name in (
                "logical_frame_px",
                "game_viewport_xywh",
                "content_layout_id",
                "ui_layout_id",
                "rotation_quarter_turns",
                "insets_px",
            ):
                add(
                    "game_display.{}".format(name),
                    requested.game_display.get(name),
                    stored.game_display.get(name),
                )
        add("game_version", requested.game_version, stored.game_version)
    return {
        "policy": "physical_static_then_software_fluid_v1",
        "ordered_fields": fields,
        "score": [int(item["score"]) for item in fields],
        "exact_count": sum(item["state"] == "exact" for item in fields),
        "mismatch_count": sum(item["state"] == "mismatch" for item in fields),
    }


class ProfileRegistry:
    """SQLite activation index plus immutable human-readable revision bundles."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = default_profile_root(root)
        self.registry_directory = self.root / ".registry"
        self.database = self.registry_directory / "profiles.sqlite3"
        database_existed = self.database.is_file()
        self.registry_directory.mkdir(parents=True, exist_ok=True)
        self._initialize()
        self.migration_report = self._recover_portable_registry(
            database_existed=database_existed
        )

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

    def _recover_portable_registry(self, *, database_existed: bool) -> Dict[str, Any]:
        """Repair the derived SQLite index from relocatable profile files.

        Immutable ``profile.json`` documents and per-variant ``active.json``
        pointers are the portable representation. SQLite is an acceleration
        and activation index. A directory move can therefore omit ``.registry``
        (common with dot-file-excluding copy tools) without losing profiles or
        their active selections.
        """

        active_paths = list(self.root.rglob("active.json"))
        with self._connect() as connection:
            revision_count = int(
                connection.execute("SELECT COUNT(*) FROM revisions").fetchone()[0]
            )
            active_rows = {
                str(row[0]): str(row[1])
                for row in connection.execute(
                    "SELECT revision_id, activated_utc FROM active_profiles"
                ).fetchall()
            }

        pointer_documents = []
        stale_authority = False
        for path in active_paths:
            try:
                pointer = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            revision_id = str(pointer.get("active_revision_id") or "")
            if not revision_id:
                continue
            pointer_documents.append((path, pointer, revision_id))
            authority = pointer.get("registry_authority")
            if authority:
                try:
                    stale_authority = (
                        Path(str(authority)).resolve() != self.database.resolve()
                    ) or stale_authority
                except OSError:
                    stale_authority = True

        active_ids = {item[2] for item in pointer_documents}
        index_incomplete = (
            not database_existed
            or revision_count == 0
            or not active_ids.issubset(set(active_rows))
            or stale_authority
        )
        report = {
            "status": "not_needed",
            "profile_root": str(self.root),
            "database_existed": bool(database_existed),
            "revisions_recovered": 0,
            "activations_recovered": 0,
            "active_pointers_rebound": 0,
        }
        if not index_incomplete:
            return report

        manifest_paths = [
            path
            for path in self.root.rglob("profile.json")
            if path.parent.parent.name == "revisions"
        ]
        manifests = {}
        for path in manifest_paths:
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            revision_id = str(manifest.get("revision_id") or "")
            identity = dict(manifest.get("identity") or {})
            context = ProfileContext.from_dict(manifest.get("context") or {})
            kind = str(identity.get("kind") or "")
            if not revision_id or kind not in PROFILE_KINDS:
                continue
            try:
                relative_directory = str(path.parent.relative_to(self.root))
            except ValueError:
                continue
            manifests[revision_id] = (
                path, manifest, identity, context, relative_directory
            )

        recovered_revisions = 0
        recovered_activations = 0
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for revision_id, item in manifests.items():
                path, manifest, identity, context, relative_directory = item
                before = connection.total_changes
                connection.execute(
                    """INSERT OR IGNORE INTO revisions
                       (revision_id, kind, owner_id, game_id, panel_signature,
                        game_signature, variant_id, review_state, created_utc,
                        relative_directory, content_sha256, camera_id, phone_id,
                        context_json, dependencies_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        revision_id,
                        str(identity.get("kind") or ""),
                        str(identity.get("owner_id") or ""),
                        str(identity.get("game_id") or "_"),
                        str(identity.get("panel_signature") or "_"),
                        str(
                            identity.get("game_display_signature")
                            or identity.get("game_signature")
                            or "_"
                        ),
                        str(identity.get("variant_id") or "_"),
                        str(manifest.get("review_state") or "review_required"),
                        str(manifest.get("created_utc") or ""),
                        relative_directory,
                        str(
                            manifest.get("content_sha256")
                            or hashlib.sha256(path.read_bytes()).hexdigest()
                        ),
                        context.camera_id,
                        context.phone_id,
                        _canonical_json(manifest.get("context") or {}),
                        _canonical_json(manifest.get("dependencies") or {}),
                    ),
                )
                if connection.total_changes > before:
                    recovered_revisions += 1

            for _path, pointer, revision_id in pointer_documents:
                row = connection.execute(
                    "SELECT * FROM revisions WHERE revision_id=?", (revision_id,)
                ).fetchone()
                if row is None:
                    continue
                previous = connection.execute(
                    """SELECT revision_id FROM active_profiles
                       WHERE kind=? AND owner_id=? AND game_id=?
                         AND panel_signature=? AND game_signature=?""",
                    (
                        row["kind"], row["owner_id"], row["game_id"],
                        row["panel_signature"], row["game_signature"],
                    ),
                ).fetchone()
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
                        revision_id,
                        str(pointer.get("activated_utc") or row["created_utc"]),
                    ),
                )
                connection.execute(
                    "UPDATE revisions SET review_state='accepted' WHERE revision_id=?",
                    (revision_id,),
                )
                if previous is None or str(previous["revision_id"]) != revision_id:
                    recovered_activations += 1

        rebound = 0
        for path, pointer, revision_id in pointer_documents:
            if revision_id not in manifests:
                continue
            authority = str(self.database.resolve())
            if str(pointer.get("registry_authority") or "") == authority:
                continue
            rebound_pointer = dict(pointer)
            rebound_pointer["registry_authority"] = authority
            _atomic_json(path, rebound_pointer)
            yaml_path = path.with_suffix(".yaml")
            write_commented_yaml(
                yaml_path,
                rebound_pointer,
                header="# Human-readable mirror; the SQLite registry is authoritative.",
                section_comments={"identity": "Display-specific active profile key."},
            )
            rebound += 1

        report.update(
            status="recovered_from_portable_manifests",
            revisions_recovered=recovered_revisions,
            activations_recovered=recovered_activations,
            active_pointers_rebound=rebound,
        )
        if recovered_revisions or recovered_activations:
            warnings.warn(
                "Recovered IRIS profile registry at {} from portable profile "
                "manifests ({} revisions, {} active selections).".format(
                    self.root, recovered_revisions, recovered_activations
                ),
                RuntimeWarning,
            )
        return report

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
        if kind in (
            "rig", "rig_game", "rig_game_color", "rig_game_orientation"
        ) and context.camera_id:
            clauses.append("r.camera_id=?")
            values.append(context.camera_id)
        if kind != "rig" and context.game_id:
            clauses.append("r.game_id=?")
            values.append(context.game_id)
        query = """SELECT r.* FROM active_profiles a
                   JOIN revisions r ON r.revision_id=a.revision_id
                   WHERE {} ORDER BY r.created_utc DESC""".format(" AND ".join(clauses))
        with self._connect() as connection:
            return connection.execute(query, values).fetchall()

    def list_candidates(
        self,
        kind: str,
        context: ProfileContext,
        *,
        active_only: bool = True,
    ) -> Sequence[Dict[str, Any]]:
        """List selectable revisions in automatic preference order."""

        if kind not in PROFILE_KINDS:
            raise ValueError("Unsupported profile kind: {}".format(kind))
        if active_only:
            rows = list(self._active_rows(kind, context))
        else:
            clauses = ["kind=?"]
            values = [kind]
            if kind in (
                "rig", "rig_game", "rig_game_color", "rig_game_orientation"
            ) and context.camera_id:
                clauses.append("camera_id=?")
                values.append(context.camera_id)
            if kind != "rig" and context.game_id:
                clauses.append("game_id=?")
                values.append(context.game_id)
            query = "SELECT * FROM revisions WHERE {}".format(
                " AND ".join(clauses)
            )
            with self._connect() as connection:
                rows = list(connection.execute(query, values).fetchall())
        candidates = []
        for row in rows:
            profile = self.revision(row["revision_id"])
            stored = ProfileContext.from_dict(profile.get("context") or {})
            profile["resolution"] = {
                "selection": "candidate",
                "compatibility": _profile_compatibility(kind, context, stored),
                "rank": _profile_match_rank(kind, context, stored),
            }
            candidates.append(profile)
        candidates.sort(
            key=lambda item: (
                tuple(item["resolution"]["rank"]["score"]),
                str(item.get("created_utc") or ""),
            ),
            reverse=True,
        )
        for index, item in enumerate(candidates, start=1):
            item["resolution"]["preference_index"] = index
            item["resolution"]["candidate_count"] = len(candidates)
        return candidates

    def resolve(self, kind: str, context: ProfileContext) -> Dict[str, Any]:
        candidates = list(self.list_candidates(kind, context, active_only=True))
        if not candidates:
            raise ProfileResolutionError(
                "No active {} profile matches camera={!r}, phone={!r}, game={!r}, "
                "panel={!r}, game_display={!r} under IRIS profile root {!s} "
                "(registry migration status: {}).".format(
                    kind, context.camera_id, context.phone_id, context.game_id,
                    context.panel_signature, context.game_display_signature,
                    self.root, self.migration_report.get("status"),
                )
            )
        result = candidates[0]
        rank = dict(result["resolution"]["rank"])
        result["resolution"] = {
            "selection": (
                "best_ranked_active_revision"
                if len(candidates) > 1
                else "active_revision"
            ),
            "candidate_count": len(candidates),
            "rank": rank,
            "compatibility": _profile_compatibility(
                kind,
                context,
                ProfileContext.from_dict(result.get("context") or {}),
            ),
        }
        return result

    def resolve_revision(
        self,
        revision_id: str,
        context: ProfileContext,
        *,
        expected_kind: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resolve an explicit revision even when its compatibility is imperfect."""

        result = self.revision(revision_id)
        kind = str((result.get("identity") or {}).get("kind") or "")
        if expected_kind is not None and kind != expected_kind:
            raise ProfileResolutionError(
                "Explicit revision {} is {}, expected {}".format(
                    revision_id, kind, expected_kind
                )
            )
        compatibility = _profile_compatibility(
            kind,
            context,
            ProfileContext.from_dict(result.get("context") or {}),
        )
        result["resolution"] = {
            "selection": "explicit_revision_override",
            "candidate_count": 1,
            "compatibility": compatibility,
        }
        return result

    @staticmethod
    def evaluate_compatibility(
        kind: str,
        requested: ProfileContext,
        profile_context: ProfileContext,
    ) -> Dict[str, Any]:
        if kind not in PROFILE_KINDS:
            raise ValueError("Unsupported profile kind: {}".format(kind))
        return _profile_compatibility(kind, requested, profile_context)

    def list_revisions(
        self, *, kind: Optional[str] = None, active_only: bool = False
    ) -> Sequence[Dict[str, Any]]:
        """List immutable revisions without consulting artifact directories."""

        if kind is not None and kind not in PROFILE_KINDS:
            raise ValueError("Unsupported profile kind: {}".format(kind))
        clauses = []
        values = []
        if kind is not None:
            clauses.append("r.kind=?")
            values.append(kind)
        if active_only:
            clauses.append("a.revision_id IS NOT NULL")
        where = " WHERE {}".format(" AND ".join(clauses)) if clauses else ""
        query = """SELECT r.*, CASE WHEN a.revision_id IS NULL THEN 0 ELSE 1 END AS active
                   FROM revisions r
                   LEFT JOIN active_profiles a ON a.revision_id=r.revision_id
                   {} ORDER BY r.created_utc DESC""".format(where)
        with self._connect() as connection:
            rows = connection.execute(query, values).fetchall()
        return [
            {
                "revision_id": row["revision_id"],
                "kind": row["kind"],
                "active": bool(row["active"]),
                "review_state": row["review_state"],
                "camera_id": row["camera_id"],
                "phone_id": row["phone_id"],
                "game_id": None if row["game_id"] == "_" else row["game_id"],
                "created_utc": row["created_utc"],
                "revision_directory": str(
                    (self.root / row["relative_directory"]).resolve()
                ),
            }
            for row in rows
        ]

    @staticmethod
    def runtime_file(profile: Mapping[str, Any], logical_name: str) -> Path:
        entry = (profile.get("runtime_files") or {}).get(logical_name)
        if entry is None:
            raise KeyError("Profile has no runtime file {!r}".format(logical_name))
        return Path(profile["revision_directory"]) / str(entry["path"])

    def resolve_adapter(
        self,
        context: ProfileContext,
        request: AdapterRequest,
        *,
        profile_revisions: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Any]:
        selected_revisions = {
            str(kind): str(revision_id)
            for kind, revision_id in dict(profile_revisions or {}).items()
        }
        unsupported = sorted(set(selected_revisions) - set(PROFILE_KINDS))
        if unsupported:
            raise ProfileResolutionError(
                "Unsupported manual profile kinds: {}".format(", ".join(unsupported))
            )

        def selected(kind: str) -> Dict[str, Any]:
            revision_id = selected_revisions.get(kind)
            if revision_id is None:
                return self.resolve(kind, context)
            return self.resolve_revision(
                revision_id, context, expected_kind=kind
            )

        rig_game = phone_game = rig_game_color = rig_game_orientation = None
        game_model = None
        resolution_warnings = []
        stale_game_color_fallback = False
        if request.requires_minimap_profile:
            if not context.game_id:
                raise ProfileResolutionError(
                    "Adapter mode {} requires a game_id".format(request.mode)
                )
            rig_game = selected("rig_game")
            dependencies = rig_game.get("dependencies") or {}
            try:
                dependent_rig_id = str(dependencies["rig"])
                dependent_phone_game_id = str(dependencies["phone_game"])
                phone_game = self.resolve_revision(
                    dependent_phone_game_id,
                    context,
                    expected_kind="phone_game",
                )
            except KeyError as exc:
                raise ProfileResolutionError(
                    "Rig-game profile has incomplete dependencies: {}".format(exc)
                )
            requested_rig_id = selected_revisions.get("rig")
            requested_phone_game_id = selected_revisions.get("phone_game")
            if requested_phone_game_id and requested_phone_game_id != dependent_phone_game_id:
                raise ProfileResolutionError(
                    "Manual phone_game revision {} conflicts with rig_game {} "
                    "dependency {}".format(
                        requested_phone_game_id,
                        rig_game["revision_id"],
                        dependent_phone_game_id,
                    )
                )
            if requested_rig_id and requested_rig_id != dependent_rig_id:
                raise ProfileResolutionError(
                    "Manual rig revision {} conflicts with rig_game {} dependency {}"
                    .format(
                        requested_rig_id, rig_game["revision_id"], dependent_rig_id
                    )
                )
            active_rig = selected("rig")
            if (
                "rig_game" not in selected_revisions
                and dependent_rig_id != str(active_rig["revision_id"])
            ):
                raise ProfileResolutionError(
                    "Active rig-game profile {} is stale: it depends on superseded "
                    "active rig {}, while the current active rig is {}. Re-publish "
                    "the current rig calibration to recompose game profiles before "
                    "opening minimap or dual mode.".format(
                        rig_game["revision_id"],
                        dependent_rig_id,
                        active_rig["revision_id"],
                    )
                )
            rig = self.resolve_revision(
                dependent_rig_id, context, expected_kind="rig"
            )
        else:
            rig = selected("rig")

        def selected_for_resolved_rig(kind: str) -> tuple[Optional[Dict[str, Any]], list[str]]:
            """Select an optional layer whose immutable rig dependency matches."""

            rig_revision = str(rig["revision_id"])
            if kind in selected_revisions:
                candidate = selected(kind)
                dependency = str((candidate.get("dependencies") or {}).get("rig") or "")
                if dependency != rig_revision:
                    raise ProfileResolutionError(
                        "Manual {} revision {} depends on rig {}, not selected rig {}"
                        .format(
                            kind,
                            candidate["revision_id"],
                            dependency or "<missing>",
                            rig_revision,
                        )
                    )
                return candidate, []
            candidates = list(self.list_candidates(kind, context, active_only=True))
            for candidate in candidates:
                dependency = str((candidate.get("dependencies") or {}).get("rig") or "")
                if dependency == rig_revision:
                    return candidate, []
            return None, [str(item["revision_id"]) for item in candidates]

        # Screen-upright orientation is independent of mini-map and color.
        # It is therefore optional for every game-scoped adapter mode and is
        # accepted only when it depends on the exact resolved rig revision.
        if context.game_id:
            try:
                game_model = selected("game_model")
            except ProfileResolutionError as exc:
                if not str(exc).startswith("No active game_model profile"):
                    raise
            candidate, stale_orientation_ids = selected_for_resolved_rig(
                "rig_game_orientation"
            )
            if candidate is None:
                resolution_warnings.append(
                    "No active game-orientation profile matches game {!r} and "
                    "resolved rig {}{}; "
                    "adapter output remains rig-calibration-display-up, which "
                    "may be Android/presenter-up rather than game-up. Run game "
                    "calibration, compose from the foreground Android surface, "
                    "or invoke runtime ADB/HIK image orientation correction."
                    .format(
                        context.game_id,
                        rig["revision_id"],
                        (
                            "; ignored stale active revisions {}".format(
                                ", ".join(stale_orientation_ids)
                            )
                            if stale_orientation_ids else ""
                        ),
                    )
                )
            else:
                rig_game_orientation = candidate

        if request.color_policy in ("auto", "game_matched") and context.game_id:
            candidate, stale_color_ids = selected_for_resolved_rig(
                "rig_game_color"
            )
            if candidate is None and stale_color_ids:
                stale_game_color_fallback = True
                resolution_warnings.append(
                    "Active game-color revisions {} do not depend on resolved rig {}; "
                    "the adapter is using rig-locked color.".format(
                        ", ".join(stale_color_ids), rig["revision_id"]
                    )
                )
            elif candidate is not None:
                rig_game_color = candidate
        elif request.requires_game_color:
            raise ProfileResolutionError(
                "Game-matched color requires a game_id"
            )

        # Older accepted rig-game revisions may carry the color payload. Keep
        # them readable while new color calibration publishes an independent
        # rig_game_color revision.
        legacy_color_profile = None
        if rig_game is not None:
            conversion = (rig_game.get("payload") or {}).get(
                "hik_bayer_conversion"
            )
            if isinstance(conversion, Mapping) and conversion.get("status") == "selected":
                legacy_color_profile = rig_game
        selected_color_profile = rig_game_color or legacy_color_profile
        if (
            request.requires_game_color
            and selected_color_profile is None
            and not stale_game_color_fallback
        ):
            raise ProfileResolutionError(
                "No active game-color calibration matches the resolved rig and game"
            )
        calibration_path = self.runtime_file(rig, "hik_camera_calibration")
        rig_game_path = (
            Path(rig_game["revision_directory"]) / "profile.json"
            if rig_game is not None
            else None
        )
        game_color_path = (
            Path(selected_color_profile["revision_directory"]) / "profile.json"
            if selected_color_profile is not None
            else None
        )
        effective_color_policy = request.color_policy
        if request.color_policy == "auto" or stale_game_color_fallback:
            effective_color_policy = (
                "game_matched" if selected_color_profile is not None else "rig_locked"
            )
        normalization = request.normalization
        game_model_plan = dict(
            (game_model.get("payload") or {})
            if game_model
            else {
                "cursor_follows": "character",
                "minimap_orientation": "unspecified",
                "source": "iris_default",
            }
        )
        if "cursor_behavior_by_acquisition" not in game_model_plan:
            game_model_plan["cursor_behavior_by_acquisition"] = (
                {
                    "zigzag": "rotating",
                    "micro_movement": "static",
                }
                if game_model_plan.get("cursor_follows") == "camera"
                else {
                    "zigzag": "static",
                    "micro_movement": "rotating",
                }
            )
        orientation_payload = dict(
            (rig_game_orientation.get("payload") or {})
            if rig_game_orientation else {}
        )
        initialization_recovery = dict(
            orientation_payload.get("initialization_recovery") or {}
        )
        initialization_surface_turns = initialization_recovery.get(
            "selected_surface_quarter_turns_clockwise_from_phone_natural"
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "resolved_utc": datetime.now(timezone.utc).isoformat(),
            "context": context.as_dict(),
            "request": request.as_dict(),
            "manual_profile_revisions": dict(selected_revisions),
            "profiles": {
                "rig": rig["revision_id"],
                "phone_game": phone_game["revision_id"] if phone_game else None,
                "rig_game": rig_game["revision_id"] if rig_game else None,
                "rig_game_color": (
                    rig_game_color["revision_id"] if rig_game_color else None
                ),
                "rig_game_orientation": (
                    rig_game_orientation["revision_id"]
                    if rig_game_orientation else None
                ),
                "game_model": (
                    game_model["revision_id"] if game_model else None
                ),
            },
            "paths": {
                "rig_calibration": str(calibration_path.resolve()),
                "rig_game_profile": str(rig_game_path.resolve()) if rig_game_path else None,
                "game_color_profile": (
                    str(game_color_path.resolve()) if game_color_path else None
                ),
                "game_orientation_profile": (
                    str(
                        (Path(rig_game_orientation["revision_directory"]) / "profile.json")
                        .resolve()
                    )
                    if rig_game_orientation else None
                ),
                "game_model_profile": (
                    str((Path(game_model["revision_directory"]) / "profile.json").resolve())
                    if game_model else None
                ),
            },
            "adapter_plan": {
                "mode": request.mode,
                "normalization": normalization,
                "rectify": normalization != "none",
                "color_order": request.color_order,
                "color_policy": effective_color_policy,
                "roi_policy": request.roi_policy,
                "mask_policy": request.mask_policy,
                "minimap_margin_px": int(request.minimap_margin_px),
                "game_upright_quarter_turns_clockwise": int(
                    (orientation_payload.get(
                        "camera_adapter_image_quarter_turns_clockwise_from_calibration_display",
                        0,
                    ))
                ) % 4 if rig_game_orientation else 0,
                "initialization_surface_quarter_turns_clockwise_from_natural": (
                    int(initialization_surface_turns) % 4
                    if initialization_surface_turns is not None
                    else None
                ),
                "game_model": game_model_plan,
                "registry_reads_per_frame": 0,
                "phone_operations": "none",
            },
        }
        selected_profiles = {
            "rig": rig,
            "phone_game": phone_game,
            "rig_game": rig_game,
            "rig_game_color": rig_game_color,
            "rig_game_orientation": rig_game_orientation,
            "game_model": game_model,
        }
        profile_compatibility = {}
        compatibility_warnings = list(resolution_warnings)
        provenance_notes = []
        for profile_kind, profile in selected_profiles.items():
            if profile is None:
                continue
            details = dict(
                ((profile.get("resolution") or {}).get("compatibility") or {})
            )
            profile_compatibility[profile_kind] = details
            compatibility_warnings.extend(
                "{}: {}".format(profile_kind, message)
                for message in details.get("warnings") or []
            )
            provenance_notes.extend(
                "{}: {}".format(profile_kind, message)
                for message in details.get("provenance_notes") or []
            )
        result["compatibility"] = {
            "policy": "automatic_compatible_or_explicit_warn_and_allow",
            "profiles": profile_compatibility,
            "warnings": compatibility_warnings,
            "provenance_notes": provenance_notes,
        }
        for message in compatibility_warnings:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
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
    "PROFILE_KINDS",
    "MASK_POLICIES",
    "ProfileRegistry",
    "ProfileResolutionError",
    "context_from_rig_calibration",
    "default_profile_root",
]
