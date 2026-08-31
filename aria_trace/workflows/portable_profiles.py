"""Export and import camera-independent phone-platform calibration bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from aria_trace.adapters.filesystem.commented_yaml import write_commented_yaml
from aria_trace.adapters.filesystem.profile_registry import (
    ProfileContext,
    ProfileRegistry,
    ProfileResolutionError,
)


PORTABLE_SCHEMA_VERSION = "1.0"
PORTABLE_PROFILE_KINDS = ("phone_game", "phone_game_color")

PORTABLE_HEADER = """# AriaTrace portable phone-platform calibration.
#
# This package contains no camera identity, sensor ROI, rectification map, or
# HIK imaging control. Import composes it with a separately calibrated local rig."""

PORTABLE_COMMENTS = {
    "portable_context": (
        "Compatibility facts in Android/panel coordinates. Source handset identity "
        "is deliberately absent."
    ),
    "payload": "Camera-independent calibration data copied from the source revision.",
    "runtime_files": "Portable masks, references, or evidence with verified hashes.",
    "source": "Traceability only; these fields never select a local camera or phone.",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_context(profile: Mapping[str, Any]) -> ProfileContext:
    stored = ProfileContext.from_dict(profile.get("context") or {})
    return ProfileContext(
        game_id=stored.game_id,
        platform=stored.platform,
        package=stored.package,
        game_version=stored.game_version,
        panel_display=stored.panel_display,
        game_display=stored.game_display,
    )


def _portable_manifest(
    profile: Mapping[str, Any], files: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    kind = str((profile.get("identity") or {}).get("kind") or "")
    if kind not in PORTABLE_PROFILE_KINDS:
        raise ValueError(
            "Only {} profiles are portable; got {}".format(
                ", ".join(PORTABLE_PROFILE_KINDS), kind or "unknown"
            )
        )
    context = _portable_context(profile)
    source_context = ProfileContext.from_dict(profile.get("context") or {})
    payload = dict(profile.get("payload") or {})
    if "shift_estimation_mask" in files:
        payload["shift_estimation_mask"] = (
            "runtime_file:shift_estimation_mask"
        )
    return {
        "schema_version": PORTABLE_SCHEMA_VERSION,
        "package_kind": "aria_trace_portable_calibration",
        "profile_kind": kind,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "portable_context": context.as_dict(),
        "payload": payload,
        "runtime_files": dict(files),
        "source": {
            "revision_id": profile.get("revision_id"),
            "content_sha256": profile.get("content_sha256"),
            "phone_provenance": {
                "id": source_context.phone_id,
                "model": source_context.phone_model,
            },
            "provenance": dict(profile.get("provenance") or {}),
        },
    }


def export_portable_profile(
    revision_id: str,
    output: Path,
    *,
    registry: ProfileRegistry,
) -> Dict[str, Any]:
    """Export one portable profile as a reviewable directory or ZIP archive."""

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Portable output already exists: {}".format(output))
    profile = registry.revision(str(revision_id))
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".portable-", dir=str(parent)))
    try:
        files = {}
        assets = temporary / "files"
        for logical_name, entry in sorted(
            (profile.get("runtime_files") or {}).items()
        ):
            source = Path(profile["revision_directory"]) / str(entry["path"])
            if not source.is_file():
                raise FileNotFoundError(
                    "Portable runtime file is missing: {}".format(source)
                )
            target_name = "{}{}".format(logical_name, source.suffix.lower())
            target = assets / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source), str(target))
            files[str(logical_name)] = {
                "path": "files/{}".format(target_name),
                "sha256": _sha256(target),
                "size_bytes": target.stat().st_size,
            }
        manifest = _portable_manifest(profile, files)
        (temporary / "portable_profile.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        write_commented_yaml(
            temporary / "portable_profile.yaml",
            manifest,
            header=PORTABLE_HEADER,
            section_comments=PORTABLE_COMMENTS,
        )
        if output.suffix.lower() == ".zip":
            temporary_zip = output.with_suffix(output.suffix + ".tmp")
            with zipfile.ZipFile(
                str(temporary_zip), "w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                for path in sorted(temporary.rglob("*")):
                    if path.is_file():
                        archive.write(str(path), str(path.relative_to(temporary)))
            os.replace(str(temporary_zip), str(output))
        else:
            os.replace(str(temporary), str(output))
            temporary = None
        return {
            "output": str(output),
            "profile_kind": manifest["profile_kind"],
            "source_revision": revision_id,
            "runtime_file_count": len(files),
        }
    finally:
        if temporary is not None:
            shutil.rmtree(str(temporary), ignore_errors=True)


def _open_package(package: Path) -> tuple[Path, Optional[Path]]:
    package = Path(package).resolve()
    if package.is_dir():
        return package, None
    if not package.is_file() or package.suffix.lower() != ".zip":
        raise FileNotFoundError(
            "Portable package must be a directory or ZIP file: {}".format(package)
        )
    temporary = Path(tempfile.mkdtemp(prefix="aria-portable-import-"))
    try:
        with zipfile.ZipFile(str(package), "r") as archive:
            for member in archive.infolist():
                relative = Path(member.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(
                        "Unsafe portable ZIP member: {}".format(member.filename)
                    )
            archive.extractall(str(temporary))
    except Exception:
        shutil.rmtree(str(temporary), ignore_errors=True)
        raise
    return temporary, temporary


def _target_context(
    portable: ProfileContext, requested: Optional[ProfileContext]
) -> ProfileContext:
    requested = requested or ProfileContext(platform=portable.platform)
    return ProfileContext(
        game_id=requested.game_id or portable.game_id,
        platform=requested.platform or portable.platform,
        package=requested.package or portable.package,
        game_version=requested.game_version or portable.game_version,
        camera_adapter=requested.camera_adapter,
        camera_id=requested.camera_id,
        phone_id=requested.phone_id,
        phone_model=requested.phone_model,
        panel_display=requested.panel_display or portable.panel_display,
        game_display=requested.game_display or portable.game_display,
    )


def import_portable_profile(
    package: Path,
    *,
    registry: ProfileRegistry,
    requested_context: Optional[ProfileContext] = None,
    activate: bool = False,
    compose_local_rig: bool = True,
) -> Dict[str, Any]:
    """Import portable data and compose mini-map geometry with the local rig."""

    root, cleanup = _open_package(package)
    try:
        manifest_path = root / "portable_profile.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "Portable package has no portable_profile.json: {}".format(root)
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("package_kind") != "aria_trace_portable_calibration":
            raise ValueError("Not an AriaTrace portable calibration package")
        kind = str(manifest.get("profile_kind") or "")
        if kind not in PORTABLE_PROFILE_KINDS:
            raise ValueError("Unsupported portable profile kind: {}".format(kind))
        portable_context = ProfileContext.from_dict(
            manifest.get("portable_context") or {}
        )
        target_context = _target_context(portable_context, requested_context)
        compatibility = registry.evaluate_compatibility(
            kind, target_context, portable_context
        )
        runtime_files = {}
        for logical_name, entry in sorted(
            (manifest.get("runtime_files") or {}).items()
        ):
            source = root / str(entry["path"])
            if not source.is_file():
                raise FileNotFoundError(
                    "Portable package file is missing: {}".format(source)
                )
            actual = _sha256(source)
            if actual != str(entry["sha256"]):
                raise ValueError(
                    "Portable file hash mismatch for {}".format(logical_name)
                )
            runtime_files[str(logical_name)] = source
        payload = dict(manifest.get("payload") or {})
        payload["portable_import"] = {
            "source_revision": (manifest.get("source") or {}).get("revision_id"),
            "compatibility": compatibility,
        }
        imported = registry.publish(
            kind,
            target_context,
            payload,
            runtime_files=runtime_files,
            provenance={
                "portable_package": str(Path(package).resolve()),
                "portable_source": dict(manifest.get("source") or {}),
                "compatibility_at_import": compatibility,
            },
            review_state="accepted" if activate else "review_required",
            activate=activate,
        )
        result = {
            "portable_profile": imported,
            "compatibility": compatibility,
            "warnings": list(compatibility.get("warnings") or []),
            "rig_game": None,
            "local_fit_required": kind == "phone_game_color",
        }
        if kind != "phone_game" or not compose_local_rig:
            return result
        if payload.get("canonical_phone_crop_xywh") is None:
            raise ValueError(
                "Portable phone-game profile has no canonical mini-map crop to compose"
            )
        rig_request = ProfileContext(
            camera_adapter=target_context.camera_adapter,
            camera_id=target_context.camera_id,
            phone_id=target_context.phone_id,
            phone_model=target_context.phone_model,
            panel_display=target_context.panel_display,
        )
        try:
            rig = registry.resolve("rig", rig_request)
        except ProfileResolutionError as exc:
            result["warnings"].append(
                "Portable profile imported without local rig composition: {}".format(
                    exc
                )
            )
            return result
        rig_context = ProfileContext.from_dict(rig.get("context") or {})
        composed_context = ProfileContext(
            game_id=target_context.game_id,
            platform=target_context.platform,
            package=target_context.package,
            game_version=target_context.game_version,
            camera_adapter=rig_context.camera_adapter,
            camera_id=rig_context.camera_id,
            phone_id=target_context.phone_id or rig_context.phone_id,
            phone_model=target_context.phone_model or rig_context.phone_model,
            panel_display=target_context.panel_display,
            game_display=target_context.game_display,
        )
        composed_payload = {
            "profile_kind": "rig_game",
            "canonical_phone_crop_xywh": payload.get(
                "canonical_phone_crop_xywh"
            ),
            "composition_rule": (
                "Use portable phone-game geometry in canonical panel coordinates; "
                "the referenced local rig supplies every camera-space conversion."
            ),
            "portable_source_revision": imported["revision_id"],
            "compatibility_at_import": compatibility,
            "capabilities": {"modes": ["minimap", "dual"]},
        }
        result["rig_game"] = registry.publish(
            "rig_game",
            composed_context,
            composed_payload,
            dependencies={
                "rig": rig["revision_id"],
                "phone_game": imported["revision_id"],
            },
            provenance={
                "portable_package": str(Path(package).resolve()),
                "local_rig_revision": rig["revision_id"],
                "compatibility_at_import": compatibility,
            },
            review_state="accepted" if activate else "review_required",
            activate=activate,
        )
        return result
    finally:
        if cleanup is not None:
            shutil.rmtree(str(cleanup), ignore_errors=True)


__all__ = [
    "PORTABLE_PROFILE_KINDS",
    "export_portable_profile",
    "import_portable_profile",
]
