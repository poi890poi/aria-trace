"""Offline retention of immutable profiles and standalone calibration evidence."""
from __future__ import annotations

import json
import math
import os
import shutil
import stat
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .atomic_write import atomic_write_text, replace_with_retry, retry_windows_access


DISPLACEMENT_KINDS = frozenset({"rig", "rig_game", "rig_game_orientation"})
EVIDENCE_MARKERS = frozenset({
    "hik_camera_calibration.json", "game_calibration_summary.json",
    "game_color_calibration.json", "game_orientation_calibration.json",
    "calibration_summary.json", "minimap_calibration.json",
    "calibration.json", "failure.json", "precheck.json",
})


@dataclass(frozen=True)
class RetentionPolicy:
    keep_portable: int = 10
    keep_rig: int = 3
    evidence_max_mb: float = 384
    evidence_min_age_hours: float = 24

    def __post_init__(self):
        for value in (self.keep_portable, self.keep_rig):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Profile retention counts must be positive integers")
        if self.keep_rig > self.keep_portable:
            raise ValueError("Rig retention must not exceed portable retention")
        for value in (self.evidence_max_mb, self.evidence_min_age_hours):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Evidence limits must be finite and nonnegative")


def _checked_path(root, path):
    """Reject escaping paths and Windows junctions as well as symlinks."""
    root = Path(root).resolve()
    path = Path(os.path.abspath(path))
    if path == root or not path.is_relative_to(root):
        raise ValueError("Purge path must be strictly inside its managed root: {}".format(path))
    current = path
    while current != root:
        if current.exists() or current.is_symlink():
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise ValueError("Purge does not follow links or reparse points: {}".format(current))
        current = current.parent
    if not path.resolve().is_relative_to(root):
        raise ValueError("Resolved purge path escapes its managed root")
    return path


def _tree_state(root, directory):
    directory = _checked_path(root, directory)
    size, newest, count = 0, directory.stat().st_mtime_ns, 0
    pending = [directory]
    while pending:
        folder = pending.pop()
        for child in folder.iterdir():
            _checked_path(root, child)
            info = child.stat()
            newest = max(newest, info.st_mtime_ns)
            if child.is_dir():
                pending.append(child)
            elif child.is_file():
                size += info.st_size
                count += 1
            else:
                raise ValueError("Unsupported filesystem entry in purge target: {}".format(child))
    return {"bytes": size, "newest_mtime_ns": newest, "file_count": count}


def _remove_tree(root, directory):
    # Verify both the resolved absolute target and every descendant before
    # recursive deletion; never traverse junctions into another storage root.
    _tree_state(root, directory)
    retry_windows_access(lambda: shutil.rmtree(str(directory)))


def _revision_path(registry, relative, revision_id):
    from .profile_registry import PROFILE_KINDS
    path = _checked_path(registry.root, registry.root / relative)
    parts = path.relative_to(registry.root).parts
    if len(parts) != 6 or parts[0] not in PROFILE_KINDS or parts[-2] != "revisions" or parts[-1] != revision_id:
        raise ValueError("Unexpected immutable revision path: {}".format(path))
    return path


def recover_pending_purge(registry, *, database_existed=True, delete_committed=False):
    """Restore pre-commit moves; finish committed deletions only on purge apply.

    A journal is outside the revisions hierarchy, so interrupted deletions
    cannot be rediscovered as portable profiles. SQLite serializes this with
    publication and activation, including another process recovering a move.
    """
    trash = registry.registry_directory / "purge-trash"
    if not trash.exists():
        return []
    warnings = []
    with registry._connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        _checked_path(registry.root, trash)
        for entry in list(trash.iterdir()):
            try:
                _tree_state(registry.root, entry)
                journal = entry / "entry.json"
                if not journal.is_file():
                    continue
                item = json.loads(journal.read_text(encoding="utf-8"))
                bundle = entry / "bundle"
                if item["kind"] == "revision":
                    original = _revision_path(registry, item["relative_path"], item["revision_id"])
                    still_indexed = connection.execute("SELECT 1 FROM revisions WHERE revision_id=?", (item["revision_id"],)).fetchone()
                    if still_indexed or not database_existed:
                        if bundle.exists():
                            if original.exists():
                                raise RuntimeError("Both original and quarantined revision exist: {}".format(original))
                            original.parent.mkdir(parents=True, exist_ok=True)
                            replace_with_retry(bundle, original)
                        _remove_tree(registry.root, entry)
                        continue
                elif item["kind"] != "evidence":
                    raise ValueError("Unknown purge journal kind")
                if delete_committed:
                    _remove_tree(registry.root, entry)
            except (OSError, ValueError, RuntimeError, KeyError) as exc:
                warnings.append("Pending purge {}: {}".format(entry.name, exc))
    return warnings


def _stage(registry, path, kind, revision_id=None):
    path = _checked_path(registry.root, path)
    _tree_state(registry.root, path)
    entry = registry.registry_directory / "purge-trash" / uuid.uuid4().hex
    _checked_path(registry.root, entry)
    entry.mkdir(parents=True)
    atomic_write_text(entry / "entry.json", json.dumps({
        "kind": kind, "revision_id": revision_id,
        "relative_path": str(path.relative_to(registry.root)),
    }))
    replace_with_retry(path, entry / "bundle")


def _evidence_plan(registry, policy, protected_paths, reviews, now):
    root = registry.root / "calibrations"
    report = {"root": str(root), "limit_bytes": int(policy.evidence_max_mb * 1_000_000),
              "before_bytes": 0, "projected_after_bytes": 0, "delete": [], "protected": [], "warnings": []}
    candidates = []
    pending = []
    if root.exists():
        try:
            state = _tree_state(registry.root, root)
            report["before_bytes"] = state["bytes"]
            pending = [root]
        except (OSError, ValueError) as exc:
            report["warnings"].append(str(exc))
    for item in reviews:
        report["before_bytes"] += item["bytes"]
        if Path(item["path"]) in protected_paths:
            report["protected"].append({**item, "reason": "referenced_by_retained_runtime"})
        elif item["newest_mtime_ns"] / 1e9 > now - policy.evidence_min_age_hours * 3600:
            report["protected"].append({**item, "reason": "recent_or_in_progress"})
        else:
            candidates.append(item)
    report["projected_after_bytes"] = report["before_bytes"]
    while pending:
        directory = pending.pop()
        children = list(directory.iterdir())
        if directory != root and EVIDENCE_MARKERS.intersection(child.name for child in children):
            state = _tree_state(registry.root, directory)
            item = {"path": str(directory), "kind": "bundle", **state}
            if any(path == directory or path.is_relative_to(directory) or (path.is_dir() and directory.is_relative_to(path)) for path in protected_paths):
                item["reason"] = "referenced_by_retained_runtime"
                report["protected"].append(item)
            elif state["newest_mtime_ns"] / 1e9 > now - policy.evidence_min_age_hours * 3600:
                item["reason"] = "recent_or_in_progress"
                report["protected"].append(item)
            else:
                candidates.append(item)
            continue
        pending.extend(child for child in children if child.is_dir() and not child.name.startswith("."))
    for item in sorted(candidates, key=lambda row: (row["newest_mtime_ns"], row["path"])):
        if report["projected_after_bytes"] <= report["limit_bytes"]:
            break
        report["delete"].append(item)
        report["projected_after_bytes"] -= item["bytes"]
    report["projected_over_limit_bytes"] = max(0, report["projected_after_bytes"] - report["limit_bytes"])
    return report


def retained_manifest(directory, manifest):
    """Keep immutable calibration identity while recording optional evidence loss.

    Only explicitly named review_evidence_* files can be pruned. Masks, models,
    references, and all other runtime_files remain mandatory. Portable exports
    obtain this effective view from ProfileRegistry.revision().
    """
    ledger = directory / "evidence-retention.json"
    if ledger.is_file():
        pruned = json.loads(ledger.read_text(encoding="utf-8")).get("pruned", {})
        missing = {
            name: item for name, item in pruned.items()
            if name.startswith("review_evidence_")
            and name in manifest.get("runtime_files", {})
            and item.get("path") == manifest["runtime_files"][name].get("path")
            and not (directory / item["path"]).exists()
        }
        if missing:
            manifest["runtime_files"] = {name: item for name, item in manifest["runtime_files"].items() if name not in missing}
            manifest["evidence_retention"] = {"status": "review_evidence_pruned", "pruned": missing}
    return manifest


def _prune_review(registry, item):
    directory = _revision_path(registry, item["relative_directory"], item["revision_id"])
    path = _checked_path(directory, Path(item["path"]))
    if path.stat().st_size != item["bytes"] or path.stat().st_mtime_ns != item["newest_mtime_ns"]:
        raise RuntimeError("Review evidence changed during purge; left in place")
    ledger = _checked_path(registry.root, directory / "evidence-retention.json")
    document = json.loads(ledger.read_text(encoding="utf-8")) if ledger.exists() else {"pruned": {}}
    document["pruned"][item["logical_name"]] = {
        "path": path.relative_to(directory).as_posix(), "size_bytes": item["bytes"],
        "sha256": item["sha256"], "purged_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # Publish the explanation before unlinking. If unlink fails, the still
    # present evidence remains visible; retrying the purge is safe.
    atomic_write_text(ledger, json.dumps(document, indent=2))
    retry_windows_access(path.unlink)


def _portable_pointer_ids(registry, rows):
    """Retain stale portable activations too, until a later activation repairs them."""
    snapshot = _checked_path(registry.root, registry.root / "active-profiles.json")
    if snapshot.is_file():
        for item in json.loads(snapshot.read_text(encoding="utf-8"))["profiles"]:
            yield item["pointer"]["active_revision_id"]
    directories = {
        _revision_path(registry, row["relative_directory"], revision_id).parents[1]
        for revision_id, row in rows.items()
    }
    for directory in directories:
        pointer = _checked_path(registry.root, directory / "active.json")
        if pointer.is_file():
            yield json.loads(pointer.read_text(encoding="utf-8"))["active_revision_id"]


def _runtime_paths(value, root, base=None):
    """Protect explicit local file references, excluding audit-only provenance."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key != "provenance":
                yield from _runtime_paths(item, root, base)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _runtime_paths(item, root, base)
    elif isinstance(value, str) and value.strip():
        try:
            path = Path(value)
            paths = [path] if path.is_absolute() else [root / path, (base or root) / path]
            for candidate in paths:
                resolved = candidate.resolve()
                if resolved.is_relative_to(root) and resolved.exists():
                    yield resolved
        except (OSError, ValueError):
            pass


def purge_profiles(registry, policy=None, *, dry_run=True):
    """Plan/apply retention at an idle maintenance boundary, never per frame."""
    policy = policy or RetentionPolicy()
    report = {"dry_run": dry_run, "policy": asdict(policy), "delete_revisions": [],
              "removed_revisions": [], "removed_evidence": [], "protected_revisions": {}, "warnings": []}
    if not dry_run:
        report["warnings"].extend(recover_pending_purge(registry, delete_committed=True))
    try:
        with registry._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = {row["revision_id"]: dict(row) for row in connection.execute("SELECT * FROM revisions ORDER BY created_utc DESC, revision_id DESC")}
            active = {row[0] for row in connection.execute("SELECT revision_id FROM active_profiles")}
            keep, group_counts = set(active), {}
            for revision_id in active:
                report["protected_revisions"][revision_id] = "active"
            for revision_id in _portable_pointer_ids(registry, rows):
                if revision_id in rows and revision_id not in keep:
                    keep.add(revision_id)
                    report["protected_revisions"][revision_id] = "portable_activation_pointer"
            for revision_id, row in rows.items():
                key = tuple(row[field] for field in ("kind", "owner_id", "game_id", "panel_signature", "game_signature"))
                limit = policy.keep_rig if row["kind"] in DISPLACEMENT_KINDS else policy.keep_portable
                group_counts[key] = group_counts.get(key, 0) + 1
                if group_counts[key] <= limit:
                    keep.add(revision_id)
            pending = list(keep)
            while pending:
                revision_id = pending.pop()
                for dependency in json.loads(rows[revision_id]["dependencies_json"]).values():
                    if dependency in rows and dependency not in keep:
                        keep.add(dependency)
                        pending.append(dependency)
                        report["protected_revisions"][dependency] = "dependency_of_retained_revision"
            runtime_paths, reviews = set(), []
            for revision_id in keep:
                row = rows[revision_id]
                directory = _revision_path(registry, row["relative_directory"], revision_id)
                _tree_state(registry.root, directory)
                manifest = retained_manifest(directory, json.loads((directory / "profile.json").read_text(encoding="utf-8")))
                runtime_paths.update(_runtime_paths(manifest.get("payload"), registry.root, directory))
                for logical_name, item in manifest.get("runtime_files", {}).items():
                    path = _checked_path(directory, directory / item["path"])
                    if logical_name.startswith("review_evidence_") and path.is_file():
                        info = path.stat()
                        reviews.append({"path": str(path), "kind": "review_file", "revision_id": revision_id,
                                        "relative_directory": row["relative_directory"], "logical_name": logical_name,
                                        "bytes": info.st_size, "newest_mtime_ns": info.st_mtime_ns,
                                        "sha256": item.get("sha256")})
                        continue
                    runtime_paths.add(path)
                    if path.suffix.lower() == ".json" and path.is_file():
                        runtime_paths.update(_runtime_paths(json.loads(path.read_text(encoding="utf-8")), registry.root, path.parent))
            for revision_id, row in rows.items():
                if revision_id not in keep:
                    directory = _revision_path(registry, row["relative_directory"], revision_id)
                    report["delete_revisions"].append({"revision_id": revision_id, "kind": row["kind"], "path": str(directory), **_tree_state(registry.root, directory)})
            report["retained_revision_count"] = len(keep)
            report["evidence"] = _evidence_plan(registry, policy, runtime_paths, reviews, time.time())
            if not dry_run:
                for item in report["delete_revisions"]:
                    _stage(registry, Path(item["path"]), "revision", item["revision_id"])
                for item in report["delete_revisions"]:
                    connection.execute("DELETE FROM revisions WHERE revision_id=?", (item["revision_id"],))
        if not dry_run:
            report["removed_revisions"] = [row["revision_id"] for row in report["delete_revisions"]]
            # Reacquire the publication lock after committing revision removal.
            # A newly published profile may reference evidence in this plan.
            with registry._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                current = {row[0] for row in connection.execute("SELECT revision_id FROM revisions")}
                evidence_items = report["evidence"]["delete"]
                if current != keep:
                    report["warnings"].append("Profiles changed during purge; evidence cleanup deferred until the next idle purge")
                    evidence_items = []
                for item in evidence_items:
                    try:
                        path = Path(item["path"])
                        if item["kind"] == "review_file":
                            _prune_review(registry, item)
                            report["removed_evidence"].append(item["path"])
                            continue
                        state = _tree_state(registry.root, path)
                        if any(state[key] != item[key] for key in state):
                            raise RuntimeError("Evidence changed during purge; left in place")
                        _stage(registry, path, "evidence")
                        report["removed_evidence"].append(item["path"])
                    except (OSError, ValueError, RuntimeError) as exc:
                        report["warnings"].append(str(exc))
    except Exception:
        if not dry_run:
            recover_pending_purge(registry)
        raise
    if not dry_run:
        report["warnings"].extend(recover_pending_purge(registry, delete_committed=True))
    report["warnings"].extend(report["evidence"]["warnings"])
    if not dry_run:
        removed = set(report["removed_evidence"])
        report["evidence"]["after_bytes"] = report["evidence"]["before_bytes"] - sum(item["bytes"] for item in report["evidence"]["delete"] if item["path"] in removed)
        report["evidence"]["over_limit_bytes"] = max(0, report["evidence"]["after_bytes"] - report["evidence"]["limit_bytes"])
        trash = registry.registry_directory / "purge-trash"
        report["pending_cleanup_bytes"] = _tree_state(registry.root, trash)["bytes"] if trash.exists() else 0
        if report["evidence"]["over_limit_bytes"]:
            report["warnings"].append("Evidence remains over budget because protected, recent, or unrecognized files were retained")
    report["status"] = "preview" if dry_run else "partial" if report["warnings"] else "complete"
    return report
