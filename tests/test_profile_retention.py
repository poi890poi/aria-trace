import io
import json
import os
import shutil
import tempfile
import time
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from unittest import mock

from rig_runtime.adapters.filesystem.profile_registry import ProfileContext, ProfileRegistry
from rig_runtime.adapters.filesystem import profile_retention as retention
from rig_runtime.adapters.filesystem.profile_retention import RetentionPolicy, purge_profiles
from rig_runtime.workflows.profile_management import main


class ProfileRetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.registry = ProfileRegistry(self.base / "profiles")
        self.policy = RetentionPolicy(keep_portable=2, keep_rig=1, evidence_max_mb=0, evidence_min_age_hours=0)

    def context(self, game="game-1"):
        return ProfileContext(camera_id="CAM", phone_id="PHONE", game_id=game,
                              panel_display={"natural_panel_px": [320, 640]},
                              game_display={"logical_frame_px": [640, 320]})

    def publish(self, kind, index, game="game-1", **kwargs):
        return self.registry.publish(kind, self.context(game), {"value": index}, **kwargs)

    def bundle(self, name, size=100, age_hours=48, marker="failure.json"):
        directory = self.registry.root / "calibrations" / name
        directory.mkdir(parents=True)
        (directory / marker).write_text("{}", encoding="utf-8")
        (directory / "review.png").write_bytes(b"x" * size)
        old = time.time() - age_hours * 3600
        for path in [*directory.iterdir(), directory]:
            os.utime(path, (old, old))
        return directory

    def test_recent_counts_are_per_profile_and_active_older_revision_survives(self):
        first = self.publish("game_model", 0, activate=True)
        for index in range(1, 5):
            self.publish("game_model", index)
        for index in range(4):
            self.publish("game_model", index, game="game-2")
        plan = purge_profiles(self.registry, self.policy)
        self.assertEqual(4, len(plan["delete_revisions"]))
        self.assertEqual(9, len(self.registry.list_revisions()))
        result = purge_profiles(self.registry, self.policy, dry_run=False)
        self.assertEqual(4, len(result["removed_revisions"]))
        self.assertEqual(5, len(self.registry.list_revisions()))
        self.assertEqual([first["revision_id"]], self.registry.active_revision_ids())
        self.assertTrue(Path(first["revision_directory"]).exists())

    def test_dependency_closure_protects_old_rig_and_portable_source(self):
        rig = self.publish("rig", 0)
        portable = self.publish("phone_game", 0, dependencies={"rig": rig["revision_id"]})
        self.publish("rig_game", 0, dependencies={"phone_game": portable["revision_id"]}, activate=True)
        for index in range(1, 5):
            self.publish("rig", index)
            self.publish("phone_game", index)
        result = purge_profiles(self.registry, self.policy, dry_run=False)
        for profile in (rig, portable):
            self.assertEqual("dependency_of_retained_revision", result["protected_revisions"][profile["revision_id"]])
            self.assertTrue(Path(profile["revision_directory"]).exists())

    def test_color_keeps_portable_count_despite_rig_kind_name(self):
        for kind in ("rig", "rig_game", "rig_game_orientation", "rig_game_color", "phone_game_color"):
            for index in range(4):
                self.publish(kind, index)
        purge_profiles(self.registry, self.policy, dry_run=False)
        for kind, expected in (("rig", 1), ("rig_game", 1), ("rig_game_orientation", 1), ("rig_game_color", 2), ("phone_game_color", 2)):
            self.assertEqual(expected, len(self.registry.list_revisions(kind=kind)))

    def test_evidence_oldest_first_and_quota_does_not_touch_outside_root(self):
        old = self.bundle("old", age_hours=72)
        new = self.bundle("new", age_hours=48)
        external = self.base / "external-evidence"
        external.mkdir()
        (external / "failure.json").write_text("{}")
        policy = RetentionPolicy(evidence_max_mb=110 / 1_000_000)
        preview = purge_profiles(self.registry, policy)
        self.assertEqual([str(old)], [row["path"] for row in preview["evidence"]["delete"]])
        self.assertTrue(old.exists())
        result = purge_profiles(self.registry, policy, dry_run=False)
        self.assertFalse(old.exists())
        self.assertTrue(new.exists())
        self.assertTrue(external.exists())
        self.assertEqual(102, result["evidence"]["after_bytes"])

    def test_recent_and_unrecognized_evidence_report_unmet_budget(self):
        recent = self.bundle("recent", age_hours=0)
        unknown = self.bundle("unknown", marker="user-note.txt")
        result = purge_profiles(self.registry, RetentionPolicy(evidence_max_mb=0), dry_run=False)
        self.assertEqual("partial", result["status"])
        self.assertGreater(result["evidence"]["over_limit_bytes"], 0)
        self.assertTrue(recent.exists())
        self.assertTrue(unknown.exists())

    def test_minimap_bundle_is_removed_as_one_unit(self):
        bundle = self.bundle("minimap-session", marker="calibration_summary.json")
        child = bundle / "android"
        child.mkdir()
        (child / "minimap_calibration.json").write_text("{}")
        result = purge_profiles(self.registry, self.policy, dry_run=False)
        self.assertEqual([str(bundle)], result["removed_evidence"])
        self.assertFalse(bundle.exists())

    def test_runtime_reference_protects_external_evidence_bundle(self):
        bundle = self.bundle("required")
        self.registry.publish("game_model", self.context(), {"required_reference": str(bundle / "review.png")}, activate=True)
        result = purge_profiles(self.registry, self.policy, dry_run=False)
        self.assertTrue(bundle.exists())
        self.assertEqual("referenced_by_retained_runtime", result["evidence"]["protected"][0]["reason"])

    def test_provenance_alone_does_not_pin_duplicate_evidence(self):
        bundle = self.bundle("provenance-only")
        self.publish("game_model", 1, provenance={"source": str(bundle / "failure.json")}, activate=True)
        purge_profiles(self.registry, self.policy, dry_run=False)
        self.assertFalse(bundle.exists())

    def test_optional_review_is_pruned_without_changing_runtime_or_manifest(self):
        review = self.base / "review.png"
        review.write_bytes(b"review" * 100)
        model = self.base / "model.npz"
        model.write_bytes(b"mandatory-runtime-model")
        profile = self.publish("phone_game", 0, runtime_files={"review_evidence_000": review, "minimap_model": model}, activate=True)
        directory = Path(profile["revision_directory"])
        manifest_before = (directory / "profile.json").read_bytes()
        result = purge_profiles(self.registry, self.policy, dry_run=False)
        self.assertEqual("complete", result["status"])
        loaded = self.registry.revision(profile["revision_id"])
        self.assertNotIn("review_evidence_000", loaded["runtime_files"])
        self.assertEqual("review_evidence_pruned", loaded["evidence_retention"]["status"])
        self.assertEqual(b"mandatory-runtime-model", self.registry.runtime_file(loaded, "minimap_model").read_bytes())
        self.assertEqual(manifest_before, (directory / "profile.json").read_bytes())
        self.assertEqual(profile["content_sha256"], loaded["content_sha256"])
        from rig_runtime.workflows.portable_profiles import export_portable_profile, import_portable_profile
        export_portable_profile(profile["revision_id"], self.base / "portable.zip", registry=self.registry)
        self.assertTrue((self.base / "portable.zip").exists())
        imported = import_portable_profile(
            self.base / "portable.zip", registry=ProfileRegistry(self.base / "imported"),
            compose_local_rig=False,
        )["portable_profile"]
        self.assertIn("minimap_model", imported["runtime_files"])
        self.assertNotIn("review_evidence_000", imported["runtime_files"])

    def test_registry_rebuild_does_not_resurrect_purged_revisions(self):
        for index in range(5):
            self.publish("game_model", index, activate=index == 4)
        purge_profiles(self.registry, self.policy, dry_run=False)
        retained = {row["revision_id"] for row in self.registry.list_revisions()}
        shutil.rmtree(self.registry.registry_directory)
        restored = ProfileRegistry(self.registry.root)
        self.assertEqual(retained, {row["revision_id"] for row in restored.list_revisions()})
        self.assertEqual(1, len(restored.active_revision_ids()))

    def test_failed_rename_rolls_back_all_revision_removals(self):
        profiles = [self.publish("game_model", index) for index in range(5)]
        original = retention.replace_with_retry
        calls = []
        def fail_second(source, target):
            calls.append(source)
            if len(calls) == 2:
                raise PermissionError("locked revision")
            return original(source, target)
        with mock.patch.object(retention, "replace_with_retry", side_effect=fail_second):
            with self.assertRaises(PermissionError):
                purge_profiles(self.registry, self.policy, dry_run=False)
        self.assertEqual(5, len(self.registry.list_revisions()))
        self.assertTrue(all(Path(item["revision_directory"]).exists() for item in profiles))

    def test_stale_portable_pointer_survives_purge_and_database_rebuild(self):
        first = self.publish("game_model", 0, activate=True)
        profiles = [self.publish("game_model", index) for index in range(1, 5)]
        with mock.patch("rig_runtime.adapters.filesystem.profile_registry._atomic_json", side_effect=PermissionError("locked mirror")):
            with self.assertWarns(RuntimeWarning):
                self.registry.activate(profiles[-1]["revision_id"])
        result = purge_profiles(self.registry, self.policy, dry_run=False)
        self.assertEqual("portable_activation_pointer", result["protected_revisions"][first["revision_id"]])
        self.assertTrue(Path(first["revision_directory"]).exists())
        shutil.rmtree(self.registry.registry_directory)
        restored = ProfileRegistry(self.registry.root)
        self.assertIn(first["revision_id"], restored.active_revision_ids())
        self.assertIsNotNone(restored.revision(profiles[-1]["revision_id"]))

    def test_new_publication_between_phases_defers_evidence_deletion(self):
        bundle = self.bundle("new-reference")
        original = self.registry._connect
        calls = []

        @contextmanager
        def connect():
            calls.append(1)
            if len(calls) == 2:
                with mock.patch.object(self.registry, "_connect", original):
                    self.registry.publish("game_model", self.context(), {"reference": str(bundle / "review.png")})
            with original() as connection:
                yield connection

        with mock.patch.object(self.registry, "_connect", connect):
            result = purge_profiles(self.registry, self.policy, dry_run=False)
        self.assertTrue(bundle.exists())
        self.assertEqual("partial", result["status"])
        self.assertTrue(any("Profiles changed" in message for message in result["warnings"]))

    def test_optional_unlink_failure_keeps_evidence_visible_until_retry(self):
        review = self.base / "review.png"
        review.write_bytes(b"review")
        profile = self.publish("phone_game", 0, runtime_files={"review_evidence_000": review})
        target = self.registry.runtime_file(profile, "review_evidence_000")
        original = Path.unlink

        def unlink(path, *args, **kwargs):
            if path == target:
                raise PermissionError("locked review")
            return original(path, *args, **kwargs)

        with mock.patch.object(Path, "unlink", unlink):
            result = purge_profiles(self.registry, self.policy, dry_run=False)
        self.assertEqual("partial", result["status"])
        self.assertIn("review_evidence_000", self.registry.revision(profile["revision_id"])["runtime_files"])
        purge_profiles(self.registry, self.policy, dry_run=False)
        self.assertNotIn("review_evidence_000", self.registry.revision(profile["revision_id"])["runtime_files"])

    def test_windows_reparse_point_is_refused_before_deletion(self):
        bundle = self.bundle("unsafe")
        target = bundle / "review.png"
        original = Path.lstat

        def lstat(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path == target:
                return type("ReparseStat", (), {"st_mode": result.st_mode, "st_file_attributes": 0x400})()
            return result

        with mock.patch.object(Path, "lstat", lstat):
            result = purge_profiles(self.registry, self.policy, dry_run=False)
        self.assertEqual("partial", result["status"])
        self.assertTrue(target.exists())
        self.assertTrue(any("reparse" in message for message in result["warnings"]))

    def test_crash_before_database_commit_restores_moved_revision_on_open(self):
        profiles = [self.publish("game_model", index) for index in range(5)]
        original = retention._stage
        calls = []
        def crash_after_first(*args, **kwargs):
            original(*args, **kwargs)
            calls.append(1)
            if len(calls) == 1:
                raise SystemExit("simulated crash")
        with mock.patch.object(retention, "_stage", side_effect=crash_after_first):
            with self.assertRaises(SystemExit):
                purge_profiles(self.registry, self.policy, dry_run=False)
        reopened = ProfileRegistry(self.registry.root)
        self.assertEqual(5, len(reopened.list_revisions()))
        self.assertTrue(all(Path(item["revision_directory"]).exists() for item in profiles))

    def test_locked_committed_trash_can_be_retried_without_resurrection(self):
        for index in range(5):
            self.publish("game_model", index)
        with mock.patch.object(retention, "_remove_tree", side_effect=PermissionError("locked trash")):
            result = purge_profiles(self.registry, self.policy, dry_run=False)
        self.assertEqual("partial", result["status"])
        self.assertGreater(result["pending_cleanup_bytes"], 0)
        reopened = ProfileRegistry(self.registry.root)
        self.assertEqual(2, len(reopened.list_revisions()))
        result = purge_profiles(reopened, self.policy, dry_run=False)
        self.assertEqual(0, result["pending_cleanup_bytes"])

    def test_escape_path_is_refused_before_any_deletion(self):
        profiles = [self.publish("game_model", index) for index in range(3)]
        with self.registry._connect() as connection:
            connection.execute("UPDATE revisions SET relative_directory=? WHERE revision_id=?", ("../outside", profiles[0]["revision_id"]))
        with self.assertRaisesRegex(ValueError, "managed root"):
            purge_profiles(self.registry, self.policy, dry_run=False)
        self.assertTrue(all(Path(item["revision_directory"]).exists() for item in profiles))

    def test_cli_defaults_to_preview_then_applies_explicitly(self):
        for index in range(5):
            self.publish("game_model", index)
        args = ["--profile-root", str(self.registry.root), "purge", "--keep-portable", "2", "--keep-rig", "1"]
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(0, main(args))
        self.assertTrue(json.loads(output.getvalue())["dry_run"])
        self.assertEqual(5, len(self.registry.list_revisions()))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(0, main(args + ["--apply"]))
        self.assertEqual(2, len(self.registry.list_revisions()))

    def test_policy_rejects_invalid_values(self):
        self.assertEqual(64, RetentionPolicy().evidence_max_mb)
        for kwargs in ({"keep_portable": 0}, {"keep_rig": 11}, {"evidence_max_mb": -1}, {"evidence_max_mb": float("nan")}, {"evidence_min_age_hours": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RetentionPolicy(**kwargs)


if __name__ == "__main__":
    unittest.main()
