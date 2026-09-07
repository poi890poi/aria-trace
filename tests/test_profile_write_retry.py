import ctypes
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from rig_runtime.adapters.filesystem import atomic_write
from rig_runtime.adapters.filesystem.profile_registry import ProfileContext, ProfileRegistry
from rig_runtime.adapters.filesystem.system_configuration import save_system_configuration


def windows_error(code=5):
    error = PermissionError(13, "Injected Windows access error")
    error.winerror = code
    return error


class ProfileWriteRetryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.real_replace = os.replace

    def test_transient_access_and_sharing_errors_recover_without_manual_retry(self):
        for code in (5, 32, 33):
            with self.subTest(winerror=code):
                path = self.root / "active.json"
                path.write_text("old")
                attempts = []

                def replace(source, destination):
                    attempts.append(source)
                    self.assertEqual("old", path.read_text())
                    if len(attempts) < 3:
                        raise windows_error(code)
                    self.real_replace(source, destination)

                with patch.object(atomic_write.os, "replace", side_effect=replace), patch.object(atomic_write.time, "sleep") as sleep:
                    atomic_write.atomic_write_text(path, "new")
                self.assertEqual("new", path.read_text())
                self.assertEqual(2, sleep.call_count)
                self.assertEqual(3, len(set(attempts)))
                self.assertEqual([], list(self.root.glob("*.tmp")))

    def test_persistent_denial_is_bounded_and_preserves_original_file(self):
        path = self.root / "active.json"
        path.write_text("old")
        with patch.object(atomic_write.os, "replace", side_effect=windows_error()) as replace, patch.object(atomic_write.time, "sleep") as sleep:
            with self.assertRaises(PermissionError) as caught:
                atomic_write.atomic_write_text(path, "new")
        self.assertEqual(6, replace.call_count)
        self.assertEqual(5, sleep.call_count)
        self.assertEqual(5, caught.exception.winerror)
        self.assertIn("6 attempts", caught.exception.__notes__[0])
        self.assertEqual("old", path.read_text())
        self.assertEqual([], list(self.root.glob("*.tmp")))

    def test_other_filesystem_errors_are_not_retried(self):
        with patch.object(atomic_write.os, "replace", side_effect=OSError(28, "disk full")) as replace, patch.object(atomic_write.time, "sleep") as sleep:
            with self.assertRaisesRegex(OSError, "disk full"):
                atomic_write.atomic_write_text(self.root / "active.json", "new")
        self.assertEqual(1, replace.call_count)
        sleep.assert_not_called()

    def test_overlapping_writers_do_not_share_staging_files(self):
        barrier = threading.Barrier(2)
        writer = threading.local()
        observed = []

        def replace(source, destination):
            observed.append((source, Path(source).read_text()))
            if not getattr(writer, "joined", False):
                writer.joined = True
                barrier.wait(timeout=5)
            self.real_replace(source, destination)

        path = self.root / "active.json"
        with patch.object(atomic_write.os, "replace", side_effect=replace):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(atomic_write.atomic_write_text, path, value) for value in ("first", "second")]
                for future in futures:
                    future.result(timeout=10)
        self.assertGreaterEqual(len(observed), 2)
        self.assertEqual(len(observed), len({item[0] for item in observed}))
        self.assertEqual({"first", "second"}, {item[1] for item in observed})
        self.assertIn(path.read_text(), {"first", "second"})

    def test_publication_retries_manifest_directory_and_active_pointer_writes(self):
        registry = ProfileRegistry(self.root / "profiles")
        counts = {}

        def replace(source, destination):
            path = Path(destination)
            kind = "revision_directory" if path.parent.name == "revisions" else path.name
            counts[kind] = counts.get(kind, 0) + 1
            if counts[kind] <= 2:
                raise windows_error()
            self.real_replace(source, destination)

        with patch.object(atomic_write.os, "replace", side_effect=replace), patch.object(atomic_write.time, "sleep"):
            profile = registry.publish("game_model", ProfileContext(game_id="game-1"),
                                       {"value": "new"}, activate=True)
        self.assertEqual([profile["revision_id"]], registry.active_revision_ids())
        self.assertTrue({"profile.json", "profile.yaml", "revision_directory",
                         "active-profiles.json", "active.json", "active.yaml"}.issubset(counts))
        self.assertTrue(all(count == 3 for count in counts.values()))
        snapshot = json.loads((registry.root / "active-profiles.json").read_text())
        self.assertEqual(profile["revision_id"], snapshot["profiles"][0]["pointer"]["active_revision_id"])
        self.assertEqual("new", registry.revision(profile["revision_id"])["payload"]["value"])

    def test_failed_revision_rename_keeps_previous_active_revision(self):
        registry = ProfileRegistry(self.root / "profiles")
        context = ProfileContext(game_id="game-1")
        previous = registry.publish("game_model", context, {"value": "old"}, activate=True)
        snapshot = (registry.root / "active-profiles.json").read_bytes()

        def replace(source, destination):
            if Path(destination).parent.name == "revisions":
                raise windows_error()
            self.real_replace(source, destination)

        with patch.object(atomic_write.os, "replace", side_effect=replace), patch.object(atomic_write.time, "sleep"):
            with self.assertRaises(PermissionError):
                registry.publish("game_model", context, {"value": "new"}, activate=True)
        self.assertEqual([previous["revision_id"]], registry.active_revision_ids())
        self.assertEqual(1, len(registry.list_revisions()))
        self.assertEqual(snapshot, (registry.root / "active-profiles.json").read_bytes())

    def test_settings_json_and_yaml_both_retry(self):
        counts = {}

        def replace(source, destination):
            counts[destination] = counts.get(destination, 0) + 1
            if counts[destination] == 1:
                raise windows_error()
            self.real_replace(source, destination)

        with patch.object(atomic_write.os, "replace", side_effect=replace), patch.object(atomic_write.time, "sleep"):
            result = save_system_configuration({"game": {"id": "game-1"}}, self.root)
        self.assertEqual("game-1", result["game"]["id"])
        self.assertEqual(2, len(counts))
        self.assertEqual({2}, set(counts.values()))

    @unittest.skipUnless(os.name == "nt", "Exercises a real Windows sharing lock")
    def test_real_windows_destination_lock_recovers_after_handle_release(self):
        from ctypes import wintypes
        path = self.root / "active.json"
        path.write_text("old")
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        # Allow readers/writers, but deny deleting/replacing the open file.
        handle = kernel.CreateFileW(str(path), 0x80000000, 3, None, 3, 0, None)
        self.assertNotEqual(ctypes.c_void_p(-1).value, handle)
        released = False

        def release(_delay):
            nonlocal released
            self.assertEqual("old", path.read_text())
            self.assertTrue(kernel.CloseHandle(handle))
            released = True

        try:
            with patch.object(atomic_write.time, "sleep", side_effect=release) as sleep:
                atomic_write.atomic_write_text(path, "new")
            self.assertEqual(1, sleep.call_count)
            self.assertEqual("new", path.read_text())
        finally:
            if not released:
                kernel.CloseHandle(handle)
