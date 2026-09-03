import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StandaloneReleaseExportTests(unittest.TestCase):
    def test_build_rejects_git_metadata_and_records_exported_tree_hash(self):
        script = (ROOT / "build-standalone-release.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("Get-PortableTreeSha256", script)
        self.assertIn("exported_python_source_tree_sha256", script)
        self.assertIn("git_metadata_included: false", script)
        self.assertIn("Release export contains forbidden Git metadata", script)

    def test_publish_recomputes_exported_tree_hash_and_rejects_git_metadata(self):
        script = (ROOT / "publish-standalone-release.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("Get-PortableTreeSha256", script)
        self.assertIn("Exported Python source-tree hash mismatch", script)
        self.assertIn("Release package contains forbidden Git metadata", script)

    def test_release_hashing_does_not_depend_on_get_file_hash(self):
        build = (ROOT / "build-standalone-release.ps1").read_text(
            encoding="utf-8"
        )
        publish = (ROOT / "publish-standalone-release.ps1").read_text(
            encoding="utf-8"
        )
        for name, script in (("build", build), ("publish", publish)):
            with self.subTest(script=name):
                self.assertIn("[System.Security.Cryptography.SHA256]::Create()", script)
                self.assertNotIn("Get-FileHash", script)
        self.assertIn("source_commit: '$SourceCommit'", build)

    def test_release_validation_is_before_any_remote_or_tag_mutation(self):
        script = (ROOT / "publish-standalone-release.ps1").read_text(
            encoding="utf-8"
        )
        validation = script.index('Invoke-Checked "Validate package identity')
        validate_only = script.index("if ($ValidateOnly)")
        github = script.index("Get-Command gh")
        tag = script.index('Invoke-Checked "Create release tag')
        upload = script.index('Invoke-Checked "Resume release upload')
        self.assertLess(validation, validate_only)
        self.assertLess(validate_only, github)
        self.assertLess(github, tag)
        self.assertLess(tag, upload)
        self.assertIn("--clobber", script)

    def test_manifest_commit_parsing_is_literal_and_quote_tolerant(self):
        script = (ROOT / "publish-standalone-release.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn('$_ -match "^source_commit\\s*:"', script)
        self.assertIn('($CommitLine -split ":", 2)[1]', script)
        self.assertIn('.Trim("\'", \'"\')', script)
        self.assertNotIn("regex]::Escape($Head)", script)

    def test_phone_target_signer_and_both_release_locations_are_in_manifest(self):
        script = (ROOT / "build-standalone-release.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn("$PhoneTargetSignerSha256", script)
        self.assertIn("-ExpectedSignerSha256 $PhoneTargetSignerSha256", script)
        self.assertIn(
            "native_phone_target: phone-target/iris-phone-target.apk", script
        )
        self.assertIn(
            "python_native_phone_target: python/android/phone-target/iris-phone-target.apk",
            script,
        )
        self.assertIn("native_phone_target_signer_sha256", script)

    def test_release_exports_only_the_neutral_runtime_package(self):
        script = (ROOT / "build-standalone-release.ps1").read_text(
            encoding="utf-8"
        )
        self.assertIn('Join-Path $ProjectRoot "rig_runtime"', script)
        self.assertNotIn("$IrisExcludedSourcePaths", script)
        self.assertNotIn('Join-Path $ProjectRoot "aria_trace"', script)

    def test_iris_entrypoints_do_not_import_trace_product(self):
        entrypoints = [
            ROOT / "iris_tools.py",
            ROOT / "hikcam.py",
            *sorted((ROOT / "packaging" / "windows" / "entrypoints").glob("*.py")),
        ]
        violations = []
        for path in entrypoints:
            if "aria_trace" in path.read_text(encoding="utf-8"):
                violations.append(str(path.relative_to(ROOT)))
        self.assertEqual([], violations)


if __name__ == "__main__":
    unittest.main()
