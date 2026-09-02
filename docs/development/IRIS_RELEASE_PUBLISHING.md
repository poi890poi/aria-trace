# IRIS release publishing

Publishing is deliberately split into source, package, validation, and upload
stages. A failure in one stage cannot be mistaken for success in another.

```bat
publish-standalone-release.bat -Tag rig-system-working-rc3 -Title "IRIS release candidate 3" -Prerelease -SkipDependencyInstall
```

The command requires a clean tracked tree whose current branch exactly matches
its upstream. It builds the package, runs every packaged executable's offline
`--help` smoke test, verifies that `release-manifest.yaml` names the current Git
commit, and verifies both archive checksums before creating a tag or release.

The upload stage uses authenticated GitHub CLI. If GitHub authentication or an
upload fails, fix the external problem and rerun with `-SkipBuild`. The command
is resumable: an existing tag must point to the same commit, and an existing
release receives the verified assets with `--clobber`.

Archives are built under temporary names and replace the prior artifacts only
after compression succeeds. Release archives remain ignored by Git and are
uploaded as GitHub Release assets, never committed to the source tree.
