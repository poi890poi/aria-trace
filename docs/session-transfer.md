# Transfer recorded sessions

In Workbench's Sessions list, check individual sessions and choose **Export
selected**, or choose **Export all finished**. Selection persists across search,
pagination, and catalog refreshes. All means all finished sessions in the
catalog, regardless of the current filter; an active recording is excluded.

Choose a ZIP using **Import sessions**. A preview lists every session in the
bundle. Check all or a subset, then choose **Import selected**. Cancel removes
the staged upload. The import result lists any renamed destination folders.
Restart Workbench once after updating to load this new feature.

The bundle retains every file inside each session, including videos, image
streams, original timestamps, keyboard/mouse inputs, annotations, and labels.
It does not include installed game/rig profiles, calibration artifacts, maps,
compiled routes, or benchmark reference caches. Install the relevant profiles
and transfer/rebuild those derived artifacts separately when needed.

Existing files are never overwritten. A colliding run folder receives the next
available number. Original manifests, run indices, session IDs, and embedded
provenance remain byte-for-byte unchanged, so an imported copy can show the same
source session number under a different folder. This preserves source identity;
importing twice intentionally creates another copy.

ZIP format `aria-trace-sessions`, version 1, contains `bundle.json` and
`sessions/<experiment>/run_<number>/<files>`. The index lists each file's size
and SHA-256. Import validates archive membership and portable paths before its
preview, then checks every selected file's checksum and all referenced frame
storage paths before publication. Failed validation or interrupted extraction
publishes no sessions. Each session is published by a directory rename; an
application error during publication rolls back the selected set. A process or
machine crash during publication can leave a subset of complete imported
sessions, which can be reviewed in the catalog. Publication uses the Workbench
catalog lock; large file I/O uses a
separate transfer lock and never holds the recorder status lock. This prevents
the transfer path from recreating the recorder/HUD contention defect.

Only complete session schema 1.0 recordings are supported. Unsafe paths, linked
files, encrypted archives, duplicate names, and unlisted entries are rejected.
Transfers stream to disk. Limits are 32 GiB compressed, 64 GiB expanded, 200,000
files, and a 32 MiB index. ZIP storage avoids recompressing already compressed
video. Allow disk space for the bundle and an extracted copy. Temporary transfers
expire after 24 hours and are removed when the next transfer begins; successful
imports and cancellations remove their staging immediately.

Verification: `python -m unittest tests.test_session_bundles
tests.test_workbench_ui tests.test_workbench -q`. Tests cover all/selected
round-trips including a second image stream, identity-preserving collisions,
active recording exclusion, corrupt payloads, unsafe embedded paths, interrupted
uploads, and the real HTTP download/upload/import endpoints.

September 6 result: all 75 tests passed. A real run 20 export/import into an
isolated temporary root also preserved all seven files by SHA-256, including its
197,662,586-byte video. Evidence:
`artifacts/poc/narrow-lap-20260906/session-transfer-roundtrip.json`.
Inline Workbench JavaScript passed Node's parser. Browser interaction with this
feature is not yet user-verified.
