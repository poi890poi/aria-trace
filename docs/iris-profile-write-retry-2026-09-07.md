# IRIS profile publication: transient Windows access failures

Type: bug fix. Baseline: `622417f`.

The user reported intermittent WinError 5 while updating profiles, resolved by
retrying the operation. Registry JSON, commented YAML, and immutable revision
directory publication used a single `os.replace` attempt. The shared `.tmp`
names in JSON/YAML writers also allowed overlapping writers to interfere.
System settings retried only the final JSON rename; their YAML companion did
not have the same protection.

The regression reproduces a Windows destination lock using `CreateFileW`
without delete sharing. The first atomic replacement fails; releasing the
handle during backoff lets the same high-level update complete. Injected
WinError 5, 32, and 33 cases separately cover access denied, sharing, and
byte-range locking. The user's particular lock holder is not identified.

The shared filesystem writer now retries those three Windows errors up to six
attempts, with 0.05, 0.10, 0.20, 0.40, and 0.80 second delays. Each text write
uses its own sibling temporary file. Revision-directory publication retries
the completed directory rename without rebuilding the profile or activating
it early. JSON and YAML profile/settings writers use the same policy.

Permanent denial stays bounded and preserves the original OS exception and
paths with an exhaustion note. Unrelated errors, such as disk full, fail
immediately. The code does not change ACLs, delete a destination to bypass a
lock, or introduce a non-atomic copy fallback. Existing registry transaction
and portable-mirror semantics remain unchanged: an exhausted mirror write
warns after live activation; it cannot roll back the authoritative SQLite
configuration. A failed revision publication leaves the prior active revision.

Scope: filesystem publication and focused regressions. Calibration algorithms,
distortion work, tracking, and unrelated workspace changes are excluded.

Tests: `tests/test_profile_write_retry.py` covers transient/persistent errors,
overlapping writers, real Windows sharing locks, all registry publication
destinations, and both system-setting formats. Broader regression results are
recorded in `artifacts/iris-profile-retry-tests.log`: **204 tests passed**, including
the Windows-only native file-lock regression. Profile registry/management,
system configuration, portable profiles, adapter export, rig reuse, the
recalibration contract, architecture, and rig calibration suites passed.
