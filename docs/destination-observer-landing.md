# Destination observer: first product slice

Type: optional feature, plus reusable data packaging and diagnostics.

Before this change, destination groups were accessible only through offline
benchmarks. This slice exposes a selected prepared destination during live route
tracking, with a scene overlay and persistent measurements. It is an observer:
it issues no movement commands and does not alter map pose, route progress or
finish decisions. No left/right travel instruction is inferred from camera bearing.

The owning boundary is `aria_trace.services.tracking.destination` and its worker.
Workbench owns selection and lifecycle. Prepared packages contain game identity,
image geometry, masking, a model identity and linked physical-goal views. Runtime
receives only the selected package and current/past images. No query XY, recorded
future frames, or benchmark imports enter production code.

The existing workbench uses Python 3.7; the optional feature runtime uses a separate
Python process. Only one request is in flight and one latest frame is pending.
Slow extraction/recovery replaces pending frames instead of backlogging capture.
Failure, stop and restart must terminate the worker and make unavailable state
explicit. Results older than 0.5 seconds cannot be presented as current tracking.
Long gaps invalidate optical flow. GPU work can still compete with the game;
separate-process isolation is not a promise of zero resource impact.

Verification: compare production observer output against frozen replay output;
exercise missing dependencies, bad packages, blank scenes, stale output, slow
worker replacement, worker failure, shutdown, and two consecutive live-style
capture runs. Validate workbench selection and existing tracker regressions.
No unattended game-control claim or production navigation policy is included.

Initial usable packages are imported from the tested outdoor/town view banks.
Import is explicit; runtime never discovers benchmark files or chooses a target
from hardcoded scene names. New-route visual annotation/preparation in the UI is
a subsequent slice. This boundary must be visible in the testing instructions.

Final verification: 48 tests passed in a clean HEAD snapshot with only this
product slice applied. The full working tree also passed 51 tests before the
independent UI request test was added, and its 10 focused destination tests
passed afterward. The clean check deliberately excludes unrelated video-control
edits. Frozen observer parity and paced GPU evidence are summarized in
`benchmarks/localization/DESTINATION_LANDING_RESULTS_20260914.md`. Browser UI
selection and the real worker's interpreter isolation were checked separately.
Live game capture plus simultaneous map/GPU processing remains unverified.
