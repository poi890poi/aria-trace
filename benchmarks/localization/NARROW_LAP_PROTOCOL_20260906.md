# Run 20 narrow-lap evaluation protocol

Type: benchmark tooling and analysis; production tracking stays frozen at
`abc08d3`. Fresh holdout: session `ae32d060-6b92-4311-a59e-69427d3c667d`, run 20.
The user describes left-biased parallel traversals, clockwise endpoint circles,
distant scene-feature aiming, and repeated map-scale transitions. These are
behavioral annotations, not measured positions or gaze labels.

Freeze source files, atlas, calibration, runtime and reference implementation
identities. Build and retain a content-addressed 5 Hz slow atlas reference.
Compare current unknown-start tracking with current-image-verified known-start
tracking on the full recording at original timestamps. The latter receives only
the first reference state as an explicitly declared starting prior; neither
receives future positions or the full lap. Keep CPU timing runs sequential.

Report first acquisition, longest lost, fresh/held/unavailable output, incorrect
fresh measurements, jumps, XY error, and jointly fresh XY/heading delivered on
time. Retain reference gaps and transition intervals; slow atlas estimates are
not external ground truth. Derive diagnostic tolerances from earlier recordings,
not run 20. Inspect native scene and mini-map frames around endpoint turns and
scale changes. Use input timing as behavior evidence, never position truth.

Repeated traversals must not be scored as coincident paths: the deliberate lane
offset and small circles are signal to preserve. Separate lap direction and
progress in qualitative review. Camera yaw, avatar heading, movement direction,
and human attention are different quantities. Without independent yaw/position
truth or gaze labels, do not claim subpixel accuracy or identify exact attended
features from mouse motion alone. Proposed aiming methods remain experimental
until evaluated in paired open-loop and then live closed-loop trials.

## Follow-up after the frozen baseline

The baseline misses world-to-town switches: visual town correlation wins, but
the controller has never armed its tiny learned position zone. Wrong-layer XY
then moves outside even the observation corridor, stopping new mode evidence.
Test removal of spatial zone authority as one variable, keeping image margins,
confirmation count, search radii, and pose acceptance unchanged. This makes
representation observation global in location, not global in image search.
The estimator still sees only current images and its previous pose. Compare
against the verified-start control. If it restores transitions, test combination
with the previously isolated confirmed-switch search reset; retain jumps and
latency costs. Run 20 becomes development evidence for this new hypothesis, not
an untouched holdout for its subsequent tuning. Do not promote either variant
from this single-session experiment.

User clarification: transition learning belongs to the map, not a route. Keep
all algorithm and atlas changes experimental. Build a separate sparse map-owned
town/world support artifact from validated earlier references (11 and 14), score
its support coverage on run 20, then retain a separately named expanded artifact
including run 20. Do not claim that the expanded artifact has a held-out test.
Use observed player-center evidence and directed crossing brackets, not the
whole mini-map footprint or an assumed complete town polygon. Unknown support
must not veto visual mode evidence. No active atlas or production tracker is
replaced.

## Independent session transfer feature

Type: new feature. Add portable ZIP export/import for all or selected retained
sessions in Workbench. Include raw videos/images, timing, inputs, annotations,
and labels byte-for-byte; derived atlases and installed profiles are outside
this session bundle. Validate archive paths and checksums before publication.
Resolve destination collisions by allocating another run directory, without
changing embedded source identity. Reject active recordings and unsafe archives.
Use temporary staging and streamed file I/O to avoid loading videos into RAM.
Test round-trip, selection, collisions, corrupt/truncated/unsafe archives, and
HTTP wiring. Keep this feature in its own commit, separate from the recording
milestone and lap findings.
