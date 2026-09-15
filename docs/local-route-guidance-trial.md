# Local route guidance trial

Type: opt-in product feature, following the continuous route-guidance contract.
The user selects a prepared route and drives. Scene recognition automatically
selects local route context; current minimap displacement and recent travel
supply a local course relation. The demonstrated paths provide an acceptable
aiming window. Actual mouse input supports current-course prediction and a
comparison between requested and current turning. No input executor is present.

This first package covers the recorded Session21 town circuit. It uses three
prepared demonstrations and the existing verified ALIKED model; no route neural
training. It is a manual trial, not validated autonomous navigation or a promise
of precise clearance. No full atlas localization is needed. Other route/rig
packages, scale transitions and new game configurations are not implicitly
supported. The first UI accepts the matching Windows game profile and window.

The live path separates slow scene recognition from fast current map/input
measurements. Both use replaceable latest frames. Unavailable context, stale
capture, contradictory map regions, insufficient movement, absent forward input,
or worker failure clears advice. A zero correction means current input fits the
observed route window, not proof of exact world position. The user can stop the
trial at any time; source and worker ownership must terminate with the workbench.

Evidence boundaries: sampled later Run21 replay, known map-image translations,
spatial-subset checks, and body calibration on Run20 support this prototype.
The later data were excluded from preparation but inspected previously. The
short centered map-motion reference used to assess delay compensation is an
approximate retrospective diagnostic; future query pixels never enter runtime.
The fitted 0.4 s regression lag includes observation-window averaging and is not
an identified actuator delay. Live game success remains a human-trial question.

Verification gate: preparation and controller parity with offline results;
causal input handling, fresh/expired advice, failed recognition and stop paths;
paced recorded-frame worker test; workbench start/stop and HUD rendering; focused
existing tracker/UI regression checks. Record capture-to-result age, fresh versus
unavailable output, longest unavailability, input and suggested turning separately.
No held value may be presented as a new measurement. Existing atlas tracker,
video settings and destination diagnostics retain their behavior.

## How to test

Restart the Workbench source process and refresh its page. No C++ build or atlas
rebuild is required. The feature runtime and prepared package are already present
in this workspace. Choose **Genshin Impact (PC POC)** and the game window, open
**Route tracer → Manual route guidance · trial**, select **Town circuit - session
21**, then **Start guidance**. Enable the normal overlay and return to the game.
Follow that town circuit with W and mouse turning, using the demonstrated minimap
scale and mouse sensitivity. The system acquires and changes route views itself.
It does not require selecting an observation target or starting at an exact XY.

The shaded band is the suggested mouse turning range; the white dot is measured
recent input. Follow the accompanying text: maintain, turn, or ease a turn.
When evidence expires, the band disappears. Stop guidance from Workbench.
Video is off by default; input and advice are always recorded. Saved trials offer
measurement ZIP and optional video downloads, including after Workbench restarts.
This package is prepared from the first three Run21 laps; a general route-preparation
UI is not part of this first trial.

## Layers and causal inputs

| Layer | Input | Output |
|---|---|---|
| Preparation | First three demonstrations, frames and input | Scene descriptors, local paths, mouse/body gain |
| Slow scene memory | Current cropped scene, prepared descriptors, previous context | Supported route view and age |
| Fast visual travel | Current and past masked minimap, teacher minimap | Local displacement, travel vector, support disagreement |
| Body state | Recent measured mouse input and calibrated gain | Compensation for the averaged motion observer's delay |
| Course guidance | Local paths from up to three demos and current travel | Suggested turning interval around a one-second lookahead |
| Human interface | Suggested interval, recent actual input, freshness | Optional input hint; no movement commands |

This is a small feedback design using a pretrained scene front end. Scene
identity supplies context, not a pixel residual to drive to zero. Physical local
map relations supply the travel geometry. The one-second goal advances along the
local path as the player moves; it is not tied to demonstration elapsed time.
Observed path variation and map support disagreement widen the input interval.
That interval is not a certified collision-free corridor.

## Evidence and decisions, 2026-09-15

Teacher data end at 124 s. Later laps had been inspected before this design;
they are excluded from preparation but are not a fresh untouched holdout.
Cached production scene retrieval exactly matched all 559 prototype decisions.
The combined controller provided 538/559 cached hints (longest unavailable 1.58 s).

| Paced native replay | Advice frames | Longest unavailable | Result age median / P95 / max |
|---|---:|---:|---:|
| Run21 124–166 s | 580 / 662 | 0.99 s | 15 / 99 / 341 ms |
| Run21 166–240 s | 1,136 / 1,213 | 1.87 s | 14 / 20 / 45 ms |
| Corner 158–166 s, video on | 69 / 74 | 0.29 s | 18 / 35 / 71 ms |

These ages end at worker output, before publication polling, HTTP and display.
The original recording supplies about 16 FPS across the two longer portions and
9.3 FPS in the corner excerpt. All source frames were processed. This does not
establish 30 FPS game capture or high-frequency live micro-correction.
There were 389 accepted scene refreshes and 1,327 accepted updates using reused
context with a new map measurement. No frozen hint is counted as a new result.
There were 25 correction-sign reversals across the two longer runs; this mixes
necessary steering reversals with possible jitter, and is not a smoothness score.

An initial video replay was invalid: the replay clock started before recorder
imports/initialization, creating up to 2.73 s of artificial capture age. Starting
the replay clock after initialization removed it. The corrected video test had
zero queue drops. Its 30 FPS file repeats 165 frames because the native source is
slower; video frame rate must not be mistaken for measurement rate.

| Candidate | Decision and confidence |
|---|---|
| Prepared scene context | Land in opt-in trial; high confidence in prototype parity, moderate in recognition coverage |
| Local map relation and advancing goal | Land in trial; known-shift P95 0.038 px, spatial regions agree; no independent world-accuracy claim |
| Recent-input compensation | Land in trial; two-session gains differ 2.6%; retrospective course P95 improves 17.96° → 10.21° on approximate map reference |
| Multiple-demo input window | Land in trial; permits demonstrated variation; clearance and human controllability remain unverified |
| Copied demonstration input | Reject as steering advice; it can turn too early and stop a turn too early |
| Ground-plane future-foot projection | Reject; too little preparation/transfer coverage |
| Autonomous control / full live reliability | Not established; manual trials must reveal response, recovery and oscillation behavior |

Verification: focused controller, package, stale-advice, raw-input failure,
subprocess stop/failure, Workbench route-only start/stop/download, existing tracker
and UI tests. Browser inspection verified route selection and controls. No live
game window was available for a human driving trial. One existing destination
test hit a temporary-file cleanup race on the first suite run; it and the full
suite passed on repeat, without changing that observer.

The final suite passed 60 tests in the working tree and 56 on the isolated
staged changes, without unrelated recording/UI edits. A real mixed-runtime check processed the
158–174 s excerpt through Python 3.12 GPU capture replay and the Python 3.7
Workbench publisher: 134/145 HUD polls showed advice, zero fresh worker hints
were incorrectly suppressed, and zero stale hints were displayed. This exposed
and corrected differing Windows performance-counter origins. Within-worker
latency remains monotonic; cross-process freshness uses epoch publication time
plus that measured latency. The original replay measurements above remain valid
as within-worker measurements.

Evidence: `artifacts/poc/local-route-control-20260915/` contains cached and paced
per-frame telemetry, native example images, input timeline, metrics and
`guidance-evidence.zip`. Package:
`artifacts/workbench/route_guides/genshin-impact-pc/town-circuit-run21/`.
