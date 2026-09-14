# Destination observer landing and fresh town laps

Decision: land optional observation, prepared-package import and live diagnostics.
Do not enable automatic steering or promote Session 21 target banks to reliable
route-control memory. They do not transfer adequately to later traversals.

## Product boundary and validation

Production receives a selected package plus present/past captured frames. ALIKED
features and local geometric fitting acquire a linked physical goal; bounded
optical flow updates it between recognition attempts. No test-time XY, lap label,
future images or benchmark import reaches the product. Bearing is image-relative,
not route displacement. The worker cannot issue game input.

Frozen Session 20 migration parity: 1,308 third-traversal frames, zero differences
in availability, result kind or coordinates. Outdoor accepts 475/808, town 151/500.
The exact match to an earlier experiment is a migration check, not ground truth.

Actual GPU worker, recorded cadence, six seconds per scene with a declared
0.6-second blackout:

| Scene/run | Processed/submitted | Fresh publications | Result latency median / P95 / max | Longest unavailable |
|---|---:|---:|---:|---:|
| Outdoor A | 170/189 | 151 | 5.46 / 59.16 / 317.01 ms | 0.716 s |
| Outdoor B | 179/189 | 160 | 4.95 / 51.99 / 287.62 ms | 0.716 s |
| Town A | 55/123 | 47 | 5.86 / 170.47 / 3178.51 ms | 3.255 s |
| Town B | 112/123 | 102 | 4.74 / 75.30 / 259.38 ms | 0.476 s |

Run A's town delay began 0.754 seconds into replay: feature extraction accounted
for 3129.49 ms of 3176.05 ms compute. It preceded the blackout. The delay did not
repeat on run B. Root cause within extraction/resource scheduling is unresolved;
do not claim it fixed by warmup or process isolation. Earlier pre-warmup results
are retained in `paced-before-warmup.json`. Pending frames are replaced, not
backlogged, and stale results do not supply a live target. All paced workers
and client threads exited. Submit P95 was below 0.76 ms across these runs.

Discovered platform bugs fixed before landing:

- The documented base Workbench PYTHONPATH points at Python 3.7 NumPy/OpenCV.
  Inheriting it in the Python 3.12 worker reproduced an immediate binary-import
  failure. Worker launch now sets a repository-only PYTHONPATH; the real feature
  interpreter loaded its own OpenCV successfully with the legacy parent setting.
- Windows private temporary-directory ACLs survived package publication. Import
  now uses a directory inheriting the artifact parent's normal permissions.
  The regression checks the published directory's DACL inheritance.
- Stop/failure races must not accept a late worker reply; acceptance rejects it.
  Overlay image and state are read under the same lock.

Workbench selection was checked in a separately owned local instance: Off is
default, both prepared destinations appear, and selection persists through page
polling. No game window was available to that check; no live-game acceptance or
combined map/GPU performance claim is made. Regression tests exercise lifecycle,
failure, restart, package validation, frame replacement, expiry and route-only
requests. See `docs/destination-observer.md` for operator instructions.

## Session 21: protocol and observations

Source: `sessions/workbench/recordings-genshin-impact-pc/run_21`, session UUID
`21b845fc-c522-4b78-9c68-55315f5f48d2`. There are 4,059 frames spanning 239.885 s.
Frame timestamps, not the 30 FPS container, are authoritative. Median interval
is 49.015 ms; P95 is 119.376 ms. This recording cannot establish a sustained
30 Hz feedback loop. Raw mouse input is retained and used only for segment review.

The route repeats through town at roughly 41-second intervals. Strong final
turn bursts occur around 38, 79, 120, 162 and 204 s. The narrow stair approach
and subsequent corner are visible in corresponding laps. The market passage
adds changing near obstruction and viewpoint; all are identified by native-frame
comparison, without treating differences between demonstrations as errors.

First, frozen Session 20 bridge/town packages tested 240 one-second samples each:
zero accepted gate observations in this different area. This is sampled negative
evidence only, not an exhaustive false-positive test.

Next, before evaluating later frames, four physical landmark groups were seeded
manually on the first lap: market canopy, building above the narrow stairs,
corner lamp, and facade after the turn (internal ID `after_turn_arch`). Fifteen
views were prepared using only first-lap images and short flow transfers. No
network training occurred. Each seed/ROI is recorded in `destination_run21.py`;
preparation and query feature caches are retained for reuse. Later native frames
41–239.885 s were evaluated causally, without threshold tuning or teacher updates.

| Landmark | Next-lap accepted / span first-to-last | Later result |
|---|---|---|
| Market canopy | 68 observations / 4.297 s | 28 on following lap, then none |
| Stair-top building | 30 / 1.391 s | No later acquisition |
| Corner lamp | 32 / 1.726 s | 23 on following lap, then none |
| Post-turn facade | brief acquisitions around each turn | 8 at 206.22–206.64 s; coverage shrinks |

Spans include internal gaps and are not successful-duration scores. Sparse
native-frame review finds the canopy still visible at 63.04 s after its last
measurement at 61.926 s; the stair-top neighborhood at 75.03 s after loss at
73.366 s; and the lamp at 79.01 s after loss at 78.855 s. The accompanying target
lifetime chart brackets exits and reserves the facade's lane through the window
because its final appearance is unverified. It uses each target's own bearing
range and reuses the row only after reviewed absence. It does not certify the
identity of every member feature.

Lighting shifts markedly toward daylight while camera/position also vary. The
current evidence shows a transfer failure, but does not isolate lighting as its
cause. Large frame-to-frame bearing motion during turns is not itself mismatch.
The first-lap memory is insufficient for dependable control of this new route.

## Design consequence

Keep route-stage memory, scene observation and motor control separate. A visible
destination can provide an aiming measurement; it does not need a mandatory
near/far pair. During a tight turn, neither these landmarks nor the old gate
packages provide continuous coverage. A bounded demonstrated turn with feedback
from accumulated rotation and a newly acquired outgoing landmark remains the
candidate design. Automatic handoff and travel alignment are not implemented.

The next useful algorithm comparison is improving first-lap-to-daylight/viewpoint
transfer, with stage timing unchanged and appearance review independent of the
estimator. Merely raising input frequency will not recover missing observations.
The optional product observer can now gather the missing live evidence without
silently turning these proposals into movement commands.

## Artifacts

- Product parity/timing: `artifacts/poc/destination-product-20260914/`.
- Fresh data cache, preparation, per-frame rows and summaries:
  `artifacts/poc/destination-run21-20260914/`.
- `lap-comparison.jpg`: first-lap regions versus later native scenes.
- `availability-bearing.png`: full subsequent-session observation availability.
- `target-lifetime.png`, `reviewed-episodes.json`, `visibility-review.jpg`:
  reviewed second-lap episode evidence and explicit exit brackets.
- `destination-landing-evidence.zip`: portable images, raw observation rows,
  metadata and report. Raw video and model weights are omitted.
