# Scene-bearing replay experiments

Decision: retain masked sparse optical flow with explicit reacquisition as a
research component. Reject the claim that retaining initial features is enough
for route steering. No scene-bearing mechanism was added to production and no
game controls were issued.

## Inputs and evaluation

Run 20, original frame timestamps, 640 x 360 grayscale, requested sampling 10 Hz
(actual approximately 8–9 Hz after selecting existing source frames). The methods
start from the same 40 corners for each clip. LK uses forward/backward optical
flow; template tracking uses bidirectional local patch correlation. Reacquisition
checks original patches for a quarter of the feature IDs per update. No reference
position, mouse input, route position, future frame, or yaw label enters tracking.

The synthetic assay translates an actual game frame by a known smooth shift and
blacks it out from 4 through 5 seconds. It tests pixel correspondence and recovery,
not the relationship between mouse motion and yaw. Real clips have no independent
feature-identity or yaw truth. “Three features available” is a redundancy diagnostic,
not a validated control quality threshold.

## A misleading result caught by images

The first assay reported three or more features throughout the straight return.
Inspection of `scene-bearing/return-straight-lk-reacquire.png` showed several final
points on HUD icons. The initial upper-central seed box also included the player’s
head. Tracking checked image bounds and texture but not semantic eligibility on
subsequent frames. This let a plausible, persistent track migrate outside the
intended scene region. That result is preserved as a failed baseline.

The corrected assay excludes known HUD and avatar regions at initialization and
requires each accepted 17 x 17 patch to remain within the geometric mask on every
frame, including reacquisition. A regression exercises LK, template tracking,
and reacquisition at excluded and permitted positions. This is not semantic
segmentation: NPCs, nameplates, moving foliage, and other dynamic scene content
can still be selected. Nor does the mask prove that an accepted point retained
its identity.

## Corrected results

Longest interval with fewer than three original features, seconds:

| Clip, source seconds | LK | Template | LK + reacquisition |
|---|---:|---:|---:|
| Outbound straight, 23–33 | 0 | 0 | 0 |
| Return straight, 45–55 | 3.349 | 4.360 | 3.241 |
| Endpoint circle, 34–44 | 8.89 | 9.54 | 6.36 |
| Later return, 121–131 | 2.117 | 2.117 | 2.117 |

The later return is a separate clip with new seeds, not a cross-lap match to
demonstrated feature identities. No parameter was fitted on this later clip.
The masks were developed after viewing the first assay; this is development
evidence, not a fresh holdout.

On the known-shift game image, P95 correspondence error was 0.102 px for LK,
4.269 px for updated-patch template tracking, and 0.222 px for LK plus original-
patch reacquisition. All three emitted zero fresh features during the blackout.
Only reacquisition recovered after it: at the first sampled visible frame, 0.1 s
after the nominal blackout endpoint. It retained 33 original IDs at the end.
Its P95 update cost was 6.491 ms in this assay. Very small latency averages after
all tracks die are empty-work results, not useful speedups.

The random-texture regression also passed, but it did not expose the template
drift seen on real game texture. The actual-image synthetic result takes priority.
Real circles change viewpoint and remove initial landmarks from view; a synthetic
translation cannot validate those cases.

## Implications for imitating the player

1. Learn repeatable gate, post, wall, and walkway features across demonstrations,
   along with their desired screen bearings. Do not infer gaze from mouse events
   or assume the player always aims at screen center. Preserve the demonstrated
   lane offset.
2. Use optical flow between image-verified landmark observations. Reacquire by
   appearance and local geometric agreement; persistence or a high correlation
   score alone must not establish landmark identity.
3. Hand off to a new visible look-ahead landmark before the old one exits view.
   Blend desired bearings around the clockwise endpoint circles. Reacquiring a
   landmark behind the player cannot substitute for that handoff.
4. Evaluate the handoff first in replay: repeatability across laps, false matches,
   longest missing target, bearing jumps and delay. Then evaluate steering with
   independently measured lane/cross-track and heading evidence. This round did
   not test a steering controller or prove sustained automatic cruising.

Confidence is high in the diagnosed HUD leakage and the known-shift comparison;
moderate in optical flow as the better local tracker on these clips; low in
unmeasured yaw accuracy or end-to-end steering performance.

Evidence: `artifacts/poc/precision-round-20260906/scene-bearing` and
`scene-bearing-masked`, containing summaries, per-feature rows, initial/final
images, and source identities. The corrected directory includes the executed
implementation snapshot. The default unmasked mode remains available solely to
reproduce the rejected baseline.
