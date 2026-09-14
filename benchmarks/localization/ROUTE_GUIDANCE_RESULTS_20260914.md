# Continuous route guidance: recognition improves, steering proposals fail

## Decision and implementation status

The route-wide memory candidate is worth retaining. **Do not land either copied
demonstration input or the tested ground-plane point transfer as live steering.**
The ILS-style manual guidance milestone is not implemented by this round.
Existing live tracking and the optional single-target diagnostic are unchanged.
No game commands were issued. The experimental code, reusable prepared memory,
raw results, and native-image report are implemented and cached locally.

This is a control-design finding: recognizing a route stage does not determine
the correction needed from the player's current path. An improved feature
extractor or higher match count alone does not fill that missing relation.

## Inputs and restrictions

Run21, session `21b845fc-c522-4b78-9c68-55315f5f48d2`: three teaching traversals
[0,41), [41,82), [82,124) seconds; 157 exemplars at approximately 0.75-second
spacing. There are 559 later cached query frames from 124 through 239.885 s.
These later laps were excluded from this memory but inspected in earlier work;
they are not an untouched holdout. Traversal boundaries are preparation
annotations, not an automatically learned segmentation result.

Runtime recognition receives the current image, prepared features and past
recognition state. It receives no query future image, future input or precise XY.
ALIKED-N16rot features were cached by the previous GPU experiment. Homography
verification supplies scene correspondence only, never a yaw/lateral error.
Normalized teaching time is an approximate ordering aid, not metric route progress.

## Recognition comparison

All four arms use the same features, mutual ratio matches, geometric gates and
top-eight candidates. The learned summary uses 32 descriptor centres and a
4,096-value residual aggregate (VLAD-style). Centre/signature preparation took
0.504 s using already extracted features; this excludes video decode, feature
extraction and neural model loading. No network was trained on the route.

| Ranking and search | Fresh / 559 | Unavailable | Longest unavailable | Recognition compute median / P95 |
|---|---:|---:|---:|---:|
| Gradient summary, global | 220 | 339 | 10.02 s | 48.94 / 75.66 ms |
| Gradient summary, local | 343 | 216 | 9.75 s | 51.19 / 82.78 ms |
| Learned summary, global | 457 | 102 | 2.82 s | 61.73 / 86.12 ms |
| Learned summary, local | 476 | 83 | 2.48 s | 59.84 / 94.49 ms |

There are no held outputs. The local learned-summary arm has 37 unavailable
episodes. Durations run from the first unavailable sample to reacquisition;
terminal episodes are censored. These are scene-recognition availability
measurements, not independently verified physical-target loss or pose accuracy.
Raw rows and `summary.json` retain mean, median, P95 and maximum timings.
No live capture-to-display latency or high-frequency control claim is supported.

The same route memory rejected all 39 cached outdoor Run20 queries from
178–208 s. This is a small unrelated-route control, not a false-acquisition rate
for similar town scenes. Native paired images support the reviewed retrievals;
the remaining accepted frames have not been individually identity-labelled.

Confidence: **moderate for a useful retrieval improvement on these town laps**;
insufficient for production accuracy or live responsiveness. Both global and
local arms improve with the new representation, so ordering alone does not
explain the result. Cost increases rather than producing a speedup.

## Why recalled input is not the guidance signal

An exemplar initially stores the teacher's next 0.6 s horizontal mouse input.
Requiring agreement from two teaching laps supplies 195/559 action priors.
Allowing one strong exemplar while rejecting comparable conflicting matches
supplies 431/559: 19 left, 302 no-turn, 110 right. There are 66 unavailable
episodes and 32 action changes between adjacent supported samples.

This larger count is not a control-quality improvement. At 160.44 s the memory
retrieves a right-turn prior while the player has only just made a left camera
adjustment. At 162.98 s it retrieves a no-turn prior while the recorded player
continues rounding the corner: the following half-second bins contain +618 and
+554 mouse counts/s. The native views and input trace document a timing/context
mismatch; they do not establish that recorded input is uniquely correct.

Neither proposal measures how the suggested action changes current travel
alignment. Showing its no-turn state as a centred ILS needle would invent
information. **Reject direct action retrieval as corrective guidance.** Retain
demonstrated direction/rate only as a bounded maneuver prior, subordinate to
independent progress/exit feedback. That integrated turn controller remains
unimplemented and unvalidated in this round.

## Physical path-point alternative: negative first screen

The candidate defines a point one second ahead on the demonstrated path using
an approximate avatar-foot anchor (320,290), with a +/-15 px vertical interval,
at 640x360. It transfers that point backward through a provisional floor plane
and then into the later-lap image. It needs no world XY or query future frames.
It is a general geometric candidate, with no gate/banner-specific rule.

The lower trapezoid is only a sensor region, not verified floor segmentation.
The foot anchor is provisional and can vary with avatar animation and camera
configuration. Stair geometry and nearby walls can also invalidate the plane.
These limitations prevent treating accepted reticles as accurate path points.

| Preparation method; same later-lap matching | Prepared / 108 | Later transfer / 187 |
|---|---:|---:|
| One-second learned feature match | 4 | 1 |
| Short native-frame flow, accumulated over one second | 41 | 1 |

The second method changes preparation only. Its later-query failures comprise:
130 without a prepared point, 29 with fewer than ten floor matches, one with
weak plane support, and 26 with the point outside the supporting inlier hull.
The latter extrapolations range from about 6 to 127 px beyond that hull. Removing
the gate would supply unsupported extrapolations, not independent path evidence.
The accepted sample differs between the two methods; neither supplies continuous
guidance. **Reject these implementations for live guidance.**

392 additional ground feature extractions were cached. GPU extraction took
70.84 ms median, 129.73 ms P95 and 2,839.26 ms maximum, including the first call.
The previous stall concern therefore remains; no root cause is claimed here.
An asynchronous slow recognition path is still necessary for a responsive loop,
but does not resolve incorrect or missing guidance information.

## Design consequence and next engineering boundary

Keep the complete loop: prepared route relations -> automatic observation ->
current travel relation -> bounded input hint -> actual human input -> visual
response. Route recognition is the memory/attention layer. It cannot also
serve as the control-error measurement merely by returning a matched frame.

The unresolved boundary is automatic preparation and observation of a useful
travel corridor/destination relation, including turn exits. Existing manually
seeded persistent destination groups remain the strongest observation evidence;
they are not yet a complete learned route. Another diagnostic target selector
or a prerecorded-input arrow would not complete the requested product.

Before integrating the live ILS display, that relation must survive independent
native-image review and controlled path/heading variations. Steering response
must then be checked using actual input and subsequent observations. This round
does not establish exact position, travel alignment, passage success, or reliable
closed-loop control, and does not claim that the full design is conclusive.

## Reuse and evidence

`artifacts/poc/route-guidance-20260914` retains the prepared bank, learned summaries,
feature caches, all four recognition arms, action proposals, both physical-point
arms, timestamps, input/model provenance and images. Existing source session and
cached frames are reused. These caches accelerate reruns; they are not precise
inferred ground truth. Source hashes are recorded in `provenance.json`.

`route_guidance_trial.py`, `route_ground_goal.py` and
`route_guidance_evidence.py` reproduce the assay. No new production dependencies,
runtime settings, atlas entries, input execution, or product default changed.
