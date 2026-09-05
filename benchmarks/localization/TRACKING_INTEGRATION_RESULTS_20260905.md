# Tracking integration and follow-up results — 2026-09-05

Direct atlas matching now runs in production without route guidance. The supported
uncertainty, first-refinement and heading-delivery corrections are integrated.
Faster representation observations reduce the remaining transition loss without
adding transition anchors or weakening confirmation. Early rejection of invalid
feature geometry removes wasted global-localization work. Reliable autonomous
cruising is still unproven: startup in the outdoor scene remains long, delivery
can miss its budget, and no closed-loop steering result is claimed.

This implementation evaluation contains 35 recorded-source runs: 30 complete
recordings and five Run17 startup clips limited to 45 seconds. They cover
1,947.53 seconds, 57,445 source frames and 56,922 processed frames. The earlier
55-replay candidate evaluation is separate and is not included in this count.

## Decisions

| Change | Decision | Evidence and confidence |
|---|---|---|
| I/J: direct atlas matching plus wider initial refinement | Landed, `83c0389` | No route package, route states or advisor are required. Existing non-refining localizers retain their fallback. Production free-roam demo/outdoor/town P95 is approximately 2.8/3.9/1.3 px. High confidence for measured recordings, moderate generalization. |
| A: observation-time transition uncertainty | Landed, `9f1eba3` | Reproduced narrow-zone arming defect. Previous Run14 loss of 4.21 s unresolved becomes 0.57 s recovered before further scheduling changes. High confidence in the diagnosed correction. |
| E/G: submit heading first, consume after XY work within source deadline | Landed, `b6c6125` | Preserves heading measurements and asynchronous fallback. Tests cover same-frame completion and an already-expired frame. Improves delivery age; does not certify a 30 Hz pipeline. |
| L: observe armed transitions at up to 30 Hz | Landed, `e7f3e04` | L gives Run9/14/11 post-acquisition losses of 0.490/0.231/0.402 s. LN repeats all three exactly. Unarmed observation cadence and confirmation thresholds remain intact. High confidence on these three crossings; moderate beyond them. |
| K: wider refinement after confirmed layer change | Retain as experimental | K improves Run9 to 0.790 s loss but does not fix its high error tail. KL shortens Run9 to 0.387 s but worsens reverse-crossing loss from L's 0.231 to 0.355 s and worst error from 35.86 to 40.98 px. The simpler L change is preferred. |
| M: bound the next startup search around the first valid hypothesis | Retain as experimental | Runs9/15 initialize near 2.1 s. Run17 takes 38.26 s and 28.03 s on repeat; unchanged comparison takes 24.20 and 33.03 s. This does not establish reliable startup improvement. |
| N: reject failed feature geometry before map correlation | Landed, `cd5cdca` | Geometry that fails existing inlier, reprojection or scale checks cannot be accepted later. Skipping its correlation preserves the checks. N/LN/final-production initialize Run17 in 22.31/22.42/22.31 s. High confidence that work is unnecessary; moderate confidence in startup benefit under variable host scheduling. |

Historical A–J experiments remain reproducible from Git revision `1386101`.
K–N experiments freeze `b6c6125` methods and save the exact executed sources and
hashes. The full-loop replay replaces only the frame source; experiments do not
inject reference positions into tracking. Benchmark-only candidates are not
production configuration flags.

## Causal comparisons

| Configuration | Run9 loss s | Run14 reverse loss s | Run11 demo loss s |
|---|---:|---:|---:|
| Integrated I/J/A/E/G production, first replay | 0.919 | 0.573 | 0.742 |
| K alone | 0.790 | 0.573 | 0.707 |
| L alone | 0.490 | 0.231 | 0.402 |
| KL | 0.387 | 0.355 | 0.402 |
| LN | 0.490 | 0.231 | 0.402 |

The preceding unchanged candidate integration had Run9 loss 1.522 s twice.
The first production replay initialized later and selected a different
trajectory, giving 0.919 s. Startup and asynchronous source sampling vary; the
report preserves that variation instead of claiming a single deterministic
speedup from selected replays.

L reduces Run9's position P95 from approximately 29–34 px to 3.45 px. LN gives
3.52 px. Short transition outliers still reach approximately 38 px; a good P95
does not erase them. The reference-based loss metric includes wrong map layer
and wrong-but-fresh position, and requires 0.5 s of fresh consistent observations
to confirm recovery. It excludes publication delay, which is reported separately.

The outdoor reference has substantial gaps. Its recurring 1.14 s post-acquisition
episode includes approximately 1.00 s without reference support. It is time to
verified recovery, not continuously proven position error. Cold acquisition,
unknown intervals and source timing remain visible in the detailed reports.

## Final production verification

| Run/configuration | First pose s | Position P95 px | Longest post-acquisition loss s | Publication P95 ms |
|---|---:|---:|---:|---:|
| 9, free-roam | 2.20 | 3.52 | 0.490 | 34.80 |
| 14 reverse, free-roam | 2.64 | 2.80 | 0.231 | 43.15 |
| 11 demo, free-roam | 2.76 | 2.79 | 0.402 | 36.59 |
| 17 startup verification, first 45 s | 22.31 | 4.03 | 0.000 | 29.38 |
| 11, route-assisted with recording | 2.17 | 2.78 | 0.302 | 236.30 |

The startup-only row cannot replace the full outdoor LN lap: that full recording
has P95 3.93 px, the 1.14 s episode with unknown time described above, and
publication P95 37.20 ms. Full town LN laps have P95 1.35 px, no measured
post-acquisition loss, and publication P95 70.69 ms. These runs use the real-time
profile. Other named profiles were not benchmarked as equivalent configurations.

Verification: 99 focused tests passed across production integration, historical
and follow-up candidates, valid and rejected global localization, map layers,
route tracking, minimap transition, cursor workers, profiles, evidence recording,
source prefetch and tracking-loss scoring. Production changes are separate
commits; unrelated workspace files are preserved. No physical-control pass is
included in the test count.

## What startup experiments establish

The slow Run17 M replay has a valid hypothesis at 26.75 s, a rejected bounded
solve at 27.53 s, and eventually initializes at 38.27 s. A bounded search does
not repair invalid geometry or guarantee consensus on moving observations.

N removes correlation and associated diagnostic raster work only after existing
feature-geometry checks have failed. Accepted geometry still undergoes the
original correlation, ambiguity, agreement and coverage checks. Invalid fixes
now report the geometry rejection without computing additional correlation
failure reasons. Existing valid-patch and bounded-search regressions pass with N.

The frozen slow reference itself first provides a Run17 position at 20.28 s.
Thus startup near 22 s is not a solved cruising requirement. Earlier reliable
acquisition needs further visual evidence and an independently validated
initializer; relaxing quality thresholds is not supported by these experiments.

## Delivery and physical validation

The decode-ahead experiment uses a bounded 30-frame queue and preserves frame
identity, original timestamps, and release order. It does not release future
images to the tracker. Prefill setup time and decoder waits are recorded
separately. Any improvement belongs to the recorded-source adapter, not physical
GDI/camera capture or tracker computation.

Town-lap decode-ahead keeps position P95 at 1.34 px and lowers publication P95
from the LN comparison's 70.69 ms to 33.79 ms. It still misses a 33.3 ms P95
target. Treat its timing benefit as preliminary on this variable host.

The final production recording run preserves position quality but fails timing:
publication P95 236.30 ms, engine P95 23.24 ms, and source release-lateness
P95 189.74 ms. Its recorder reports 219 dropped and 392 repeated frames while
writing 1,800 output frames without an encoding error. This is successful file
creation, not a successful sustained-recording quality result. The source
comparison with recording enabled is reported separately below.

| Transport/configuration | Position P95 px | Publication P95 ms | Fresh XY + heading within 33.3 ms / all source frames after first pose |
|---|---:|---:|---:|
| Ordinary decode, Run11 route recording | 2.78 | 236.30 | 64.0% |
| Decode-ahead, Run11 route recording | 2.77 | 33.96 | 91.8% |
| Decode-ahead, Run18 free-roam laps | 1.34 | 33.79 | 91.5% |

The recording comparison keeps longest position loss at 0.302 s. Decode-ahead
records zero queue drops and 213 repeated output frames, so zero queue drops
does not imply every output frame is a new source observation. Both recording
files contain 1,800 1280×720 H.264 frames, and first/middle/last frames decode.
Prefill took 103 ms for town laps and 140 ms for recording, before the source
clock starts. Those setup costs are excluded from publication latency and are
not concealed as a tracker speedup. The similar accuracy, improved source timing
and two configurations support a replay-transport benefit with moderate
confidence; neither configuration reaches the 95% joint deadline criterion.

The ordinary final free-roam runs also miss that joint criterion: Run9 88.4%,
Run14 84.1%, and demo Run11 89.8%. Full-source deadlines include overwritten
frames and stale heading measurements. Fresh, timely pairs still do not prove
heading correctness or steering capability.

Genshin Impact is running and its window can be enumerated with desktop access,
but it was not foreground. The current GDI adapter reads the visible client
region. A request for a fresh user-driven cruise is pending; no fresh moving
holdout or closed-loop steering test has been performed in this evaluation.
The inspected tracking loop publishes observations and does not implement an
autonomous route controller. Position accuracy cannot substitute for measuring
wrong turns, steering oscillation, path clearance, interventions and longest
continuous control loss.

## Reproducibility

Evidence root: `artifacts/poc/tracking-integration-20260905`.
Each cohort contains original source timestamps, actual Workbench telemetry,
scored rows, source identities, cache identities and per-run reports. The slow
references are frozen, hash-checked and reused; they are same-atlas inferred
estimates, not external ground truth. Startup-only Run17 comparisons are limited
to the first 45 s and must not be presented as complete lap runs. Full lap
validation is stored separately in `combined-LN`.

Use `tracking_candidates --variant production` for the current runtime,
`tracking_followups --variant <letters>` for frozen follow-ups, and
`prefetched_replay` for the explicitly separate transport experiment. Always
choose a new output directory to preserve evidence.
