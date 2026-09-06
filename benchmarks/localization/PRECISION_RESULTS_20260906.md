# Precision, transition handoff, and scene-bearing benchmarks

This round retains promising experiments and rejects shortcuts; it does not
change the production tracker or active atlas. The known-good release tag remains
unchanged. Best measured canonical XY continuity comes from subpixel refinement
combined with intermediate-scale matching, but its publication cost is too high
for the intended visual-control loop.

## Evaluation and limits

Baseline revision: `9e8de66`. The matched control already includes the preceding
experimental removal of hard spatial transition gates. It is not the earlier
production baseline that missed five of six Run 20 transitions. Current images
verify a declared first-state starting proposal on runs 11, 14, and 20. Subsequent
reference states are scoring only; no route progress or future XY enters tracking.

Recorded-source Workbench replay uses original source timestamps and bounded
decode-ahead. This includes tracker scheduling and publication, not physical
capture, browser paint, actuator response, or closed-loop cruise success. The
Python 3.12.10 runtime uses OpenCV 5.0.0 with six OpenCV threads. Timed E2E replays
ran sequentially without concurrent vision workloads.

References are cached slow SIFT estimates from the same atlas, not external
ground truth. P95 and worst values below mean disagreement with that proxy.
Loss uses the frozen leave-session-out reference-resolution envelope and includes
wrong discrete map mode; held XY is reported separately. The envelope is not a
validated gameplay tolerance. No new pixel tolerance was chosen to make a
candidate pass. Unknown reference intervals remain explicit.

Run 20 has 5,931 frames over 239.903 s, approximately 24.7 Hz rather than 30 Hz,
with 118 capture gaps exceeding 100 ms. Its six crossings and three time windows
are repeated development evidence, not independent statistical trials. Runs
11/14 are historical direction checks after fixing the candidates. There is no
fresh cruise recorded after these hypotheses were proposed.

## Full narrow-lap results

| Run 20 candidate | P95 / worst px | Longest lost s | Held XY | Maximum step px | Publication P95 ms |
|---|---:|---:|---:|---:|---:|
| Matched control | 2.910 / 34.638 | 1.229 | 85 | 24.534 | 23.428 |
| Subpixel alone | 2.254 / 34.296 | 1.229 | 92 | 24.329 | 23.509 |
| 17 scales alone | 2.807 / 5.111 | 0.422 | 0 | 5.461 | 68.378 |
| Subpixel + 17 scales | 2.194 / 3.352 | 0.422 | 0 | 2.836 | 66.743 |
| Expand below endpoint score 0.55 | 2.838 / 31.172 | 1.075 | 35 | 19.926 | 27.393 |

The combination remains below 3.36 px proxy disagreement in every measured
crossing window. It has zero fresh results exceeding 20 px, versus seven for
the control. Its maximum output step drops about 88%, but jointly fresh XY and
heading delivered within 33.3 ms falls from 95.73% to 50.33% of all source frames.
Thus smooth-looking spatial output does not establish timely control quality.

All full Run 20 variants confirm six mode changes. The combination still has
32 fresh rows whose discrete mode differs from the available reference mode;
its continuously selected scale supplies XY while the unchanged controller
confirms the discrete mode later. This explains the remaining 0.422 s longest
loss despite accurate XY and zero held outputs. Approximately 7.44 s has unknown
reference status. Neither unknown intervals nor skipped source frames count as
verified accurate tracking.

Subpixel improves ordinary precision in all three 80-second windows, and the
known fractional-shift regression independently verifies reduced raster error
without changing integer score or coverage acceptance. However, it is not a
standalone continuity fix: the first inbound transition's worst error increases
from 17.11 to 32.78 px and held rows increase. Its strongest case is in combination
with the transition representation change.

## Directed recording checks

| Session / candidate | P95 / worst px | Longest lost s | Held XY | Publication P95 ms |
|---|---:|---:|---:|---:|
| 11 control | 2.786 / 24.175 | 0.402 | 20 | 24.743 |
| 11 subpixel | 2.198 / 24.400 | 0.369 | 17 | 29.363 |
| 11 combination | 2.164 / 3.099 | 0.302 | 0 | 69.735 |
| 14 control | 2.800 / 39.060 | 1.906 | 13 | 41.905 |
| 14 subpixel | 1.930 / 36.507 | 0.231 | 16 | 33.580 |
| 14 combination | 1.854 / 2.982 | 0 | 0 | 71.889 |

The combination's accuracy and timing behavior reproduce in both directions.
Single-run OS scheduling differs: for example, Run 14 control is slower than
its subpixel candidate. That is not evidence that adding subpixel fitting speeds
up the tracker. The much larger 17-scale cost repeats across recordings and
appears inside local matching itself: Run 20 local-template P95 rises from
3.746 to 40.357 ms. Fifteen extra layers require 192,496,960 bytes (183.6 MiB)
of arrays, plus roughly 0.8 s measured initial pyramid construction in the clip.

The historical Run 17 check uses unknown-start acquisition, with no later
reference state supplied as an initial hint. Both control and combination first
publish XY after 22.312 s and have 24.431 s initial loss to verified recovery.
On scored portions, P95 / worst disagreement is 3.941 / 6.020 px for control and
3.118 / 5.986 px for the combination. Longest post-acquisition loss changes from
1.140 to 1.025 s; both have no held XY after acquisition. Publication P95 rises
from 22.584 to 65.541 ms. Only about 40.2% of processed frames have a reference
error, and 85.646 s remains unknown in the loss evaluation. These results do not
certify the unscored town portions or fix the slow initial acquisition.

## Negative causal tests

**Score-triggered expansion:** using the existing 0.55 acquisition score limits
full sweeps to 100 of 5,835 processed rows and restores ordinary latency. But a
wrong-scale image match can exceed that threshold; the third crossing still
reaches 31.17 px error. Reject this shortcut. Do not increase or tune thresholds
on the withheld recordings to conceal the miss.

**Remove intermediate scales:** native frame 577 at 19.184 s shows the control
still matching town scale while the successful candidate matches world scale.
To separate early endpoint handoff from intermediate zoom handling, the first
25 seconds were repeated using only the existing town/world layers. Worst error
was 27.344 px, versus 3.895 px with 17 scales over that interval. Endpoint-only
handoff is insufficient; intermediate representation support contributes. This
failed ablation was stopped before combinations or a full-lap run.

The full combination was also given a deliberately wrong starting proposal,
offset 200 canonical pixels in X. It accepted no XY and produced no pose during
the five-second test. The larger scale hypothesis set did not turn that particular
wrong prior into a fresh measurement. This is one negative-start test, not proof
against every repeated-texture false acquisition.

The previous post-confirmation search reset had mixed jump results. It was not
combined again: the scale-aware candidate already has no held XY outputs, and
a reset after confirmation cannot repair the earlier wrong-scale measurements.

## Atlas support and cache reuse

The candidate atlas stores observed town/world support and directed crossing
brackets as map features. Runtime remains visual-only in this round. The expanded
candidate was trained on 11/14/20 before the Run 17 lookup; it remains inactive.
At Run 17's 158 stable inferred positions, 98 lookup as world and 60 as unknown,
with no contradictory label among those samples. This is a best-case lookup at
reference XY, not an E2E test of a map prior. No town reference states were accepted
in Run 17, so it cannot validate town-region generalization.

Keep unknown regions and overlapping support explicit. The data supports sparse
observations and directed hysteresis, not a complete town polygon. Adding a new
hard gate or arbitrary prior weight is unjustified. A future soft map prior can
schedule or order image searches near learned crossings; it must neither veto
unexpected image evidence nor supply a pose.

Run 17's current-calibration cache contains 158 accepted states from 348 samples,
beginning at 20.280 s. Its 33.29 s reference build was followed by a verified cache
hit; the second CLI invocation completed in about 1.6 s including startup. Cached
source, calibration, atlas, configuration, implementation and output identities
are checked. Matching completed inferred references are saved and reused; stale
or incomplete entries are not silently accepted.

## Decisions and confidence

| Candidate | Decision | Confidence and practical limit |
|---|---|---|
| Fractional peak fitting | Retain with transition work | High for synthetic raster precision; moderate for native accuracy; transition continuity can worsen alone |
| Intermediate-scale XY matching | Retain as accuracy reference | Moderate-high across eight crossings in three recordings; high confidence current full sweep is too slow |
| Fractional + scale matching | Preferred spatial benchmark | Best measured error/steps; fails timely joint-control delivery; no production promotion |
| Endpoint-only handoff | Reject as sufficient transition solution | Clear first-crossing failure |
| 0.55 score-triggered sweep | Reject | Confident wrong-scale matches defeat the trigger |
| Atlas-owned support | Retain observations, explicit unknowns | Sparse world support generalizes partly; town/soft-prior benefit not established |
| Scene LK + reacquisition | Research component only | Synthetic recovery works; real feature persistence and identity are insufficient for steering |

See `SCENE_BEARING_RESULTS_20260906.md` for the masked optical-flow/template
comparison, preserved HUD-leakage failure, and the proposed demonstrated-bearing
and landmark-handoff approach. No yaw accuracy or automatic control claim is
made from feature persistence.

Next engineering priority is reducing scale-search cost while preserving the
measured transition behavior, then testing coherent XY/scale publication and
landmark handoff on fresh recordings. A hierarchical image search and an
atlas-owned soft scheduling prior are hypotheses, not validated improvements.

## Evidence and reproduction

Root: `artifacts/poc/precision-round-20260906`. `comparison.json` contains latency
decomposition, source delivery, fresh/held/unavailable counts, all-source joint
delivery, and input identities. `precision-analysis.json` contains per-crossing
and per-window results. Each cohort retains raw/scored telemetry, original
source timestamps, report identities and executed experiment snapshots.

The post-run paired native check also evaluates integer and fractional peaks on
six identical recorded minimaps with the same bounded proposal, mask and layer.
Scores, validity and coverage match exactly on all six. This isolates the native
measurement change but is not an independent position-accuracy test. Results and
the executed source are in `native-peak-check.json` and `native_peak_source.py`.

`native-comparison` contains native frame plus masked minimap/control/candidate
atlas panels on identical frames. Selection is the control's worst error within
each crossing window, restricted to frames processed by both methods. These
images distinguish a representation mismatch from a general atlas-stitching
failure; they are not a complete audit of every atlas seam. The late endpoint
example selects world scale 3.863 while the control still uses town 1.302.

Run the pinned Python environment with `-m benchmarks.localization.precision_candidates`
and the ordinary `template_cpu` replay arguments. Candidate switches are
`--subpixel`, `--scale-sweep`, and the rejected `--selective-scale`. The
`endpoint_scale_ablation` wrapper removes the 15 intermediate layers. The
protocol and per-cohort reports record the exact session, start policy and input
index. `report.git_revision` and `implementation` identify actual executed code;
the inherited `experiment.base=f607fa6` label is historical template-harness
metadata and is not the revision used for this round.

Verification: 16 recorded-source E2E runs across four recordings (including two
25-second ablations and the five-second negative-start test), 30 scene assays
including synthetic tests and both mask policies, and 29 focused tests covering
fractional precision, score/coverage gates, mode ownership, scene-mask enforcement,
loss/recovery, demonstrated starts and map-layer behavior. All completed without
runtime errors. Failed quality/timing candidates remain in the evidence.
