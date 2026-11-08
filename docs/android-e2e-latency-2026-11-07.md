# Warm end-to-end Android latency investigation

**The final balanced holdout reduced median warm task time from 7.531 to
4.561 seconds (39.4%) and capture-to-completed-input time from 914.8 to
472.1 ms (48.4%). All eight tasks passed. The changes are saved in source;
the installed 0.4.0 wheel has not been replaced. Further tested alternatives
did not produce a reliable gain. Startup/model warmup are excluded.**

The subsequent [full Airtest comparison](android-airtest-current-comparison-2026-11-07.md)
exposed a 40.98-second warm task with delayed arrival confirmation. The 4.561-second
median below remains the result of this specific holdout; it is not a latency
guarantee across changing game scenes or lighting. Use the newer comparison when
assessing consistency and practical alternatives.

This performance investigation continues the
[component experiments](android-steady-latency-2026-11-07.md). The measured task
starts in verified Genshin gameplay and ends after the same loading transition
and two consecutive verified arrival observations. Each run must issue exactly
three task taps and one Teleport command. App startup, model initialization and
independent evaluation screenshots are outside this clock. Capture-to-completed
tap is also recorded, with acquisition, recognition and dispatch costs separated.

Only current captured pixels, the existing explicit UI rules and current device
state may authorize input. Evaluation screenshots and expected tap counts must
not provide runtime targets. Full-screen error recognition, ambiguity policy,
coordinate checks, focus checks, overlay handling and bounded retries remain.
CPU inference only. Raw evidence and the frozen starting source stay ignored in
`artifacts/android-e2e/`.

## Protocol and stopping rule

Test direct acquisition and batched device checks independently, then combine
survivors. Investigate polling, evidence I/O and recognition only after measuring
their remaining contribution. Record failures and negative comparisons. Accept a
task-time change when its median improves by at least 3% and 50 ms in a fresh
repeat, without a p95 regression exceeding 5% or any correctness failure. Small
samples are operational evidence on this phone, not statistical or cross-device
proof. A component gain without a repeatable task gain does not justify added
runtime complexity by itself.

Stop when the measured bottlenecks have been tested and further justified
candidates fail to reproduce an improvement under this contract. This establishes
a practical plateau for the tested workflow, not a global performance minimum.

## Initial baseline

Two warm tasks using the previous turn's persistent ADB channel and histogram
preprocessing passed. Median task time was **7.436 s** (7.328–7.544 s); the six
capture-to-input measurements had median **991.7 ms**, p95 **1096.3 ms**. The
initial app/model warmup was recorded separately and is excluded. Source hashes,
per-frame states, stage timings and independent arrival images are saved in
`baseline-live/`.

## Development evidence

The first seven-task cohort isolated each change. All tasks passed the original
three-input, loading and arrival contract, and the independent arrival images
were inspected. Median task / capture-to-input times were:

| Path | Task | Capture-to-input |
|---|---:|---:|
| Current baseline, 3 runs | 7.573 s | 1020.6 ms |
| Direct capture only, 2 runs | 6.022 s | 803.0 ms |
| Batched guards only, 2 runs | 6.999 s | 942.6 ms |

Batched guards read display/overlay state once, then read focus closest to
injection and dispatch on the device. Their first shell implementation still
started several pipelines. A compact `awk` version reduced that overhead. The
underlying control is never sent after overlay recovery: recovery returns False
and requires a new observation. Registered generic Session popup/busy policies
retain their existing lookup path.

In the subsequent 14-task development cohort, direct capture plus the first
batched guards took 5.756 s median. Compact guards alone did not establish an E2E
gain (5.886 s), despite lower input latency; this result required further checks.
Using a 50-ms interactive wait with compact guards took 5.148 s. A 10-ms wait was
slower at 5.529 s. Overlapping the three evidence-image writes with game
transitions took 4.643 s; all writes were joined inside the task clock. Narrower
focus diagnostics with full-query fallback were also tested, with a separate
comparison before selecting a final path. These development figures are not the
final frozen-source result.

CPU OCR checks used ten immutable images with interleaved repeated calls. Four
threads took 117.5 ms median; two took 162.6 ms and eight took 115.9 ms. Eight
threads did not provide a material gain. Disabling orientation classification
took 114.4 ms but changed the map output in all three measured passes; reject it.
Full detection and classification remain active. The installed 1.4.4 source was
inspected; the current [RapidOCR usage guide](https://rapidai.github.io/RapidOCRDocs/main/en/install_usage/rapidocr/usage/)
also documents independent detection/classification/recognition stages. This is
not a claim that its newer API is interchangeable with the installed adapter.

One more acquisition candidate retains a software bitmap and uses
[`Bitmap.sameAs`](https://developer.android.com/reference/android/graphics/Bitmap#sameAs(android.graphics.Bitmap))
instead of copying all pixels into another Java byte array for comparison.
A 24-pair read-only probe with full JPEG encoding reduced median capture time
from 87.7 to 77.9 ms. It was advanced to an E2E comparison rather than credited
with an inferred task saving. Both acquisition methods passed fresh-UI-change
checks and portrait/landscape/portrait transitions with one persistent helper.

Development failures were retained: a malformed Base64 response exposed an
existing missing ValueError catch in capture fallback; the regression now checks
that a new PNG is acquired instead of retaining an old frame. An early-exit awk
query caused a broken-pipe diagnostic from Android; consuming the complete dump
fixed that response failure. Initial unit tests hit the sandbox's external temp
directory permissions and were rerun with an ignored workspace temp directory.

The 12-task refinement cohort gave 5.196 s for the byte-copy direct path and
4.716 s for retained bitmaps. A 10-ms wait caused 26–29 observations instead of
18–20, adding work while the game was still transitioning. A separate plateau
cohort stopped before its first measured task: the unchanged world detector
intermittently rejected the visible health bar after startup. Its evidence is
retained in `plateau-probes/`. The harness now allows up to 15 seconds of no-input
readiness observations before timing; it still refuses to start outside verified
gameplay. Neither runtime classification nor timed arrival checks were weakened.

The repeat completed all 12 tasks. The bitmap/narrow-focus/50-ms reference took
5.158 s median. Full focus diagnostics took 4.918 s, 100-ms polling took 4.887 s,
10-ms polling took 5.179 s, and the original multi-pipeline guards took 5.265 s.
Together with the previous cohort's 4.831 s at 100 ms versus 5.196 s at 50 ms,
these results support the longer wait. Narrow focus queries did not reproduce an
improvement; the source implementation keeps the full query. Compact guards
reduced capture-to-input time versus the multi-pipeline guards (432 versus
521 ms in this cohort).

## Implementation and impact

This is a performance change to on-demand acquisition and the Genshin example's
input/evidence scheduling, plus a capture-error fallback fix. `Session(auto)`
now attempts native bitmap acquisition only on Android 13 with one physical
display. Exceptions disable that capability and acquire a fresh legacy frame;
helper failures acquire fresh PNG. `jpeg` explicitly selects legacy acquisition.
No approximate image reuse is used. The report identifies the actual backend.

Shared `androidtestkit.guarded.checked_tap` owns batched geometry/overlay/focus
checks. A missing `awk` can select the old path only before input. Unknown output,
focus or geometry mismatch, timeout and lost acknowledgement never cause replay.
The example retains the generic Session path when extra popup/busy rules exist.
Overlay recovery returns False and requires reacquisition before a task tap.
The production guard passed native-counter tests in all three fixture variants,
including focus/geometry rejection and overlay withholding.

Evidence writes retain their original captured array and overlap transitions.
The task joins writes on success and failure, and reports write failures. Arrival
PNG remains synchronous. The Genshin wait is 100 ms for interactive states;
download/preparation backoff and full-screen OCR remain. Generic Session timing,
OCR defaults and the installed 0.4.0 wheel are unchanged. New source APIs do not
import experiment code. Final validation must exercise the actual production
Phone/Runner rather than benchmark substitutions; its control loads the frozen
starting profile, and hashes guard against edits during a cohort.

## Remaining candidates and stopping decision

The production-alternative cohort completed eight tasks. Production at 100 ms
took 5.202 s median; 50 ms took 4.874 s. The existing scrcpy 4.1 stream took
8.063 s despite quicker individual captures. It performed 27–30 observations
and 2.9–3.2 s of OCR, versus 16–17 observations and about 1.3 s of OCR in the
neighboring production tasks. Its continuous decoded frames did not use the
on-demand helper's exact-repeat suppression.

An isolated exact full-frame comparison tested that causal explanation without
approximate reuse. The next eight-task cohort found zero identical streaming
frames and still took 7.635 s. Its production control took 4.726 s; 50-ms polling
took 5.136 s. Thus streaming and shorter polling do not provide a repeatable
task gain. The isolated experimental comparator stays outside production.

| Candidate | Decision |
|---|---|
| Direct acquisition and native bitmap equality | Use, with explicit capability/fresh fallback and rotation checks |
| Batched compact input guards | Use; native counters and failure-path checks pass |
| Narrow focus diagnostics | Omit; no repeatable E2E gain over the full query |
| Concurrent evidence PNG writes | Use; join inside task completion and propagate errors |
| 100-ms interactive wait | Use; 50-ms advantage did not repeat, 10 ms added observations |
| Two/eight OCR threads | Keep four for this validated task; no improvement justified a change |
| Disable OCR orientation classification | Reject; changed map recognition |
| Streaming, with/without exact frame comparison | Reject for this task; slower completion |

The tested acquisition, transport, guard, recognition, scheduling and evidence
costs now have either a supported change or a recorded negative comparison.
Further progress would need a different mechanism and fresh evidence, not more
poll-frequency tuning. This is the practical stopping point for this phone,
scene, CPU OCR configuration and unchanged workflow contract. It does not prove
that other hardware, models, game scenes or capture designs cannot be faster.

## Final holdout

Source was frozen before the final balanced order B P P B P B B P. The control
uses the starting profile with legacy capture, synchronous evidence and a 250-ms
wait; it already includes the previous investigation's persistent shell and
histogram preprocessing. Production uses the actual source classes with default
auto acquisition, compact full-query guards, overlapping evidence and a 100-ms
wait. Both use CPU OpenVINO OCR, four threads and a 1,280-pixel input limit on the
same Samsung SM-G781B, Android 13, Windows/Python 3.12.10 host.

| Metric | Frozen baseline | Production |
|---|---:|---:|
| Warm task count | 4 | 4 |
| Task mean | 7.385 s | 4.565 s |
| Task median | 7.531 s | 4.561 s |
| Task empirical p95 | 7.730 s | 4.757 s |
| Task maximum | 7.750 s | 4.772 s |
| Capture-to-input count | 12 | 12 |
| Capture-to-input mean | 934.9 ms | 484.4 ms |
| Capture-to-input median | 914.8 ms | 472.1 ms |
| Capture-to-input empirical p95 | 1028.5 ms | 588.7 ms |
| Capture-to-input maximum | 1033.1 ms | 619.0 ms |

Mean task capture cost fell from 2.464 to 1.259 s; mean dispatch cost fell from
1.251 to 0.665 s. Mean full-screen OCR cost stayed similar, 1.076 versus 1.034 s.
Remaining time includes polling, image processing, evidence and game transitions;
these cannot be treated as independent additive game/runtime savings because
the game continues while observations and evidence writes execute.

All tasks issued exactly three successful inputs, one Teleport, an observed
loading transition and two consecutive verified arrivals, with no recovery or
retry during timing. Independent ADB PNGs were taken after each task and visually
checked: all eight show gameplay at the same waypoint. There were 39 baseline
and 60 production acquisitions, including 4 and 20 exact repeats respectively.
These repeats follow new acquisitions with identical pixels; they are not stale
images returned after errors. No acquisition failure occurred. Actual acquisition
was `screencap` for every baseline and `direct-bitmap` for every production task.
Battery temperature ranged from 37.4 to 37.8 C across the interleaved cohort;
this is context, not an energy or thermal-improvement claim.

The final source passed 88 focused Android, Genshin and OCR-policy tests on
Windows. Live native-counter/geometry/focus/overlay and default-capture rotation
checks supplement those contracts. Across development and final comparisons,
71 timed warm tasks passed; the separately recorded readiness failure occurred
before timing. Samples are small and one-device only, so empirical p95 values
are descriptive, not population tail guarantees. No human-speed, cold-start or
cross-device improvement is claimed.

Raw images, source hashes, per-observation states/timings and metadata remain
ignored under `artifacts/android-e2e/`. Key cohorts are `final-holdout/`,
`production-alternatives/`, `stream-exact-repeat/`, `plateau-repeat/`,
`production-guard-fixture/` and `production-auto-freshness/`. Concise results are
kept here instead of committing raw research outputs. Changes are uncommitted
workspace source; run from this repository with `python -m androidtestkit` to use
them. The installed wheel and unrelated workspace changes were preserved.
