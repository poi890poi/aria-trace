# Precision and scene-bearing benchmark round

Type: experimental algorithm/benchmark work only. Production tracker, active
atlas, capture behavior, and live game controls remain unchanged. Baseline code
revision: `9e8de66`; baseline policy is the preceding experimental removal of
hard spatial transition gates plus current-image-verified starting XY.

Run 20 is development data. Compare subpixel peak interpolation alone, local
intermediate-scale matching alone, then their combination only if each has
support. Retain visual mode confirmation and XY continuity gates. The prior
post-switch reset remains a separate, previously mixed candidate. Use cached
same-atlas reference geometry only for scoring, except the explicitly declared
first-state start proposal. Runtime may use current images, original timestamps,
previous estimated pose and atlas pixels, never future reference states.

First validate fractional translation on synthetic known shifts and native
recorded minimaps with the same bounded proposal for paired methods. Then run
survivors through the actual recorded-source Workbench loop. Preserve failures,
unknown reference intervals, fresh/held/unavailable provenance, longest lost,
incorrect fresh poses, jumps, latency distributions and delivery coverage.
Do not tune gameplay pixel tolerances or score proximity to supplied routes as
absolute accuracy. Scene-bearing persistence and input-idle stability are not
independent heading truth or closed-loop steering success.

Subpixel candidate: separable three-point quadratic peak interpolation, bounded
to half a localization pixel on each axis. Keep original integer footprint
coverage and correlation acceptance. Intermediate-scale candidate: compare a
fixed logarithmic pyramid between atlas town/world endpoint scales using only
bounded local image matches. Preserve distinct mode ownership: intermediate
scale can supply current-image XY, while the visual controller still confirms
the discrete rendering mode. Record extra hypotheses, runtime, and memory/setup
cost. Reject a clearly worse single candidate before testing combinations.

Development amendment after the first 25-second scale-sweep clip: all-scale
matching reduced the first crossing's maximum proxy disagreement to 3.895 px,
but publication P95 was 70.317 ms. Complete the full lap to diagnose consistency,
then test selective expansion as a separate scheduling hypothesis: evaluate the
active endpoint first and expand only below the existing 0.55 image-acquisition
score. This is a reuse of a pre-existing threshold, not a newly fitted gameplay
tolerance. Preserve the all-scale negative timing result. Freeze this policy
before testing other sessions. A strong wrong-scale match can defeat this trigger;
test all crossings and retain that failure if observed.

Development amendment after inspecting the matched native frame at 19.184 s:
the successful 17-scale candidate selects the other endpoint scale before the
discrete controller confirms it. Therefore the full sweep changes both the
available scale set and which layer can supply current-image XY. Isolate these:
repeat with only the two existing endpoint layers, retaining image-verified XY
selection and unchanged discrete mode confirmation; then combine with subpixel.
Do not attribute the full improvement to intermediate scales until this ablation.

No cruise newer than run 20 exists; run 19 is full-map capture. Use runs 11/14
for directed-crossing checks and a previously collected cruise such as run 17
as a withheld check after configurations are fixed. Rebuild stale cache inputs
rather than bypass integrity validation. These are historical checks, not fresh
post-hypothesis holdouts. Do not promote production behavior from this round.

Run 17 reference rebuild produced only 158 accepted states from 348 samples,
all world mode, beginning at 20.280 s. Do not use that later first state as a
starting-position prior at source time zero. Any full Run 17 replay must use
unknown-start acquisition and report the large unscored intervals. Runs 11/14
provide the matched known-start direction checks. The frozen expanded atlas
support (trained on 11/14/20) may be scored at Run 17's inferred positions only
as a best-case support lookup, never as an input to runtime localization.

Scene-bearing experiment: track masked scene features on straight movement and
endpoint turns, separating map pose, avatar heading, camera motion and desired
lane. Compare sparse optical flow with local template tracking from identical
initial image features. Use held-out laps only for correspondence evaluation;
do not use their reference positions as runtime predictions. Test loss and
reacquisition explicitly. Accumulated town/world support remains atlas-owned,
with unknown areas explicit and no route-index authority.
