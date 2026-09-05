# Tracking implementation evaluation

Type: supported estimator behavior change, followed by isolated transition,
initialization and scheduling fixes/experiments. User authorized the plan after
the 55-replay candidate report.

The runtime may use current/past frames, calibration, the atlas and explicitly
selected route proposals. Frozen slow references are scoring-only in free roam.
The direct matcher must work without a route package; other map localizers keep
their existing fallback. Global acceptance checks and route guidance ownership
remain intact. No input controls are issued by the recorded benchmark.

First slice: promote I/J and the independently supported A correction. Next,
promote E/G delivery scheduling and test transition recovery and acquisition
separately before combinations. Preserve historical candidate methods from Git
revision 1386101 so production edits do not silently redefine past experiments.

Acceptance: reproduce the supported trajectory improvement on demo, forward and
reverse transitions and laps, without extending longest reference-based loss;
report startup and all-source publication age separately. Target 95% fresh XY
and heading within 33.3 ms; do not hide misses in engine-only timing. Preserve
negative variants. Pixel error is a diagnostic proxy, not cruising acceptance.
Cached same-atlas references have gaps and no independent heading truth.
Fresh physical sessions and closed-loop steering require available live capture
and control infrastructure; inspect those paths before claiming such validation.

Risks: wider seed searches can choose wrong peaks; local matching can drift on
the wrong layer during transition; stale inputs can look smooth. Meaningful
regressions cover no-route operation, unsupported-map fallback, first refinement,
observation-time uncertainty and asynchronous delivery. Land independent slices
in separate commits and record their measured evidence here.

Integrated slices: `83c0389` direct matching and first refinement; `9f1eba3`
observation uncertainty; `b6c6125` heading submission/consumption. The production
replay cohort is `artifacts/poc/tracking-integration-20260905/production-AEGIJ`.

Follow-up experiments freeze b6c6125 methods: K widens position refinement after
a visually confirmed layer switch while retaining continuity gating; L raises
representation observation cadence only while armed; M reuses the first valid
initialization hypothesis as a 150 px bounded proposal for the next solve.
Invalid observations still clear initial hypotheses, restoring unrestricted
search. Test K and L separately on 9/14/11, then KL if results support it.
Test M on 9/15 and the first 45 s of 17, explicitly as a startup experiment.
No hypothesis supplies an accepted pose without current image measurement.

A separate recorded-source experiment primes at most 30 decoded frames and
releases each at its original timestamp. It reports setup and decoder wait
separately. It tests replay transport, not tracker compute or physical capture.
Frame identity/order and non-early release have a behavioral regression.
