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
