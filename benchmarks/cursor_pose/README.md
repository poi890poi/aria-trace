# Cursor pose benchmark protocol

This package is the standard offline benchmark for cursor pose publication. It
is intentionally outside `aria_trace`: benchmark mechanisms cannot become a
production dependency.

## Contract

The unit under test is chronological and causal:

```text
recorded frame
  -> one-frame primary pose estimator
  -> one final confidence + temporal-innovation acceptance gate
  -> one publication action
  -> published cursor state
```

The evaluator independently estimates the whole-session direction of travel
from cursor-masked mini-map phase correlation. This label is valid only for a
controlled forward-only session, so the input audit requires `W`, rejects
`A/S/D`, and rejects relative mouse movement. It is strictly a post-run label
and is never passed to an estimator or publication policy. A session with phase
response below `0.05` or end-to-end displacement below `1.5 px` is rejected
rather than assigned a weak direction label.

This is a functional E2E travel reference, not a per-frame pixel pose label.
Initial alignment, collision, or path curvature can make a correct cursor pose
disagree with the travel vector. Use it to compare complete publication
behavior and report E2E worst error. Do not tune a pose-outlier gate from this
disagreement alone. Outlier-gate validation additionally needs labeled pose
failures or a separately specified corruption protocol; agreement with the
accurate estimator is an audit, not independent truth.

## Real-time control premise

The control target is 30 fresh pose fixes per second with a 33.3 ms
capture-to-publication deadline.  The minimum usable tier is 15 fresh fixes per
second with a 66.7 ms deadline.  Held and predicted outputs are continuity
signals and never count as fresh real-time availability.

Offline estimator timing is not complete pipeline timing.  Reports must label
it as estimator-only evidence until capture, decode, extraction, worker IPC,
scheduling, fusion, rendering, and publication are timestamped end to end.

## Rate names and denominators

Never use an unqualified field or column named `rate`.

| Metric | Numerator | Denominator |
|---|---|---|
| `primary_candidate_produced_rate` | One-frame pose candidates produced by the first layer | All chronological primary attempts |
| `primary_measurement_accepted_rate` | Candidates passing the final confidence and temporal outlier gate | All chronological primary attempts |
| `primary_measurement_rejected_rate` | Missing, low-confidence, or temporally implausible candidates | All chronological primary attempts |
| `fallback_invocation_rate` | Rejected primary measurements passed to a fallback | All chronological primary attempts |
| `fallback_output_success_rate` | Fallback invocations that produced a held/predicted state | Fallback invocations only |
| `final_output_available_rate` | Fresh plus fallback-produced states | All chronological primary attempts |
| `output_provenance_rate.*` | One mutually-exclusive provenance class | All chronological primary attempts |

Accepted and rejected rates are complements. Output provenance is exactly one
of `fresh_measurement`, `held`, `predicted`, or `unavailable`, and the four
rates must sum to 100%. Internal paths such as pixel validation and a
Gaussian fitter fallback are reported as algorithm path invocation rates, not
as pose acceptance rates.

## Accuracy and continuity

Accuracy is absolute circular screen-heading error in degrees. The full E2E
output reports mean, median, P95, and worst. `worst` is only reported for the
complete chronological E2E result. Fresh-only and fallback-only diagnostics
report mean/median/P95 but intentionally omit a maximum.

Unavailable states are never assigned zero error. They are represented by:

- final availability;
- unavailable frame count;
- unavailable episode count; and
- longest unavailable episode in frames.

## Complete solutions, not fallback stacks

Every solution has two decision layers only:

1. The selected one-frame estimator produces a candidate on nearly every frame.
2. One final gate applies the profile confidence threshold and compares the
   candidate with the last accepted state. Ordinary-size innovations
   pass normally; a larger innovation must also exceed the calibration's median
   pose confidence. Both thresholds come from the saved calibration.

Rejected candidates never update gate or publication state. Each complete
solution then has exactly one action: publish unavailable (`reject`), reuse the
previous accepted state (`hold`), or extrapolate the last two accepted states
(`predict`). There is no chained filter or fallback-of-fallback.

The default matrix is orthogonal: accurate/real-time/fast estimator profiles
use the same confidence+hold policy; confidence-only and strict temporal gates
use the same real-time estimator and hold policy; hold/predict/unavailable use
the same real-time estimator and confidence-only gate. The human summary can
therefore attribute each change to one layer. Negative controls remain in
machine-readable results for traceability. There is no unrestricted Cartesian
product.

Natural confidence decisions are the authoritative acceptance/rejection result.
Deterministic outage scenarios hide the same otherwise accepted frames for every
strategy. They are robustness stress tests, not natural rejection measurements.

## Reproducibility and code management

All evaluated algorithm implementations are retained for future research,
including rejected and inconclusive candidates.  Production code must not
import the POC harness, and promoting one candidate must not delete another.
Each result manifest hashes the exact implementation so later work can repeat
or compare it.  Removal is allowed only after the implementation and its
evidence have been preserved in an explicitly named archive commit or tag.

Every run writes:

- `benchmark_config.json`: immutable run parameters and thresholds;
- `method_manifest.json`: fully-qualified implementations, source files, source
  line numbers, file SHA-256, and implementation SHA-256;
- `input_manifest.json`: calibration/model/session metadata and content hashes;
- `primary_measurements.csv`: raw estimator results before fallback;
- `e2e_rows.csv`: every published state, provenance, label, and error;
- `results.json`: machine-readable aggregates and environment provenance; and
- `REPORT.md`: narrow-screen result cards and definitions.

`results.json` records the Git revision and separately records whether any
tested source is dirty. A run with `tested_source_dirty=true` is diagnostic and
must not be used as a release comparison. Commit benchmark and method changes,
then rerun. Repository-wide unrelated dirtiness is retained as an audit field
but does not invalidate a run when tested source files are clean.

Tested methods are changed one at a time. A report is never overwritten: use a
new output directory containing a date/run identifier. Negative results remain
in source control or archived artifacts with their exact method ID and hashes.

## Run

From the repository root, using the standalone Python environment:

```powershell
$python = '.tools\standalone-release-py31210\Scripts\python.exe'
& $python -m benchmarks.cursor_pose.run_e2e `
  artifacts\workbench\minimap_calibrations\genshin-impact-pc\segments-df624035-833-bd07601f-708\calibration.json `
  sessions\workbench\recordings-genshin-impact-pc `
  artifacts\poc\cursor-pose-e2e-YYYYMMDD-HHMMSS
```

Use repeated `--solution NAME` arguments to restrict the complete solutions.
Video hashing is enabled by default;
`--skip-video-hash` is only for a quick local diagnostic.

After running the experimental method screen and this chronological E2E
benchmark, combine them without merging their different evidence semantics:

```powershell
& $python -m benchmarks.cursor_pose.build_layered_report `
  artifacts\poc\cursor-pose-method-screen\results.json `
  artifacts\poc\cursor-pose-e2e\results.json `
  artifacts\poc\cursor-pose-layered-report
```

The combined `REPORT.md` is phone-readable. `layered_results.json` preserves
the compared rows and hashes both authoritative source result files.

Build the plain-text real-time control report from an existing complete
candidate benchmark without rerunning the estimators:

```powershell
& $python -m benchmarks.cursor_pose.build_realtime_control_report `
  artifacts\poc\cursor-pose-candidate-e2e-YYYYMMDD-HHMMSS `
  sessions\workbench\recordings-genshin-impact-pc `
  artifacts\poc\cursor-pose-realtime-control-YYYYMMDD-HHMMSS
```

This report measures source timing and estimator-only deadline completion.  It
does not claim capture-to-publication compliance.
