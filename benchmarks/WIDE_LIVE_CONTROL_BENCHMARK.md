# Wide live-control benchmark

This benchmark selects pose and localization mechanisms for a continuous
30 FPS control loop. It keeps cold-start localization separate from high-rate
continuation and never counts a held state as a fresh visual fix.

## Evidence contract

- Target: 30 fresh fixes per second, with a 33.3 ms per-stage compute budget.
- Minimum failure boundary: 15 fresh fixes per second or 66.7 ms.
- Pose absolute evidence comes from forward-only end-to-end motion direction.
- Repeated-lap input and scene KLT are independent temporal-response evidence,
  not pose truth.
- Localization references are sparse post-run evidence and never feed the
  tracker after the single declared starting pose.
- Cross-method consistency is diagnostic when truth is unavailable; it is not
  promoted to accuracy.
- Decode/capture and application publication latency are reported separately
  from algorithm compute.

## Candidate structure

The desired runtime stays at two layers:

1. One current-frame visual measurement.
2. One calibration-derived physical innovation gate. A rejection holds the
   prior state but is reported as held, not fresh.

Continuous smoothing, confidence hysteresis, and alternate matchers remain in
benchmark code as traceable negative or research candidates. Production must
not import this benchmark package.

## Reproduction

Run `benchmarks.localization.run_wide_temporal`, then
`benchmarks.cursor_pose.run_wide_temporal`, then
`benchmarks.build_wide_live_control_report`. Each command requires explicit
session, calibration, atlas/reference, and output paths. Machine-readable
results and chronological telemetry are written under the requested output.
