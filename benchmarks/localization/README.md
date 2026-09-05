# Localization real-time benchmark

This package standardizes the high-rate XY localization comparison. It keeps
initial absolute localization separate from the chronological local tracker.

The tested live-control shape has only two per-frame decision layers:

1. one current-frame local map matcher produces a measured XY candidate;
2. one physical-continuity gate either accepts that measurement or holds the
   last accepted state.

Held state is never counted as a fresh localization fix. The target is at
least 95% of chronological frames accepted within 33.3 ms (30 FPS); the
minimum tier uses 66.7 ms (15 FPS). The measured core includes mini-map
extraction and local matching, but not live capture, worker scheduling,
fusion, rendering, or publication.

When replay rows contain decode timing, the report additionally measures a
serial recorded-video path from decode through mini-map extraction to the XY
result. This is stronger than core-only timing, but it is still not camera/GDI
capture-to-consumer latency; that requires live timestamp instrumentation.

Experimental matchers remain in `poc/benchmark_route_tracer.py`. Production
must not import this benchmark package.

Arrange replay results as
`<input>/<candidate>/<replay>/{report.json,telemetry.jsonl}`, then run:

```powershell
$python = '.tools\standalone-release-py31210\Scripts\python.exe'
& $python -m benchmarks.localization.build_realtime_control_report `
  artifacts\poc\localization-candidates-YYYYMMDD `
  artifacts\poc\localization-realtime-control-YYYYMMDD
```

The output is a machine-readable JSON file and a narrow-screen `REPORT.txt`.
The report recommends only candidates with independent post-run reference
coverage; unlabeled holdouts can support latency and availability but cannot
prove localization accuracy.

For sessions that have only a sparse offline atlas anchor, add several matcher
families under the same candidate/replay layout and build the consistency
report:

```powershell
& $python -m benchmarks.localization.build_cross_session_report `
  artifacts\poc\localization-cross-session-candidates-YYYYMMDD `
  artifacts\poc\localization-cross-session-report-YYYYMMDD
```

This report uses the offline anchor and a leave-one-family-out median. A family
may vote only when its worst anchor disagreement stays inside its configured
local search basin; availability is reported separately so a sparse but precise
method can corroborate overlapping frames. Search radii of the same
gradient-correlation implementation count as one family, so minor parameter
variants cannot manufacture consensus. Held outputs do not vote. The report
also separates wrong map-scale/mode fixes from same-mode localization error.
Agreement is explicitly not labeled as external ground-truth accuracy.

Record repeated traversals for direct cross-pass repeatability evidence using
the [route repeatability session protocol](ROUTE_REPEATABILITY_SESSION.md).
The [method and findings report](REPEATABILITY_BENCHMARK.md) defines the
reusable contracts, causal boundaries, metrics, reproduction commands, and
current Run 17/18 evidence.
`build_repeatability_report.py` discovers recurring complete-state waypoints
without lap boundaries and subtracts reference path variation before scoring.
`benchmarks.temporal_turns` diagnoses temporal hysteresis at sharp reversals by
fitting sign and time lag between independent control, KLT scene, and candidate
signals instead of assuming equal frame indexes.
`benchmarks.build_turn_evidence` extracts those three timestamped channels from
a recorded session. Cursor pose is optional; raw input and KLT remain useful as
independent behavior evidence when no pose candidate is supplied.
