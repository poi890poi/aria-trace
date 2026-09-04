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
