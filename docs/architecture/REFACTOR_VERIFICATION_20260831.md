# Refactor verification - 2026-08-31

## Scope

This verification covers the completed structural migration from the
`acquisition` monolith to the layered `aria_trace` packages. It does not change
map-stitching matching behavior, rig feature metrics, or live-tracker timing.

## Environment

```text
Runtime: .tools/standalone-release-py31210/Scripts/python.exe
Command: python -m unittest discover -s tests
Hardware opened: no physical hardware
Tests discovered: 405
Elapsed: 37.112 seconds
```

## Result

- 402 passed in the combined run.
- 2 deterministic failures are identical to the pre-refactor baseline in
  [`BASELINE_20260830.md`](BASELINE_20260830.md): map-stitch oriented-gradient
  polarity chooses `(50, 20)` instead of `(5, 20)`, and rig feature
  repeatability is `0.7397899649941657` instead of greater than `0.90`.
- 1 Workbench live-tracker telemetry timing assertion failed only in the full
  combined run (`telemetry_rows == 0`). It passed immediately in an isolated
  fresh process (`1/1`, 0.188 seconds). This remains a documented order/timing
  flake and is not hidden or changed as part of the refactor.
- The focused HIK/rig and architecture suite passed `134/134` after the final
  dependency split.

No additional deterministic failure was introduced. The two algorithm failures
must be diagnosed as separate evidence-backed behavior changes. The telemetry
test requires its own timing/root-cause investigation rather than a weakened
assertion.

## Structural checks

- `git diff --check` passed before each refactor/documentation commit.
- Production imports from `acquisition` and `poc` were scanned and none remain
  under `aria_trace` or `replay`.
- Architecture tests enforce compatibility identity and dependency direction.
- Concurrent mutable rig/profile data and the local stress script were not
  staged or modified.
