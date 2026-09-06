# User-verified route tracing — September 6, 2026, 19:27

The user reports successful route tracing and requests commit, tag and push.
Tag: `route-tracking-known-good-20260906-1927`. Preserve the earlier
`route-tracking-known-good-20260906` tag.

The verified route-assisted run is `20260906T112701016685Z-d55ff98b`, from
19:27:01 to 19:28:10 Asia/Taipei. It completed without a runtime error and stopped
with `route-finish-confirmed`. All three required current-image endpoint
confirmations passed, the endpoint mode matched, and the final distance was
11.130 px inside the existing 14 px gate. The user verification establishes
route completion, not flawless localization, 30 Hz delivery or autonomous cruise.

The following run, `20260906T112827531557Z-ba1179df`, is a separate free-roam
unknown-start test from 19:28:27 to 19:30:32. The user reports good outdoor tracking
and awful town tracking, and describes live tracking as cumbersome.

## What the saved evidence establishes

The route run has 1378 telemetry rows with no dropped evidence records: 1350 fresh
XY outputs, 28 held at the trained transition, and zero unavailable poses.
Across its 66.955 s telemetry span, observed publication-stream capture cadence
is 20.57 updates/s. Capture-to-publication median/P95/worst is
54.673/150.828/509.529 ms. Only 30 of 1378 saved rows have joint fresh XY and
heading delivered within 33.3 ms.

Its world and town sections are each contiguous; the labels below are the
tracker's published map mode. These cadence values measure processed observations,
not independently measured game-render FPS.

| Route section | Processed updates/s | Publication median / P95 ms | Engine P95 ms | Fresh / held |
|---|---:|---:|---:|---:|
| World/outdoor mode | 24.50 | 50.431 / 92.424 | 37.889 | 1032 / 28 |
| Town mode | 13.40 | 100.119 / 228.258 | 100.841 | 318 / 0 |

The sluggishness therefore includes both reduced update rate and delayed output.
Time outside the engine also matters: publication age minus engine elapsed time
has median/P95 42.071/102.729 ms overall. This residual does not isolate capture,
queueing and publication; it must not be called capture duration without further
instrumentation. Route-video encoding reports 491 dropped and 1102 repeated
frames, so its encoded 30 FPS is not a measurement of live update rate.

The free-roam run processed 3242 frames but retained only 2620 telemetry rows,
with 622 evidence records dropped. Its saved town-mode rows contain 949 fresh
outputs and 168 `held:continuity-jump` outputs. Holds recur across the sustained
town interval, including relative windows 40–50, 50–60, 70–80 and 80–90 seconds.
This is not just a brief transition problem. Dropped evidence prevents complete
freshness/loss-duration claims, and accepted outputs are not independent proof
of accurate town position. No new localization fix is claimed in this milestone.

## Inputs and scope

Both runs use atlas `08b6f2d6-820a-4bfd-875a-6a55d1986a4e`, minimap calibration
`segments-df624035-833-bd07601f-708`, scene-yaw calibration
`01dbaa74-8e00-4763-a215-9ea37e18b1b2`, and the real-time profile. The verified
route uses package `362f9d53-e80f-4eeb-9cdb-f89959835cd2-72904af0` and demonstrated
startup with current-frame map correlation. The free-roam run has no route package.

The running process does not embed its Git revision. This tag marks the current
repository checkpoint associated with the user's verification, not a recovered
exact process binary identity. Recent parallel-scale and replay-wait experiments
remain outside normal live tracking; they are not credited with this success.

Original manifests, telemetry, event frames and video remain under
`artifacts/workbench/live_tracking/genshin-impact-pc/<tracking-id>/`.
The read-only review, input SHA-256 identities and script are saved under
`artifacts/poc/live-review-20260906T1127/`.

Next diagnosis priorities: sustained live capture-to-publication delay and town
localization on native frames, including why current-image proposals jump while
town mode is already selected. Preserve the successful outdoor behavior and
route completion; do not infer that weakening continuity checks fixes town XY.

Type: release documentation and evidence review. Verification: user-reported
pass, complete route manifest, 3/3 finish metadata, raw telemetry and timing
analysis. Scope: no runtime, atlas or route-compilation changes.
