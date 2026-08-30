# Reference benchmarks

These numbers are references, not acceptance thresholds. They were measured on
the development HIK/Android rig on 2026-08-30 and are traceable to repository
artifacts. The rig is offline for this release build, so no values below were
re-measured or silently replaced.

Source calibration and adapter run:
`artifacts/rig-adapter-stress-20260830-183419/run-01/`.

| Critical node | Reference result | Scope |
|---|---:|---|
| Complete headless rig calibration | 144.69 s | One full run including target presentation, imaging, geometry, benchmarks, and save |
| ChArUco verification | 12/12 frames | Final geometry frames |
| Reprojection error p95 | 1.093 camera px | Saved rig geometry |
| Hardware ROI payload reduction | 19.7% | 1156×1080 ROI versus 1440×1080 sensor |
| HIK adapter read p50 / p95 | 31.0 / 47.0 ms | 24 reads after hardware ROI |
| Camera frame interval p50 / p95 | 33.62 / 37.81 ms | 23 intervals |
| Adapter construction | 20.34 ms | Registry/profile resolution included |
| Adapter open | 653.99 ms | Camera open and locked configuration |
| Adapter per-frame read p50 / p95 | 21.74 / 31.12 ms | Three-frame smoke cycle |
| Adapter observed throughput | 47.48 fps | Short three-frame read sequence; not a sustained-rate claim |
| Adapter close | 64.83 ms | Camera release |
| Display request to stable camera response p50 / p95 | 1065.63 / 1082.85 ms | Two accepted trials; reference-only, host clock |

The response-latency metric is deliberately not an ADB clock subtraction. It
uses one host monotonic clock from display request to camera observation. ADB
acknowledgement timestamps remain telemetry only.

MVS fused Bayer gamma/color conversion reference:
`artifacts/mvs-selected-bayer-conversion-probe-20260830/results.json`.

| Feature | Identity | Selected gamma + CCM |
|---|---:|---:|
| Median read p50 | 33.354 ms | 33.286 ms |
| Median read p95 | 35.496 ms | 35.664 ms |

The measured difference is within run variation and supports the design claim
that MVS performs gamma/color conversion inside its existing Bayer-to-BGR path,
without another Python image pass.

## What must be benchmarked on a release machine

- Rig calibration wall time and reprojection distribution.
- Sustained adapter frame interval for each `full`, `minimap`, and `dual` mode.
- Rectification on/off read time at the actual saved ROI dimensions.
- Zigzag capture frame counts, drop counts, ADB clock fit, and video duration.
- Mini-map calibration wall time and evidence quality on a representative
  session.

Never reuse a reference number as a gate without first measuring its variance
on the target camera, phone, USB controller, CPU, and game display mode.
