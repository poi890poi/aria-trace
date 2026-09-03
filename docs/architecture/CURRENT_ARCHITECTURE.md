# Current architecture

This document describes the implemented architecture after the 2026-08-31
structural refactor. It is the directory-ownership reference for new work.

## Dependency direction

```text
apps -> workflows -> services -> domain
          |             |
          v             v
       adapters -----> ports

experiments/POC -> production packages (allowed)
production packages -> experiments/POC (forbidden)
```

The repository has two product boundaries: `rig_runtime/` owns the neutral IRIS
acquisition/calibration runtime, while `aria_trace/` owns integrated gameplay
tracing and tracking. Neither package is hidden behind a `src/` wrapper.

## Package ownership

| Package | Owns | Must not own |
| --- | --- | --- |
| `rig_runtime/domain` | Neutral schemas for identity, time, space, provenance, quality, execution and portable records | Trace-product state, devices, files or UI |
| `rig_runtime/ports` | Replaceable source/transform/estimator/sink/repository/control protocols | Implementations or workspace discovery |
| `rig_runtime/services` | Rig, mini-map, cursor and scene calibration plus reusable vision mechanisms | Workbench, mapping, localization or tracking policy |
| `rig_runtime/adapters` | Android, HIK, rig-device and filesystem integrations | Integrated gameplay tracing or tracking |
| `rig_runtime/workflows` | IRIS recording, calibration, profile and evidence orchestration | Workbench/route/teleport orchestration |
| `rig_runtime/evidence` | Neutral calibration and space-aware media evidence | Trace-product workflow decisions |
| `rig_runtime/apps` | IRIS HIK commands and the standalone rig calibrator | Workbench, review or tracker UI |
| `aria_trace` | Workbench, recorder/review, Windows capture, mapping, localization, route/teleport and realtime tracking | IRIS runtime implementation |
| `acquisition` | Compatibility import and CLI facades only | New implementation |
| `poc` | Experiments and retained compatibility aliases to promoted production services | Dependencies from production code |

## Implemented capability clusters

```text
rig_runtime/
  domain/                 neutral contracts and calibration records
  ports/                  replaceable component interfaces
  adapters/
    android/               capture, control, display and phone integration
    hik/                   MVS driver and calibrated camera integration
    rig/                   dual-source/device composition
    filesystem/            sessions, annotations, video and profile stores
  services/
    calibration/
      minimap/             single canonical boundary/transition service
      cursor/              dynamics, rigid shape, pose and worker
      scene_yaw/           angular-scale calibration
      rig/                 device-independent geometry and measurements
    vision/                reusable visual estimators
  workflows/              recording, calibration and profile use cases
  evidence/               calibration, feature and rig evidence builders
  apps/
    rig_calibrator/        standalone Windows GUI

aria_trace/
  adapters/windows.py      desktop gameplay capture
  services/
    mapping/               stitching, layers and references
    localization/          route and teleport mechanisms
    tracking/              runtime, fusion and performance profiles
  workflows/              route, teleport and input-verification use cases
  evidence/               tracking and integrated POC evidence
  apps/
    workbench/             API, jobs, state, catalog, capture and live tracking
    record/review          integrated trace acquisition and inspection
```

## Universal data flow

Sources produce typed packets or envelopes with explicit timing and image-space
identity. Workflows validate capabilities and select inputs. Services calculate
results from those declared inputs. Evidence code renders diagnostics without
changing decisions. Filesystem adapters persist results and trace records.

```text
device/session source
  -> adapter packet
  -> workflow selection and capability validation
  -> service input
  -> service result
  -> evidence records + artifact references
  -> repository/application response
```

Callers—not calibration algorithms—select frames and semantic segment roles.
The mini-map boundary service therefore has no knowledge of Android, HIK,
zigzag acquisition, sessions or the Workbench.

## Configuration and mutable data

- `config/games/` and `config/routes/` are source-controlled definitions.
- `profiles/phone_game/`, `profiles/rig/`, `profiles/rig_game/`, and
  `profiles/rig_game_color/` are mutable,
  machine/device-specific profile stores.
- `profiles/.registry/settings.{json,yaml}` stores operator defaults. One named
  rig-repeatability policy owns both reuse and GUI-save gates.
- `profiles/calibrations/` stores source rig bundles when no explicit output is
  supplied; production resolution still reads immutable runtime copies from
  profile revisions.
- `sessions/` is operator working data.
- `artifacts/` is generated analysis, evidence and compiled data.

Code must not guess between these roots. Resolvers receive their root explicitly
or use the one documented default for that data class.

## Canonical entry points

```powershell
python -m aria_trace.apps.workbench
python -m rig_runtime.apps.rig_calibrator
python -m rig_runtime.apps.hik_rig_calibration
python -m rig_runtime.apps.hik_stream
python -m iris_tools setup show
```

Compatibility commands remain available as documented in
[`COMPATIBILITY.md`](COMPATIBILITY.md), but new code and documentation use the
canonical owners above.

## Enforced boundaries

`tests/test_architecture_boundaries.py` verifies:

- neutral domain and ports do not depend on legacy/platform packages;
- `rig_runtime` never imports the `aria_trace` product;
- the standalone IRIS release exports `rig_runtime` and no `aria_trace` code;
- production does not import `poc`;
- Workbench shell contains no calibration/mapping algorithms;
- HIK calibration services do not import hardware adapters, workflows or apps;
- legacy symbols are exact aliases to canonical owners;
- source definitions are separate from mutable profiles.

The canonical non-hardware regression command is:

```powershell
E:\workspace\aria-trace\.tools\standalone-release-py31210\Scripts\python.exe -m unittest discover -s tests
```

Hardware verification is opt-in and must identify and exclusively own the
camera, phone or input adapter it uses.
