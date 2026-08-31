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

The repository uses `aria_trace/` directly rather than a `src/` wrapper.

## Package ownership

| Package | Owns | Must not own |
| --- | --- | --- |
| `aria_trace/domain` | Platform-neutral schemas for identity, time, space, provenance, quality, execution and portable records | OpenCV, NumPy, devices, files, UI, Workbench state |
| `aria_trace/ports` | Component lifecycle and source/transform/estimator/sink/repository/control protocols | Implementations or workspace discovery |
| `aria_trace/services` | Calibration, mapping, localization, tracking and vision mechanisms | Device discovery, HTTP/GUI state, session selection |
| `aria_trace/adapters` | Android, Windows, HIK, rig-device and filesystem integrations | Product workflow policy or estimation algorithms |
| `aria_trace/workflows` | Explicit composition of services, repositories and adapters | HTTP/GUI translation or new estimation math |
| `aria_trace/evidence` | Deterministic evidence calculation, serialization and review records | Input selection or workflow decisions |
| `aria_trace/apps` | Workbench, recorder, review, HIK CLI and rig-calibrator entry points | Calibration/localization algorithms |
| `acquisition` | Compatibility import and CLI facades only | New implementation |
| `poc` | Experiments and retained compatibility aliases to promoted production services | Dependencies from production code |

## Implemented capability clusters

```text
aria_trace/
  domain/                 universal contracts and gameplay records
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
    mapping/               stitching, layers and references
    localization/          route and teleport mechanisms
    tracking/              runtime, fusion and performance profiles
    vision/                reusable visual estimators
  workflows/              recording, calibration, route and teleport use cases
  evidence/               trace, feature and rig evidence builders
  apps/
    workbench/             API, jobs, state, catalog, capture and live tracking
    rig_calibrator/        standalone Windows GUI
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
- `sessions/` is operator working data.
- `artifacts/` is generated analysis, evidence and compiled data.

Code must not guess between these roots. Resolvers receive their root explicitly
or use the one documented default for that data class.

## Canonical entry points

```powershell
python -m aria_trace.apps.workbench
python -m aria_trace.apps.rig_calibrator
python -m aria_trace.apps.hik_rig_calibration
python -m aria_trace.apps.hik_stream
```

Compatibility commands remain available as documented in
[`COMPATIBILITY.md`](COMPATIBILITY.md), but new code and documentation use the
canonical owners above.

## Enforced boundaries

`tests/test_architecture_boundaries.py` verifies:

- domain and ports do not depend on legacy/platform packages;
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
