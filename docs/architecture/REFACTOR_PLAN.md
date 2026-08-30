# AriaTrace Refactor Plan

## Status and rollback point

This plan starts from the verified repository state tagged
`teleport-learning-known-good-20260830` at commit `2e212fb`.

The refactor is structural unless a commit explicitly declares a behavior or
performance change. Current acquisition formats, command-line entry points,
Workbench HTTP endpoints, profile resolution, calibration artifacts, and the
`hikcam` compatibility facade remain supported during migration.

## Why this refactor is required

The design documents describe replaceable capture, calibration, localization,
tracking, replay, and control components, but the implementation has accumulated
several conflicting ownership models:

- `acquisition` owns hardware, storage, calibration, analysis, tracking,
  rendering, HTTP transport, and application state;
- `acquisition/workbench.py` coordinates nearly every workflow and also contains
  behavior that belongs to those workflows;
- production tracking and scene-yaw code import mechanisms from `poc`;
- replay code imports acquisition storage rather than a neutral session contract;
- algorithms commonly select inputs, calculate results, write artifacts, and
  render evidence in one call;
- mutable dictionaries carry time, space, quality, and provenance differently
  between components;
- compatibility fallbacks and implicit defaults obscure which path produced a
  result;
- runtime profiles, source-controlled configuration, and generated evidence are
  mixed in the same hierarchy;
- tests cover considerable behavior but are concentrated around internal details
  of large modules instead of stable component contracts.

Moving files without correcting these boundaries would preserve the fragility.
The primary goal is therefore explicit and universal data flow.

## Required invariants

1. An algorithm receives all required information through its declared input.
2. An algorithm does not discover devices, sessions, profiles, files, or UI state.
3. A caller selects frames and source roles. Calibration does not infer an
   acquisition protocol.
4. Time and coordinate spaces are explicit. Pixels or timestamps from different
   spaces cannot be combined without a referenced transform.
5. Raw measurements remain distinct from accepted, fused, or corrected state.
6. Quality records observations and checks, not an unexplained confidence scalar.
7. Every result identifies the component, configuration, source evidence, code,
   and dependency versions that produced it.
8. Evidence rendering and artifact persistence do not change the algorithm result.
9. An optional input cannot silently become mandatory. Capability requirements are
   declared and validated before execution.
10. A fallback is named in the result and has an explicit eligibility rule. It is
    never selected because an unrelated exception happened.
11. Production packages never import experimental packages.
12. Slow analysis, UI, or persistence work cannot block the latest-frame tracking
    loop.

## Universal component input and output

AriaTrace should not force every component into one synchronous method. A camera,
an image transform, a stateful tracker, and an artifact sink have different
lifecycle needs. They should instead share one universal envelope and a small set
of port families.

### Envelope

```text
Envelope[T]
  schema                 Stable payload schema ID and version
  value                  Typed component-specific payload
  identity               Envelope ID and optional parent IDs
  timing                 Capture/source/receive/publication times with clock IDs
  space                  Raster or coordinate-space ID and transform references
  provenance             Producer, configuration, code, dependencies, source refs
  quality                Named measurements, checks, warnings, and decision
  diagnostics            Structured diagnostic values and artifact references
```

The envelope metadata is immutable. Large image arrays remain referenced or shared
in memory; provenance is a chain of identifiers rather than recursively copied
documents. Serialization is owned by codecs and artifact stores, not by domain
objects.

### Port families

```text
Source[T]       start() / read() -> Envelope[T] / stop()
Transform[I,O] process(Envelope[I], Context) -> Envelope[O]
Estimator[I,O] update(Envelope[I], Context) -> Envelope[O]
Sink[T]         write(Envelope[T]) -> ArtifactRef
Repository[T]   resolve(query) / load(ref) / save(value)
ControlBackend  connect() / submit(intent) / release_all() / disconnect()
```

`Context` carries trace ID, cancellation, an explicit immutable configuration,
and services such as clocks. It must not expose Workbench state or a workspace
root.

### Compatibility and capability declaration

Components declare requirements such as accepted payload schema, image channel
order, required coordinate space, required calibration capability, and optional
inputs. A workflow validates these requirements before starting. This replaces
late failures caused by implicit assumptions about source, device, crop, scale,
orientation, session label, map revision, or artifact layout.

## Target package hierarchy

```text
src/aria_trace/
  domain/
    envelope.py
    identity.py
    timing.py
    spaces.py
    quality.py
    provenance.py
    artifacts.py
    errors.py

  ports/
    sources.py
    transforms.py
    repositories.py
    control.py

  services/
    capture/
    calibration/
      rig/
      minimap/
      cursor/
      scene_yaw/
    mapping/
      stitching/
      layers/
      transitions/
    localization/
      global_map/
      local_motion/
      route/
      teleport/
    tracking/
    demonstration/

  adapters/
    android/
    windows/
    hik/
    opencv/
    filesystem/

  workflows/
    recording.py
    calibration.py
    mapping.py
    teleport.py
    route.py
    tracking.py

  evidence/
    schemas/
    recorder.py
    renderers/

  apps/
    workbench/
      api/
      jobs/
      state/
      static/
    rig_calibrator/

experiments/
tests/
  unit/
  contract/
  integration/
  replay/
  hardware/
```

Dependencies point inward:

```text
Apps -> Workflows -> Services -> Domain
             |          |
             v          v
            Ports <- Adapters
```

- Domain has no OpenCV, filesystem, network, process, UI, or device dependency.
- Services may depend on domain and declared ports.
- Adapters implement ports and may depend on platform libraries.
- Workflows compose services and adapters but contain no estimation algorithms.
- Apps translate HTTP/GUI requests into workflow calls.
- Experiments may import production components; production cannot import
  experiments.

## Fundamental capabilities to freeze

The following behaviors need named characterization suites before extraction:

1. Session write/read round trip, raw input integrity, frame timing, and incomplete
   session recovery.
2. Android, Windows, HIK, and calibrated dual-source lifecycle and safe shutdown.
3. Clock mappings and all declared raster/coordinate-space transforms.
4. Mini-map boundary, cursor center, rigid cursor shape, cursor pose, and dynamics.
5. Scene-yaw angular scale and closure quality.
6. Map stitching, localization derivative construction, observed coverage, layers,
   and scale transitions.
7. Global initialization, continuous local tracking, asynchronous correction, and
   performance profiles.
8. Demonstrated-route compilation, proposal generation, route-independent
   estimation, and post-run route similarity.
9. Teleport input-phase parsing, target selection, post-load arrival consensus,
   and learned artifact persistence.
10. Workbench job state, cancellation, restart recovery, API compatibility, and
    overlay lifecycle.
11. Control timeout/error behavior and unconditional input release.

Golden data must be small, immutable, and identified by content hash. Development
data may guide diagnosis; at least one independent recording must validate any
behavior or performance change.

## Migration sequence

### Phase 0 - Baseline and safeguards

- preserve a known-good tag and backup mutable Workbench data;
- inventory public imports, CLIs, HTTP endpoints, profile paths, and artifact
  schemas;
- add characterization tests and baseline timing for the fundamental capabilities;
- record expected compatibility surfaces and removal criteria.

### Phase 1 - Project spine

- introduce `src/aria_trace/domain` and `src/aria_trace/ports`;
- add the universal envelope, timing, space, provenance, quality, diagnostics,
  artifact reference, context, and component error contracts;
- add codecs between new contracts and current dictionaries;
- introduce project metadata and one canonical test command;
- add architecture tests that reject forbidden dependency directions.

No production algorithm moves in this phase.

### Phase 2 - One vertical proof

Use mini-map boundary calibration as the first vertical extraction because its
intended boundary is already clear: the caller supplies an image stack and the
algorithm returns a boundary model.

- wrap the current unchanged algorithm in a typed transform;
- move frame/session selection into a workflow adapter;
- move evidence persistence into an artifact sink;
- compare the old and new entry points on the same verified frame arrays;
- retain the old import as a compatibility facade.

The proof passes only if numerical results and native evidence remain equivalent.

### Phase 3 - Workbench decomposition

Extract, without changing endpoints:

1. session and artifact catalog;
2. source and adapter selection;
3. capture workflow;
4. analysis job state machine;
5. artifact compatibility service;
6. live-tracker supervisor;
7. HTTP request/response translation;
8. UI state serialization.

Workbench becomes an application shell. It cannot import OpenCV algorithms.

### Phase 4 - Algorithm extraction

Move one independently reviewable capability per commit:

1. cursor pose and dynamics;
2. scene yaw;
3. map stitching and localization derivative;
4. map layers and transitions;
5. global and local localization;
6. route compilation and route assistance;
7. teleport analysis.

For each capability: freeze current behavior, extract the pure core, add typed
ports, compare old/new outputs, migrate callers, then retain or remove the facade
according to the compatibility policy.

### Phase 5 - Tracking runtime

Split the live tracker into independently scheduled components:

- latest-frame ingestion;
- mini-map extraction;
- local translation;
- cursor pose;
- global localization;
- route proposal;
- continuity/fusion gate;
- tracker state machine;
- evidence publication;
- overlay rendering.

Messages carry source time and publication time. A held estimate is distinct from
a fresh estimate. Expensive global, representation, persistence, and rendering
work stays off the high-rate loop.

### Phase 6 - Data and profile separation

- keep source-controlled game and route definitions under `config/`;
- keep mutable device, rig, and game-device profiles under a workspace data root;
- keep generated analysis under an artifact root;
- keep only compact, licensed, content-addressed evidence under `tests/fixtures`;
- migrate current data with manifests and checksums rather than path guessing.

### Phase 7 - Compatibility removal

Only after callers, artifacts, packaged applications, and migration tests prove the
new boundary:

- remove production imports from `poc`;
- remove obsolete stores and implicit path resolution;
- remove transitional dictionary and import facades;
- consolidate packaging and dependency declarations;
- archive superseded documentation and entry points.

## Test strategy

### Unit

Pure algorithms with deterministic arrays and typed values. No filesystem, devices,
Workbench, or subprocesses.

### Contract

Every implementation of a port runs the same lifecycle, timing, capability, error,
and shutdown tests. Artifact codecs round-trip current and new schemas.

### Integration

Workflows run against fake sources, in-memory repositories, and temporary artifact
stores. HTTP tests assert public behavior rather than Workbench internals.

### Replay

Recorded, content-addressed samples exercise initialization, steady-state tracking,
transitions, route assistance, loss, and recovery. Reports separate fresh, held,
unavailable, and rejected measurements.

### Hardware

Hardware tests are opt-in, exclusive, and identify the claimed camera/phone/input
adapter. They never run as an unannounced consequence of unit or integration tests.

### Packaged applications

PyInstaller smoke tests verify resources, imports, profile resolution, and clean
startup without opening devices unless requested.

## Traceability and accountability

Every workflow has a trace ID. Every component invocation records:

- component and algorithm ID;
- component version and Git revision;
- configuration ID and content hash;
- input envelope IDs and source artifact hashes;
- clock and coordinate-space IDs;
- start, completion, and publication time;
- status: queued, running, completed, canceled, rejected, or failed;
- quality checks and explicit decision;
- structured error type and causal chain;
- output and diagnostic artifact references.

Every refactor commit documents type, intended outcome, affected and unaffected
boundaries, risk, rollback, and verification. Architecture decisions live under
`docs/architecture/decisions/`. Rejected experiments remain in evidence reports;
they are not silently deleted.

## Change gates

A migration slice may land only when:

- old and new behavior have been compared on named inputs;
- schema and public API compatibility are tested;
- no new reverse dependency is introduced;
- algorithms have no new implicit I/O or global state;
- timing and memory do not regress beyond a declared tolerance;
- failure, cancellation, and safe shutdown behavior is tested;
- generated evidence identifies which implementation path ran;
- unrelated concurrent work is absent from the commit.

## Completion criteria

- production imports form the declared inward dependency graph;
- Workbench owns presentation and orchestration, not algorithms;
- core algorithms accept explicit typed inputs and perform no hidden discovery or
  persistence;
- every result is traceable through time, space, provenance, quality, and evidence;
- optional data remains optional and capability validation happens before work;
- mutable runtime data is separated from source-controlled configuration;
- fundamental replay/calibration suites pass through both source and packaged entry
  points;
- compatibility facades have named owners, expiry conditions, and migration tests;
- production contains no dependency on `poc` or `experiments`.
