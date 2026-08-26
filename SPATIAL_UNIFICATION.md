# AriaTrace Spatial and Temporal Unification

## 1. Purpose

AriaTrace receives spatial evidence from phone screenshots, physical cameras, mini-maps, stitched area/global maps, and recordings made by different devices and sessions. These sources must interoperate without forcing any producer to know the components that may consume its output later.

Spatial unification is therefore a graph of named coordinate frames connected by explicit, directed, quality-bearing transforms. It is not one mandatory global bitmap and it does not rewrite source evidence into a supposedly universal coordinate system at ingestion time.

The design must support:

- canonical phone-screen coordinates;
- raw, undistorted, and normalized camera images;
- phone/ADB and physical-camera mini-map observations;
- stitched local-area, regional, and global maps;
- dynamic mini-map-to-map pose estimates;
- recordings imported from other devices, rigs, resolutions, orientations, and sessions;
- clock alignment and measured causal latency;
- uncertainty, partial validity, multiple hypotheses, and complete provenance.

## 2. Architectural Rule

Every producer exports a self-describing fragment:

```text
coordinate frames
    + directed transform observations
    + valid regions and time ranges
    + uncertainty and confidence
    + source/evidence provenance
```

The producer has no dependency on the central graph or on peer producers. A registry adapter validates and imports fragments. Consumers query the registry by frame ID and time; they do not reach into calibration, stitching, mini-map, or dataset implementation internals.

```text
Rig calibration -----------+
Mini-map calibration ------+
Map stitcher --------------+--> fragment validator --> spatial registry --> query/resolver
Live pose estimator -------+
Dataset importer ----------+
Timing calibration --------+
```

This keeps rig calibration independently testable. It can produce a correct camera-to-phone transform and latency profile even when no game profile, mini-map extractor, global map, or recorded route exists.

## 3. Frame Graph, Not Implicit Conventions

A **frame** defines a coordinate system. A **transform observation** maps coordinates from one frame to another. Identity or compatibility must never be inferred only because two rasters happen to have the same dimensions.

Suggested stable frame identifiers are URI-like strings:

```text
aria://rig/<rig-id>/camera/raw
aria://rig/<rig-id>/camera/undistorted
aria://device/<device-id>/screen/<orientation>/<layout-revision>
aria://game/<game-profile-id>/minimap/local
aria://artifact/<map-id>/area/<area-id>
aria://artifact/<map-id>/global
aria://dataset/<dataset-id>/session/<session-id>/camera/<stream-id>
aria://dataset/<dataset-id>/session/<session-id>/minimap
```

IDs name frames; metadata describes them. Names do not grant compatibility.

### 3.1 Raster coordinate convention

For image frames:

- `(0, 0)` is the centre of the top-left pixel.
- positive X points right;
- positive Y points down;
- integer coordinates address pixel centres;
- the continuous raster extent is `[-0.5, width-0.5] x [-0.5, height-0.5]`.

This matches OpenCV point and homography conventions and removes half-pixel ambiguity. Phone-screen coordinates use the same convention as the declared ADB screenshot orientation.

Map frames may instead use metres or explicit `map_unit`s and may use Y-up. Their axis directions and handedness must be declared rather than normalized by assumption.

### 3.2 Frame descriptor

Every frame declares:

- stable frame ID and revision;
- dimensions: normally 2D for current planar work;
- frame type: raster, planar metric, or abstract planar map;
- units;
- origin and axis directions;
- raster size and pixel-centre convention when applicable;
- optional valid mask or boundary;
- owner artifact and provenance;
- compatibility keys such as phone orientation, game/map revision, and camera mode.

## 4. Directed Transform Contract

For a planar transform from frame A to frame B:

```text
[x_B' y_B' w_B']^T = matrix_3x3 * [x_A y_A 1]^T
x_B = x_B' / w_B'
y_B = y_B' / w_B'
```

Every transform declares:

- `from_frame` and `to_frame`;
- transform model and matrix direction;
- static, piecewise-static, or time-varying behavior;
- source timestamp and validity interval;
- source-clock ID;
- valid domain polygon/mask and expected overlap;
- covariance or another explicit uncertainty model;
- confidence and quality measurements;
- estimator name, version, configuration hash, and evidence references;
- status: proposed, accepted, warning, rejected, or superseded.

Supported initial models are:

- `translation_2d`;
- `rigid_2d` or SE(2);
- `similarity_2d` or Sim(2);
- `affine_2d`;
- `projective_2d`;
- `raster_remap`, referenced as a preprocessor rather than masquerading as a 3x3 matrix.

The simplest model supported by evidence should be preferred. A similarity transform is preferable to a projective transform for map alignment when scale, rotation, and translation explain the data.

## 5. Static, Dynamic, and Hypothesis Edges

### Static edges

Examples:

- undistorted camera image to canonical phone screen;
- phone screen to a fixed mini-map crop;
- stitched area-map raster to its area-map coordinates;
- area-map coordinates to a global-map frame after accepted registration.

### Dynamic edges

Examples:

- current mini-map content to a stitched area map;
- live camera mini-map to a route-local map;
- a dataset session's moving observation to a global map.

Dynamic transforms are timestamped samples or intervals. Interpolation is allowed only when the model and maximum gap are declared.

### Multiple hypotheses

Map matching can be ambiguous. The registry must preserve a weighted set of hypotheses instead of publishing only the largest peak as truth. A hypothesis set includes:

- a shared source observation;
- candidate transform matrices;
- normalized weights or likelihood scores;
- candidate-specific validity and uncertainty;
- ambiguity metrics;
- the evidence required to accept or reject candidates later.

A consumer may request `accepted_only`, `best_hypothesis`, or the full hypothesis set. The default safety behavior is not to collapse an unresolved set silently.

## 6. Spatial Layers

### 6.1 Phone screen

The phone screen is the principal device-level canonical frame. Its resolution, orientation, system-bar policy, display scaling, and layout revision form part of the frame identity.

ADB screenshots normally arrive directly in this frame. If an ADB source changes orientation or includes different system insets, it exports a different frame plus an explicit transform or crop to the desired canonical screen.

### 6.2 Camera image

Rig calibration exports:

- raw and undistorted camera frame descriptors;
- the lens-remap relationship;
- an undistorted-camera-to-normalized-output matrix;
- explicit normalized-output origin and scale in phone-screen coordinates;
- valid mask and uncertainty over the required region.

The spatial adapter may register a direct camera-undistorted-to-phone-screen transform. The rig module itself does not know why the phone coordinates will be used.

### 6.3 Mini-map observations

The game-specific mini-map calibrator distinguishes at least three concepts:

- a source-specific `minimap_view_px` raster cropped from an ADB/phone or camera-derived observation;
- a calibrated local observation frame, often cursor-centred and masked, into which source views can be normalized;
- dynamic map-content coordinates whose relationship to an area/global map changes with pose.

It exports static relationships among the screen, source view, and calibrated local observation frames. It declares:

- crop or boundary geometry;
- circular/shape mask;
- cursor rotation centre;
- local origin, scale, and orientation;
- whether content rotates, translates, scales, or changes projection;
- calibration and game-layout revision.

An ADB mini-map and a camera-derived mini-map remain distinct source observation frames. They become comparable through their respective edges to the same canonical phone-screen and calibrated local definitions. Their pixels must not be equated merely because the crop sizes match. Mini-map-to-area/global-map pose is a separate dynamic hypothesis edge, never folded into the static crop transform.

### 6.4 Stitched area and global maps

A stitcher exports one frame per immutable map artifact or revision. Source tiles retain their own frames and tile-to-mosaic transforms. An area map links to a global map only through an evidence-backed transform.

The contract preserves:

- map unit and axis convention;
- raster resolution or metres/map-units per pixel;
- covered polygon and holes;
- source tile transforms;
- inaccessible or unknown regions;
- layer, floor, region, and map revision;
- stitch residuals and spatially varying confidence.

There is no requirement that a global map be north-up, metric, or complete. Those properties are declared only when measured or supplied.

### 6.5 Imported devices, sessions, and datasets

An importer creates a namespace for every immutable dataset and session. It preserves original pixels, timestamps, calibration references, and coordinate conventions. Registration to current frames is stored as new graph edges; source files are not rewritten.

Cross-session or cross-device alignment normally proceeds through evidence:

```text
dataset camera frame
    -> dataset phone screen, if its rig calibration exists
    -> dataset mini-map observation
    -> area/global map candidate transforms
    -> current session or route frame through the accepted map relationship
```

If old data lacks rig calibration, direct image registration may publish a lower-confidence camera/dataset-to-map edge. Its provenance must say that canonical screen geometry was unavailable.

## 7. Time and Latency Are Part of the Contract

Spatial transforms are meaningful only at known times. Every timestamp names its clock. The system maintains two different concepts.

### Clock transforms

A clock transform maps timestamp coordinates:

```text
t_destination = scale * t_source + offset
```

It may include uncertainty and a validity interval because offset and drift can change.

### Causal latency

Latency describes how long an action or physical state change takes to become observable. It is a distribution, not a clock offset:

```text
observation_time = mapped_event_time + causal_latency
```

Examples include:

- control issue to phone presentation;
- phone presentation to camera exposure;
- camera exposure to host receive;
- host receive to normalized observation;
- complete control-to-camera-perception latency.

Rig calibration may only measure the end-to-end baseline when intermediate timestamps are unavailable. It publishes exactly the endpoints it observed and does not invent a decomposition.

Alternating complementary signals with pseudorandom dwell lengths provide transition correspondences and avoid periodic phase ambiguity. The result retains median, tails, jitter, missed transitions, endpoint criterion, timestamp quantization, and evidence.

Spatial queries at an event time may request either raw observation time or latency-compensated effective time. The resolver never applies latency compensation implicitly.

## 8. Composition and Uncertainty

To resolve A to C through B:

```text
H_A_to_C = H_B_to_C * H_A_to_B
```

The resolver validates frame IDs, units, transform direction, time intervals, valid regions, and input-space compatibility before composition. It returns:

- composed matrix or hypothesis set;
- valid source/destination overlap;
- propagated uncertainty;
- confidence and warnings;
- exact transform chain and artifact revisions;
- source evidence references.

Matrix composition alone is insufficient for uncertainty. Initial propagation may use Jacobian linearization and covariance, while bootstrap samples or candidate ensembles should be retained when errors are non-Gaussian. Confidence is never multiplied into coordinates or substituted for covariance.

The resolver should prefer shorter, accepted, lower-uncertainty paths but must allow a caller to request a particular artifact/revision chain for reproducibility.

## 9. Independent Module Interfaces

The following conceptual interfaces keep dependencies one-way:

```text
SpatialFragmentProducer.export_spatial_fragment() -> YAML-compatible object
SpatialFragmentValidator.validate(fragment) -> validation report
SpatialRegistry.import_fragment(fragment, artifact_ref) -> immutable revision
SpatialResolver.resolve(from_frame, to_frame, at_time, policy) -> result
TemporalResolver.map_time(from_clock, to_clock, timestamp) -> result
```

Producers may write fragment files without linking the registry library. The validator and importer understand the common schema; they do not import producer implementations.

Rig calibration's generic inputs are similarly narrow:

```text
FrameStream.read() -> timestamped image
TargetPresenter.present(target_id, state) -> timestamped event
AlternatingStimulus.set_state(state, token) -> timestamped event
SignalObserver.observe(frame) -> timestamped probabilities
```

Game, ADB, camera, touch, and UI integrations are adapters outside the calibration core.

## 10. YAML Fragment Example

Artifacts remain authoritative in their domain-specific YAML files. A standard `spatial_exports` section makes their outputs discoverable without forcing a monolithic file format.

```yaml
# Self-describing spatial exports from one immutable artifact revision.
spatial_schema_version: "1.0"
artifact_id: "rig-logitech-c920-pixel7-20260826T120000Z"

frames:
  - frame_id: "aria://rig/logitech-c920-pixel7/camera/undistorted"
    revision: "20260826T120000Z"
    kind: raster_2d
    unit: pixel
    size_px: [1920, 1080]
    origin: top_left_pixel_center
    axes: {x: right, y: down}
    pixel_center_convention: integer_is_pixel_center

  - frame_id: "aria://device/pixel7/screen/portrait/layout-1080x2400"
    revision: "layout-1080x2400"
    kind: raster_2d
    unit: screen_pixel
    size_px: [1080, 2400]
    origin: top_left_pixel_center
    axes: {x: right, y: down}
    pixel_center_convention: integer_is_pixel_center

transforms:
  - transform_id: "camera-undistorted-to-phone-screen"
    from_frame: "aria://rig/logitech-c920-pixel7/camera/undistorted"
    to_frame: "aria://device/pixel7/screen/portrait/layout-1080x2400"
    model: projective_2d
    behavior: static
    direction: from_to
    matrix_3x3:
      - [1.234, 0.012, -322.1]
      - [-0.008, 1.229, -104.7]
      - [0.00001, -0.00002, 1.0]
    valid_mask_file: "valid_screen_mask.png"
    uncertainty:
      model: bootstrap_point_error
      p95_px_at_required_roi: 0.58
    confidence: 0.96
    status: accepted
    estimator:
      name: "aria_rig_calibration"
      version: "1.0"
    evidence: ["charuco_detection.png", "screen_overlap.png"]

clocks:
  - clock_id: host_monotonic_ns
    unit: nanosecond

latencies:
  - latency_id: "control-to-camera-perception"
    from_event: control_issue
    to_event: first_stable_camera_observation
    from_clock: host_monotonic_ns
    to_clock: host_monotonic_ns
    median_ns: 81600000
    p05_ns: 65300000
    p95_ns: 112400000
    endpoint: first_stable
    scope: display_and_camera_pipeline_baseline
    source: measured
```

The example numbers are illustrative. Domain artifacts may reference large masks, map tiles, remap arrays, covariance samples, and evidence by relative path and content hash.

## 11. Queries Used by Integrating Components

Representative queries are:

- Normalize a UVC frame into canonical phone-screen coordinates.
- Convert an ADB mini-map cursor centre into a camera-image location for review.
- Project a live mini-map displacement into an area-map frame.
- Compare camera and ADB observations from different sessions in one phone-screen frame.
- Resolve an imported session's mini-map hypothesis into the current global-map revision.
- Find all accepted paths between a route-local frame and a stitched global map.
- Retrieve the transform chain and uncertainty used for a particular pose estimate.
- Map a control timestamp to a camera observation window with explicit latency tails.

Each result is immutable and traceable to the exact artifact revisions used in resolution.

## 12. Review UI

The workbench should visualize the graph only when relationships aid review. Useful views include:

- selected source and destination frames;
- the chosen transform chain;
- side-by-side and overlaid rasters;
- valid overlap and masked regions;
- uncertainty ellipses or heatmaps;
- competing map-match hypotheses;
- time alignment and latency windows;
- artifact revision and evidence links.

The review UI may suggest accepting, rejecting, or superseding an edge. It does not overwrite original measurements. Human decisions are append-only review records.

## 13. Versioning and Invalidation

Frames and transforms are immutable revisions. A new calibration, phone orientation, game UI layout, map stitch, or dataset registration creates a new revision. Superseded edges remain available for reproducing old results.

Compatibility gates include:

- camera hardware and actual mode;
- lens settings and physical mounting;
- phone resolution, orientation, and system-inset policy;
- game UI and mini-map layout revision;
- stitched map content revision, layer, and region;
- estimator and matcher configuration;
- timestamp clock and synchronization interval.

A registry may mark an edge inactive for current use without deleting it.

## 14. Initial Verification

Automated tests should cover:

- transform direction and multiplication order;
- the top-left pixel-centre convention;
- full-screen and cropped normalization origin/scale;
- unit and axis mismatch rejection;
- partial valid-mask composition;
- time-varying edge selection and maximum interpolation gap;
- clock conversion without absorbing causal latency;
- latency-window calculation from measured distributions;
- covariance propagation through composed transforms;
- multiple-hypothesis preservation;
- cross-device/session namespaces and immutable revisions;
- round-trip transforms where a valid inverse exists;
- evidence/provenance retention in every resolved result.

The first implementation can use a directed graph and local matrix/covariance composition. Global graph optimization is optional later work and must not change the producer contracts.
