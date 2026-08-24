# AriaTrace - Software Design Specification

## 1. Design Principles

- Implement the simplest useful version first.
- Hide replaceable mechanisms behind small interfaces.
- Store raw observations and timestamps.
- Store raw measurements separately from reconstructed or corrected state.
- Prefer explicit confidence and safe stopping over guessed control.
- Use demonstration observations and controls as references, never as an open-loop schedule.
- Preserve full-scene provenance even when a subsystem derives observations from the mini-map.
- Allow observation producers to run at different rates; do not require one monolithic synchronized pose.

## 2. System Architecture

```text
Recorded Session -> DemonstrationCompiler -> ReplayPackage
                                                |
FrameSource -> ScreenNormalizer -> ObservationHub -> ReplayAligner
                                      |       |          |
                                      |       +-> Recorder
                                      |                  |
                                      |-> StateEstimator |
                                      |                  |
                                      +----------> ReplayCoordinator
                                                       |
                                                LocalController
                                                       |
                                                 ControlIntent
                                                       |
GameProfile -> ControlMapper -> TouchStateEngine -> ControlBackend
                                      |
                                   Recorder
```

The MVP PC runtime is Python on Windows. OpenCV provides calibration and image processing. The scrcpy client/server version is pinned with the project.

The diagram describes the eventual replay system. The next implemented slice stops before live replay:

```text
Controlled Probes -> GameProfiler -> GameProfile + behavior evidence
                                      |
In-game Map Viewer -> FullMapAcquirer -> FullMapArtifact + coverage diagnostics
                                      |
Calibration Frames -> MiniMapCalibrator -> MiniMapCalibration + diagnostics
Cruise Session -----> MiniMapExtractor --> shift/facing/stability observations
Route Session ------> Keyframe Selection -> Local Submaps -> RouteDescriptor
                            |                                      |
                            +---- source frame/input/timing refs ---+

RouteDescriptor + source evidence -> Acquisition Workbench review
```

Mini-map algorithms in this section are proposals to test. The stable architecture is the evidence boundary: reusable calibration, normalized observations, quality-bearing measurements, source provenance, and derived geometry that can be recomputed.

## 3. Core Interfaces

```text
FrameSource.start() / latest_frame() / stop()
ScreenNormalizer.normalize(raw_frame) -> ScreenFrame
TouchRecorder.start() / events() / stop()
DemonstrationCompiler.compile(session) -> ReplayPackage
MiniMapCalibrator.calibrate(frames, motion_provenance) -> MiniMapCalibration
MiniMapExtractor.extract(frame, calibration) -> MiniMapObservation
MiniMapShiftEstimator.estimate(previous, current) -> RelativeShiftMeasurement
CharacterFacingEstimator.estimate(observation, cursor_model) -> FacingMeasurement
MiniMapCruiseModeler.fit(observations, measurements) -> MiniMapCruiseModel
GameProfiler.run(probe_plan, observations, inputs) -> GameProfile
FullMapAcquirer.acquire(frame_source, control_backend, game_profile) -> FullMapArtifact
RouteDescriptorCompiler.compile(session, full_map, calibration, cruise_model) -> RouteDescriptor
StateInitializer.initialize(observation, start_prior) -> GameplayState
StateEstimator.update(observation, controls) -> GameplayState
ReplayAligner.update(observation, state, replay) -> ReplayAlignment
ReplayCoordinator.next_target(alignment, state, replay) -> ReplayTarget
LocalController.step(frame, state, target, action_prior) -> ControlIntent
RecoveryPolicy.step(alignment, state, progress) -> RecoveryDecision
ControlMapper.map(intent, game_profile) -> DesiredTouchState
ControlBackend.connect() / submit(events) / release_all() / disconnect()
```

## 4. Core Data Types

```text
ScreenFrame       {image, capture_time, receive_time, calibration_id}
TouchContact      {id, x_norm, y_norm, pressure, active}
TouchSample       {source_time, receive_time, contacts[]}
PoseEstimate      {x, y, character_yaw, camera_yaw, covariance, confidence}
GameplayState     {pose, velocity, locomotion_mode, interaction_state, uncertainty}
ReplayStage       {id, reference_range, landmarks, interaction, completion_evidence}
ActionPrior       {motion, view_change, actions, demonstrated_duration}
ReplayAlignment   {stage_id, progress, reference_ids[], confidence, alternatives[]}
ReplayTarget      {stage_id, local_goal, visual_targets[], completion_test}
RecoveryDecision  {mode, target_stage, reason, limits}
LocalGoal         {x, y, bearing, target_type, tolerance}
ControlIntent     {desired_motion, desired_view_change, actions[]}
DesiredTouchState {contacts[]}
```

Coordinates used by capture and control are normalized unless explicitly labeled otherwise.

### Mini-map and route descriptor records

The following are conceptual records and relationships for the next milestone, not a frozen serialization schema:

```text
MiniMapCalibration {id, source_session, circle, valid_mask, method_config,
                    quality, diagnostics[], created_time}
MiniMapObservation {id, source_stream, source_frame_index, capture_time,
                    receive_time, crop_transform, calibration_id, artifact_refs[]}
RelativeShiftMeasurement {from_id, to_id, dx, dy, quality, overlap,
                          intermediate_refs[], original_value, review_state}
FacingMeasurement {observation_id, angle, quality, cursor_model_id,
                   intermediate_refs[], original_value, review_state}
MiniMapCruiseModel {calibration_id, cursor_model, stability_model,
                    movement_statistics, rotation_statistics, provenance}
ControlBinding {action, device, control, activation_semantics, parameters}
BehaviorMeasurement {behavior, probe_id, value, quality, timing, evidence_refs[]}
GameProfile {id, version, capture_defaults, control_bindings[], behaviors[],
             map_viewer_procedure, limitations[], human_edits[]}
FullMapArtifact {id, game_profile_id, source_captures[], view_states[],
                 transforms[], coverage, quality, tile_or_mosaic_refs[]}
RouteKeyframe {observation_id, source_frame_ref, input_refs[], measurement_refs[]}
LocalSubmap {id, keyframe_ids[], relative_measurement_ids[], estimated_geometry}
RouteCenterlineSample {progress_s, estimated_position, facing, keyframe_or_submap_ref}
RouteDescriptor {source_session, game_profile_id, full_map_id, calibration_id,
                 cruise_model_id, keyframes[], submaps[], raw_measurements[],
                 centerline_samples[], annotations[]}
```

The route centerline's `progress_s` is cumulative progress/distance along the demonstrated route and provides a one-dimensional index `R(s)` into route state and observations. It complements rather than replaces reconstructed spatial position. Measured pairwise relationships are immutable evidence; positions and submap geometry are replaceable estimates derived from them. Revisited locations may later contribute correction constraints without changing these source records.

Keyframes remove nearly identical frame-by-frame redundancy while retaining source-frame references. Nearby keyframes form short local submaps whose size and construction policy remain experimental. A giant stitched map is never authoritative. Sparse tiles, stitched local maps, route strips, and trajectory plots are derived visualizations.

Two error classes remain separate. An individual shift or facing measurement may be catastrophically wrong; its original output is retained while append-only review/validity records may flag or reject it. Future checks may use estimator quality, correlation-peak shape, overlap, neighboring-frame or short-window consistency, and physically plausible speed, acceleration, or rotation. Accumulated drift can arise even when every local measurement is reasonable; local submaps limit its immediate effect, and preserved revisit relationships may later support geometric correction. No global optimizer or loop-closure algorithm is selected for the first descriptor.

## 5. Component Design

### Capture and normalization

- `AdbFrameSource` is used during early development.
- `UvcFrameSource` is the final capture implementation.
- A ChArUco calibration estimates camera intrinsics and distortion.
- A screen-plane homography maps the undistorted image to the canonical phone screen.
- Rectification maps are precomputed. Capture uses a single latest-frame slot, not a queue.

### Recorder

- `GetEventRecorder` parses physical multi-touch slots when permitted by the device.
- WindowsRawKeyboardMouseSource uses a message-only Raw Input receiver to preserve keyboard make/break transitions and true relative mouse motion without requiring recorder actions during gameplay.
- `InjectedEventRecorder` records all events sent by a control backend.
- A PC touch proxy is a fallback for demonstrations when physical input events cannot be read.
- Device monotonic time is mapped to PC monotonic time while both original timestamps are retained.
- Session storage uses video plus JSONL sidecars for frames, touches, controls, and telemetry.
- Primary pixels use replaceable compressed-video sinks; the MVP defaults to H.264/Matroska and keeps MJPEG only as a compatibility fallback.
- Online features are extracted before video encoding and retained as session evidence with source frame indices and extractor metadata.
- Features generated later from compressed video are separately labeled as versioned, regenerable caches.
- `AnnotationStore` records append-only frame markers. The reviewer supplies the MVP authoring UI; a future `WorldReadyDetector` may add the same marker type without changing consumers.
- Full-take acquisition has no in-game authoring controls. Frame and input sources run only for the queued take and discard pre-start observations. The first qualifying input received becomes session time zero and starts the configured-duration clock. There is no separate focus or input-test gate; take boundaries are confirmed or corrected only after gameplay. Landmarks are visual observations, not demonstrator inputs.
- `PortalInitializationExtractor` selects the `world_ready` to `route_start` interval and preserves whether exported pixels came from raw lossless evidence or decoded compressed video.

### Game profiling

- `GameProfiler` extends the existing versioned `profiles/games` concept; it does not create a parallel profile system. The current profiles contain capture/input adapter defaults and coarse control summaries. Profiling adds an evidence-backed control specification, measured behaviors, map-viewer procedure, limitations, and human edits.
- Control bindings map semantic actions to physical controls and activation semantics. A keyboard/mouse game might specify `W/A/S/D` movement, mouse camera axes, `Space` jump, and a right-mouse dash, while also specifying whether dash is a single click, hold, toggle, timed press, or context-dependent action. Controller and touch profiles express the same semantics using their native buttons, axes, contacts, ranges, and dead zones.
- Profiles record compatibility and concurrency where it matters: which actions can be held together, chords, mutually exclusive states, locomotion modes, and transitions such as press/release or hold/repeat. Route-specific instructions remain in route profiles.
- Semi-automatic profiling first reuses one short basic gameplay sample containing ordinary cruise, rotation-only, and movement-only segments. It estimates movement speed and direction, turning/camera response, acceleration and stopping, camera-character coupling, and input-to-observed-response latency where observable. Additional jump/dash, cooldown/recovery, collision, or other probes are requested only when the shared sample cannot support a needed field. This list is extensible and does not require every game to expose every behavior.
- Each inferred value retains the probe definition, source frames, recorded inputs, timestamps, intermediate measurements, quality/confidence, and profiler version. Human review may accept, correct, or supply values and comments without deleting the measured result. Profiles must distinguish measured, manually supplied, assumed, and unknown values.
- The profiler should generate inspectable probe summaries and highlight missing or contradictory fields. Automation may propose a profile, but a human-editable result is the authority used by later acquisition and control mapping.

### Full in-game map acquisition

- `FullMapAcquirer` obtains the complete reference map from the game's full map viewer. The semi-transparent mini-map is a live observation source and is not the preferred source for complete map texture.
- Once the game profile identifies how to open, navigate, zoom, switch available regions/layers, and close the viewer, acquisition automatically traverses the viewer, captures sufficient overlapping source views, and verifies coverage. The exact scan path, feature registration, and viewer-specific UI handling remain experimental.
- Captures use a zoom/detail level whose effective map resolution exceeds the mini-map. Resolution, scale, UI occlusions, overlap, and coverage are measured rather than inferred from screenshot dimensions alone.
- The reusable artifact retains original viewer captures, relevant input/control events and timestamps, viewer state or zoom where observable, tile transforms, a coverage/completeness mask, quality diagnostics, and derived tiles or mosaics. A derived mosaic may be convenient, but source captures and transforms remain authoritative evidence.
- Coordinate relationships among the full map, mini-map observations, and route geometry are explicit calibration estimates with provenance and confidence. Do not assume identical scale, crop, rotation, projection, icon layers, or map revision.
- Automated acquisition should detect incomplete or low-quality coverage and retry or request human correction through the existing workbench. Human edits to viewer controls or coverage exclusions become versioned profile/artifact data.
- Locked, undiscovered, occluded, or otherwise inaccessible regions are recorded explicitly and prevent a false completeness claim; they are not silently treated as acquired map area.

### Mini-map calibration and cruise modeling

- Calibration reuses the rotation-only segment of the basic gameplay sample. Its frames and recorded camera-input timing remain synchronized; no separate player-facing calibration sequence is required by default.
- The working assumption is that the mini-map remains fixed while camera orientation changes. Temporal averaging or a stability/heatmap calculation suppresses changing scene content; thresholding, edge extraction, and a circle detector such as Hough circle detection then propose the mini-map boundary.
- Calibration produces a reusable structured artifact rather than coordinates alone. At minimum, its diagnostic set includes an average/stability/heatmap image, threshold result, edge image, and the detected circle overlaid on a reference frame. The exact aggregation, thresholds, detector, and artifact schema remain subject to experiment.
- `MiniMapExtractor` uses the calibration to produce a normalized circular observation and valid mask. Pixels outside the circle are excluded from matching.
- The initial XY proposal uses Fourier/phase correlation between overlapping consecutive observations. It publishes displacement together with peak, overlap, and other quality evidence rather than only `(dx, dy)`.
- Character facing comes from the central mini-map cursor and is distinct from camera facing and movement direction. The cursor is game-specific and may be a triangle, chevron, a circle with a direction marker, or a combination of approximately symmetric simple shapes.
- The initial facing-model proposal polar-transforms observations about the calibrated center, using a precomputed transform map where useful, so cursor rotation becomes translation. Differently oriented cruise observations can then be rotationally normalized and combined to identify stable cursor structure. The exact model remains open.
- The cruise model reuses the short-cruise and movement-only segments of that same sample. It records speed and rotation-rate distributions and aligns observations by estimated map motion to measure temporal stability. Stable map texture and cursor structure can then be separated statistically from icons, floating/animated elements, and temporary effects; unreliable regions can be down-weighted without recognizing every icon.
- The first reusable capability boundary is extraction, XY shift estimation, and character-facing estimation. Each output retains confidence, timing, provenance, and relevant intermediate measurements.

### Demonstration compilation and alignment

- `DemonstrationCompiler` converts an annotated session into a versioned `ReplayPackage`; the source session remains authoritative evidence.
- For the recorder milestone, compilation selects useful observations as keyframes, groups nearby evidence into local submaps, preserves raw pairwise measurements, reconstructs replaceable local/spatial estimates, and assigns cumulative route progress. Exact keyframe and submap policies are chosen experimentally.
- A replay package or route descriptor indexes reference observations, action priors, route stages, landmarks, interactions, and completion tests. It does not flatten them into a timed macro or one stitched map.
- Every mini-map keyframe remains linked to the original full scene, exact timestamps, relevant input/control state, calibration, measurements, and diagnostic artifacts. This preserves the option to investigate later full-scene route-relative localization without changing the recording contract.
- Later full-scene work may estimate progress relative to a demonstrated viewpoint, lateral displacement from the demonstrated trajectory, orientation difference, or alignment with paths, openings, obstacles, and landmarks. These are possible control-relevant observations, not a scene-localization design for the current milestone.
- `ReplayAligner` maintains several plausible demonstration locations when evidence is ambiguous and combines visual retrieval, route order, recent controls, transition evidence, and coarse pose.
- Alignment advances only when live evidence supports progress. It may remain stationary, skip an already-completed stage, or return to a prior stage during recovery.

### Localization

- `KnownStartInitializer` supplies the portal/start-region prior and its uncertainty.
- A portal profile binds a portal ID, spawn-position prior, route frame, and multi-view initialization submap. Live matching is spatial and requires consistent hypotheses across several frames rather than equal recording timestamps.
- The completed feature-map and portal-initialization work remains evidence about view coverage and hypothesis gating; it is not the committed next localization stack.
- Mini-map route recording first establishes relative XY movement and character-facing observations. Their actual failure modes will determine how they are combined with full-scene observations in later replay work.
- The feature map may contain multiple submaps and must cover the route from relevant travel directions; proximity without view overlap is insufficient.
- Absolute localization returns a hypothesis rather than directly resetting pose. A fusion gate checks coarse minimap region, predicted heading, temporal continuity, inlier quality, and pose-jump limits before applying a correction.
- The MVP gate is an explicit planar predictor and consistency checker, not a general-purpose fusion framework. Camera and character headings remain distinct in the game state even when a ground-running profile temporarily constrains them.

### Planning and local control

- The demonstrated route is a sequence of observable stages with metric goals where useful, visual references, action priors, transition evidence, and recovery anchors.
- `ReplayCoordinator` converts the current alignment into the next target; it does not generate touch events.
- The local controller starts from the demonstrated action prior and uses the latest main view to correct alignment, pass obstacles/openings, and approach interactable objects.
- The initial local controller may be rule-based and later replaced by a learned policy.
- The basic human-like loop is orient, run briefly, correct from direct visual feedback, and confirm progress at a landmark. Increasing uncertainty shortens or stops motion and triggers observation or reorientation.
- `RecoveryPolicy` first releases unsafe controls, then chooses observe, reorient, local relocalize, backtrack to an anchor, or stop. Recovery has explicit attempt and time limits.

### Touch generation and injection

- `ControlMapper` converts generic intents using a per-game profile.
- `TouchStateEngine` owns the authoritative contact state and derives valid down/move/up transitions.
- `ScrcpyBackend` is the MVP injector and supports independent pointer IDs.
- Future backends may include minitouch, MaaTouch, ADB input, or external hardware.

## 6. Runtime and Safety

- Capture, localization, local control, control dispatch, and recording run independently.
- Slow planning never blocks the local control loop.
- Pose is propagated from capture time to control-send time using recent controls.
- Stale frames, low confidence, backend failure, shutdown, and uncaught errors trigger `release_all()`.
- Latency telemetry records each pipeline stage and reports median and tail latency.
- Observation records distinguish capture, receive, processing-start, estimate-publication, control-send, and observed-response times where available. Unknown delay remains unknown rather than being folded into a nominal timestamp.
- Independent mini-map, full-scene, input, and future estimator streams retain their own cadence and freshness. Consumers select sufficiently recent evidence instead of waiting for a heavyweight all-source pose.

## 7. Review and Diagnostic Workbench

The existing Acquisition Workbench and reviewer are the integration point for profiling, full-map acquisition, and recorder diagnosis. Planned views include game-profile fields with probe evidence, full-map source tiles and coverage diagnostics, the original gameplay frame, mini-map crop and masks, XY shift and facing estimates, confidence/quality, reconstructed trajectory, route progress, keyframe/submap membership, controls, and diagnostic images. Selecting a suspicious result must reveal neighboring source frames and associated measurements.

Review judgments are structured records attached to the affected measurement or route result. They preserve the original value and support `correct`, `suspicious`, and `wrong` states plus a comment. The intended debugging bundle is: annotation, affected observation, original frames, diagnostics, and nearby measurements, so later tools or Codex sessions can diagnose systematic failures from evidence rather than screenshots alone.

## 8. Post-recorder Decision Boundary

Do not specify the autonomous replay architecture until mini-map cruise and gameplay-route recordings can be evaluated. Likely investigation areas include live localization along the route, mini-map/full-scene combination, route-relative error, closed-loop correction, constrained traversal, perception/control rate and latency, divergence recovery, and learning from repeated failures. These are directions, not committed modules or algorithms.

The development loop is `record -> model -> inspect -> evaluate -> choose next problem -> improve`. In particular, no optimizer, loop-closure method, controller, detailed pose representation, or difficulty metric is selected by this document. Local submaps and preserved revisit evidence leave room for later correction if accumulated drift proves material.
