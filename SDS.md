# AriaTrace - Software Design Specification

## 1. Design Principles

- Implement the simplest useful version first.
- Hide replaceable mechanisms behind small interfaces.
- Store raw observations and timestamps.
- Prefer explicit confidence and safe stopping over guessed control.
- Use demonstration observations and controls as references, never as an open-loop schedule.

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

## 3. Core Interfaces

```text
FrameSource.start() / latest_frame() / stop()
ScreenNormalizer.normalize(raw_frame) -> ScreenFrame
TouchRecorder.start() / events() / stop()
DemonstrationCompiler.compile(session) -> ReplayPackage
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

## 5. Component Design

### Capture and normalization

- `AdbFrameSource` is used during early development.
- `UvcFrameSource` is the final capture implementation.
- A ChArUco calibration estimates camera intrinsics and distortion.
- A screen-plane homography maps the undistorted image to the canonical phone screen.
- Rectification maps are precomputed. Capture uses a single latest-frame slot, not a queue.

### Recorder

- `GetEventRecorder` parses physical multi-touch slots when permitted by the device.
- `InjectedEventRecorder` records all events sent by a control backend.
- A PC touch proxy is a fallback for demonstrations when physical input events cannot be read.
- Device monotonic time is mapped to PC monotonic time while both original timestamps are retained.
- Session storage uses video plus JSONL sidecars for frames, touches, controls, and telemetry.
- Primary pixels use replaceable compressed-video sinks; the MVP defaults to H.264/Matroska and keeps MJPEG only as a compatibility fallback.
- Online features are extracted before video encoding and retained as session evidence with source frame indices and extractor metadata.
- Features generated later from compressed video are separately labeled as versioned, regenerable caches.
- `AnnotationStore` records append-only frame markers. The reviewer supplies the MVP authoring UI; a future `WorldReadyDetector` may add the same marker type without changing consumers.
- `PortalInitializationExtractor` selects the `world_ready` to `route_start` interval and preserves whether exported pixels came from raw lossless evidence or decoded compressed video.

### Demonstration compilation and alignment

- `DemonstrationCompiler` converts an annotated session into a versioned `ReplayPackage`; the source session remains authoritative evidence.
- A replay package indexes reference observations, action priors, route stages, landmarks, interactions, and completion tests. It does not flatten them into a timed macro.
- `ReplayAligner` maintains several plausible demonstration locations when evidence is ambiguous and combines visual retrieval, route order, recent controls, transition evidence, and coarse pose.
- Alignment advances only when live evidence supports progress. It may remain stationary, skip an already-completed stage, or return to a prior stage during recovery.

### Localization

- `KnownStartInitializer` supplies the portal/start-region prior and its uncertainty.
- A portal profile binds a portal ID, spawn-position prior, route frame, and multi-view initialization submap. Live matching is spatial and requires consistent hypotheses across several frames rather than equal recording timestamps.
- The first `StateEstimator` contains a pose estimator using a predict-and-correct model:
  - predict from recent controls and a calibrated motion model;
  - correct using recorded main-view keyframes;
  - use minimap matching for coarse drift correction.
- The estimator publishes uncertainty and may later be replaced by visual SLAM or global relocalization.
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

## 7. Planned Extension Points

- `GlobalRelocalizer` for random starts
- Alternate metric pose estimators
- Learned local visual controller
- Additional games and control layouts
- Alternate camera and control backends
- Semantic object and interaction mapping
- Learned demonstration alignment and adaptive replay policies
