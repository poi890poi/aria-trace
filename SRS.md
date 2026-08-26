# AriaTrace - Software Requirements Specification

## 1. Purpose

Build an MVP that records a human gameplay route and adaptively follows it in a later session using live images and ordinary multi-touch controls. The design shall favor simple implementations behind replaceable interfaces.

Replay is closed-loop route reproduction. It shall not assume that recorded controls can be issued at their original timestamps.

## 2. MVP Scope and Assumptions

- One unrooted physical Android phone, one game profile, one demonstration, and one predefined route.
- Known portal/start region; exact live pose may vary. Arbitrary random-start localization is future work.
- PC-hosted runtime; ADB is permitted.
- Development capture uses ADB screenshots; final capture uses a fixed UVC camera and bracket.
- Camera focus, exposure, gain, and white balance are adjusted manually and locked.
- Navigation excludes combat.
- The target domain includes first-person shooters and third-person MMORPG-style navigation. A simple sandbox may validate plumbing but not target-level performance.

The current milestone is the observation and game-modeling side of this MVP: short capture-plan-labeled gameplay segments for profiling, UI observation, mini-map calibration, and cruise modeling; automated full-map acquisition; and a gameplay route recorder/descriptor. Adaptive route replay remains the product goal, but its detailed estimator and controller design is deferred until recordings expose the dominant errors.

## 3. Functional Requirements

### Capture and calibration

- **CAP-01:** Acquire timestamped frames through a replaceable `FrameSource`.
- **CAP-02:** Provide ADB screenshot and UVC camera frame sources.
- **CAL-01:** Calibrate camera intrinsics and distortion using a standard ChArUco workflow.
- **CAL-02:** Rectify the phone screen into canonical screen coordinates using a screen-plane homography.
- **CAL-03:** Store calibration, optical settings, validation error, and camera format in a reusable profile.
- **CAL-03A:** Store the authoritative rig calibration as a commented YAML artifact. Its consumer contract shall declare the undistorted input space, input-to-output 3x3 transformation matrix, output size, canonical top-left phone-screen origin, output scaling, and valid mask so downstream agents do not need to infer coordinate conventions.
- **CAL-03B:** Provide a no-interpolation 1:1 camera-pixel inspection area for manual focus and image-quality review, with raw and undistorted views and current-versus-best comparison.
- **CAL-03C:** Measure camera-view/phone-screen coverage, camera utilization, screen-view IoU, task-ROI coverage, geometry uncertainty, and end-to-end matchable resolving power using both ADB-to-camera and camera-to-camera references where available.
- **CAL-03D:** Measure control-to-camera-perception latency with timestamped alternating visual signals. Preserve clock conversion separately from the causal latency distribution and report latency tails, jitter, missed/ambiguous transitions, endpoint criterion, and inspectable evidence.
- **CAL-04:** Automatically estimate the mini-map's position and circular boundary and store the result, method/configuration provenance, confidence/quality, and inspectable diagnostic images as a reusable structured artifact.
- **CAL-05:** The initial calibration experiment shall use a short rotation-only session labeled by its selected capture plan, then use temporal aggregation and circle detection to propose the mini-map boundary. This is an experimental procedure, not a fixed algorithm requirement.
- **CAL-06:** Record a straight-forward/no-turn session long enough to produce notable mini-map translation. Preserve it as evidence relating cursor heading, commanded forward motion when available, and observed mini-map shift direction.

### Spatial and temporal unification

- **SPU-01:** Represent phone screens, raw/undistorted camera images, mini-map observations, stitched area/global maps, and imported device/session datasets as explicitly named, versioned coordinate frames.
- **SPU-02:** Exchange directed transform observations through a producer-independent contract containing matrix/model direction, units, axes, validity region/time, uncertainty, confidence, provenance, and artifact revision.
- **SPU-03:** Compose compatible transforms without requiring any producer to know integrating components. Reject ambiguous coordinate conventions and retain the exact transform chain used by every result.
- **SPU-04:** Treat clock mappings and causal control-to-perception latency as separate models. Never hide measured latency inside a clock offset or compensate it without an explicit consumer request.
- **SPU-05:** Preserve imported datasets in their original frames and add registration edges rather than rewriting source pixels or timestamps.
- **SPU-06:** Preserve multiple spatial hypotheses for ambiguous matching until evidence or review accepts or rejects them.

### Full-map acquisition

- **MAP-01:** Automatically acquire complete coverage of all accessible map regions/layers from the game's full in-game map viewer rather than reconstruct the reference map solely from the semi-transparent mini-map; explicitly report inaccessible or undiscovered areas.
- **MAP-02:** Acquire source views at an effective spatial/detail resolution greater than the mini-map and retain enough overlap or other registration evidence to combine them reliably.
- **MAP-03:** Store a versioned map artifact containing source captures, capture/view state, transforms, coverage/completeness information, quality, and derived tiles or mosaic products.
- **MAP-04:** Keep the acquired full map distinct from live mini-map observations and from reconstructed route geometry. Preserve any mapping among their coordinate systems as an explicit calibrated estimate.
- **MAP-05:** Use a profiled map-viewer control procedure to automate opening, navigating, capturing, verifying coverage, and exiting the viewer. Game-specific UI details shall remain editable profile data.

### Game profiling

- **GPR-01:** Provide a semi-automatic game profiler that produces a versioned, human-editable game profile from controlled probes, recorded evidence, and reviewer corrections.
- **GPR-02:** Represent semantic actions and their physical bindings, including movement, camera control, jump, dash, interaction, and game/map UI actions as applicable.
- **GPR-03:** Represent activation semantics such as press, release, single click, hold, toggle, chord, duration, analog range, dead zone, and mutually compatible simultaneous actions where applicable.
- **GPR-04:** Measure relevant behaviors such as movement and rotation response, acceleration/stopping, camera-character coupling, jump/dash behavior, cooldown or recovery timing, collision response, and end-to-end input response where observable.
- **GPR-05:** Preserve the source frames, input events, probe definition, timing, inferred value, confidence, and human edit history behind each modeled behavior. A manual correction shall not erase the measured result.
- **GPR-06:** Keep adapter/capture defaults, control specification, measured behavior, map-viewer procedure, and known limitations in one game-profile relationship without coupling route-specific instructions into the game profile.
- **GPR-07:** Prefer separate bounded sessions for ordinary cruise, rotation-only, movement-only, and straight-forward/no-turn evidence. The capture plan shall store the semantic label so a reviewer need only confirm the requested motion is visible. Additional probes require an evidence-quality reason.
- **GPR-08:** A short labeled session shall support timer-triggered capture with input evidence optional, while route and other input-dependent stages may continue to require first-input start and healthy input capture.

### Demonstration recording

- **REC-01:** Record raw and normalized video with frame timestamps.
- **REC-02:** Record arbitrary physical or injected multi-touch events without requiring knowledge of the game control scheme.
- **REC-03:** Preserve source timestamps, PC receive timestamps, calibration ID, and session metadata.
- **REC-04:** Support deterministic playback and analysis of recorded sessions.
- **REC-05:** Persist the feature observations actually used online from pre-encoding frames, tied to their exact frame index, extractor configuration, and calibration.
- **REC-06:** Add and remove portal lifecycle markers without rewriting prior annotation history.
- **REC-07:** Record an uninterrupted human take without requiring recorder controls, markers, or labels during gameplay; route boundaries, stages, and landmarks are derived or corrected afterward.
- **REC-08:** For the PC keyboard/mouse POC, preserve raw keyboard transitions and relative mouse motion with per-event timestamps; absolute cursor polling is not sufficient evidence for locked-camera behavior.
- **REC-09:** Start capture sources only for the requested session, discard queue/switch input during a short explicit settling interval, and then start the bounded session clock automatically. Do not add a separate focus or input-test gate.
- **REC-10:** Append any number of sessions across workbench restarts. List existing session data, allow each session to be assigned a role from a controlled label list, and move deleted sessions to recoverable trash.

### Demonstration model

- **DEM-01:** Compile a recorded session into a versioned replay package containing synchronized observations, route stages, landmarks, interactions, action priors, and completion evidence.
- **DEM-02:** Retain the original video, controls, timestamps, calibration, annotations, and feature evidence referenced by the replay package.
- **DEM-03:** Permit manual correction of route stages and success/failure annotations without rewriting source evidence.
- **DEM-04:** Extract a normalized circular mini-map observation from a source game frame while excluding pixels outside its valid map mask.
- **DEM-05:** Estimate relative XY mini-map shift and character facing with quality/confidence and relevant intermediate measurements. Character facing shall not be assumed equal to camera facing or movement direction.
- **DEM-06:** Build route descriptors from selected keyframes grouped into local submaps, retaining a route centerline with both reconstructed spatial position and cumulative progress along the demonstrated route.
- **DEM-07:** Preserve raw relative measurements separately from reconstructed or corrected positions, including original values and validity/review state, so bad measurements can be rejected or replaced later.
- **DEM-08:** Keep each selected observation traceable to its original full game frame, timestamp, input/control state, calibration, measurements, and diagnostic artifacts.
- **DEM-09:** Treat stitched maps, route strips, sparse tiles, and trajectory plots as derived inspection products, never as the authoritative route representation.
- **DEM-10:** Permit future constraints from revisited locations without requiring loop closure or global optimization in the current milestone.

### Mini-map cruise model

- **MMC-01:** Use consecutive mini-map observations from role-labeled cruise and movement-only sessions, with sufficient overlap to estimate relative translation. Use the straight-forward/no-turn session to determine the sign and angular relationship between cursor heading and observed map translation.
- **MMC-02:** The initial XY proposal shall evaluate Fourier/phase correlation over the valid circular map area; the implementation shall preserve peak/overlap quality and remain replaceable.
- **MMC-03:** Facing estimation shall support game-specific central cursors rather than hard-code one cursor shape. Polar normalization and aggregation across different orientations are the initial proposal, not a mandated final algorithm.
- **MMC-04:** Measure movement-speed and rotation-rate distributions without assuming movement direction and facing are identical.
- **MMC-05:** Estimate temporal stability after movement alignment so transient icons, animations, floating elements, and effects can be down-weighted without requiring exhaustive icon recognition.

### Adaptive replay

- **RPL-01:** Estimate live progress against the demonstration without requiring elapsed-time alignment.
- **RPL-02:** Treat recorded actions as priors; adapt their direction, magnitude, duration, and timing from live feedback.
- **RPL-03:** Detect lack of progress, overshoot, unexpected camera-character state, collision, and loss of alignment.
- **RPL-04:** Support pausing, observing, relocalizing, local detours, backtracking, skipping completed stages, and rejoining the route.
- **RPL-05:** Confirm important transitions and final completion from observations rather than from the number of controls issued.
- **RPL-06:** Record replay alignment, selected references, issued controls, observed responses, uncertainty, and recovery decisions.

### Localization and navigation

- **LOC-01:** Initialize the MVP from a configured portal/start-region prior.
- **LOC-02:** Estimate position, character heading, camera heading, and uncertainty.
- **LOC-03:** Combine main-view evidence, coarse minimap evidence, control history, and recorded-route references.
- **LOC-04:** Reject absolute-pose hypotheses that conflict with coarse location, expected heading, temporal continuity, or configured pose-jump limits.
- **LOC-05:** Initialize a portal arrival by matching several live observations to a portal-specific map; recorded and live sessions shall not require time alignment.
- **NAV-01:** Follow a predefined metric route through local goals.
- **NAV-02:** Use direct visual feedback for obstacle avoidance, narrow openings, interactions, and short-range correction.
- **NAV-03:** Stop safely when observations are stale or confidence is insufficient.
- **NAV-04:** Optimize for demonstrated route completion while allowing bounded local deviation and later rejoin.

### Control

- **CTL-01:** Express navigation output as game-independent motion, view, and action intents.
- **CTL-02:** Map intents to game-specific touch layouts through configuration.
- **CTL-03:** Maintain independent multi-touch contacts and support concurrent movement, camera, and action inputs.
- **CTL-04:** Use a replaceable control backend; the MVP backend is a pinned scrcpy client/server implementation.
- **CTL-05:** Release all active contacts on stop, error, timeout, or disconnection.

### Timing and telemetry

- **TIM-01:** Use a common PC monotonic timeline and retain original device timestamps.
- **TIM-02:** Measure capture, processing, control-send, and observed-response latency.
- **TIM-03:** Process the newest frame and discard stale queued frames.
- **TIM-04:** Target 30 Hz local control and less than 100 ms capture-to-control latency for the initial replay profile, subject to selected camera hardware.
- **TIM-05:** Retain enough provenance to compute observation age, estimator update rate, processing latency, capture-to-estimate latency, capture-to-control latency, stale-observation use, and game response after control without treating unknown delay as zero.
- **TIM-06:** Permit observation sources and estimators to operate at different rates without requiring a single heavyweight synchronized pose output.

### Review and diagnosis

- **REV-01:** Extend the existing review tooling to synchronize source frames, mini-map crops and masks, shift/facing estimates, confidence, route progress, trajectory, keyframe/submap membership, controls, and diagnostic images where available.
- **REV-02:** Make suspicious measurements traceable to neighboring source frames and all associated provenance.
- **REV-03:** Store correct/suspicious/wrong judgments and comments as structured, machine-readable annotations without destroying original measurements.

## 4. Quality Requirements

- No root access or permanent Android application is required.
- Capture, recording, demonstration compilation, replay alignment, state estimation, replay coordination, local control, game mapping, and input injection are independently replaceable.
- Compressed source video is retained for review and future reprocessing. Exact online features and selected lossless keyframes are retained when encoder artifacts could change an evaluated observation.
- A successful feature/PnP solve is not sufficient confidence by itself; accepted poses require independent consistency checks.
- Runtime failures must default to no active touch contacts.
- Replay quality shall be evaluated by completion, route deviation, stage alignment, recovery, intervention, false correction, and latency; pose error alone is insufficient.

## 5. Out of Scope for MVP

- Arbitrary/random-start global localization
- FPS aiming and combat
- Multiple games or phone layouts
- Automatic camera optical adjustment
- Fully autonomous semantic exploration
- Circumvention of game security or anti-cheat systems

For the current recorder milestone specifically, autonomous replay, scene-based route-relative localization, constrained-traversal control, divergence recovery, loop closure, and a prescribed global optimization method are also out of scope. They remain candidate directions to evaluate after real recorder results exist.
