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

## 3. Functional Requirements

### Capture and calibration

- **CAP-01:** Acquire timestamped frames through a replaceable `FrameSource`.
- **CAP-02:** Provide ADB screenshot and UVC camera frame sources.
- **CAL-01:** Calibrate camera intrinsics and distortion using a standard ChArUco workflow.
- **CAL-02:** Rectify the phone screen into canonical screen coordinates using a screen-plane homography.
- **CAL-03:** Store calibration, optical settings, validation error, and camera format in a reusable profile.

### Demonstration recording

- **REC-01:** Record raw and normalized video with frame timestamps.
- **REC-02:** Record arbitrary physical or injected multi-touch events without requiring knowledge of the game control scheme.
- **REC-03:** Preserve source timestamps, PC receive timestamps, calibration ID, and session metadata.
- **REC-04:** Support deterministic playback and analysis of recorded sessions.
- **REC-05:** Persist the feature observations actually used online from pre-encoding frames, tied to their exact frame index, extractor configuration, and calibration.
- **REC-06:** Add and remove portal lifecycle markers without rewriting prior annotation history.

### Demonstration model

- **DEM-01:** Compile a recorded session into a versioned replay package containing synchronized observations, route stages, landmarks, interactions, action priors, and completion evidence.
- **DEM-02:** Retain the original video, controls, timestamps, calibration, annotations, and feature evidence referenced by the replay package.
- **DEM-03:** Permit manual correction of route stages and success/failure annotations without rewriting source evidence.

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
