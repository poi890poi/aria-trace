# AriaTrace Constitution

## Purpose

Build a minimally intrusive system that follows a recorded human gameplay route in a later session through ordinary game controls. Optimize for repeatable route completion under cross-session variance, not for exact input playback or an impressive pose estimate.

## First Principles

- Study the closed loop before choosing algorithms: touches change game state, the game renders observations, and observations guide the next touches.
- Treat the human demonstration as a structured reference, not a clocked macro. Recorded controls are priors; live observations decide what to do and when.
- Align by route progress, landmarks, and state transitions rather than elapsed time. Permit local deviation and rejoin when it better preserves the demonstrated intent.
- Model character position, character heading, camera heading, camera pitch, speed, and locomotion mode separately. Camera pose is not character pose.
- Treat game physics as measured behavior. Calibrate movement, turning, stopping, camera response, automatic camera motion, collision, and latency for each game profile.
- Give every game a versioned, human-editable control and behavior profile. Control bindings and activation semantics are game data, not assumptions embedded in navigation code; automated probes must retain their evidence and confidence.
- Acquire the complete reference map automatically from the game's full map viewer at greater effective detail than the semi-transparent mini-map. Preserve source captures, coverage, and transforms so the map can be regenerated and inspected.
- Use the information humans use: landmarks and route stages for orientation, short-term motion for continuity, and direct visual feedback for steering through paths, obstacles, and openings.
- A landmark is an observed visual reference, never an input the demonstrator must provide. Recording must not require markers, hotkeys, or other authoring actions during gameplay.
- Do not require globally precise metric pose for every control decision. Metric pose supports route progress; local visual control handles immediate traversal.
- Represent a route with coordinates where useful, but also with visible landmarks, corridors, decision points, expected transitions, and recovery observations.
- Preserve both spatial relationships and cumulative progress along the demonstrated route. Route progress is an index into route evidence, not a replacement for spatial position.
- Treat measured relative motion and facing as observations; reconstructed or later-corrected route geometry is an estimate. Never make a derived stitched map the only route record.
- Keep character facing, camera facing, and movement direction distinct even when a game or experiment temporarily couples them.

## Engineering Rules

- Implement the simplest useful mechanism behind a small replaceable interface.
- Keep capture, normalization, recording, localization, planning, local control, control mapping, and input injection independently replaceable.
- Keep full-map acquisition, mini-map observation, and route reconstruction distinct. The full map is reference data from the map viewer; the mini-map is a live observation source; route geometry is an estimate derived from recorded measurements.
- Preserve raw frames, touch events, timestamps, and intermediate hypotheses so experiments can be replayed.
- Keep every selected route observation traceable to its full game frame, recorded controls, timing, calibration, measurements, and diagnostic artifacts. A mini-map-derived route must not become mini-map-only evidence.
- Preserve the live-to-demonstration alignment, selected reference, action prior, issued control, observed response, and recovery decision for every replay step.
- Separate prediction from correction. A visual pose solve is a hypothesis; it must pass independent consistency checks before changing navigation state.
- Make uncertainty operational. Increasing uncertainty causes shorter actions, slower movement, observation, reorientation, or stopping.
- Prefer deterministic replay tests before hardware integration, then verify the same behavior on the physical device.
- Judge changes by route progress, deviation, false corrections, recovery, latency, and safe stopping—not by detector confidence alone.
- Record assumptions and limitations with every experiment. Synthetic evidence must never be presented as measured game behavior.
- Preserve capture and processing times without concealing unknown delay. Estimators may run at different rates and must not be forced into one heavyweight synchronized pose result.
- Use structured human review annotations so a questionable result can be traced to source observations and reused as machine-readable debugging evidence.
- Follow an evidence-led cycle: record, model, inspect, evaluate, choose the next problem, and improve. Do not commit detailed replay algorithms before recordings show which failure modes dominate.

## MVP Boundaries

- PC-hosted logic, an unrooted Android device, replaceable ADB control, and final USB-camera capture.
- One game profile, known portal/start region, one recorded route, ordinary ground navigation and interactions, and no combat.
- Use aligned human demonstrations to learn game behavior and evaluate navigation when hardware becomes available.
- Use Minecraft only to validate plumbing. Claims relevant to top-tier MMORPG or FPS navigation require representative scene and control complexity.

## Safety and Compatibility

- Interfere with the phone and game as little as possible.
- Release every active touch on stop, timeout, disconnection, stale observation, or unhandled failure.
- Do not circumvent game security or anti-cheat systems.
- When evidence is insufficient, stop or gather information instead of issuing a guessed movement.
