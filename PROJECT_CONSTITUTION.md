# AriaTrace Constitution

## Purpose

Build a minimally intrusive system that follows a recorded human gameplay route in a later session through ordinary game controls. Optimize for repeatable route completion under cross-session variance, not for exact input playback or an impressive pose estimate.

## First Principles

- Study the closed loop before choosing algorithms: touches change game state, the game renders observations, and observations guide the next touches.
- Treat the human demonstration as a structured reference, not a clocked macro. Recorded controls are priors; live observations decide what to do and when.
- Align by route progress, landmarks, and state transitions rather than elapsed time. Permit local deviation and rejoin when it better preserves the demonstrated intent.
- Model character position, character heading, camera heading, camera pitch, speed, and locomotion mode separately. Camera pose is not character pose.
- Treat game physics as measured behavior. Calibrate movement, turning, stopping, camera response, automatic camera motion, collision, and latency for each game profile.
- Use the information humans use: landmarks and route stages for orientation, short-term motion for continuity, and direct visual feedback for steering through paths, obstacles, and openings.
- Do not require globally precise metric pose for every control decision. Metric pose supports route progress; local visual control handles immediate traversal.
- Represent a route with coordinates where useful, but also with visible landmarks, corridors, decision points, expected transitions, and recovery observations.

## Engineering Rules

- Implement the simplest useful mechanism behind a small replaceable interface.
- Keep capture, normalization, recording, localization, planning, local control, control mapping, and input injection independently replaceable.
- Preserve raw frames, touch events, timestamps, and intermediate hypotheses so experiments can be replayed.
- Preserve the live-to-demonstration alignment, selected reference, action prior, issued control, observed response, and recovery decision for every replay step.
- Separate prediction from correction. A visual pose solve is a hypothesis; it must pass independent consistency checks before changing navigation state.
- Make uncertainty operational. Increasing uncertainty causes shorter actions, slower movement, observation, reorientation, or stopping.
- Prefer deterministic replay tests before hardware integration, then verify the same behavior on the physical device.
- Judge changes by route progress, deviation, false corrections, recovery, latency, and safe stopping—not by detector confidence alone.
- Record assumptions and limitations with every experiment. Synthetic evidence must never be presented as measured game behavior.

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
