# AriaTrace Project Definition

## Name and Story

**AriaTrace** combines three ideas:

- **Ariadne's thread:** a human demonstration leaves a guide through the game world's labyrinth. A later run can locate itself on that thread, follow it, and rejoin it after a deviation.
- **Aria:** a performance is interpreted in its present conditions rather than reproduced as mechanical timing.
- **Trace:** the synchronized visual, control, and route record preserved from the human session.

The name expresses the project purpose: follow the demonstrated trace while adapting the performance to the live world.

## Mission

AriaTrace reproduces a human-demonstrated gameplay route in a later session using only live screen images and ordinary game controls. It adapts to cross-session differences instead of replaying the recorded inputs on a fixed clock.

## Meaning of Replay

A demonstration provides:

- the intended route and its observable stages;
- reference views, landmarks, interactions, and completion evidence;
- recorded controls as action priors;
- examples of camera-character-control behavior.

During a live run, the system estimates demonstration progress, predicts the effect of candidate controls, observes the result, and corrects or recovers. It may change control direction, magnitude, duration, or timing; pause for evidence; locally detour; skip already-completed steps; and rejoin the demonstrated route.

This is **adaptive, closed-loop replay**, not video matching, pose tracking in isolation, or deterministic macro playback.

## Core Problem

Cross-session variance includes loading time, frame and control latency, initial camera/character state, movement response, small collisions, accumulated displacement, dynamic objects, effects, illumination, UI changes, and incomplete visual overlap. The system must preserve the demonstrated intent despite these differences.

The replay loop has four responsibilities:

1. **Align:** determine the current route stage and likely reference observations.
2. **Estimate:** maintain useful character, camera, motion, and uncertainty state.
3. **Adapt:** choose controls from the demonstration prior and live visual error.
4. **Recover:** stop, observe, relocalize, backtrack, or rejoin when progress diverges.

## Target and MVP

The target domain is top-tier mobile MMORPG and FPS navigation. Minecraft is only a pipeline smoke test; representative offline FPS and third-person environments must be used before claiming target relevance.

The MVP uses one unrooted Android phone, PC-hosted logic, a known portal/start region, one recorded route, ADB control, and initially ADB capture followed by a fixed USB camera. It covers ordinary traversal and route interactions, not combat.

## Success

Primary success is repeatable route completion. Supporting measures are route deviation, stage-alignment accuracy, recovery rate, intervention rate, completion time, false corrections, stale-control incidents, and capture-to-control latency. Pose accuracy is diagnostic, not the objective.
