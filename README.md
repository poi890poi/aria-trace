# AriaTrace

AriaTrace is PC-hosted, vision-guided gameplay replay for an unrooted Android device. A human demonstration defines the route, observable stages, and action priors. A later live run aligns itself to that demonstration and changes controls from visual feedback to overcome cross-session variance.

The name joins Ariadne's guiding thread, an aria interpreted as a live performance, and the recorded trace that connects both sessions. See [PROJECT_DEFINITION.md](PROJECT_DEFINITION.md) for the complete purpose and story.

## Ariadne's thread

![Ariadne gives Theseus a ball of thread before he enters the Labyrinth](assets/ariadne-thread.jpg)

*Ariadne gives Theseus the thread before he enters the Labyrinth.*

In Greek mythology, Ariadne gave Theseus a ball of thread before he entered the Labyrinth to confront the Minotaur. Theseus secured one end at the entrance and unwound the thread as he travelled through the maze. After the confrontation, the path he had laid down guided him out again.

That is the central metaphor for AriaTrace. A human demonstration lays down a perceptual thread: route observations, landmarks, actions, timing, and uncertainty remain connected as evidence. During a later run, AriaTrace does not blindly repeat a timed sequence. It watches the current scene, finds where it is relative to the recorded trace, and adjusts its actions to keep following that thread toward the destination.

The **aria** in the name emphasizes that every run is a new performance rather than an identical playback. The **trace** is both the route left by the demonstration and the inspectable evidence connecting what was seen, what was done, and why the replay chose its next action.

This is not timed macro playback. Reliable route completion is the objective; localization is one supporting signal.

Genshin Impact PC is the first POC game. The Acquisition Workbench records an unrestricted number of sessions, then lets the operator classify each one from a short label list: ordinary cruise, rotation-only, slow horizontal 360° scene turn, movement-only, straight-forward/no-turn, full-map coverage, or route demonstration. The workbench calibrates the circular mini-map and scene-relative yaw, verifies pose and shift evidence, stitches the observed map, and can run a two-rate live tracker using low-rate absolute map fixes plus high-rate relative shift and rotation. It persists selected-source provenance, quality metrics, and task-specific visual evidence for review.

Start the integrated PC acquisition flow from the repository root:

    $env:PYTHONPATH=((Resolve-Path .tools).Path + ';' + (Resolve-Path .).Path)
    python -m aria_trace.apps.workbench

Open http://127.0.0.1:8765/. Choose the game, visible game window, input type, and duration, then select **Start recording**. Switch to the game during the three-second settling countdown; recording begins on the first control received while that game has focus and stops after the selected duration. The optional overlay hides whenever the game loses focus and can be shown or hidden from the Workbench. Every successful, non-empty session appears in the list, where it can be labeled or moved to recoverable trash. Failed, canceled, zero-duration, and frameless attempts are discarded automatically.

The process that starts the Workbench owns its lifecycle. Its instance ID, PID, endpoint, start time, and data roots are exposed by `/api/instance`, with the essential identity also shown in the page header. A second launch on the same port reports the existing instance and exits without replacing it; stop the owner explicitly with Ctrl+C before restarting.

## IRIS integration

AriaTrace consumes normalized, space-aware camera and Android data from IRIS,
the Invariant Rig System. IRIS owns rig acquisition, calibration, profile
management, evidence, and HIK-compatible camera adaptation in the neutral
`rig_runtime/` package. The `aria_trace/` package owns the integrated gameplay
Workbench, recording/review, mapping, localization, and realtime tracking.

See [IRIS_README.md](IRIS_README.md) for setup, calibration, profile deployment,
and camera-adapter usage. The enforced dependency direction is
`aria_trace -> rig_runtime`; the neutral runtime never imports AriaTrace.
