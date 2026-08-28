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
    python -m acquisition.workbench

Open http://127.0.0.1:8765/. Choose the game, visible game window, input type, and duration, then select **Start recording**. Switch to the game during the three-second settling countdown; recording begins on the first control received while that game has focus and stops after the selected duration. The optional overlay hides whenever the game loses focus and can be shown or hidden from the Workbench. Every successful, non-empty session appears in the list, where it can be labeled or moved to recoverable trash. Failed, canceled, zero-duration, and frameless attempts are discarded automatically.

The process that starts the Workbench owns its lifecycle. Its instance ID, PID, endpoint, start time, and data roots are exposed by `/api/instance`, with the essential identity also shown in the page header. A second launch on the same port reports the existing instance and exits without replacing it; stop the owner explicitly with Ctrl+C before restarting.

Camera-to-phone rig calibration has a separate Windows desktop application:

    python -m acquisition.rig_calibration.app

Its USB camera, phone target, and optional ADB access are operator initiated and
do not touch recorder sessions. See
[the app guide](acquisition/rig_calibration/app/README.md) for the guided
geometry, exact-pixel focus, standardized e-SFR/MTF design, established feature
matching measurements, latency, YAML, adapter, and isolated PyInstaller build
workflows. The current source replaces the former project-defined resolving
power screen with display-referred e-SFR/MTF and ground-truth feature matching.
It reports its primary e-SFR/MTF result in cycles per display
pixel; cycles per camera pixel is retained only as the native analysis axis,
while physical cycles/mm is optional and requires measured display pitch. A
one-pixel alternating phone pattern is `0.5 cycles/display-pixel`, not an
unqualified `1 line/pixel` result.

For the Genshin POC, record each useful motion as its own session. After **CAPTURE COMPLETE**, return to the list and choose the matching label. Selecting a label is the single review action that promotes a successful recording to usable evidence. The machine-readable index at `artifacts/workbench/poc_evidence/genshin-impact-pc/evidence_index.json` links those labels to source sessions, confirmation markers, timing/count summaries, and profile provenance.

Start with:

- [PROJECT_DEFINITION.md](PROJECT_DEFINITION.md) — concise scope and meaning of gameplay replay
- [PROJECT_CONSTITUTION.md](PROJECT_CONSTITUTION.md) — first principles and governing engineering rules
- [PROJECT_STATUS.md](PROJECT_STATUS.md) — tested state, commands, results, limitations, and next work
- [SRS.md](SRS.md) — brief requirements
- [SDS.md](SDS.md) — brief architecture and replaceable interfaces
- [RIG_CALIBRATION.md](RIG_CALIBRATION.md) — USB-camera/phone geometry, focus, matchability, UI, and YAML contract
- [SPATIAL_UNIFICATION.md](SPATIAL_UNIFICATION.md) — coordinate-frame graph for screens, cameras, mini-maps, maps, datasets, and time
- [RECORDER_GUIDE.md](RECORDER_GUIDE.md) — minimal GUI workflow for recording repeated routes
- [poc/README.md](poc/README.md) — pose-estimation POC commands
- [poc/RESULTS.md](poc/RESULTS.md) — measured results

The normal recorder workflow is documented in [RECORDER_GUIDE.md](RECORDER_GUIDE.md). Detailed session format, diagnostic CLI, inspection, and review information is in [acquisition/README.md](acquisition/README.md).

Generated datasets, models, and plots are under ignored `data/` and `artifacts/` directories. Local tools and Python packages are under ignored `.tools/`.
