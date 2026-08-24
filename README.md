# AriaTrace

AriaTrace is PC-hosted, vision-guided gameplay replay for an unrooted Android device. A human demonstration defines the route, observable stages, and action priors. A later live run aligns itself to that demonstration and changes controls from visual feedback to overcome cross-session variance.

The name joins Ariadne's guiding thread, an aria interpreted as a live performance, and the recorded trace that connects both sessions. See [PROJECT_DEFINITION.md](PROJECT_DEFINITION.md) for the complete purpose and story.

This is not timed macro playback. Reliable route completion is the objective; localization is one supporting signal.

Genshin Impact PC is the first POC game. The Acquisition Workbench includes a Genshin workflow wizard, a human-editable control-profile draft, and distinct evidence captures for behavior profiling, the full map viewer, mini-map calibration, mini-map cruise modeling, and the repeatable route. The current slice records and labels the required evidence; automated map traversal and the map/mini-map estimators remain subsequent implementation work. Autonomous replay design follows evidence from that recorder rather than preceding it.

Start the integrated PC acquisition flow from the repository root:

    $env:PYTHONPATH=((Resolve-Path .tools).Path + ';' + (Resolve-Path .).Path)
    python -m acquisition.workbench

Open http://127.0.0.1:8765/. The normal GUI asks for the game profile, game window, input type, and guided capture or route preset; technical overrides stay under **Advanced settings**. Queue each uninterrupted capture, return focus to the game, and follow the displayed stage instructions. A click-through HUD appears at the game window's upper-right with waiting, countdown, finalizing, complete, and failed states. Recording stops automatically; return to the workbench when the HUD says **CAPTURE COMPLETE**.

For the Genshin POC, choose **Genshin Impact (PC POC)**, confirm the editable controls, select the first wizard stage, and follow the displayed instructions. Arm it, queue the capture, switch to Genshin, perform the guided evidence take, then return and confirm it. Repeat for the remaining wizard stages.

The wizard shows confirmed progress for every stage. Its machine-readable index is written to `artifacts/workbench/poc_evidence/genshin-impact-pc/evidence_index.json`; it links stage status to source sessions, confirmation markers, timing/count summaries, and the control-profile draft used for the capture.

Start with:

- [PROJECT_DEFINITION.md](PROJECT_DEFINITION.md) — concise scope and meaning of gameplay replay
- [PROJECT_CONSTITUTION.md](PROJECT_CONSTITUTION.md) — first principles and governing engineering rules
- [PROJECT_STATUS.md](PROJECT_STATUS.md) — tested state, commands, results, limitations, and next work
- [SRS.md](SRS.md) — brief requirements
- [SDS.md](SDS.md) — brief architecture and replaceable interfaces
- [RECORDER_GUIDE.md](RECORDER_GUIDE.md) — minimal GUI workflow for recording repeated routes
- [poc/README.md](poc/README.md) — pose-estimation POC commands
- [poc/RESULTS.md](poc/RESULTS.md) — measured results

The normal recorder workflow is documented in [RECORDER_GUIDE.md](RECORDER_GUIDE.md). Detailed session format, diagnostic CLI, inspection, and review information is in [acquisition/README.md](acquisition/README.md).

Generated datasets, models, and plots are under ignored `data/` and `artifacts/` directories. Local tools and Python packages are under ignored `.tools/`.
