# AriaTrace

AriaTrace is PC-hosted, vision-guided gameplay replay for an unrooted Android device. A human demonstration defines the route, observable stages, and action priors. A later live run aligns itself to that demonstration and changes controls from visual feedback to overcome cross-session variance.

The name joins Ariadne's guiding thread, an aria interpreted as a live performance, and the recorded trace that connects both sessions. See [PROJECT_DEFINITION.md](PROJECT_DEFINITION.md) for the complete purpose and story.

This is not timed macro playback. Reliable route completion is the objective; localization is one supporting signal.

Genshin Impact PC is the first POC game. The Acquisition Workbench records an unrestricted number of sessions, then lets the operator classify each one from a short label list: ordinary cruise, rotation-only, movement-only, straight-forward/no-turn, full-map coverage, or route demonstration. The workbench calibrates the circular mini-map from the labeled sessions and preserves the forward segment as the cursor-heading-to-mini-map-shift reference for pose refinement. It persists the masked model, confidence metrics, and boundary, cursor-center, cursor-shape, polar, pose, and quality evidence for review. Automated map traversal and cruise/route estimators remain subsequent work.

Start the integrated PC acquisition flow from the repository root:

    $env:PYTHONPATH=((Resolve-Path .tools).Path + ';' + (Resolve-Path .).Path)
    python -m acquisition.workbench

Open http://127.0.0.1:8765/. Choose the game, visible game window, input type, and duration, then select **Start recording**. Switch to the game during the three-second settling countdown; recording begins on the first control received while that game has focus and stops after the selected duration. The optional overlay hides whenever the game loses focus and can be shown or hidden from the Workbench. Every successful, non-empty session appears in the list, where it can be labeled or moved to recoverable trash. Failed, canceled, zero-duration, and frameless attempts are discarded automatically.

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
