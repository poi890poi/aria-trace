# AriaTrace

AriaTrace is PC-hosted, vision-guided gameplay replay for an unrooted Android device. A human demonstration defines the route, observable stages, and action priors. A later live run aligns itself to that demonstration and changes controls from visual feedback to overcome cross-session variance.

The name joins Ariadne's guiding thread, an aria interpreted as a live performance, and the recorded trace that connects both sessions. See [PROJECT_DEFINITION.md](PROJECT_DEFINITION.md) for the complete purpose and story.

This is not timed macro playback. Reliable route completion is the objective; localization is one supporting signal.

The immediate POC uses a selectable game profile and replaceable PC capture/input adapters to exercise the same recorder, session format, replay compiler, and closed-loop interfaces intended for later Android/UVC operation.

Start the integrated PC acquisition flow from the repository root:

    $env:PYTHONPATH=((Resolve-Path .tools).Path + ';' + (Resolve-Path .).Path)
    python -m acquisition.workbench

Open http://127.0.0.1:8765/, choose any visible game window, select or omit a game profile, define a fixed capture duration, and queue repeated uninterrupted takes. Return focus to the game and play naturally; recording starts and stops automatically without in-game recorder controls. Combat Master is one profile-data candidate, not a recorder dependency.

Start with:

- [PROJECT_DEFINITION.md](PROJECT_DEFINITION.md) — concise scope and meaning of gameplay replay
- [PROJECT_CONSTITUTION.md](PROJECT_CONSTITUTION.md) — first principles and governing engineering rules
- [PROJECT_STATUS.md](PROJECT_STATUS.md) — tested state, commands, results, limitations, and next work
- [SRS.md](SRS.md) — brief requirements
- [SDS.md](SDS.md) — brief architecture and replaceable interfaces
- [poc/README.md](poc/README.md) — pose-estimation POC commands
- [poc/RESULTS.md](poc/RESULTS.md) — measured results

The synchronized recording and review workflow is documented in [acquisition/README.md](acquisition/README.md).

Generated datasets, models, and plots are under ignored `data/` and `artifacts/` directories. Local tools and Python packages are under ignored `.tools/`.
