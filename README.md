# AriaTrace

AriaTrace is PC-hosted, vision-guided gameplay replay for an unrooted Android device. A human demonstration defines the route, observable stages, and action priors. A later live run aligns itself to that demonstration and changes controls from visual feedback to overcome cross-session variance.

The name joins Ariadne's guiding thread, an aria interpreted as a live performance, and the recorded trace that connects both sessions. See [PROJECT_DEFINITION.md](PROJECT_DEFINITION.md) for the complete purpose and story.

This is not timed macro playback. Reliable route completion is the objective; localization is one supporting signal.

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
