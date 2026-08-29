# Teleport behavior evidence

Teleportation is an explicitly recorded gameplay behavior. It is not inferred by
the live tracker and does not change localization state.

The Genshin PC workflow records the complete sequence from normal gameplay,
through opening the global map and clicking a teleport target, to the loaded
destination becoming ready. Later analysis produces one reusable
`TeleportBehaviorSample` containing:

- the clicked target in a named canonical global-map coordinate space;
- the observed post-load destination in the same coordinate space;
- the optional pre-teleport origin;
- exact source frame and input evidence;
- map/localization artifact provenance and measurement quality.

Keeping target and destination separate permits learning systematic arrival
offsets without treating an apparent runtime position jump as teleport detection.
