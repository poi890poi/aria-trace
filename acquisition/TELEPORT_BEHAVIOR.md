# Teleport behavior evidence

Teleportation is an explicitly recorded gameplay behavior. It is not inferred by
the live tracker and does not change localization state.

The Genshin PC workflow records the complete sequence from normal gameplay,
through opening the global map and clicking a teleport target, to the loaded
destination becoming ready. Later analysis produces one destination-oriented
`TeleportBehaviorSample` containing:

- the clicked target in a named canonical global-map coordinate space;
- the observed post-load destination in the same coordinate space;
- an ordered, visually guarded phase sequence from map opening through world-ready;
- the measured arrival distribution and heading at the destination;
- exact source frame and input evidence;
- map/localization artifact provenance and measurement quality.

The origin is intentionally not modeled. Keeping target and destination separate
permits learning systematic arrival offsets without treating an apparent runtime
position jump as teleport detection.

The behavior is a guarded state machine, not a fixed mouse macro. Map opening may
use a hotkey or the mini-map. Zoom and pan continue until the target is visible in
the localized map viewport. Target selection must expose the expected target
panel. Confirmation is an optional branch because some targets transition from
the Teleport button directly to loading. Destination readiness requires a stable
post-load mini-map localization consensus.

`teleport_analysis.py` separates drags from stationary clicks, aligns the selected
map target to the original stitched-map raster, localizes the first stable arrival
with the calibrated mini-map, and writes three review images plus the reusable
JSON record. Quality is reported with direct measurements (feature inliers,
reprojection error, localization scores, arrival sample count, and arrival
spread); it does not synthesize a proprietary confidence score from them.

Arrival localization prefers the normal feature-plus-correlation fix. An offline
fallback may use the feature-proposed center only when correlation is the sole
rejection source, the center is inside observed coverage, each proposal has at
least 8 inliers, 0.75 inlier ratio, and at most 2 px p95 reprojection error, and
three sampled frames agree within 12 px over no more than 2 seconds. The artifact
records which source supplied every accepted arrival sample. This fallback is not
used by live tracking and does not weaken its localization gates.
