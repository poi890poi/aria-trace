# Calibration and map-analysis Workbench requirements

## Purpose

The Workbench must turn retained, labeled recording sessions into inspectable
calibration and map artifacts. Recording remains unrestricted and independent
of analysis: users record any number of sessions, label them afterward, and run
analysis when enough suitable evidence exists.

## Reuse constraint

Existing, verified AriaTrace algorithms are authoritative building blocks. The
Workbench must orchestrate `minimap_calibration`, `cursor_pose`, the session
reader, and their existing evidence artifacts instead of replacing them with
parallel implementations. New code should cover only missing discovery,
selection, full-map stitching, verification, visualization, and UI plumbing.

## Session discovery and selection

- Analysis tools discover complete, non-superseded sessions from their labels,
  game profile, frame availability, and input health.
- If exactly one suitable session exists for a role, it is selected
  automatically.
- If several suitable sessions exist, the Workbench shows every candidate and
  permits explicit user selection. The automatic default must be visible and
  changeable.
- Missing roles are reported by name without implying that the player must
  perform an unrequested action.
- Every artifact records the selected source session IDs and paths.

## Mini-map and cursor calibration

- A user can run the existing circular mini-map boundary, cursor-center,
  cursor-shape, and cursor-pose calibration from labeled rotation-only and
  movement-only sessions.
- Straight-forward data is not consumed by this task. It belongs to the
  separate pose-verification task below.
- The Workbench displays quality metrics and the existing evidence images,
  including boundary, center, cursor shape, polar response, and pose quality.
- Calibration results remain reviewable; numerical confidence never silently
  substitutes for user review.

## Scene-relative yaw calibration

- A separate `scene_rotation_360` session records a stationary character while
  the camera turns slowly on the horizontal axis through at least 360 degrees.
- Scene pixels, not mini-map rotation or raw input magnitude, provide the
  relative-yaw measurement. UI regions may be masked but remain preserved in
  the source recording.
- Calibration must detect and visualize the full-turn loop closure, report
  accumulated rotation, direction, closure error, tracked/inlier features,
  confidence, timing, and rejected frames, and retain per-frame estimates.
- Scene-yaw calibration and verification have their own Workbench tab and
  source selector. They do not modify mini-map/cursor calibration.

## Straight-forward shift and pose verification

- The system estimates the mini-map content displacement from the beginning to
  the end of the selected straight-forward session.
- The result preserves the raw start-to-end shift vector, response/confidence,
  source frame indices/timestamps, estimated cursor screen heading, and their
  angular relationship.
- Review output visualizes the two masked observations, registration overlay,
  shift vector, and pose/shift relationship.
- This reference may improve pose-orientation sign and angular-offset
  estimation; it must not claim resolved world heading without supporting
  evidence.
- The dashboard reports observed pose-estimator wall time per frame and masked
  shift-estimator wall time as median and p95. Pose timing excludes model load
  and evidence rendering; shift timing uses repeated complete calls after one
  warm-up. Benchmarking must not change estimator math or acceptance thresholds.

## Full-map stitching

- A user can run automatic stitching on a labeled full-map session.
- The stitcher extracts useful map-view frames, estimates overlap and pairwise
  transforms, rejects unusable transitions, and composes a mosaic without
  discarding raw measurements.
- The artifact reports accepted/rejected frames, overlap/registration quality,
  spatial coverage, source-frame provenance, and warnings for incomplete or
  weak coverage.
- The Workbench displays the mosaic, coverage map, registration-quality plot,
  and representative alignment evidence.
- Automatic stitching does not by itself certify that every game region or
  layer was recorded; coverage claims are limited to observed frames.

## Two-rate live tracking

- Live tracking consumes three explicit, same-game artifacts: a reviewed
  mini-map/cursor calibration, a scene-relative-yaw calibration, and an
  observed full-map mosaic. The Workbench shows all three artifact IDs before
  capture starts and never substitutes another platform's geometry.
- A high-cost masked, multi-scale, multi-angle mini-map-to-mosaic search runs at
  a configurable low rate and supplies absolute position, rotation, scale,
  score, margin, and wall time. It runs off the frame-processing thread.
- At capture rate, the existing calibrated scene-yaw estimator measures
  relative rotation and masked mini-map registration measures relative shift.
  These updates continue while an absolute search is in flight.
- Absolute fixes are fused through the existing pose correction gate. The
  output retains whether each fix initialized, corrected, or was rejected;
  uncertainty and degraded/relocalizing modes remain visible.
- A compact overlay draws the fused pose, heading, recent trail, mode, absolute
  and relative confidence, update time, global-search time, and uncertainty on
  the selected observed mosaic. It hides when the target game loses focus and
  remains user-closeable through the existing Workbench overlay control.
- Tracker coordinates are pixels in the observed stitched mosaic. They are not
  world coordinates and do not imply coverage beyond recorded map frames.
- Live tracking and session recording are mutually exclusive because both own
  the selected capture source. Starting either while the other is active is
  rejected with a visible explanation.

## Workbench interaction

- Calibration and stitching are available from the same simple session-list
  Workbench; stage-selection recording flow is not reintroduced.
- Mini-map/cursor calibration, pose verification, map stitching, and live
  tracking have separate task tabs, as does scene-relative yaw calibration. Each tab
  identifies the exact input role and session.
- A tab displays a result only when its recorded provenance exactly matches the
  currently selected sessions. Historical POC artifacts and results from other
  selections are not mixed into the task view.
- Each tab shows a small, task-specific evidence set; it does not dump every
  image present in an artifact directory.
- Analysis controls show automatic candidates and manual selectors only when
  there is a choice.
- Long-running analysis reports running, complete, or failed state and does not
  block session management.
- Every session row has an **Open folder** action that opens the exact validated
  session directory in the platform file manager.
- Evidence images open at reviewable resolution from the Workbench.

## Acceptance criteria

- No failed, canceled, zero-duration, frameless, or unhealthy required-input
  recording is promoted to analysis input.
- Automatic and manual selection resolve to paths inside the configured session
  root.
- All generated artifacts have machine-readable summaries, deterministic source
  provenance, and human-viewable quality evidence.
- Existing mini-map/cursor tests continue to pass, and new orchestration,
  stitching, verification, folder-opening, API, and UI behavior has focused
  coverage.

## Android + HIK game calibration profiles

- A reusable calibration profile is identified by the exact tuple **rig + game
  + image source**. Android/scrcpy and HIK observations from the same run are
  separate profiles and never share raw pixel geometry.
- Small phone/camera displacement is handled by an optional new headless HIK
  rig-calibration revision. Older observations remain immutable and the new
  geometry revision records its source rig calibration.
- Mini-map calibration records Android/scrcpy and rectified HIK frames in one
  acquisition session. Android presentation timestamps are mapped from Android
  `CLOCK_MONOTONIC` to the PC performance counter; HIK receive timestamps use
  that same host counter and retain the raw device frame counter.
- Cross-source visual delay is estimated from the shared changing game content
  and stored as calibration-only timing. It is not a live-tracker dependency
  and does not collapse the two source profiles into one.
- The automatic camera-view control is one continuous horizontal sweep. Its
  vertical increments repeat **up, down, down, up**, producing absolute pitch
  targets horizon, sky, horizon, ground, horizon. Every issued touch event is
  stored beside the two streams.
- This zigzag task calibrates only mini-map isolation: source crop, complete
  circular boundary, the actual mask used for shift estimation, and the visible
  game `N` compass marker. It does not fabricate cursor, pose, or global-map
  evidence; those existing verified tasks remain separate.
- The current Android surface rotation, saved ChArUco image-viewer orientation,
  and game-render orientation are three separate observations. Game orientation
  is represented explicitly. Cross-source alignment is derived from the saved
  rig geometry, camera-visible phone region, and recorded Android surface
  rotation; synchronized ADB and HIK pixels verify that alignment rather than
  inventing a second transform. Neither phone nor viewer orientation is
  substituted for game orientation.
  North is measured independently for each image source from the `N` marker
  using screen-angle convention `0=right,
  90=down, 180=left, 270=up`; phone rotation is never substituted for map
  north.
- Each sampled calibration frame records its detected north angle, confidence,
  and the OpenCV rotation needed to make north point up. A stable north marker
  also produces a static 2x3 north-up transform. If the marker moves, the
  result retains per-frame angles and configures live detection. If it cannot
  be identified confidently, north remains explicitly unresolved.
- The production HIK adapter supports `minimap`, `full`, and `dual` modes.
  Mini-map-only mode applies the smallest aligned hardware ROI for low USB
  throughput. Dual mode performs one full-camera acquisition and derives the
  full and mini-map products with the same timestamp and frame number.
- Mini-map rectification is optional. Disabling it returns the aligned sensor
  crop for minimum processing latency. A future verified north-up transform
  applies only to rectified mini-map output; raw sensor crops retain their
  reported source orientation and avoid the extra processing. The production
  adapter does not infer north from unreviewed evidence.
