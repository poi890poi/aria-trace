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

## Workbench interaction

- Calibration and stitching are available from the same simple session-list
  Workbench; stage-selection recording flow is not reintroduced.
- Mini-map/cursor calibration, pose verification, and map stitching have
  separate task tabs. Each tab identifies the exact input role and session.
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
