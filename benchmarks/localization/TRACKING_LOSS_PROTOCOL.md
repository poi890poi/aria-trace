# Tracking loss evaluation protocol v1

Type: benchmark behavior change. The previous headline reported the longest
interval without an accepted XY measurement. Confident wrong tracking could
therefore report zero loss. The new headline measures elapsed recorded-source
time from observed tracking failure to verified recovery. Production tracking,
acceptance gates, atlas construction and reference inference are unchanged.
All reference information remains evaluation-only.

## Visual-control objective

The product objective is autonomous completion of a learned route. Control is
lost when the combined observations no longer support the next navigation action:
correct route stage/direction, progress to the next decision point, and timely
steering within the traversable path. A smooth wrong trajectory is a failure;
temporary global-map uncertainty need not interrupt cruising if local visual
steering and route-stage recognition remain effective.

The eventual control evaluation must measure longest continuous control loss,
distance travelled without reliable guidance, missed/wrong turns, steering
oscillation, recovery interruptions and route completion without intervention.
Loss ends when action and observed response demonstrate recovered guidance.
Cross-track uncertainty must be compared with observed path clearance; progress
and heading uncertainty with turn timing, speed and measured turn/stop response.
The generic 35 px route corridor is a matcher setting, not measured clearance.

These replays run the tracking loop, not a closed-loop cruise controller. They
do not validate local visual guidance, heading truth, actual steering decisions,
or turn/braking dynamics. Consequently `visual_control.longest_control_lost_s`
is null, explicitly unmeasured. The localization proxy below diagnoses failures;
it cannot certify control loss duration or successful autonomous cruising.

## Definition

- A sample is lost if XY is unavailable, the output uses the wrong reference
  map layer, or position error exceeds the declared canonical-map-pixel limit.
- Derive the headline distinguishability envelope from OTHER recordings' slow
  references, separately per map scale, without using any tracking output.
  Leave each evaluated recording out. Remove a reference sample, predict its
  position from the adjacent same-mode samples, and measure the residual. Reject
  triples crossing mode changes or exceeding normal reference support gaps.
  Use the larger of that residual's P99 and sqrt(2) times median reference map
  scale (the relative integer-raster quantization bound of two map matches).
  At least ten residuals and a known scale are required; otherwise position
  agreement is unknown. Store all input hashes, sample counts and derived limits.
- This envelope captures reference interpolation/motion and raster uncertainty;
  it is not an empirically validated gameplay tolerance or a confidence interval
  for absolute truth. Fixed 5/10/20 px results remain sensitivity diagnostics only.
  An optional CLI override supports analysis, not a cruising acceptance claim.
- An available pose without a usable reference is unknown. Unknown intervals
  neither establish failure nor verify recovery. They keep an existing loss
  episode open, with unknown seconds explicitly reported in that episode.
- Recovery requires fresh, reference-consistent XY on the correct layer for
  at least 0.5 s of consecutive evaluated samples. A held pose within tolerance
  is not position loss, but cannot verify recovery. One isolated good sample
  cannot reset an episode. Once confirmed, recovery is dated to the beginning
  of that successful window; its confirmation timestamp is recorded separately.
- Processed telemetry gaps beyond the reference interpolation support
  (1.5/reference Hz seconds) interrupt recovery confirmation and become unknown.
- The headline includes initial acquisition. Also report first verified
  acquisition and longest loss after acquisition; the latter is unavailable
  if acquisition was never verified. First available pose remains separate.
- An open episode at recording end is explicitly unresolved/right-censored.
  Its duration is only the elapsed interval observed so far, not recovery time.
- Zero means no established loss under this protocol, not proof of reliability
  through unknown intervals. Entirely unscorable available output yields n/a.

Durations are computed on recorded source timestamps, with sample states applied
until the next sample subject to the maximum gap above. They exclude publication
delay; capture-to-publication and heading age retain their separate timing
metrics. Sparse references limit episode boundary precision. An episode spanning
unknown intervals is time to verified recovery, not continuously proven error.
The references are inferred from the same atlas, not independent external truth.
The demonstration reference also supplies route proposals in its own replay.

## Outputs and reproducibility

`tracking_loss` contains the headline configuration, episode records, longest
loss, longest loss after acquisition, reference-unknown duration and recovery
status. `tracking_loss_sensitivity` retains all evaluated tolerances. Each scored
sample includes `tracking_loss_state` and `tracking_loss_reasons` at the headline
tolerance. Former `loss_episodes` / `longest_loss_s` fields are replaced by the
accurately named `nonfresh_xy_episodes` / `longest_nonfresh_xy_s`; these remain
secondary diagnostics, not the headline quality metric.

Run `python -m benchmarks.localization.rescore_workbench <evidence-root>` to
rescore immutable production telemetry using cached references. The command
preserves pre-v1 reports and scored rows, runtime source identities, and records
the evaluation source hashes and Git state. It does not rerun tracking or change
its measurements. This is a revised analysis of existing sessions, not a fresh
holdout or evidence for landing a production fix.

Verification covers confident wrong poses, wrong layers at matching XY,
intermittent apparent recovery, held correct output, unavailable initialization,
unknown references, telemetry gaps, end-of-recording censoring and tolerance
boundaries. The known Run14 reverse-transition failure must become a multi-second
unresolved episode instead of a 0.13 s nonfresh-output interval.
