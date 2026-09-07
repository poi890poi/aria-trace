# Unattended distortion correction for a fixed display rig

Type: behavior change and calibration capability. This work is separate from
the profile publication and rebuilding reliability fixes.

## Problem and boundary

The previous policy had only `off` and `guided`. Guided collection rejected
headless operation and instructed the operator to move/tilt the phone and press
capture for each view. Simply capturing the same pose repeatedly would not
provide the multi-pose evidence used by the existing physical-intrinsics fit.
[OpenCV's calibration documentation](https://docs.opencv.org/4.7.0/d9/d0c/group__calib3d.html)
describes that multi-view estimator and its radial/tangential model.

The new default, `auto`, measures correction for the stationary phone display.
It jointly fits a free plane homography and four radial/tangential coefficients
using a centered square normalization matrix. The matrix is explicitly labeled
as a coordinate gauge, not estimated physical camera intrinsics. The scope is
the fixed camera/display plane. No phone motion or capture prompt is involved.

## Evidence and completion policy

Collection is bounded to three times the requested frame count, with distinct
capture timestamps and at least 36 shared corners. It rejects observed motion
or unstable corners above 2 camera pixels at p95. A deterministic third of the
shared corner identities is excluded from every fit, even if detection order
changes. Training uses median observations from all but the final frame.

The exact candidate is evaluated on withheld corners in the final frame. It
must improve p95 by at least 5% (configurable) and 0.05 display pixels, without
increasing maximum error. Full-sensor probes reject folding and unstable
inverse mappings. Accepted candidates are saved without refitting on holdout
data. Evidence, candidate parameters and rejection reasons accompany the model.

Automatic fitting follows positioning confirmation. Interactive recalibration
refreshes it. Geometry acquisition checks the measured placement again and
disables the correction if it changed, clearing any already corrected geometry
candidates so raw and corrected coordinate spaces cannot be mixed.

Runtime still composes distortion into its existing single remap. Rejected
correction leaves explicit homography-only output. It does not claim that
distortion was successfully measured. Hardware acquisition errors propagate.
The guided diagnostic and explicit `off` remain available. Rig reuse keeps
the saved model; it does not silently refit an unchanged rig.

## Experiments and limitations

The initial isolated experiment used known OpenCV projections of one stationary
pose with spatial holdouts. At zero added noise, p95 decreased from 1.221 to
0.015 display pixels. A distortion-free case worsened and was rejected.

A subsequent seed, reserved from implementation, tested three stationary poses
at two noise levels. Inputs and results are recorded in
`artifacts/iris-auto-distortion-experiment.json`:

| Synthetic condition | Decision | Held-out p95 display pixels |
|---|---|---|
| Distorted, 0.03 camera-pixel noise, three poses | Accept all three | 1.904–2.290 to 0.072–0.080 |
| Distorted, 0.15 camera-pixel noise | Accept one; reject two unstable inverses | Accepted: 2.115 to 0.345 |
| No distortion, both noise levels, six cases | Reject all six | No sufficient improvement |

The two unstable candidates remain negative results; the inverse gate was not
weakened to accept them. Fit/evaluation took about 20–159 ms in that run,
excluding capture and target presentation. This is an observation, not a
performance guarantee.

Tests also evaluate unseen dense spatial samples against independently
generated projection coordinates, enforce holdout exclusion, exercise the
headless collection boundary without input/preview calls, and check bounded
failure, explicit-off behavior, and interactive repositioning order.

The resumed review adds a serialized-model test through the runtime's combined
remap: p95 raw-camera sampling error on unseen synthetic pixels must be below
0.15 pixels. Integration tests verify that duplicate timestamps cannot count as
independent observations, camera-read failures propagate, and geometry captured
before a later placement change is discarded when correction is disabled.

Final validation: **322 tests passed** in 41.346 seconds, covering automatic and
guided distortion, the real geometry handoff, calibration configuration,
headless profile rebuilding, public camera modes, adapter export, registry and
Windows write retries, and release-tool lookup/export checks. Log:
`artifacts/iris-automatic-distortion-final-tests.log`. These release-related
tests do not constitute a packaged-executable run.

No physical device experiment or packaged-executable validation has been run.
Synthetic evidence supports the software contract and the tested distortion
family; it does not establish accuracy on the user's optical setup or outside
the observed phone plane. Persisted correspondence arrays increase profile
size. No new numeric dependency or per-frame runtime estimation is introduced.
