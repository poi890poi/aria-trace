# CPU XFeat evaluation protocol

Base production revision: `f607fa6`. Experimental tooling only; production tracking
and its configuration remain unchanged. Official XFeat revision:
`e92685f57f8318b18725c5c8c0bd28c7fe188d9a`, pretrained `weights/xfeat.pt`.

Compare the current SIFT global proposal to XFeat sparse features with the same
L2 nearest-neighbor ratio test (0.80), then separately the upstream mutual
nearest-neighbor matching rule. Preserve geometry, map correlation, coverage,
initialization consensus, transition logic, local image refinement and publication
scheduling. CPU-only PyTorch, one inference thread initially, maximum 8,000
features as in the SIFT baseline. Mask feature locations using the existing mask;
XFeat descriptors can still see surrounding image context, as SIFT descriptors do.
Model load and atlas extraction are setup, reported separately from query timing.

Use identical decoded frames for proposal comparison before E2E: Run17 startup
0–45 seconds and Run11 demo development samples. Compare accepted/rejected fixes,
reasons, feature counts, reference error where known, wall and CPU time. Continue
surviving candidates on reverse transitions, demo and outdoor/town laps. A full
replay of a clearly inferior proposal may document the control-level consequence,
but does not justify parameter sweeping or weakening acceptance gates.

E2E uses the actual Workbench and original-cadence recorded source, as in the
previous evaluation. Compare fresh/held/unavailable output, initialization,
reference-based longest lost, jumps, position error, heading freshness and total
publication age. Decompose global solve time from decode and foreground update.
Repeat promising differences; compare combinations only with evidence of need.

Frozen slow references are reused from
`artifacts/poc/workbench-rebuilt-atlas-20260905/references/references.json` and
hash checked. They are scoring inputs only. They share the atlas and SIFT-derived
inference with the baseline, so agreement is not independent ground truth. Unknown
outdoor startup cannot establish XFeat false positives or improvement; validate
such disagreements visually or label them unverified. No route state or reference
location is allowed into free-roam inference. A fresh moving holdout is not yet
available; no autonomous-control readiness claim follows from these replays.

Selection requires sustained accuracy/continuity improvement without worse false
recoveries or deadline delivery. Report negatives and inconclusive outcomes,
preserve raw telemetry, source/weight hashes and environment. CPU milliseconds
are measurements on this host, not hardware-independent promises.

## Diagnostic follow-up, frozen after native-resolution failure

Both XFeat proposal variants accepted zero of the 92 development samples.
Identical-image controls match exactly, but native-to-full-atlas correspondences
are mostly wrong. Reference-centered crops improve correspondence but still
fail the unchanged geometry checks. The native input is only 138×138. Test one
additional factor: resize both query and atlas by 2 before extraction, returning
keypoint coordinates to original pixels. Keep the 8,000 feature limit, matching
rule and all gates. This is a combined XFeat/resolution candidate, not evidence
that XFeat alone helped. Preserve original results. Reference-centered crops
remain diagnostics only and never enter the estimator.

The 2× ratio variant accepts 11/46 demo frames but no outdoor frames. Mutual
matching finds some correct correspondences mixed with many outliers. Test the
official XFeat + LighterGlue weights as a bounded final matcher comparison, with
native-resolution features first, default confidence 0.1 and all final pose gates
unchanged. Start with one-second clips from each development session before
spending on longer replays. Verify strict weight loading; record matcher setup,
CPU latency and negative results. Do not introduce reference-selected map crops
or loosen pose checks to make learned matching succeed.
