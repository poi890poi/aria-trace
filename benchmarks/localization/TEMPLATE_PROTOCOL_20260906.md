# Known-start and translation-only template evaluation

The current production integration already uses local gradient template matching
for steady XY. Its global initializer still requires SIFT geometry before map
correlation. Test removing that feature dependency and fitting translation only
within each existing normalized atlas layer. Production base is f607fa6; current
experiment history is 7555a76. No production changes precede evidence.

Candidates: masked gradient-magnitude TM_CCORR_NORMED, and separately masked
grayscale TM_CCOEFF_NORMED. Fix in-layer scale to 1 and rotation to 0; world/town
layers supply the distinct calibrated scales. Preserve query images, masks,
canonical transforms, search-window proposals, runtime consensus, local XY
refinement, transitions and publication. In unknown-start free-roam tests no
reference location or route proposal enters inference. The known-start tests
below explicitly supply the demonstrated first position even when the harness
uses its free-roam mode to isolate startup from full-route assistance.

Feature inliers and feature/correlation agreement no longer exist and must not
be fabricated. Use the existing correlation score0.55 and distinct-peak margin
0.06 as initial experimental gates, plus the existing local-template coverage
criterion of at least75% observed support. Those numerical cutoffs are not
equivalent validation across representations. Report false accepted positions,
wrong layers and unknown reference intervals rather than calling acceptance
accuracy. Include a repeated-pattern rejection control and a known-translation
control with masked corrupted pixels.

The initial unknown-start plan (superseded by the clarification below) was to
start with identical 46 development samples each from Run11 demo and Run17
outdoor startup (first45s). Compare against the preserved and repeated SIFT
controls and previous XFeat results. Record wall/CPU time including whole-layer
search, setup separately, and post-run reference error. Surviving candidates
continue through actual Workbench replay, paired full demo/reverse/outdoor/town
sessions where useful, with acquisition and longest post-acquisition loss,
fresh/held/unavailable output, discontinuities and delivery age visible.

The cached references are SIFT-derived same-atlas proxies. In that initial plan
they were scoring-only; the known-start exception is explicitly recorded below.
Hash-check their outputs and atlas/calibration inputs. Inspect early accepted
positions lacking reference support directly against native images. No fresh
moving holdout is currently available; no autonomous-control readiness claim is
permitted. Preserve negative comparisons and land only supported behavior.

## User clarification: route tracing starts at the demonstrated start

Prioritize this explicitly authorized assumption instead of an unknown-position
global initializer. Use only the first stored Run11 demonstration state as the
start coordinate/layer. Preserve its provenance and label it as a prior, not
current visual evidence. Compare existing SIFT, a translation template within
the start window, and seeding the existing local template tracker directly.
The last option can begin current-image refinement on the first frame without
waiting for two global fixes. Mark fresh versus held output accurately.

First test full Run11 demo replay, then perturb only the prior by40 canonical
pixels within the existing55-pixel first-refinement radius. Those offsets are
synthetic prior perturbations, not quality tolerances. Additional recordings
have different starts, so do not secretly initialize them from their scoring
references. Any same-recording demonstration/start test is a development replay,
not a fresh or cross-session route-following validation. Unknown-position global
experiments remain a separate optional scope and have not yet been measured.

Test the full route-assisted demo both without and with the start prior to
check its interaction with route transition anchors. Also compare Run14 reverse
without and with its own explicitly declared first-state demonstration prior;
that is a second development recording with a town start, not a repeat tracing
of Run11. Keep a deliberately gross 200px wrong-start clip as an assumption
violation control, not a required accuracy tolerance or a supported start range.

This is experimental tooling and documentation. It patches startup only within
the benchmark process; production APIs, stored route packages and startup
defaults do not change. Risks are falsely treating the supplied start as visual
confirmation and applying the wrong representation layer. Inspect first-frame
freshness and image correction, preserve failed starts, and distinguish startup
improvements from steady tracking and control delivery. Compare bounded SIFT
against bounded translation matching separately from the direct local seed.
