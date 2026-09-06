# Route startup and transition follow-up

Behavior change: expose an explicit demonstrated-start route path, retaining
unknown-position startup for existing requests and free roaming. A start supplies
only a search proposal; fusion and public pose must wait for a current-image
measurement. Preserve masks, heading provenance, atlas coordinates and steady XY
matching. Do not count a held prior as verified acquisition. Reject incompatible
requests before capture begins, and permit restart without reusing previous state.

First reproduce Run11 route-assisted and Run14 start-only baseline/prior pairs
with the existing bounded decode-ahead recorded source. This separates decode
stall variation from algorithm behavior; it does not improve physical capture.
Report prefetch setup, source lateness and buffering separately. Run sequentially,
without concurrent benchmark or test jobs. Keep original source times; no future
image or scoring state enters the tracker. Run14 explicitly uses its own first
demonstration state, not Run11's start. Frozen references and input hashes remain
checked and reused. Both recordings are development data, not fresh holdouts.

Implement startup in its owning tracker layer, with Workbench request validation
and an explicit route-start UI option. Test missing/current measurements, invalid
mode, one-time start use and unchanged unknown-start behavior. Compare the final
production path against the previously measured prior experiment, including its
interaction with route transition anchors. A gross wrong-start negative control
must remain visible; scores alone do not establish absolute correctness.

Diagnose the first transition divergence from raw timestamps, active layers,
representation evidence and held/fresh output. State a falsifiable cause before
trying a transition fix; compare one change at a time and retain negative results.
Do not weaken loss scoring or position acceptance to make the report pass.

Acceptance focuses on first verified acquisition, longest loss (startup and
post-acquisition separately), wrong fresh output, jumps, inferred-reference error
and joint XY/heading delivery. Any claimed continuity improvement must reproduce
on both directions. Unknown reference spans stay unknown. No claim of live
closed-loop auto-cruise readiness without fresh capture and steering validation.

Scope: route-start behavior, replay source instrumentation, diagnosed transition
changes and focused tests. Preserve unrelated camera/device work. Land independently
reviewable supported slices and document unproven or rejected candidates.

Observed transition diagnosis: with decode-ahead, reverse baseline and prior
both lose 0.2314058s. World is confirmed at 15.694s, but the direct tracker keeps
its old-layer timestamp and 12px search, rejecting corrections as continuity
jumps until 16.017s. Runtime metadata says reset_local_reference, yet only the
legacy previous_minimap is cleared. Hypothesis: clearing the template tracker's
measurement timestamp after a confirmed switch will use its existing first-frame
55px refinement and avoid comparing across representations. Preserve the last
position as a search proposal and trained transition anchors; no displacement
is applied by the observer. Test both directions before promoting this fix.

The forward reference also estimates intermediate map scales (1.671 at 37.270s,
1.391 at 37.498s, 1.304 at 37.701s). Its first town evidence is not already at the
fixed town layer's 1.303 scale. A timestamp reset alone cannot address this
transient scale mismatch, so retain forward losses and unknown intervals.

Reset-only reverse evidence removes reference-observed loss but increases the
maximum output jump from 19.69 to 32.02px. Do not promote based on reference loss
alone. A second one-variable experiment holds XY only while the existing
controller is armed and the two normalized layer scores differ by less than
its existing minimum margin. Compare that hold alone, then its combination with
the reset. This adds no new numerical threshold. Preserve the longer nonfresh
gap as a cost, including gaps without sufficient reference evidence.

The route UI defaults its visible starting-position selector to the demonstrated
beginning under the user's assumption. Existing API requests that omit the new
policy retain unknown-position behavior; the free-roam UI always sends that
policy. Startup uses the local refiner's existing 0.55 score gate, not the steady
tracker's permissive real-time threshold. This is an intentional startup behavior
change and is tested separately from transition experiments.
