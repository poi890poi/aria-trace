# Tracking correction experiments

Type: benchmark-only causal experiment. Production defaults remain unchanged
until a candidate has supporting full-loop evidence. Frozen baseline revision:
1386101. Atlas and cached references are those in the rebuilt-atlas report.
References are evaluation-only except the declared Run11 route proposals.

## Hypotheses

| ID | Mechanism | Expected benefit | Main risk / dependency |
|---|---|---|---|
| A | Carry observation-time position uncertainty into transition controller | Arm the reverse crossing that currently misses a tiny learned zone | Earlier arming can hold or switch too soon; may need B for a valid target-layer position |
| B | Reverse trained transition endpoints as spatial search proposals | Reacquire against the new layer using current pixels | Requires confirmed layer change; holding an imperfect anchor may worsen A |
| C | Latest consensus pose instead of averaging positions across motion | Reduce initial/recovery position lag | Latest solve may be noisier; worker delay remains |
| D | Retain recovery hypotheses while recovery is latched despite accepted local motion | Permit two global recovery fixes to reach consensus | Moving fixes still need alignment; C and D can interact |
| E | Submit/consume cursor in the same frame, bounded by remaining 33.3 ms | Reduce heading age and XY/heading time mismatch | Waiting may increase XY latency/drops; late results keep asynchronous fallback |

The experiment runner installs explicit edited methods on the real production
classes before constructing AcquisitionWorkbench. Saved generated method source,
hashes and the variant definition identify the actual executed implementation.
The capture adapter alone supplies recorded frames at original cadence. It does
not supply reference XY, heading or future observations to tracking.

## Evaluation protocol

1. Behavioral regressions reproduce missed arming and recovery-consensus erasure;
   baseline and all-edits regression suites check adjacent invariants.
2. Transition matrix: baseline, A, B, AB on Run14; compare the demonstration
   crossing too. Test E alone and with the supported transition candidate.
3. Initialization/recovery matrix: baseline, C, D, CD on the full outdoor Run17,
   with Run11 initialization evidence as a second scene where needed.
4. Repeat promising candidates, then run additional cruises and both lap scenes.
   Avoid concurrent timing benchmarks on this host. Preserve negative variants.
5. Report loss, acquisition, wrong-layer duration, error distributions, jumps,
   accepted/held/unavailable output, all-source deadline fractions and heading
   publication age. Use source timestamps and original frozen references.

Combination effects require their constituent singles and baseline. A component
that is ineffective alone can remain useful if its paired comparison establishes
the dependency. Do not sweep arbitrary combinations without a mechanism.

Confidence is qualitative and scoped: high for a reproduced mechanism supported
by regression and repeated evidence; moderate for measured improvement across
recordings with remaining uncertainty; low for one replay or mixed results.
No recorded replay establishes physical capture/HUD/control performance or
successful autonomous cruising. All recordings predate these candidates, so
fresh post-change holdout and closed-loop control confidence remain unestablished.
