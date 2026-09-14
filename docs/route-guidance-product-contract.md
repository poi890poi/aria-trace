# Continuous visual route guidance: active product contract

This contract supersedes single-destination selection as the intended user
workflow. That implemented viewer is a diagnostic component. The next useful
product is continuous manual guidance, like an ILS display; automatic navigation
comes after that guidance has been validated.

Latest evidence: [route-wide guidance assay, September 14](../benchmarks/localization/ROUTE_GUIDANCE_RESULTS_20260914.md).
Prepared learned summaries improve scene recognition on later town laps, but
directly recalled mouse actions and the tested automatic ground-path point
transfer fail the guidance gate. These are offline implementations, not a landed
ILS feature. The guidance and feedback rows below remain unimplemented.

## Goal and user workflow

The user demonstrates a route, prepares reusable route memory, selects that
route, and drives while following live input hints. The system manages observation
targets and transitions automatically. It should help sustain smooth travel,
negotiate narrow passages, and finish turns correctly. It does not require exact
world XY or identical trajectories across repeated demonstrations.

Preparation should work from one to three demonstrations, with cached features
and relations. Use small pretrained primitives or cheaper deterministic methods
when they support the loop. Route-specific neural training is not a prerequisite.

## Whole loop and ownership

| Layer | Input | Output | Current status |
|---|---|---|---|
| Route preparation | Demonstration images, recorded input, repeated views, relative map motion | Ordered route stages, observable groups, useful relations, overlaps, turn priors and expected exit conditions | Partial offline experiments; no complete route-guidance package builder |
| Fast observation | Current/past frames and selected prepared memory | Feature-group identity, bearing, motion/expansion, support, age, uncertainty; relative motion where available | One selected destination observer exists; route-wide observation does not |
| Stage and attention management | Recent observations, route memory and action history | Current stage, relevant groups, automatic acquisition/recovery and handoff | Not integrated or validated across complete routes |
| Guidance | Validated visual relations, motion, stage and actual recent input | Suggested next input and strength, turn continuation/termination cue, availability and uncertainty | Not implemented; this is the next product milestone |
| Human response | Live guidance | Actual keyboard/mouse input and resulting scene motion | Human drives; input logging already exists |
| Feedback check | Suggested input, actual human response, subsequent observations | Whether the correction helped, whether to continue/change/withdraw it | Not implemented as a complete guidance loop |

The human-response layer is eventually replaceable by a bounded input executor.
Do not build a separate automatic-navigation policy while the guidance loop is
still unvalidated. Guidance must observe actual input, not assume the human
executed its suggestion.

## Minimum information and relations

- Track only enough stable groups to support the current decision. Group identity
  and relationships matter more than maximizing feature count.
- One suitable target may support an approach. A near/far pair is useful when
  relative motion resolves ambiguity; pairs are not mandatory.
- Camera bearing, travel heading and lateral route relation are different
  quantities. A centred target alone does not establish that travel is centred.
- Relative map shift can support motion/progress reasoning without demanding
  accurate global XY. Any depth/progress interpretation needs its calibration.
- Expected feature motion, expansion, occlusion and boundary exit can support
  handoff. They are inputs to a controller, not an isolated collection of tests.
- Different lap positions are normal. Their feature-coordinate differences are
  not ground-truth steering errors or a desired zero for the live controller.

## Continuous guidance display

The main display answers: **What input should I give now, and is that advice
currently supported?** It should show a simple left/right correction cue with
strength, maintain/straighten indications through a turn, and a clear unavailable
state. A separate route-relative alignment cue is shown only when supported.
The meaning and polarity must remain stable when tracking targets change.

The user chooses a route and enables guidance. Individual target selection,
feature dots, IDs and detailed timelines belong in optional diagnostics. The
existing once-per-second diagnostic scene preview is insufficient as the final
responsive guidance display.

There is no valid input hint merely because a feature match exists. Unknown
travel relation or stale/conflicting evidence must not be rendered as centred,
straight-ahead, or a confident turn instruction. Suggested input units and gains
must correspond to the actual game's camera/movement response.

## Turns and narrow passages

On an approach, stable target relations can support frequent small corrections.
At a tight turn, a distant target may be absent. The candidate design uses the
demonstrated turn direction/rate as a prior, observed accumulated rotation and
actual input to track turn progress, and outgoing-scene acquisition to confirm
the transition back to approach guidance. Exact timing alone is insufficient.
This combination remains to be validated in the game.

Narrow-passage evaluation must check whether the advice improves passage and
preserves useful clearance. Neither a target centred in the camera nor agreement
with a previous lap is proof of successful travel alignment.

## Evidence gates and known limits

First validate the guidance on held-out recorded segments with explicit unknowns;
then run it while a human drives. Record advice, actual input, observations and
outcomes separately. Recorded human actions are useful context but not the unique
correct action at each instant, and replay cannot prove closed-loop success.

Report advice availability and longest loss, unsupported or wrong-direction
advice, handoff discontinuities, capture-to-visible-hint delay, unnecessary
oscillation, human intervention, passage outcomes and route completion. Use
native-image review and independent outcome evidence, not estimator consistency
as accuracy. Preserve target lifetime/own-bearing timelines with reviewed exits.

Current limits: Session 21 first-lap groups transfer poorly to later lighting and
viewpoint conditions. A three-traversal route-wide memory recognizes 476/559 later
samples but still has a 2.48 s longest unavailable interval. Feature extraction
has an unresolved intermittent multi-second stall; travel alignment and automatic
stage/handoff control are unvalidated.
Adding more targets or raising update frequency alone does not fix these limits.

Progression: continuous observation and stage memory -> useful manual input
guidance -> measured human-guided route trials -> automatic input execution.
The single-gate viewer is supporting infrastructure, not completion of the first
two steps. No runtime guidance or movement change is made by this document.
