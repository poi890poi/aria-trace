# Local map relation plus scene context: control screen

Type: offline design/algorithm experiment. No live guidance, atlas changes or
game input execution. Continue the continuous-manual-guidance product contract.

Hypothesis: a north-fixed map supplies a physical local travel relation that
whole-scene correspondence cannot. A current query's displacement relative to
a teaching image and the displacement to a future teaching point give a
current-to-goal vector. Past-only map motion supplies current travel direction.
Their signed angular difference is a candidate course correction. It is not
full atlas localization and does not require world XY.

Preparation: Run21 first three traversals, before124 s; compare one- and
three-second demonstrated lookahead. Current frames >=124 s are excluded from
preparation but have been inspected previously. The existing causal scene
retrieval selects the teaching context. No future query frames, inputs or
outcome labels may select or refine a runtime hint. Input is diagnostic only
in this first screen. Bound scene-context reuse to one second and distinguish
reused context from a freshly measured current map relation.

Use the existing calibrated minimap mask and even-DFT phase correlation as the
baseline. First test identical images and known pixel translations, then
native map registration and subregion consistency. Keep measured correlation,
spatial-subset disagreement, update cost and explicit unavailable output. A
large correlation score is not independently verified physical accuracy.

Assess observable direction and interval scale before choosing control gains.
For native examples, inspect minimap correspondence and scene context, especially
narrow passes and tight turns. A long lookahead may cut across a corner; a
short lookahead may be dominated by map quantization. Do not infer correct
advice from agreement with demo input or from forcing the output onto the route.
Known image perturbations and a declared feedback fixture can test response,
not establish successful game travel. If the relation survives this screen,
compare feedback, maneuver prior and their combination on the same test cases.

Record source timestamps; cache native crops for reuse. Report availability,
longest unavailability, discontinuities and processing latency separately from
capture/display. No high-frequency or closed-loop product claim from replay.
