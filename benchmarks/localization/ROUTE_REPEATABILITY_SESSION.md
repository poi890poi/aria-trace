# Route repeatability session

Use the `route_repeatability` session label for one uninterrupted recording
containing repeated traversals of the same short route. This is benchmark data,
not a demonstrated route and not calibration input.

## Recording protocol

1. Choose a short route covered by the current map atlas, with a visually
   distinctive start point. Avoid combat, menus, teleportation, and intentional
   camera framing changes.
2. Start stationary at the boundary point for about one second.
3. Record at least three complete passes; five or more gives a more useful tail
   distribution.
4. Either run closed laps in one direction, or alternate between the same two
   endpoints. Briefly pause at every lap boundary or turnaround.
5. Timing and speed may vary naturally. Finish stationary at a boundary point,
   then label the recording **Repeated route laps / back-and-forth**.

Input events are optional evidence for pass segmentation and direction. They
must never be treated as global-position truth.

## Evaluation contract

Passes are segmented after capture. Candidate algorithms never receive pass
number, reference timing, demonstrated motion, or another method's position.
Dynamic UI and the player cursor are masked before reciprocal visual sequence
matching aligns frames across passes. Alignment must support both forward and
reversed sequences and must not assume equal speed or timing.

For sessions without external truth, repeatability is measured from direct
canonical-XY disagreement on independently matched observations of the same
place. Report fresh, held, and unavailable outputs separately, plus mean,
median, P95, and worst repeatability error in canonical map pixels. Also report
per-pass closure error, loss episodes, longest loss, wrong map-scale decisions,
and the full processing latency distribution.

Use leave-one-family-out cross-method consensus as corroboration. Parameter
variants of one implementation receive one vote, and a candidate may not vote
on its own score. Cross-pass or cross-method agreement estimates repeatability;
it does not establish absolute localization accuracy. Any player path variation
or uncertain frame correspondence must be reported as an evidence limitation.
