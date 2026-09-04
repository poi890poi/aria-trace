# Route repeatability session

Use the `route_repeatability` session label for one uninterrupted recording
containing repeated traversals of the same short route. This is benchmark data,
not a demonstrated route and not calibration input.

## Recording protocol

1. Choose a short route covered by the current map atlas, with a visually
   distinctive start point. Avoid combat, menus, teleportation, and intentional
   camera framing changes.
2. Begin moving naturally; no stationary start or boundary gesture is needed.
3. Record at least three complete passes; five or more gives a more useful tail
   distribution.
4. Either run closed laps in one direction, or alternate between the same two
   endpoints. Do not pause or signal lap boundaries for the recorder.
5. Timing, speed, steering, and the exact path may vary naturally. Label the
   recording **Repeated route laps / back-and-forth** afterward.

Input events are optional evidence for control timing and direction. They must
never be treated as global-position or player-heading truth.

Sharp reversals in a back-and-forth recording are also temporal-response tests.
Compare candidate heading change against timestamp-aligned raw turn input and
KLT scene rotation. Input is control intent, KLT is observed camera motion, and
neither is player-heading truth. Fit time lag and sign convention first. Report
onset lag, wrong-direction duration, and missed reversals; declare a pose defect
only when independent evidence agrees, otherwise report the event inconclusive.

## Evaluation contract

No lap segmentation or boundary cue is required. Post-run reference states are
grouped into repeated nearest-waypoint neighborhoods using canonical XY, map
mode, and heading as one complete state. A continuous passage through a
neighborhood counts once. Opposite traversal directions remain separate
waypoints. Candidate algorithms never receive waypoint identity, reference
timing, demonstrated motion, or another method's position.

For sessions without external truth, repeatability is the dispersion of
candidate-minus-reference error inside each repeated-state subgroup. Subtracting
the reference displacement prevents natural differences between laps from being
misreported as estimator error. Report position in canonical pixels and heading
in degrees separately, along with fresh, held, unavailable, latency, and map-mode
results.

Use leave-one-family-out cross-method consensus as corroboration. Parameter
variants of one implementation receive one vote, and a candidate may not vote
on its own score. Cross-pass or cross-method agreement estimates repeatability;
it does not establish absolute localization accuracy. Any player path variation
or uncertain frame correspondence must be reported as an evidence limitation.
