# Route-wide manual guidance trial

Type: experimental route-memory and input-hint design; no game actuation.
This precedes any production guidance change. The existing destination viewer
and map tracker remain unchanged during the assay.

Question: can a compact memory select relevant scene groups throughout a lap,
and retrieve useful demonstrated turn actions without treating inter-lap image
offsets as steering error?

Development input: Run21 first three traversals, [0,41), [41,82), [82,124) s.
These boundaries are manually supplied experiment annotations. Build 0.75-second
scene exemplars with ALIKED descriptors already cached by previous experiments.
Only demonstration input may annotate exemplar actions. No network is trained.
Run21 [124,239.885] s is withheld from the memory, though previously inspected;
it is a new memory test, not an untouched final holdout. Run20 provides a
different-route negative control. Recorded timestamps are authoritative.

Retrieval: low-resolution normalized gradient signature ranks candidates;
mutual descriptor matching and geometric spread verify current scene identity.
The image transformation is used only for recognition and feature display, never
as a lateral or heading error. Local stage search and bounded global recovery
must be compared against unrestricted recognition. Input hints retrieve actions
from demonstrated context; they are action priors, not validated corrections.

Report fresh recognized observations, unavailability and longest loss; proposal
latency excluding cached extraction; stage consistency; action disagreement
between matching demonstrations; and disagreement with query input as diagnostic
only. Inspect native frames around turn suggestions. Absence of disagreement
does not prove correct guidance. Do not relabel neutral/no-turn demo input as a
proof of correct alignment. No precise XY enters this assay.

Promotion gate: recognition must cover the intended contexts with reviewed scene
identity and refuse unrelated controls. Demonstration actions must remain visibly
distinct from an ILS correction; actual correction needs an independently tested
travel relation. If the gate fails, do not expose confident steering suggestions
merely to make the display look complete.

Sequential follow-up, preserving the initial outputs: replace only gradient
ranking with a 32-centre residual aggregate of the same cached ALIKED descriptors.
Fit the centres on 12,000 teacher descriptors, with seed 914; no neural training.
Keep the same top-eight verification candidates and geometry. Test both global
and local search to distinguish representation gain from the ordering rule.
Then remove mandatory two-demo action voting as a separate diagnostic variant:
accept a sole strong exemplar, but withhold when candidates with at least 85%
of the best support disagree in action or span more than 0.13 normalized phase.
These cutoffs are declared prototype settings, not calibrated probabilities.
