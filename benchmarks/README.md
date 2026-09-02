# Offline benchmark policy

Benchmarks live here when they are expected to make or reject an engineering
decision. Production packages must not import this package.

Every benchmark must:

1. declare the complete input/output contract and whether processing is causal;
2. name every rate with its numerator and denominator;
3. keep fresh measurements, held/predicted state, and unavailable output separate;
4. report mean, median, and P95 accuracy, with worst error only for a complete
   chronological E2E output;
5. preserve raw per-sample rows and loss episodes rather than only aggregates;
6. separate independent truth, proxy agreement, and functional E2E evidence;
7. use frozen development and holdout session lists;
8. write configuration, input identity, environment, Git revision, and exact
   implementation/source hashes with every run; and
9. retain negative results under stable method IDs instead of deleting code or
   silently changing an existing method.

A report produced from dirty tested source is diagnostic only. Commit the
benchmark/method implementation and rerun before using it for a release or
landing decision.

Current protocols:

- [Cursor pose causal E2E benchmark](cursor_pose/README.md)

