# Genshin Seq-046 yaw POC results

The portal-start experiment is documented in [PORTAL_INITIALIZATION_RESULTS.md](PORTAL_INITIALIZATION_RESULTS.md). Its bidirectional reference map confirmed all 6/6 held-out arrival episodes; a one-direction reference confirmed 4/6 and missed both opposite-facing episodes.

## Offline reference

- COLMAP model: one coherent reconstruction
- Registered frames: 216 / 216
- Sparse points: 54,617
- Mean reprojection error: 0.991 px
- Estimated camera: SIMPLE_RADIAL, `fx = fy = 1042.31 px`, `cx = 718`, `cy = 498`, `k = -0.002286`

The COLMAP reference uses incremental local view-yaw rotations. It is a strong offline pseudo-reference, not engine-provided ground truth.

## Angular-flow backend at 30 Hz

- Tracking failures: 0 / 2,159 updates
- Runtime: 35.55 ms mean, 37.08 ms p95
- Sparse-interval yaw correlation: 0.912
- Sparse-interval yaw MAE: 0.701 degrees
- Accumulated yaw MAE: 17.40 degrees
- Maximum accumulated error: 42.64 degrees

This is useful as a fast local turning signal, but its small interval biases accumulate too far for global heading.

## Essential-matrix backend

At 30 Hz, all 299 probe updates failed pose recovery because consecutive frames had insufficient translational baseline.

At a 10-frame / 3 Hz baseline:

- Failed updates: 42 / 215
- Runtime: 43.29 ms mean per processed pair
- Sparse-interval yaw correlation: 0.741
- Sparse-interval yaw MAE: 0.590 degrees
- Accumulated yaw MAE: 7.91 degrees
- Final yaw error: -14.88 degrees

This backend is not reliable enough to replace the angular-flow backend or act as a global correction source.

## Decision

Keep angular flow for immediate visual control. Obtain global heading corrections from absolute visual localization, the minimap, or another world-referenced observation. Do not rely on integrating either relative backend indefinitely.

## Absolute relocalization

An even/odd split used 108 sparse frames to build a map and held out 108 frames as queries. All query-to-query match edges were removed before registration, so queries could connect only to the fixed map.

- Map: 108 / 108 registered, 34,829 points, 0.955 px mean reprojection error
- Held-out registration: 108 / 108
- Offline batch registration time: 4.07 s
- Query rotation error: 0.041 degrees median, 0.075 degrees p95, 0.218 degrees maximum
- Query position RMSE: 0.00892 reconstruction units, 0.0247% of reference path length

The independently reconstructed map was aligned to the full pseudo-reference with a Sim(3) fit using map frames only. Query frames were excluded from alignment.

This is strong evidence that stored-map pose correction works on nearby views from the same recording. It is not engine ground truth and does not test route deviation, a second session, dynamic scene changes, or USB-camera capture. Monocular position remains scale-ambiguous and is not yet metric.

## Cross-traversal results

TartanAir V2 `ArchVizTinyHouseDay/Data_easy` provides different metric trajectories through the same environment. P000 supplied the fixed 116-image map.

### P003 balanced route

- 152 held-out query images
- 67 poses returned by SIFT/COLMAP
- 59 valid poses below 0.25 m and 5 degrees: 38.8% total valid coverage
- Median error over returned poses: 0.0054 m and 0.087 degrees
- Eight false poses, all from queries more than 2 m outside the mapped trajectory
- Valid coverage by nearest-map distance: 22/22 below 0.5 m, 6/8 at 0.5-1 m, 31/62 at 1-2 m, and 0/60 at 2-4 m

### P005 reverse-view stress test

P005 is positionally close but looks in nearly the opposite direction; its median closest map-view direction difference is 146 degrees.

- SIFT returned one false pose from 118 queries.
- ALIKED/LightGlue returned 36 poses with oracle candidate retrieval, but all 36 were false.
- The learned backend's median returned error was 1.34 m and 166.9 degrees.

### Decision

Feature-map localization is viable only inside the map's position-and-view coverage. Build maps with both travel directions. Treat every PnP output as a hypothesis and gate it against coarse position, expected heading, temporal continuity, and pose-jump limits before fusion.

## Replay-time fusion and false-pose rejection

The replay uses a minimal planar predictor plus an absolute-pose consistency gate. It is not a full EKF. TartanAir ground truth and the previously produced PnP hypotheses are measured inputs. Local control motion, relative-heading noise, and the coarse prior are explicitly simulated because no phone control log exists.

Across 100 deterministic-noise trials:

- P003 supplied 59 valid and 8 false hypotheses per trial. The gate accepted 99.983% of valid hypotheses and 0 false hypotheses. The only valid rejection in 5,900 opportunities was caused by a simulated coarse-prior outlier.
- P003 gated position error was 0.273 m mean p95 across trials and heading error was 4.775 degrees mean p95. Trusting every PnP output instead produced 6.447 m and 114.91 degrees mean p95, with a worst 4.72 m instantaneous correction.
- P005 supplied 36 reverse-view false hypotheses per trial and no valid hypotheses. The gate rejected all 3,600 false hypotheses. Trusting every output produced 3.003 m and 177.81 degrees mean p95, with a worst 8.31 m correction.
- With no valid P005 absolute correction, uncertainty moved the navigation mode from `TRACK` toward `CAUTIOUS`, `RELOCALIZE`, and occasionally `STOP` instead of pretending that the pose was known.

The result demonstrates the purpose of the gate under the tested noise model. It does not validate Genshin motion physics, minimap accuracy, camera/character coupling, or real-time latency. Those require aligned game video and controls.
