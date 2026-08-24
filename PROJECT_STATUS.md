# Project status and continuation guide

Last updated: 2026-08-24 (Asia/Taipei)

## Current decision

AriaTrace remains vision-guided, closed-loop reproduction of a human-demonstrated gameplay route rather than fixed-time macro playback. The immediate milestone is now the observation and game-modeling foundation: build a semi-automatic, human-editable game profiler; automatically acquire the complete high-resolution map from the full in-game map viewer; calibrate the circular mini-map; model useful mini-map motion and character-facing signals from a cruise; and compile a human gameplay recording into an inspectable route descriptor.

The route descriptor will select keyframes, group local evidence into short submaps, retain raw relative measurements separately from reconstructed geometry, and maintain a route centerline indexed by cumulative progress as well as spatial position. Every selected mini-map observation must remain traceable to the original full game frame, exact timing, controls, and diagnostics. One stitched map image is never authoritative.

The completed KLT, feature-map relocalization, portal initialization, and fusion-gate POCs below remain valid experimental evidence. They are not the committed immediate replay/localization stack. Their main retained conclusions are that relative estimates drift, view coverage matters, pose solves can be catastrophically wrong, and independent quality/consistency evidence is necessary.

Do not design the autonomous replay subsystem in detail yet. After real route recordings exist, use observed failures to choose among live route localization, mini-map/full-scene combination, route-relative state, closed-loop correction, latency/rate work, constrained traversal, and divergence recovery.

## Environment

- Workspace: the current checkout (`Resolve-Path .`)
- Project root: repository-relative commands assume the current directory
- OS/shell: Windows / PowerShell
- Python: 3.7.9
- Local Python dependencies: `.tools`; set `$env:PYTHONPATH=(Resolve-Path .tools).Path`
- GPU: NVIDIA GeForce GTX 1650, 4 GB, compute capability 7.5
- CUDA COLMAP 4.1.1: `.tools\colmap-cuda\bin\colmap.exe`
- CPU COLMAP 4.1.1: `.tools\colmap\bin\colmap.exe`
- COLMAP source used for diagnostics: `third_party\colmap-src`, tag 4.1.1

Install Python dependencies in a fresh environment with:

```powershell
python -m pip install -r requirements-poc.txt
```

## Test data and artifacts

- Genshin Seq-046 video: `data\gid\Seq-046\Seq-046.mp4`
- Video: 1436x996, 30 fps, 72 s, 2160 frames
- Sparse images: `data\gid\Seq-046\Frames-Sparse` (216 images at 3 fps)
- Full feature database: `artifacts\colmap_seq046\database_4k.db`
- Full reference model: `artifacts\colmap_seq046\sparse\0`
- Reference text export: `artifacts\colmap_seq046\reference_text`
- Relocalization experiment: `artifacts\relocalization_seq046`
- TartanAir V2 archive: `data\tartanair2\ArchVizTinyHouseDay\Data_easy\image_lcam_front.zip`
- Extracted TartanAir trajectories: `data\tartanair2\ArchVizTinyHouseDay\Data_easy\P000` through `P006`
- Balanced cross-traversal experiment: `artifacts\relocalization_tartanair_tinyhouse_p000_p003`
- Reverse-view stress test: `artifacts\relocalization_tartanair_tinyhouse_p000_p005`
- Fusion replay: `artifacts\fusion_replay`
- H.264 acquisition smoke: `artifacts\acquisition_smoke_h264`
- Raw-frame feature acquisition smoke: `artifacts\acquisition_smoke_features_v2`
- Portal annotation recording smoke: `artifacts\portal_annotation_smoke`
- Extracted lossless portal interval: `artifacts\portal_init_smoke`
- TartanAir bidirectional portal initialization: `artifacts\portal_init_tartanair_bidirectional`
- TartanAir one-direction portal control: `artifacts\portal_init_tartanair_one_direction`

`data/`, `artifacts/`, `.tools/`, and `third_party/` are local working assets; verify their presence on a different checkout.

The supplied Genshin orientation columns were rejected as ground truth: their quaternion norms ranged from 0.000124 to 0.369 and contained large discontinuities. The full COLMAP model is therefore only a pseudo-reference.

## Experiments completed

### Canonical acquisition recorder: Windows adapters

The canonical acquisition recorder now has dependency-free Windows adapters behind its existing source interfaces; the same recorder and session schema will accept the later Android/UVC sources. `WindowsWindowFrameSource` selects one visible window by an exact title or unambiguous substring and captures its client area through GDI. `WindowsKeyboardMouseSource` records foreground keyboard state, mouse buttons, and absolute client cursor changes on the common PC monotonic timeline. The standard recorder exposes these as `--window TITLE --pc-input`; sessions use the same video, JSONL, inspection, review, and annotation workflow as the mobile-oriented sources.

A one-second real-desktop H.264 smoke captured 31 frames at 1920x1020 with zero drops, 34.03 ms median frame interval, two input-state observations, and a complete inspectable manifest. Deterministic tests cover ambiguous title rejection, frame metadata, foreground state, keyboard/button evidence, and machine-independent tool discovery.

Limitations: GDI captures visible screen pixels, so the game window must remain unobstructed and at a fixed size. WindowsRawKeyboardMouseSource is now the faithful keyboard/mouse path: a message-only Raw Input receiver records keyboard make/break scan codes, extended-key flags, raw relative mouse deltas, button/wheel transitions, device handles, foreground state, and PC monotonic timestamps. WindowsXInputSource remains the faithful controller path. The earlier polled keyboard/absolute-cursor source is retained only as a compatibility fallback.

### Generic acquisition workbench

The local Acquisition Workbench is the primary PC MVP entry point. It lists visible Windows windows, loads independent game and route profiles, and configures AcquisitionRecorder without importing game-specific code. A custom/unprofiled game remains a first-class choice.

Acquisition is now zero-interruption. Queueing a take pre-arms frame and input sources while the workbench is still focused, but pre-start observations are discarded. After the selected game has focus, the first qualifying gameplay input is retained at session time zero and starts the configured-duration clock. This removes the source-startup gap without spending recording time merely because the game gained focus. There are no gameplay hotkeys or stage/completion inputs. Early focus loss marks the take failed. Successful captures receive take_start and take_end evidence boundaries; route_start and route_complete are created only through explicit post-take confirmation. Compilation remains blocked until each take is confirmed.

Landmarks are visual reference observations used by later alignment, not buttons or actions expected from the player. They are derived or corrected after recording. Raw Input is the recommended keyboard/mouse evidence source because it preserves key timing and locked-camera relative mouse motion; XInput is recommended for controller play because it preserves analog locomotion speed, camera axes, triggers, buttons, and timing. A configured input stream with zero control events is now rejected instead of being confirmable as ready, and final Raw Input receive/filter counters are stored in the manifest for diagnosis.

Start with python -m acquisition.workbench and open http://127.0.0.1:8765/. Arbitrary-game tests use a fake Popular Game A profile to prove that the workflow contains no Combat Master coupling and no in-game recorder commands.
### Data acquisition suite

`acquisition/` now records concurrent frame streams and raw Android or synthetic input events on a common PC monotonic timeline. It writes exact timing to JSONL sidecars, defaults to replaceable H.264/Matroska storage through FFmpeg, retains MJPEG as an explicit fallback, reports storage rates, and serves a local frame/input reviewer. `AdbGetEventSource` maps Android kernel timestamps using lowest-RTT `/proc/uptime` samples while retaining the original timestamp.

Online frame processors run on the decoded raw frame before video encoding. The implemented `OnlineSiftRecorder` samples at a configurable cadence into `evidence/online_sift_v1/features.sqlite3`; each observation records the exact video frame index, raw-frame SHA-256, keypoints, losslessly represented descriptors, extractor configuration, and optionally the corresponding lossless PNG. Later features regenerated from H.264 belong under `derived/` and must not be confused with this evidence.

Measured on Genshin Seq-046 at 1436x996:

- Three-second H.264 smoke: 91 frames, zero drops, 1.34 MiB video versus 12.54 MiB for the earlier MJPEG smoke (9.4x smaller).
- Five-second H.264 plus online SIFT at 1 Hz: 152 frames, zero drops, five feature observations / 16,352 keypoints. Video was 2.91 MiB and the feature database 1.82 MiB. The short-sample projection was 3.34 GiB/hour total, including about 1.27 GiB/hour of SIFT evidence.
- Current full-suite checkpoint: 42/42 passing, including H.264 frame-index decode, exact lossless keyframe recovery, raw-feature sampling, clock mapping, multistream recording, deferred first-input session start, Raw Input decoding with the workbench focus gate as authority, XInput behavior capture, post-take confirmation, capture-safe HUD behavior, arbitrary-game profile orchestration, replay compilation/alignment, and pose-fusion gates.

During the first feature smoke, shutdown triggered an OpenCV/FFmpeg decoder assertion because the main thread released a source while its capture thread was inside `read()`. Shutdown now signals the worker, waits for normal exit, and only force-stops a still-blocked source. The repeated smoke completed cleanly.

Reproduce:

```powershell
$env:PYTHONPATH=((Resolve-Path .tools).Path + ';' + (Resolve-Path .).Path)
python -m acquisition.record --output artifacts\acquisition_smoke_features_v2 --video genshin=data\gid\Seq-046\Seq-046.mp4 --synthetic-input --duration 5 --online-features sift --feature-rate 1
python -m acquisition.inspect_session artifacts\acquisition_smoke_features_v2
python -m acquisition.review artifacts\acquisition_smoke_features_v2
python -m unittest discover -s tests -v
```

Hardware validation and crash-safe long-session chunking remain open. Do not infer UVC timing, controller visibility, or sustained capture performance from file-video smoke tests.

Portal lifecycle annotation and initialization-set extraction are also implemented. The reviewer can append or delete `teleport_start`, `world_ready`, and `route_start` markers tied to portal/route IDs and exact frames. `acquisition.extract_portal_init` exports the marked interval, preferring lossless raw-frame evidence and explicitly labeling H.264 fallback. The end-to-end smoke recorded 122 frames, marked `world_ready` through `route_start`, and exported three lossless raw observations with 3,005, 3,153, and 3,582 SIFT keypoints. The five lossless feature samples made the evidence database 9.48 MiB, projecting 8.24 GiB/hour for that short sample; use this mode only for short portal captures or selected events. This prepares portal-map inputs; it does not yet build the portal 3D map or perform live multi-frame initialization.

### Portal-start camera initialization

GID has no repeated portal arrivals, so TartanAir `ArchVizTinyHouseDay/Data_easy` was used as a controlled surrogate. Six trajectories pass within 0.35 m of a synthetic portal near `[0.8, 0.46, -0.85]` m with nearly opposite camera headings. P000/P002 are map sessions and P003-P006 are held-out arrivals. Query-to-query match edges were removed.

- One-direction P000 map: 39/52 valid query frames, 4/6 eligible arrival episodes confirmed, zero false confirmation. Both opposite-facing P005 episodes failed completely (0/13 frames).
- Bidirectional P000+P002 map: 52/52 valid frames, 6/6 episodes confirmed, zero false confirmation.
- Bidirectional errors: 0.0020 m position median / 0.0058 m p95; 0.053 degree rotation median / 0.112 degree p95.
- Confirmation required three consecutive poses within the portal prior and bounded inter-frame motion. All successful episodes confirmed on their first three frames.
- Test suite: 15/15 passing.

Conclusion: the known portal is a useful position prior but does not compensate for missing view coverage. Record portal arrivals from every plausible camera heading and require multi-frame confirmation. This POC estimates camera pose only. It does not test actual teleport/loading transitions, Genshin changes, physical-camera recapture, real-time retrieval, or character heading. Full details are in `poc/PORTAL_INITIALIZATION_RESULTS.md`.

### Full offline reference

COLMAP reconstructed 216/216 sparse frames as one model with 54,617 points, mean track length 12.20, and mean reprojection error 0.991 px. Its optimized camera is `SIMPLE_RADIAL`, 1436x996, focal length 1042.31 px, principal point (718, 498), radial parameter -0.002286.

### Relative yaw

- KLT angular flow at 30 Hz: no tracking failures; 35.55 ms mean, 37.08 ms p95; interval yaw correlation 0.912 and MAE 0.701 degrees; accumulated MAE 17.40 degrees.
- KLT essential matrix at 30 Hz: all 299 probe updates failed due insufficient baseline.
- KLT essential matrix at 3 Hz: 42/215 failures; interval MAE 0.590 degrees; accumulated MAE 7.91 degrees.

Conclusion: angular flow is useful locally, but neither relative backend supplies global heading.

### Held-out absolute relocalization

The 216 sparse images were split even/odd into 108 map images and 108 held-out queries. The map was reconstructed from map images only. Query-to-query match edges were then removed, so held-out images could register only through the fixed map.

- Map build: 108/108 frames, 34,829 points, 0.955 px mean reprojection error, 130.9 s.
- Relocalization: 108/108 held-out queries, 4.07 s for the offline batch.
- Query rotation error against the aligned pseudo-reference: 0.041 degrees median, 0.075 degrees p95, 0.218 degrees maximum.
- Query position RMSE: 0.00892 reference-model units, or 0.0247% of the reconstructed query path length.
- Unit tests: 4/4 passed.

The two reconstruction gauges were aligned with a Sim(3) fitted only to map camera centers. Query errors were then evaluated without using query poses for alignment.

This proves same-recording relocalization under a dense alternating split. It does not establish performance under a different route, a different play session, dynamic scene/UI changes, image recapture from a physical camera, or strict online latency. The positional units are not meters.

### Contiguous Genshin traversal split

A traversal means one continuous path/playthrough through an area. Seq-046 has no independent second recording, so its first 108 frames were mapped and its later 108 frames were queried. Exhaustive map-to-query matching was used and query-to-query edges were removed.

- Registered: 108/108 later frames.
- Rotation error: 0.184 degrees median, 0.323 degrees p95.
- Position RMSE: 0.0306 reconstruction units.

This is harder than alternating frames but remains one session around one landmark. GID's published route map shows that its separate sequences do not repeat Seq-046's location, so TartanAir was used for a true multi-trajectory test.

### TartanAir cross-traversal localization

Source: [official TartanAir V2 documentation](https://tartanair.org/) and [official Hugging Face dataset](https://huggingface.co/datasets/theairlabcmu/tartanair2). `ArchVizTinyHouseDay/Data_easy` supplies 640x640, 90-degree-FoV images, metric poses, and seven different trajectories through the same house. The tested camera model is `SIMPLE_PINHOLE`, focal length 320 px, principal point (320, 320).

P000 is the fixed map traversal. Its supplied metric poses were used to triangulate the map, deliberately isolating query localization from map-building errors. Query-to-query edges were absent.

#### Balanced route: P000 map, P003 queries

- Images: 116 map, 152 query.
- Backend: 4,096-feature SIFT, exhaustive offline matching, PnP registration.
- Verified map-to-query pairs: 1,575 / 17,632 possible.
- Solver returned poses: 67/152 (44.1%).
- Valid poses under 0.25 m and 5 degrees: 59/152 (38.8%).
- False returned poses: 8.
- Within 0.5 m of the mapped trajectory: 22/22 valid.
- From 0.5 to 1 m: 6/8 valid.
- From 1 to 2 m: 31/62 valid.
- From 2 to 4 m: 0/60 valid; eight solver outputs in this band were false.

The result shows accurate cross-traversal localization where the map has view coverage, and unsafe false positives outside that coverage. A PnP success flag must never directly reset navigation pose.

#### Reverse-view stress test: P000 map, P005 queries

P005 is spatially close to P000 but looks in nearly the opposite direction: the median closest map-view direction difference is 146 degrees, and no query has both <1 m distance and <30 degree view overlap.

- SIFT: only 17 exhaustive cross-pairs verified; 1/118 query poses returned and it was false.
- ALIKED/LightGlue with oracle metric nearest-neighbor candidates: 824/1,180 pairs verified and 36/118 poses returned, but 0/118 met the 0.25 m / 5 degree validity threshold.
- Median returned ALIKED pose error: 1.34 m and 166.9 degrees.
- ALIKED extraction took 114.7 s on CPU; 1,629 LightGlue candidate pairs took 445.2 s on CPU.

This is a map-coverage failure, not merely a feature-count failure. A route map must record relevant locations from both travel directions or use a representation that supports extreme viewpoint change.

The installed COLMAP ONNX CUDA provider failed because `cublasLt64_12.dll` is missing, so learned extraction/matching used CPU. Classic SIFT continued using CUDA normally.

## Reproduce the relocalization test

The commands below assume `database_4k.db` and the full reference model already exist. Use a new output directory or remove old generated artifacts deliberately before rerunning; the database preparation script refuses to overwrite its output.

```powershell
python poc\prepare_relocalization_split.py `
  --images data\gid\Seq-046\Frames-Sparse `
  --output artifacts\relocalization_seq046 `
  --map-stride 2

New-Item -ItemType Directory artifacts\relocalization_seq046\map_sparse | Out-Null
.\.tools\colmap-cuda\bin\colmap.exe mapper `
  --database_path artifacts\colmap_seq046\database_4k.db `
  --image_path data\gid\Seq-046\Frames-Sparse `
  --output_path artifacts\relocalization_seq046\map_sparse `
  --Mapper.image_list_path artifacts\relocalization_seq046\map_images.txt `
  --Mapper.multiple_models 1 `
  --Mapper.max_num_models 5 `
  --Mapper.min_model_size 10 `
  --Mapper.ba_refine_focal_length 1 `
  --Mapper.ba_refine_extra_params 1 `
  --Mapper.num_threads 6

python poc\prepare_relocalization_database.py `
  --input artifacts\colmap_seq046\database_4k.db `
  --output artifacts\relocalization_seq046\map_query_only.db `
  --query-list artifacts\relocalization_seq046\query_images.txt

New-Item -ItemType Directory artifacts\relocalization_seq046\registered_map_only_edges | Out-Null
.\.tools\colmap-cuda\bin\colmap.exe image_registrator `
  --database_path artifacts\relocalization_seq046\map_query_only.db `
  --input_path artifacts\relocalization_seq046\map_sparse\0 `
  --output_path artifacts\relocalization_seq046\registered_map_only_edges `
  --Mapper.ba_refine_focal_length 0 `
  --Mapper.ba_refine_extra_params 0 `
  --Mapper.num_threads 6

New-Item -ItemType Directory artifacts\relocalization_seq046\registered_map_only_edges_text | Out-Null
.\.tools\colmap-cuda\bin\colmap.exe model_converter `
  --input_path artifacts\relocalization_seq046\registered_map_only_edges `
  --output_path artifacts\relocalization_seq046\registered_map_only_edges_text `
  --output_type TXT

$env:PYTHONPATH=((Resolve-Path .tools).Path + ';' + (Resolve-Path .).Path)
python poc\evaluate_relocalization.py `
  --reference-images artifacts\colmap_seq046\reference_text\images.txt `
  --registered-images artifacts\relocalization_seq046\registered_map_only_edges_text\images.txt `
  --map-list artifacts\relocalization_seq046\map_images.txt `
  --query-list artifacts\relocalization_seq046\query_images.txt `
  --output artifacts\relocalization_seq046\evaluation_map_only_edges
python -m unittest discover -s tests -v
```

Machine-readable results are in `artifacts\relocalization_seq046\evaluation_map_only_edges\summary.json` and per-image errors are in `errors.csv`.

## Diagnostic note

An initial every-third-frame split failed with `No images with matches`. It was not a filename/newline problem. The existing COLMAP sequential matcher had generated temporal offsets 1, 2, 4, 8, 16, 32, 64, and 128; every-third selection therefore had zero map-to-map verified pairs. The even/odd split has 627 map-to-map and 215 map-to-query verified pairs after removing 629 query-to-query pairs.

## Cross-traversal reproduction entry points

The P003 workflow is composed from these scripts, in order:

```powershell
python poc\prepare_tartanair_relocalization.py --data-root data\tartanair2\ArchVizTinyHouseDay\Data_easy --map-trajectory P000 --query-trajectory P003 --output artifacts\relocalization_tartanair_tinyhouse_p000_p003
python poc\prepare_relocalization_database.py --input artifacts\relocalization_tartanair_tinyhouse_p000_p003\database.db --output artifacts\relocalization_tartanair_tinyhouse_p000_p003\map_query_only.db --query-list artifacts\relocalization_tartanair_tinyhouse_p000_p003\query_images.txt
python poc\create_tartanair_ground_truth_model.py --data-root data\tartanair2\ArchVizTinyHouseDay\Data_easy --image-list artifacts\relocalization_tartanair_tinyhouse_p000_p003\map_images.txt --output artifacts\relocalization_tartanair_tinyhouse_p000_p003\gt_map_input
python poc\evaluate_tartanair_relocalization.py --data-root data\tartanair2\ArchVizTinyHouseDay\Data_easy --registered-images artifacts\relocalization_tartanair_tinyhouse_p000_p003\registered_text\images.txt --map-list artifacts\relocalization_tartanair_tinyhouse_p000_p003\map_images.txt --query-list artifacts\relocalization_tartanair_tinyhouse_p000_p003\query_images.txt --output artifacts\relocalization_tartanair_tinyhouse_p000_p003\evaluation
```

Between these scripts, the executed COLMAP stages were `feature_extractor`, `exhaustive_matcher`, `point_triangulator`, `image_registrator`, and `model_converter`. Exact option values and timings are retained in the experiment's `logs/` directory and machine-readable summaries are in `evaluation/summary.json`.

The learned P005 upper bound additionally uses `prepare_tartanair_candidate_pairs.py --query-neighbors 10`, ALIKED N16ROT extraction, and `matches_importer --match_type pairs --FeatureMatching.type ALIKED_LIGHTGLUE`.

## Prior pose-stack POC outcome

The implemented replay-time fusion and rejection gate in `poc/pose_fusion.py` and `poc/replay_pose_fusion.py` tested KLT-style relative prediction, a synthetic coarse route prior, retrieved feature-map/PnP hypotheses, and an independent consistency gate. Across 100 trials it accepted no false P003 or P005 pose while accepting 99.983% of valid P003 poses. A naive reset-to-every-PnP baseline produced multi-meter and near-opposite-heading failures. Machine-readable results are in `artifacts/fusion_replay/summary.json`; seed-zero frame logs are beside it.

This remains a gate POC, not validated gameplay replay. Its motion and coarse-prior observations are synthetic. KLT and feature/PnP may later become useful asynchronous signals, but the project no longer prescribes that combination before mini-map route recordings reveal actual needs.

## Implemented foundation and current gaps

Genshin Impact PC is now the first POC game profile. The integrated workbench exposes three evidence stages: one basic gameplay sample shared by behavior, ordinary UI, mini-map calibration, and cruise modeling; full-map viewer capture; and repeated route recording. The basic sample asks for a short cruise, rotation-only, and movement-only segment. Profile drafts are stored under workbench artifacts without rewriting source profiles. A versioned POC evidence index tracks confirmed/required progress and links each stage to source sessions, marker state, timing/count summaries, drop counts, and the profile-draft snapshot. A Windows in-game HUD reports waiting for focus, waiting for first input, recording countdown, finalizing, completion, and failure without stealing focus; it runs in an isolated process and fails closed unless Windows confirms capture exclusion. Non-route evidence uses distinct capture markers and is blocked from route compilation; the route stage retains the existing reference/held-out compile flow.

The canonical recorder preserves full-frame video, exact frame timing, raw PC or Android input evidence, source configuration, drop counts, and append-only markers. Existing `profiles/games` records capture/input defaults and coarse control summaries, while route-specific setup remains under `profiles/routes`. The current `ReplayPackage` compiler samples reference frames, descriptors, route stages, and action priors with links to source-frame indices.

The following milestone capabilities are not implemented yet:

- automatic inference and validation of detailed control/behavior fields from the recorded profiler probes;
- automated map-viewer input, complete high-resolution coverage verification, registration, and full-map artifact construction;
- automatic circular mini-map calibration or its structured/debug artifacts;
- normalized masked mini-map extraction;
- quality-bearing XY shift and character-facing estimators;
- cruise movement/rotation statistics or stability modeling;
- keyframe/local-submap route descriptor with raw measurement relationships and cumulative progress;
- synchronized route diagnostics and result-level correct/suspicious/wrong review annotations.

The existing marker-only annotation model cannot yet attach review state to an estimator measurement. The current `ReplayPackage` also uses time-sampled reference frames and a monotonic progress index; it is a useful foundation, not yet the selected keyframe/local-submap descriptor described in `SDS.md`.

## Next milestone: profiled map and route recorder

Work proceeds from three player-facing evidence stages, with the first session supplying several modeling tasks:

1. **Basic gameplay sample and game profiler.** Record one short cruise, one rotation-only segment, and one movement-only segment with ordinary gameplay UI visible. Extend the existing game profile from this shared input/observation evidence and human editing. Record semantic actions, physical bindings, activation semantics, compatible combinations, movement/camera response, relevant timing, confidence, provenance, and unknowns. The same frames and controls also seed mini-map calibration and cruise modeling; those are downstream analyses, not extra instructions for the player.
2. **Full-map acquisition.** Use the profiled map-viewer controls to open and automatically traverse the full in-game map, capture overlapping views at greater effective detail than the mini-map, verify complete coverage, and produce source tiles, transforms, coverage/quality diagnostics, and derived map products. Do not assume the full-map and mini-map coordinate systems are identical.
3. **Gameplay route descriptor.** Record a human route, select keyframes with source-frame references, group them into experimentally sized local submaps, retain raw relative measurements independently of replaceable geometry, and maintain both spatial position and cumulative progress `s`. Add derived route-strip/local-map/trajectory views and structured review annotations without making a stitched route image authoritative.

From stage 1, the mini-map calibration task uses the rotation-only frames to evaluate temporal stability, threshold/edge processing, and circle detection. The cruise-model task uses the cruise and movement-only frames to evaluate overlapping mini-map views, masked Fourier/phase correlation for `(dx, dy)`, game-specific cursor modeling, movement/rotation distributions, and motion-aligned temporal stability. Both retain confidence, provenance, timing, and intermediate products; neither requires another player-facing recording unless evidence quality later proves insufficient.

Individual catastrophic measurements and gradual accumulated drift are separate evaluation categories. The first descriptor must allow measurements to be flagged or rejected without deleting original values. Local submaps bound immediate drift; revisit evidence may be retained for later correction, but no loop-closure or optimization method is required now.

Timing evaluation begins with the recorder: observation age, estimator rate, processing and capture-to-estimate latency, stale evidence, and source provenance must remain visible. Future observation sources may run asynchronously.

After this milestone, inspect real recordings and choose the next problem. Do not pre-commit a global optimizer, replay pose representation, controller, constrained-traversal policy, or recovery design. The development sequence is `record -> model -> inspect -> evaluate -> choose next problem -> improve`.

Minecraft may validate plumbing only. Genshin Impact PC is the selected first third-person POC game; Combat Master remains an additional FPS profile-data candidate rather than the immediate milestone. No simple-environment result establishes target-game performance. Physical Android/UVC timing and capture validation also remain outstanding.
