# Portal Initialization POC

## Question

Can a known arrival location plus several visual matches initialize camera position and heading when a new session does not follow the recorded route?

## Dataset surrogate

GID contains different Genshin locations rather than repeated portal arrivals, so it cannot answer this question. The test uses TartanAir V2 `ArchVizTinyHouseDay/Data_easy`, which provides independent RGB trajectories and metric camera poses. Six trajectories pass within 0.35 m of a common point while viewing it from nearly opposite directions.

The synthetic portal center is `[0.8, 0.46, -0.85]` meters. Frames within 0.75 m form arrival episodes. P000 and P002 are reference sessions; P003 through P006 are held out. Query-to-query match edges are removed, so held-out frames localize only against the fixed portal map.

This is a valid test of spatial camera initialization. It is not a test of teleport/loading detection, Genshin UI or weather changes, a physical-camera recapture, or third-person character heading.

## Results

| Reference coverage | Valid query frames | Confirmed arrival episodes | False confirmations |
|---|---:|---:|---:|
| P000, one viewing direction | 39/52 | 4/6 | 0 |
| P000 + P002, opposing directions | 52/52 | 6/6 | 0 |

The one-direction map failed on all 13 P005 frames and both eligible P005 arrival episodes, even though their positions were inside the same portal region. Their cameras faced the opposite way.

With both directions represented, query camera error was 0.0020 m median / 0.0058 m p95 and 0.053 degrees median / 0.112 degrees p95. Every eligible episode confirmed after its first three frames.

Confirmation required three consecutive returned poses, all inside the portal prior, with at most 0.5 m translation and 15 degrees rotation between adjacent estimates. No query-to-query feature matching was available.

## Decision

A portal profile must store views from all camera headings that may occur after arrival. The live initializer should:

1. Use the selected portal as a position prior.
2. Match raw live frames only against that portal's local map.
3. Estimate camera pose independently for each frame.
4. Require three consistent poses before navigation starts.
5. Keep character heading unknown until game-specific motion reveals it.

## Artifacts and reproduction

- Bidirectional result: `artifacts/portal_init_tartanair_bidirectional/portal_evaluation.json`
- One-direction control: `artifacts/portal_init_tartanair_one_direction/portal_evaluation.json`
- Split preparation: `poc/prepare_tartanair_portal.py`
- Multi-frame evaluation: `poc/evaluate_portal_initialization.py`

Both pipelines use COLMAP GPU SIFT feature extraction, exhaustive matching, a TartanAir ground-truth-pose map, point triangulation, query-to-query edge removal, image registration, and metric evaluation. Exact image lists and machine-readable results are retained inside each artifact directory.

Dataset references: [TartanAir documentation](https://tartanair.org/) and [official tools](https://github.com/castacks/tartanair_tools). GID coverage is described by its [official repository](https://github.com/zhaoxuhui/Genshin-Impact-Dataset).
