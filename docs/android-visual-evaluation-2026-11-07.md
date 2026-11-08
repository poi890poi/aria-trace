# Visual evaluation and intent results — November 7, 2026

Naming update, November 8: the current package is SwifTest (`swiftest`). Names,
hashes and results below describe the measured pre-rename source. To replay that
frozen source with the current probe, pass `--package androidtestkit`; the default
package is now `swiftest`.

Implemented in the general `androidtestkit` runtime for black-box Android games:
exact-region reuse for templates/OCR, actual regional OCR, bounded polling with
scrcpy frame notifications, `wait_any`, and structured intent outcomes. No game
SDK or UI hierarchy is required. The game-specific profile was not changed.

## Behavior and impact

| Area | Change | Boundary |
| --- | --- | --- |
| Repeated evaluation | Fresh acquisition, exact pixel comparison, cached pure result | Changed pixels/configuration rerun evaluation; errors never become cached success |
| Storage | At most 16 MiB of pixel copies and 64 entries per session | Result objects have additional overhead; close releases pixels |
| OCR regions | Infer on padded crop; translate boxes back; filter original scope | Crop context changes recognition; default padding is eight pixels |
| Waiting | Default sleep at most 100 ms; no growing backoff; no extra sleep after streaming observations | Capture/inference can exceed a deadline; no game-state event source |
| Alternatives | `wait_any` checks named conditions on a shared observation | First unique match in mapping order wins; it sends no input |
| Results | Per-case/file success, last completed result or explicit checkpoint, failed intent/error | Dispatch is not verified success; abrupt process termination has no final report |

Existing target uniqueness, focus, geometry, observation age and input retry
policies remain active. Python groups use `session.intent`; JSON steps accept an
`intent` label. A checkpoint inside a failed group preserves partial progress.
Cases reset their own results, and a later success cannot hide an earlier failure.
Returned Python values become the final result. Text-entry payloads are omitted
from automatic details; explicit checkpoint data and errors are not redacted.

The main costs are pixel comparison/copying, more frequent on-demand acquisition
than exponential backoff, and possible OCR accuracy changes from cropping. Energy
use was not measured. Cached results do not authorize reuse of an old acquisition.

## Controlled comparison

Before edits, 60 framework tests passed and source was frozen locally. The probe
is `benchmarks/android/visual_evaluation_probe.py`. Baseline and candidate use
the same process environment: Windows, Python 3.12.10, NumPy 2.5.4, OpenCV 5.0.0,
RapidOCR CPU with two threads. Each run starts a new process.

The template workload has 24 generated 1280×720 frames. Twenty contain unchanged
region pixels while pixels outside animate; the final four move, duplicate,
remove and restore the target. Ground truth is the inserted tile's coordinates,
not the matcher score. Seed 341 ran baseline first; fresh seed 991735 ran candidate
first. All 48 expected box sets passed in both versions.

| Template workload | Baseline median / mean | Candidate median / mean | Matching calls |
| --- | ---: | ---: | ---: |
| Seed 341 | 16.51 / 17.49 ms | 0.97 / 4.30 ms | 24 → 5 |
| Fresh seed 991735 | 15.74 / 15.82 ms | 1.00 / 4.54 ms | 24 → 5 |

This is deliberately repeat-heavy. Changed frames still cost about 16–17 ms in
these runs. Disabling the candidate cache restored 24 matcher calls and a 16.82 ms
median at seed 341. These are local evaluation times, not device capture or E2E
game completion times.

Synthetic real OCR used three `READY 42` frames followed by `WAIT 42`, with outside
animation. Both versions produced all four expected decisions at each seed.
Candidate OCR input was 376×656 rather than 720×1280, with two inference calls
instead of four. Cached calls took 0.34–0.42 ms. Changed/crop-only warm calls took
about 281–290 ms; full-screen warm calls ranged 286–362 ms. That small sample does
not establish a reliable crop-only speed gain. Model initialization was excluded
from those warm timings.

## Real screenshot checks and negative results

Two previously captured game frames were checked visually: a selected waypoint
with a visible `傳送` button, and a black loading frame without that button. Both
baseline and candidate missed the positive when given only the Traditional
Chinese spelling: OCR produced `传送`. Explicit aliases matched it in both versions;
the loading frame remained negative. Cropped confidence and box geometry differed
slightly, so cropped inference is not output-equivalent to full-screen inference.

The default detector enlarges small crops to a minimum side length. On the narrow
button region, the warm negative took about 755 ms cropped versus 543 ms full-screen.
Thus cropping alone is not a universal optimization. The runtime does not silently
change the existing OCR scaling policy.

With the existing `LocalOCR(max_side=1280)` option applied to **both** versions,
the detector instead uses its maximum-side policy. A different positive screenshot
and the same loading image were replayed three times with caching disabled:

| Warm inference, excluding first model initialization | Full-screen baseline | Regional candidate |
| --- | ---: | ---: |
| Button present, two repeats | 480–483 ms | 35.0–35.4 ms |
| Button absent, three repeats | 305–307 ms | 16.5–16.6 ms |

All six decisions agreed with expected presence/absence in both versions. This is
one selector on two stored images, not a fresh live-task benchmark. The different
positive is a local holdout; the identical loading image is not an independent
negative holdout. Region `(0.75, 0.86, 0.97, 0.96)`, default padding, and aliases
`["傳送", "传送"]` were held fixed. Check other labels/languages before adopting
the size option.

## Verification and provenance

83 framework tests and 28 related game/benchmark tests pass. Coverage includes
movement, duplicate targets, configuration/asset changes, capture failure, cache
limits, crop coordinates, shared alternatives, deterministic polling, frame-driven
pacing, partial checkpoints, JSON validation, case isolation and explicit exits.
The wheel is checked against source and tested separately. `pip check` reports no
broken requirements in this environment. This does not certify all declared
Python/NumPy/OpenCV combinations or establish live phone response time.

Generated frames, raw timings, replay scripts, source snapshot and wheel remain
ignored under `.tmp/visual-evaluation/` and `.tmp/intent-improvements-baseline/`.
Baseline SHA-256: `session.py`
`019f66f0090ee8f778a6d51f7442ff7bfc44d1bb6a2be926d0cc5800885f7734`;
`targets.py` `f164bc96cf7f3c1d91015db0f5eb91b9e7b95f3f5d3a9e58c62c0442318ff089`.
Replay positive hashes begin `72cb29ca5544` (initial) and `9ba0612aaa31` (holdout);
loading hash begins `cb2555924d1f`. Full hashes accompany ignored results.

Reproduce the generated workload with the project interpreter:

```text
python benchmarks/android/visual_evaluation_probe.py --source-root PATH_TO_SOURCE --output .tmp/probe.json --seed 341 --real-ocr
```

Use `--disable-cache` on the candidate for the ablation. Instructions and reporting
examples are in [the usage guide](android-usage.md). The next useful qualification
is a live, repeated black-box game flow with independent completion checks,
including animation inside target regions and small text at crop edges.
