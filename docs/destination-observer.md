# Observe a prepared destination while route tracing

The Route tracer now has **Observe a destination (optional)**, initially **Off**.
It follows a prepared physical target in the game scene and shows its bearing,
supporting features, observation age, and longest unavailable interval. You drive.
This release does not steer, estimate left/right travel error, decide passage
completion, or change the existing map tracker and route finish decision.

## Try it on this workstation

1. Stop the existing Workbench from its owning terminal with Ctrl+C, then restart
   it using the normal README launch command. Refresh the browser. No C++ build,
   map rebuild, or route recompile is required for this observer.
2. Open **Route tracer**, select the usual compiled route and visible game window.
3. Under **Observe a destination (optional)** choose **Bridge approach gate**
   (outside town) or **Town exit gate** (looking outward from inside town).
   These packages came from Session 20. They do not cover every town landmark.
4. Start route tracing and drive manually toward the selected landmark. A separate
   worker loads the feature model. Normal map tracking continues while it loads.
5. Inspect the destination panel: yellow cross = prepared physical goal;
   green dots = supporting observations. Turn away, turn back, and stop/restart
   to check loss, recovery, and clean initialization. **Download observation
   telemetry** saves the per-frame JSONL for this run.

The scene panel refreshes with the Workbench's approximately one-second page poll;
it is a diagnostic preview, not the feature worker's frame rate. Each image carries
the age of the processed frame. The service returns no target after 500 ms of
measurement age. Flow updates count as new image measurements, not copied holds.
The longest unavailable interval starts after first acquisition and can include
the target legitimately leaving view; it is not automatically a tracking failure.

Bearing is `(x - image_center) / image_half_width`: zero means camera centre.
It is not degrees and is not left/right displacement from the demonstrated route.
Travel alignment remains explicitly unknown.

## Runtime and reusable data

The base Workbench keeps its existing Python environment. Observation runs in
`.tools/feature-gpu/Scripts/python.exe`, with the installed LightGlue ALIKED source
in `.tools/lightglue-upstream` and verified `aliked-n16rot.pth` in
`.tools/model-hub/checkpoints`. CUDA is selected when available, otherwise CPU.
This workstation's verified GPU runtime uses Python 3.12, PyTorch 2.11/CUDA 12.8,
and a GTX 1650. CPU real-time performance has not been established.

For another installation, set `ARIA_TRACE_FEATURE_PYTHON` to the feature
interpreter and `ARIA_TRACE_FEATURE_ROOT` to the directory containing those source
and model folders. Install dependencies into that interpreter, not the base
Workbench. The worker replaces inherited `PYTHONPATH` so legacy binary packages
cannot override its own NumPy/OpenCV installation. It does not download weights
or train a network during route tracing.

Prepared targets live in
`artifacts/workbench/destination_targets/<game>/<destination>/` as
`destination.json` and `bank.npz`. They include linked views of the same physical
goal, model identity, masks, geometry, provenance and a bank checksum. They are
reused on subsequent runs. Initial scene masking and image geometry are for the
tested 16:9 Genshin setup; different HUD layouts need separately prepared data.

The explicit importer is `python -m
aria_trace.services.tracking.destination_packages --help`. It takes a linked-view
bank and metadata; it is not an automatic route-learning tool. New-route target
selection/preparation in the UI is not implemented yet. Packages are local data
artifacts, not embedded in the source commit.

Each tracking run writes `destination/observations.jsonl`, `worker.log`, and
`destination_observer.json` below its existing live-tracking evidence folder.
There is one worker request in flight and one replaceable pending frame. Startup
failure, model error, incompatible geometry and stale output are visible. Stopping
tracking closes the worker; restarting creates fresh state.

## Evidence and limitations

The product observer exactly reproduced the frozen Session 20 third-traversal
outputs on 1,308 frames: 475/808 accepted outdoor observations and 151/500 town
observations, zero availability/kind differences and zero coordinate differences.
This proves migration parity, not absolute positioning accuracy.

Two six-second paced GPU replays per scene tested the actual worker and frame
replacement, including a declared 0.6-second blackout. Outdoor capture-to-result
P95 was 52–59 ms. Town P95 was 75–170 ms, but one town run had a 3.18-second
maximum caused predominantly by feature extraction; its repeat peaked at 259 ms.
The intermittent stall is unresolved. Process isolation prevents queue growth,
but cannot prevent contention with the game. These are recorded-frame worker
checks, not a combined live-game/map-tracker performance certification.

Session 21 adds previously unseen town laps and lighting changes. The frozen
gate packages rejected all 240 one-second samples each, as expected for an area
without those gates. This is sampled negative evidence, not a claim that every
frame or every repeating pattern is safe.

The product slice is suitable for optional observation and gathering live
evidence. Automatic steering and turning control remain unlanded experiments.
