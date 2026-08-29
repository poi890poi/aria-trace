# Data Acquisition Suite

This package records synchronized gameplay observations and raw inputs, inspects session health, and provides a local frame-by-frame reviewer. Its primary role is to preserve human demonstrations that can later be compiled into adaptive replay packages. The format is intentionally independent of a game, controller, capture device, and control backend.

Recorded controls are evidence of what the human did, not a schedule that the live system must repeat. Replay alignment and control adaptation are downstream responsibilities.

`AcquisitionRecorder` and the session schema are the canonical recorder for both the PC POC and later Android/UVC operation. Windows window/input, Android/ADB, UVC, and future hardware integrations are replaceable sources feeding the same recorder; they are not separate recording products.

## Session format

```text
session/
  manifest.json
  frames.jsonl
  inputs.jsonl
  annotations.jsonl          # append-only marker operations
  video_<stream>.mkv
  evidence/                 # online outputs computed from raw frames
  derived/                  # optional, reproducible caches
```

Game-profiler evidence, full-map captures, mini-map calibration models, cruise models, and route descriptors are planned artifacts with explicit source-session/frame references. They do not replace this acquisition session. Exact artifact schemas and paths will be versioned when the first experiments establish what must be retained.

`manifest.json` is versioned and records source configurations, completion status, counts, drops, and Android-to-PC clock mapping. JSONL files are line-buffered. Every record retains its original source timestamp when available and a PC monotonic session timestamp.

annotations.jsonl stores append-only add/delete operations. Marker edits never rewrite earlier history. Marker kinds include take boundaries, generic evidence-capture boundaries/failure, portal lifecycle, route boundary/stage/failure, and note records; each remains tied to an exact stream, frame index, and session time, with portal or route IDs where applicable.

The default is H.264 (`libx264`, CRF 20) in Matroska. Exact time is in `frames.jsonl`, not the container playback rate. MJPEG AVI remains available with `--video-encoding mjpeg` as a large compatibility fallback. The encoder is isolated behind a video-sink interface so hardware encoding or another container can replace it later.

### Storage policy

- Durable evidence: compressed primary video, exact frame timestamps, raw inputs, calibration references, annotations, and the feature observations actually consumed online. Online features are extracted from the raw decoded frame before H.264 encoding and stored under `evidence/` with the source frame index, raw-frame hash, extractor version, and configuration.
- Selected raw pixels: optional lossless PNGs may accompany online feature observations for map keyframes and diagnostic events. Keeping every frame losslessly is intentionally not the default.
- Derived cache: features, embeddings, optical flow, and localization databases generated later from recorded video belong under `derived/<pipeline-name>-<version>/`. They may be deleted and regenerated and are never labeled as raw-frame evidence.
- Redundant views: retain only when an experiment requires them; an internal capture and external camera can otherwise double the dominant storage cost.

`python -m acquisition.inspect_session` reports actual bytes and a projected GiB/hour. Rate depends strongly on motion and detail. In the checked 1436x996 Genshin smoke sample, 91 frames used 1.34 MiB of H.264 versus 12.54 MiB of MJPEG (9.4x smaller), projecting 1.57 GiB/hour for that short sample. Frame timestamps were about 27 MiB/hour at 30 fps; raw input sidecars were much smaller.

A second five-second smoke recorded online SIFT at 1 Hz before encoding. Five observations containing 16,352 keypoints occupied 1.82 MiB, projecting about 1.27 GiB/hour for features alone; H.264 plus sidecars and features projected 3.34 GiB/hour for that sample. Saving 30 Hz SIFT would therefore be wasteful. The default evidence cadence is 1 Hz and should later be driven by the actual relocalizer/keyframe policy.

Lossless pixels are substantially larger. In the four-second portal smoke, five SIFT observations with their PNG frames made the evidence database 9.48 MiB and projected about 8.24 GiB/hour; the complete session projected 10.18 GiB/hour. Use lossless sampling for short portal captures or selected diagnostic/keyframe events, not indiscriminately throughout long runs.

Long-session chunking is not implemented yet. Record bounded sessions for now; a hard process termination may damage the active video container even though prior JSONL records remain readable. Five-minute crash-safe chunks are the next storage change before unattended collection.

## Planned game, map, and mini-map acquisition

The Acquisition Workbench exposes separate short Genshin POC stages for ordinary cruise, rotation-only, movement-only, and straight-forward/no-turn motion, followed by full-map viewer evidence and the repeated route. Each short stage supplies `segment_label`, structured motion semantics, timer-start behavior, and its input requirement. The workbench stores a human-editable control-profile draft and records each stage with its label, capture kind/ID, workflow-stage ID, frames, optional raw controls, timing, and confirmation markers. It also maintains `artifacts/workbench/poc_evidence/<profile-id>/evidence_index.json`, a structured stage-progress inventory that links confirmed captures to their source sessions, marker state, timing/count summaries, drops, and profile-draft provenance. Mini-map calibration selects confirmed sessions by role instead of asking for timestamp boundaries.

### Game profile session

The existing files under `profiles/games` provide adapter defaults and coarse control summaries. The workbench profile editor now saves a reviewable draft under `artifacts/workbench/game_profiles/<profile-id>/draft.json` without rewriting the source profile. Later automatic profiling extends the same relationship with measured behavior; it does not introduce a separate configuration product.

A semi-automatic profiling session runs controlled actions while recording inputs and game frames. It should infer or help a reviewer enter:

- semantic controls such as movement, camera, jump, dash, interaction, and map/menu actions;
- physical bindings such as WASD, mouse axes/buttons, Space, controller sticks/buttons, or touch regions;
- activation details including press/release, single click, hold, toggle, chord, analog range, dead zone, duration, and compatible simultaneous actions;
- movement, turning, acceleration/stopping, camera-character coupling, jump/dash timing, cooldown/recovery, collision response, and input-to-visible-response timing where measurable.

Every inferred field retains its probe definition, source frames and input records, timing, intermediate measurements, confidence, and profiler version. Human edits are versioned and preserve the original measured value. The profile distinguishes measured, manually supplied, assumed, and unknown values. Route-specific setup remains in `profiles/routes`.

### Full map-viewer acquisition

Complete map texture should be acquired from the game's full map viewer, not reconstructed only from the smaller semi-transparent mini-map. The current recorder accepts a manually navigated take labeled **Full-map coverage** afterward. The next automation uses the confirmed viewer open/close, pan, zoom, region/layer switching, and related UI controls to navigate and capture every accessible area. Locked, undiscovered, occluded, or otherwise inaccessible areas are reported explicitly rather than silently counted as complete.

Capture at a zoom/detail level whose effective map resolution exceeds the mini-map, with sufficient overlap or other registration evidence. The resulting versioned artifact retains original source frames, viewer/control state and timing, transforms, coverage/completeness information, quality diagnostics, and derived tiles or mosaics. Coverage gaps or poor registration should trigger an automated retry or an explicit request for human review.

The full-map artifact, live mini-map observations, and reconstructed route geometry are separate coordinate spaces. Any scale, rotation, crop, projection, or alignment relationship among them is an explicit calibrated estimate with provenance and confidence. Do not assume UI icons, occlusions, or map revisions are identical between the viewer and mini-map.

### Calibration evidence from labeled short sessions

1. Use the entire confirmed rotation-only session selected by its manifest label; do not ask the player to mark a time range.
2. Retain the full game frames and recorded camera inputs on the normal session timeline.
3. Aggregate the frames under the working assumption that the mini-map does not rotate with camera orientation. Changing scene content should become less stable while the fixed mini-map remains identifiable.
4. Evaluate temporal averages/stability maps, thresholding, edges, and a circle detector such as Hough circle detection to propose the mini-map position and boundary.

The exact motion amplitudes, duration, aggregation, thresholds, and detector parameters are deliberately unset pending experimentation. A calibration run succeeds by producing a reusable, versioned artifact with source provenance, circle/mask data, method configuration, and quality—not by returning coordinates only.

Its artifact set must be inspectable by both people and later scripts. At minimum it should retain:

- an average, stability, or heatmap image;
- the binary/threshold result;
- the edge image;
- the detected circle overlaid on an original or reference frame;
- a machine-readable manifest linking those products to the source session, frames, controls, and calibration method/configuration.

### Cruise and heading-motion evidence from labeled sessions

Use the ordinary-cruise and movement-only sessions at enough cadence that consecutive circular mini-map observations overlap. The straight-forward/no-turn session must produce notable translation and constrains movement to cursor heading, establishing the relationship between estimated cursor angle and observed mini-map shift direction. Preserve the entire game frame and any available input stream; mini-map crops, valid-circle masks, shift/facing measurements, and diagnostic outputs are derived records.

The initial experiments will:

- estimate `(dx, dy)` using Fourier/phase correlation within the valid circular map area and retain peak, overlap, and confidence/quality evidence;
- observe the central character-facing cursor across multiple orientations, without assuming its shape or equating facing with camera direction or movement;
- evaluate a polar transform about the mini-map center, with a reusable precomputed transform map where helpful, so rotation becomes translation and differently oriented observations can be normalized and combined;
- measure movement-speed and rotation-rate distributions;
- align observations by estimated map motion and measure temporal stability so stable map texture can be distinguished from icons, animations, floating elements, and temporary effects.

These algorithms are proposals. The durable output is a cruise model and quality-bearing observations that support normalized mini-map extraction, relative XY shift estimation, and character-facing estimation through replaceable implementations.

### Gameplay route take and descriptor

The human records the route through the existing uninterrupted-take workflow. Compilation should avoid treating every nearly identical video frame as an independent permanent observation. It selects useful keyframes while retaining their original stream/frame indices, timestamps, input/control references, and measurements; groups nearby keyframes into experimentally sized local submaps; and records cumulative route progress alongside reconstructed spatial position.

Raw relative measurements remain separate from reconstructed or corrected geometry. An individual bad shift or facing estimate can therefore retain its original value while being flagged, rejected, or replaced. Revisited locations may later add correction constraints, but the first recorder does not require loop closure or global optimization.

The source session remains authoritative. Sparse map tiles, stitched local maps, a route-strip image, and trajectory plots are useful derived views, but no giant stitched map becomes the route record.

## Integrated PC workbench

For the PC-only MVP, use the local acquisition workbench:

    $env:PYTHONPATH=((Resolve-Path .tools).Path + ';' + (Resolve-Path .).Path)
    python -m acquisition.workbench

Open http://127.0.0.1:8765/. Select a game profile, visible game window, control type, and duration, then select **Start recording**. The screen intentionally has no guided-stage or arming state: every valid completed capture is appended to the session list and can be classified afterward with a label dropdown. Failed, canceled, zero-duration, and frameless attempts are discarded rather than saved; unwanted successful sessions are moved to recoverable trash. See [the recorder guide](../RECORDER_GUIDE.md) for the complete player workflow.

On Windows, the normal command also starts a small isolated HUD process. It polls the lightweight `/api/hud` contract and shows **ARMING**, **SWITCH TO GAME**, **PLAY TO START**, recording countdown, finalization, completion, and failure at the selected game window's upper-right. The window uses no-activate and click-through styles, hides whenever the selected game loses focus, and starts only when Windows applies and reads back `WDA_EXCLUDEFROMCAPTURE`. This is required because the current `win32_gdi_visible_client_v1` source copies visible desktop pixels. The HUD process is isolated so a desktop-UI failure cannot take down the recorder. The Workbench exposes **Hide overlay** / **Show overlay**; `--no-hud` starts hidden. Exclusive fullscreen may suppress ordinary topmost overlays, so borderless/windowed fullscreen is preferred.

Use Windows raw keyboard and mouse for keyboard/mouse play: it preserves make/break scan codes, relative camera deltas, button/wheel transitions, device handles, and per-event timing. Use XInput for controller play: it preserves locomotion and camera axes, triggers, buttons, magnitude, and timing. Legacy keyboard/cursor polling remains available only for compatibility.

Select **Start recording** and switch to the selected game during **SWITCH TO GAME**. The workbench starts the selected sources, discards queue/switch residue for three seconds, then waits for the first active control while the selected game has focus. That control becomes session time zero. There is no recorder hotkey, stage button, or completion gesture during gameplay. Recordings append under a stable per-game experiment folder, so the operator can create any number of sessions across restarts.

The recorder preserves the entire take. It derives a provisional take start from the first observed control and retains the captured tail for completion evidence. After gameplay, choosing a non-empty session label is the explicit content-review action. Visual landmarks are recognizable reference observations derived or annotated after recording; they never require player actions.

The workbench is an orchestrator, not another recorder. Game defaults live under profiles/games, route instructions live under profiles/routes, and replaceable Windows, UVC, and ADB sources feed the same AcquisitionRecorder and session schema.

The independent [`rig_calibration`](rig_calibration/README.md) package implements camera/phone geometry, normalization, matchability, alternating-signal latency, commented YAML artifacts, and spatial export. It deliberately has no dependency on the workbench, camera/ADB drivers, game profiles, maps, or dataset registries; those systems integrate through small adapters.
## Standalone recorder (advanced)

The workbench above is the normal recorder interface. Use this command-line interface for adapter diagnostics, scripted acquisition, and non-GUI sources.

Record a short Windows PC-game route:

```powershell
$env:PYTHONPATH=((Resolve-Path .tools).Path + ';' + (Resolve-Path .).Path)
python -m acquisition.record `
  --output sessions\pc_demo_001 `
  --window "Unique Game Window Title" `
  --pc-input `
  --route-id short-route-a `
  --duration 20
python -m acquisition.inspect_session sessions\pc_demo_001
python -m acquisition.review sessions\pc_demo_001
```

The selected game client must remain visible, unobstructed, foreground, and at a fixed size. Title matching rejects ambiguous substrings; use --exact-window-title when needed. --pc-raw-input is the faithful keyboard/mouse path and records keyboard transitions plus raw relative mouse motion. --pc-input retains the earlier polled keyboard/absolute-cursor adapter as a compatibility fallback. Raw input can be bursty, so the recorder queue now defaults to 4,096 events and the workbench uses 8,192; input drops remain explicit in the manifest.

Smoke-test with an existing video and synthetic gamepad state:

```powershell
$env:PYTHONPATH=((Resolve-Path .tools).Path + ';' + (Resolve-Path .).Path)
python -m acquisition.record `
  --output sessions\smoke `
  --video genshin=data\gid\Seq-046\Seq-046.mp4 `
  --synthetic-input `
  --online-features sift `
  --feature-rate 1 `
  --feature-lossless-frames `
  --portal-id mondstadt-gate `
  --route-id gate-to-hotel `
  --duration 5
```

Record an unobstructed phone with a UVC camera and raw gamepad events from Android:

```powershell
python -m acquisition.record `
  --output sessions\gamepad_demo_001 `
  --camera phone=0 `
  --camera-width 1920 `
  --camera-height 1080 `
  --camera-fps 30 `
  --getevent
```

Use `--duration SECONDS` or press `Ctrl+C`. `--serial` selects one Android device. Multiple `--video` and `--camera` options create multiple frame streams. `--adb-screenshot` provides a slow development capture source; it is not a replacement for the future continuous internal-video source.

H.264 needs FFmpeg on `PATH` or supplied with `--ffmpeg PATH`. `--video-crf` controls quality and size (lower means higher quality and larger files); keep the default 20 until physical-camera feature stability is measured.

`--online-features sift` stores up to 4,096 raw-frame SIFT observations at `--feature-rate` Hz. Descriptor values are stored as `uint8` only when the conversion is numerically lossless. Use `--feature-stream ID` to restrict extraction and `--feature-lossless-frames` when exact pixels are needed for every sampled observation. Online extraction is optional in this standalone recorder; the eventual navigation runtime must register the feature processor it actually uses.

`--portal-id` and `--route-id` provide defaults for later annotations. They do not assert that loading has completed or that localization succeeded.

`AdbGetEventSource` stores raw Linux input events. It samples `/proc/uptime`, chooses the lowest-round-trip clock measurement, and maps kernel event timestamps to PC monotonic time. If clock sampling fails, it explicitly falls back to host receive time. Device-specific gamepad normalization and touch-slot interpretation remain downstream decoders so raw evidence is never lost.

### Continuous Android screen and input capture

The continuous Android adapter uses the pinned scrcpy 4.1 phone-side server over
ADB. One full display encode is decoded on the PC and fanned out into repeatable
ROI streams. Every ROI derived from a source frame carries the same Android
MediaCodec presentation timestamp; raw `getevent -lt` timestamps use Android
`CLOCK_MONOTONIC`, and one lowest-RTT `/proc/uptime` calibration maps both onto
the PC session clock.

Each `--android-roi` is `ID=x,y,w,h[,output_w,output_h[,crf]]`. Repeat it for
multiple streams. A zero width or height extends to the frame edge. ROI
coordinates refer to the captured orientation at start, which is locked for the
take. Output width/height performs host-side downsampling, and CRF selects that
stream's stored H.264 quality (lower is higher quality).

```powershell
python -m acquisition.record `
  --adb E:\Android\Sdk\platform-tools\adb.exe `
  --scrcpy-server .tools\scrcpy-win64-v4.1\scrcpy-server `
  --android-roi full=0,0,0,0,1280,720,22 `
  --android-roi minimap=24,24,420,420,256,256,16 `
  --duration 20 `
  --output sessions\android_demo
```

Raw Android input is enabled automatically. Add `--no-android-input` only for a
screen-only diagnostic. `--android-save-phone` (or
`--android-save-phone /sdcard/SomeDirectory`) additionally ZIPs the complete
finalized session, pushes it to the phone, and prints the resulting device path.
The ZIP contains the videos plus `frames.jsonl`, `inputs.jsonl`, and the clock
mapping; copying only a video would discard the synchronization authority.

### Synchronized Android + HIK mini-map calibration

For a complete new HIK setup, use the single entry point:

```powershell
.\calibrate-hik-game-camera.bat
```

It calls the existing tools in order: headless rig calibration, automated game
launch plus rig-normalized HIK/ADB zigzag capture and mini-map calibration, then the
profiled HIK adapter demo in dual mode. Explicit output directories connect the
stages; the mini-map summary supplies the exact published rig-game profile to
the demo. A failed stage stops the sequence and retains completed artifacts.
Use `-NoDemo` to finish after calibration, or `-GameId`, `-CameraId`, and
`-PhoneSerial` only when defaults cannot select the intended setup.

With the game awake and ready and an existing rig result, run capture with:

```powershell
.\capture-game-minimap-zigzag.bat --rig-calibration .\artifacts\hik-calibration-YYYYMMDD-HHMMSS
```

The command waits for Enter before it injects input. Acquisition always opens
the HIK camera through the saved rig-calibrated adapter and simultaneously
records the full Android logical display. The continuous gesture sweeps
horizontally while alternating horizon/sky/horizon/ground returns.

Each long diagonal stroke starts on the unobstructed central look-control
surface, away from the right-side action cluster. It emits progressive MOVE
events while the pointer remains down, matching the first verified camera-drag
transport. Twelve strokes use 486 vertical pixels and 243 horizontal pixels
per stroke. Duplicating each direction from the original pattern produces six
up and six down strokes, returns pitch to the horizon, and caps the gesture at
120 touch events.

Before showing the prompt, the command sends Android's non-toggling
`KEYCODE_WAKEUP` through ADB and dismisses the keyguard. If Android explicitly
reports a remaining swipe keyguard, it performs one remote upward swipe. The
physical power button is never used, so waking the phone cannot move the rig.

For the default `--game-id genshin-impact`, the command resolves the installed
international or mainland-China Genshin package, launches its Android launcher
activity, and verifies that the package is foreground before showing the
prompt. Use `--android-package <package>` for another build or
`--no-launch-game` to retain a completely manual launch step. Starting the app
does not inject game input or modify camera/calibration controls.

To record source data without analysis, use:

```powershell
.\capture-game-minimap-zigzag.bat --rig-calibration .\artifacts\hik-calibration-YYYYMMDD-HHMMSS
```

Dual-source capture requires the saved rig bundle. The `android_phone` stream
keeps the complete Android logical display. The `hik_phone` stream uses the
calibrated HIK adapter, so it contains only the camera-visible phone region,
rectified by the saved rig geometry and quarter-turned into the same logical
game orientation as ADB. Odd dimensions are padded only at the video encoder
edge and the unpadded content size remains in frame metadata.

Each session also contains `cross_source_check/`. At a low sampling rate it
crops ADB with the saved rig-visible rectangle and reuses the rig calibration's
cross-source scorer on synchronized game frames. The summary, side-by-side
image, edge overlay, valid mask, and difference heatmap are diagnostic only;
low confidence never rejects or changes a recording.

Game control uses the reusable `ScrcpyTouchController` in
`acquisition/scrcpy_control.py`. It maintains one pinned scrcpy 4.1 control
socket for the gesture instead of starting one ADB process per touch event.
Acquisition ends only after the controller reports the complete DOWN/MOVE/UP
sequence and the configured tail has elapsed; incomplete control is rejected.

The downstream calibration result boundaries remain deliberately separate:

- `artifacts/game-minimap-calibration-*` contains source-specific mini-map
  results, stacked heatmaps, complete fitted boundaries, and exact shift masks.
- `profiles/phone_game/<phone>/<game>` stores the camera-independent mini-map
  crop in the phone's natural display coordinates.
- `profiles/rig_game/<rig>/<game>` is created only when a base rig calibration
  is supplied. It references the base rig and phone-game revision and adds no
  optical transform.

Every JSON result has a commented YAML companion. All new revisions remain
`review_required`; cursor, pose, tracking, game modeling, and north estimation
are outside this product.

The production HIK adapter consumes the base rig plus the optional rig-game
profile:

```python
from acquisition.rig_calibration.hik import ProfiledHikGameCamera

with ProfiledHikGameCamera(
    "artifacts/hik-calibration-...",
    "profiles/rig_game/<rig>/<game>/current.json",
    mode="dual",                 # "minimap", "full", or "dual"
    rectify_minimap=True,
) as camera:
    frame_set = camera.read_streams()
    full = frame_set.streams["full"]
    minimap = frame_set.streams["minimap"]
```

Dual products come from one acquisition and therefore share the frame number
and timestamps. Its `full` product is the base rig's normalized complete
visible-phone output, and `minimap` is a crop in that same normalized space.
Mini-map-only mode applies a small hardware ROI to reduce camera and USB
throughput.

The existing `import ... as hikcam` facade accepts the same options and keeps
`get_frame()` as the default single-frame interface; `get_frames()` returns the
two synchronized products in dual mode:

```python
import acquisition.rig_calibration.hik.camera as hikcam

with hikcam.HikCamera(config={
    "calibration": "artifacts/hik-calibration-.../hik_camera_calibration.json",
    "minimap_calibration": "profiles/rig_game/<rig>/<game>/current.json",
    "mode": "dual",
    "rectify": True,
}) as camera:
    streams = camera.get_frames()
```

## Inspect

```powershell
python -m acquisition.inspect_session sessions\gamepad_demo_001
```

The report includes frame counts, resolution, duration, median/p95 frame interval, recorded drops, input counts, online feature artifacts, exact file sizes, and projected GiB/hour.

## Review

```powershell
python -m acquisition.review sessions\gamepad_demo_001
```

Open `http://127.0.0.1:8765/`. The local reviewer supports stream selection, scrubbing, stepping, playback, keyboard left/right navigation, inspection of raw inputs within 100 ms, and per-frame online-feature metadata. At a selected frame, enter the portal/route IDs and add `teleport_start`, `world_ready`, or `route_start`. Markers can be deleted; deletion is recorded as an append-only tombstone. It does not upload session data.

The planned profiler/map/mini-map/route extension belongs in this reviewer rather than a disconnected UI. It should expose game-profile fields with their probe evidence, full-map source captures and coverage diagnostics, and synchronized gameplay views containing the full frame, mini-map crop and masks, shift/facing values and quality, trajectory and route progress, keyframe/submap membership, controls, and available diagnostics. Selecting a suspicious result should expose neighboring source frames.

Review of an estimator result will be stored as a structured annotation attached to the affected observation or measurement, preserving the original output. The review states are `correct`, `suspicious`, and `wrong`, with an optional comment. This is not implemented by the current marker-only `AnnotationStore`; its schema/UI must be extended so a later diagnostic session can consume the annotation together with the source observations, frames, artifacts, and nearby measurements.

## Extract a portal initialization interval

After marking `world_ready` and `route_start`:

```powershell
python -m acquisition.extract_portal_init sessions\gamepad_demo_001 `
  --output portal_maps\mondstadt-gate\arrival-001 `
  --portal-id mondstadt-gate `
  --route-id gate-to-hotel `
  --require-lossless
```

The extractor selects the annotated interval and prefers PNG frames retained from the raw pre-encoding feature path. Its manifest identifies the source session, marker IDs, exact frame indices, hashes, feature database, and source quality. Without `--require-lossless`, it falls back to decoded H.264 frames and labels them `decoded_compressed_video`; this fallback must not be represented as raw evidence.

Recorded and live sessions will be aligned spatially rather than by elapsed time. These extracted arrivals are the inputs for a later portal-specific feature map. Live initialization will search only the selected portal map, require several consistent pose hypotheses, and use the configured portal spawn as the character-position prior.

## Hardware validation still required

- UVC timestamp stability and requested camera-mode support
- Android permission to read the connected controller with `getevent`
- Mapping of controller event device, axes, ranges, dead zones, and buttons
- Clock offset stability over long sessions
- Sustained dual-stream recording performance
- Continuous clean internal capture for direct-touch human demonstrations
