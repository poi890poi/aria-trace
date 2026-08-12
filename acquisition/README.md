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

`manifest.json` is versioned and records source configurations, completion status, counts, drops, and Android-to-PC clock mapping. JSONL files are line-buffered. Every record retains its original source timestamp when available and a PC monotonic session timestamp.

annotations.jsonl stores append-only add/delete operations. Marker edits never rewrite earlier history. Marker kinds include take_start, take_end, portal lifecycle, route boundary, route stage, route failure, and note records; each remains tied to an exact stream, frame index, session time, portal ID, and route ID.

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

## Integrated PC workbench

For the PC-only MVP, use the local acquisition workbench:

    $env:PYTHONPATH=((Resolve-Path .tools).Path + ';' + (Resolve-Path .).Path)
    python -m acquisition.workbench

Open http://127.0.0.1:8765/. The normal GUI asks only for the visible game window, control type, and route preset. A route preset supplies its setup instructions, take count, and duration. For an unprofiled game, choose **Custom route / any game** and enter a short route name. Technical IDs and overrides remain available under **Advanced settings**, but are not required. See [the recorder guide](../RECORDER_GUIDE.md) for the complete player workflow.

Use Windows raw keyboard and mouse for keyboard/mouse play: it preserves make/break scan codes, relative camera deltas, button/wheel transitions, device handles, and per-event timing. Use XInput for controller play: it preserves locomotion and camera axes, triggers, buttons, magnitude, and timing. Legacy keyboard/cursor polling remains available only for compatibility.

Queue a take, return focus to the selected game, and perform the route naturally. The workbench pre-arms frame and input sources before you switch windows, starts the take clock on the first selected-game focus, and stops after the configured duration. This prevents the first keyboard or mouse event from falling into a source-startup gap. There are no recorder hotkeys, stage buttons, or completion gestures during gameplay. Losing game focus early invalidates the take instead of silently contaminating it.

The recorder preserves the entire take. It derives a provisional take start from the first observed control and retains the captured tail for completion evidence. After gameplay, confirm the full-take route boundaries or use the reviewer to correct them. Visual landmarks are recognizable reference observations derived or annotated after recording; they never require player actions. Compilation is blocked until route boundaries are confirmed.

The workbench is an orchestrator, not another recorder. Game defaults live under profiles/games, route instructions live under profiles/routes, and replaceable Windows, UVC, and ADB sources feed the same AcquisitionRecorder and session schema.
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
