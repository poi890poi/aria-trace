# Recorder guide

The Acquisition Workbench is the normal way to record a route. It uses the canonical AriaTrace recorder: the PC window and input adapters used by the current POC can later be replaced by Android/UVC adapters without changing the session format or recording workflow.

## Start the recorder

1. Start the game and put it in windowed or borderless-windowed mode. Leave the game at the repeatable starting state for the route.
2. From the AriaTrace repository root, run:

       $env:PYTHONPATH=((Resolve-Path .tools).Path + ';' + (Resolve-Path .).Path)
       python -m acquisition.workbench

3. Open <http://127.0.0.1:8765/>.

There are no required command-line parameters. The default output locations are `sessions/workbench/` for recordings and `artifacts/workbench/` for compiled results.

When the selected game runs as administrator, the Workbench process serving port 8765 must also run as administrator. Stop any existing non-elevated Workbench first: launching a second elevated copy cannot take over a port that the first copy still owns. The Start action checks this before it creates a session and reports the mismatch on the page.

## Record and organize sessions

1. Select the game profile and visible game window.
2. Choose **Windows Raw Keyboard + Mouse** unless intentionally recording a controller or visual-only evidence.
3. Set the duration and select **Start recording**.
4. Switch to the game while the HUD says **ARMING** or **SWITCH TO GAME**. Recording begins after the three-second settling countdown.
5. Perform one coherent sample. Recording stops automatically; **Stop** is available when an early end is necessary.
6. Return after **CAPTURE COMPLETE** and choose a label in that session's row. The available labels cover ordinary cruise, rotation-only, movement-only, straight-forward/no-turn, full-map coverage, and route demonstrations.
7. Repeat as often as needed. There is no stage selection, arming step, take target, or fixed session limit.

Choosing a non-empty label is the single post-capture review action. Rotation-only and movement-only sessions feed mini-map calibration, while straight-forward/no-turn preserves the cursor-heading-to-observed-map-shift relationship. **Delete** moves an unwanted session under `sessions/workbench/.trash/` so the operation remains recoverable.

Only complete recordings with positive duration and at least one video frame are committed to the session list. Privilege/input failures, recorder errors, cancellations, zero-duration attempts, and frameless attempts are discarded automatically; their error remains visible in the Workbench instead of becoming a session.

The HUD is enabled automatically on Windows. It is always on top, click-through, does not activate or unfocus the game, and is accepted only when Windows applies `WDA_EXCLUDEFROMCAPTURE`; otherwise it remains disabled so it cannot silently enter the recorded frames. Use borderless/windowed-fullscreen mode if an exclusive-fullscreen presentation hides ordinary desktop overlays. `python -m acquisition.workbench --no-hud` disables it explicitly.

Label changes refresh `artifacts/workbench/poc_evidence/genshin-impact-pc/evidence_index.json`. This index identifies each source session and its selected role, markers, timestamps, frame/input counts, and drops. It is an evidence inventory, not a claim that the captured map or mini-map data has already been modeled successfully.

Full-map and route stages require healthy input evidence. Short labeled stages declare input optional and use a timer start, so they can be recorded with an adapter for synchronized controls or with **No input capture** for visual-only evidence. Raw Input receive, acceptance, and foreground-rejection counters are retained whenever that adapter is enabled.

For an automated receiver/recorder check that does not consume a gameplay sample, run `python -m acquisition.verify_windows_input --window "Genshin Impact" --inject-f24 --settle 1.5 --wait 3`. It injects F23 during settling and F24 afterward, and passes only when F23 is discarded, F24 starts the recorder at session time zero, and a complete verification session is durable.

The normal screen asks for four choices:

1. **Game profile** — choose a profiled game or the custom/unprofiled option.
2. **Game window** — the visible window containing gameplay.
3. **How are you playing?** — choose **Windows Raw Keyboard + Mouse** for keyboard/mouse or **XInput Controller** for a controller.
4. **Seconds** — the bounded duration of this session.

## Record each take

1. Put the game at the same chosen starting state.
2. In the workbench, select **Start recording**.
3. Switch to the selected game window during **ARMING** / **SWITCH TO GAME**. The three-second settling interval discards the queue click and switch residue, then starts the bounded recording. There is no focus gate.
4. Play the complete route naturally. Preserve the route, speed, camera movement, pauses, and all other behavior you intend the machine to learn.
5. Simply play the sample. The take stops automatically when the countdown ends. If you finish early, remain in the game until the HUD reports completion.
6. Return to the workbench after **CAPTURE COMPLETE** and choose the session label.
7. Repeat for any number of sessions.

Run 1 becomes the reference demonstration. Later takes are held out for evaluation. Visual landmarks and route boundaries are derived or corrected after gameplay; the player never performs extra marker actions during a take.

## Input choices

| Choice | Use it for | Fidelity |
| --- | --- | --- |
| Windows Raw Keyboard + Mouse | Normal keyboard/mouse play | Raw key transitions, relative mouse motion, buttons, wheel, devices, and event timing |
| XInput Controller | Xbox-compatible controllers | Sticks, triggers, buttons, magnitudes, and timing |
| Legacy Keyboard + Mouse | Compatibility fallback only | Polled keys and absolute cursor state; less faithful than Raw Input |
| No input capture | Short timer-start visual segments and frame-only diagnostics | Preserves visual motion but not the human control evidence; unavailable for input-triggered route/full-map stages |

## What the recorder preserves

Each take stores synchronized video frames, exact frame timestamps, raw input events, source configuration, completion state, drop counts, and post-take annotations. Recorded controls are evidence of human behavior, not a fixed playback schedule: the later replay system uses live visual feedback to adapt that behavior to the current run.

The standalone `python -m acquisition.record` interface remains available for adapter diagnostics and non-GUI automation. It is not required for the normal PC MVP workflow.
