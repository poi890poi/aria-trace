# Recorder guide

The Acquisition Workbench is the normal way to record a route. It uses the canonical AriaTrace recorder: the PC window and input adapters used by the current POC can later be replaced by Android/UVC adapters without changing the session format or recording workflow.

## Start the recorder

1. Start the game and put it in windowed or borderless-windowed mode. Leave the game at the repeatable starting state for the route.
2. From the AriaTrace repository root, run:

       $env:PYTHONPATH=((Resolve-Path .tools).Path + ';' + (Resolve-Path .).Path)
       python -m acquisition.workbench

3. Open <http://127.0.0.1:8765/>.

There are no required command-line parameters. The default output locations are `sessions/workbench/` for recordings and `artifacts/workbench/` for compiled results.

## Configure a recording

### Genshin POC wizard

For the first POC:

1. Select **Genshin Impact (PC POC)** as the game profile and select the visible Genshin window.
2. Use **Windows Raw Keyboard + Mouse** unless intentionally profiling a controller.
3. Open **Confirm/edit game controls**, verify the bindings used by the current account, add behavior or map-viewer notes, and save the draft.
4. Select wizard stage 1, read its displayed instructions, and select **Arm recorder**.
5. Select **Queue next capture** and switch to Genshin. The upper-right HUD says **ARMING**, then **PLAY TO START**. The first active input received starts the session and the `REC · mm:ss` countdown.
6. In this single basic sample, perform a short cruise, stop and rotate the camera only, then hold the camera still and move only. Do not operate the recorder or mark anything while playing.
7. When the HUD says **CAPTURE COMPLETE**, return to the workbench and confirm the capture. If it says **CAPTURE FAILED**, return and rerecord it.
8. Select **Change setup** and continue with the full-map and route stages.

The first sample is the shared source for basic behavior, normal UI, mini-map calibration, and cruise evidence; it is not four separate player tasks. The current workbench preserves and indexes that evidence but does not yet run behavior inference, map stitching, mini-map calibration, or cruise estimators. The route stage retains the existing three-take compile/evaluate flow.

The HUD is enabled automatically on Windows. It is always on top, click-through, does not activate or unfocus the game, and is accepted only when Windows applies `WDA_EXCLUDEFROMCAPTURE`; otherwise it remains disabled so it cannot silently enter the recorded frames. Use borderless/windowed-fullscreen mode if an exclusive-fullscreen presentation hides ordinary desktop overlays. `python -m acquisition.workbench --no-hud` disables it explicitly.

Each stage card shows `confirmed / required` progress. Confirmation refreshes `artifacts/workbench/poc_evidence/genshin-impact-pc/evidence_index.json`. This index is the handoff across stages: it identifies each source session and its capture kind/ID, markers, timestamps, frame/input counts, drops, and control-profile draft provenance. It is an evidence inventory, not a claim that the captured map or mini-map data has already been modeled successfully.

When an input adapter is enabled, a capture with zero control events is rejected automatically and cannot remain **READY**. Raw Input receive, acceptance, and foreground-rejection counters are retained in the session manifest to diagnose an empty stream. Match the recorder's privilege level to the game or try the legacy adapter as a diagnostic fallback if Raw Input remains empty.

For an automated receiver/recorder check that does not consume a gameplay sample, run `python -m acquisition.verify_windows_input --window "Genshin Impact" --inject-f24 --wait 2`. It writes a bounded verification session and passes only when Raw Input is accepted, the recorder starts, and the first durable input is session time zero.

The normal screen asks for four choices:

1. **Game profile** — choose a profiled game or the custom/unprofiled option.
2. **Game window** — the visible window containing gameplay.
3. **How are you playing?** — choose **Windows Raw Keyboard + Mouse** for keyboard/mouse or **XInput Controller** for a controller.
4. **Guided capture or route preset** — choose a workflow stage or known route. Choose **Custom route / capture** and enter a short name when no preset exists.

The selected workflow stage or route preset supplies the instructions, number of captures, and duration. The green summary shows those values before recording. Open **Advanced settings** only when intentionally overriding them; experiment folders and internal capture IDs are generated automatically.

Select **Arm recorder** when the summary is correct. Arming only prepares the take slots—it does not record gameplay yet.

## Record each take

1. Put the game at the same chosen starting state.
2. In the workbench, select **Queue next take**.
3. Switch to the selected game window. The HUD changes from **ARMING** to **PLAY TO START**. The first qualifying control input received is retained at session time zero and starts the countdown. There is no focus gate.
4. Play the complete route naturally. Preserve the route, speed, camera movement, pauses, and all other behavior you intend the machine to learn.
5. Simply play the sample. The take stops automatically when the countdown ends. If you finish early, remain in the game until the HUD reports completion.
6. Return to the workbench after **CAPTURE COMPLETE** and select **Confirm full take boundary** when the captured start and end represent the full route.
7. Repeat until every take is ready, then select **Compile reference and evaluate**.

Run 1 becomes the reference demonstration. Later takes are held out for evaluation. Visual landmarks and route boundaries are derived or corrected after gameplay; the player never performs extra marker actions during a take.

## Input choices

| Choice | Use it for | Fidelity |
| --- | --- | --- |
| Windows Raw Keyboard + Mouse | Normal keyboard/mouse play | Raw key transitions, relative mouse motion, buttons, wheel, devices, and event timing |
| XInput Controller | Xbox-compatible controllers | Sticks, triggers, buttons, magnitudes, and timing |
| Legacy Keyboard + Mouse | Compatibility fallback only | Polled keys and absolute cursor state; less faithful than Raw Input |
| No input capture | Frame-only diagnostics | Cannot preserve the human control demonstration |

## What the recorder preserves

Each take stores synchronized video frames, exact frame timestamps, raw input events, source configuration, completion state, drop counts, and post-take annotations. Recorded controls are evidence of human behavior, not a fixed playback schedule: the later replay system uses live visual feedback to adapt that behavior to the current run.

The standalone `python -m acquisition.record` interface remains available for adapter diagnostics and non-GUI automation. It is not required for the normal PC MVP workflow.
