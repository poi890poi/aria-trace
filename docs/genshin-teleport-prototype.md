# Genshin example

The game-specific rules live in `examples/genshin/profile.py`. The user's script at `examples/genshin/teleport.json` contains the app and three UI targets. Run it through the general Python framework from this checkout.

```text
python -m pip install ".[ocr]"
python -m androidtestkit run examples/genshin/teleport.json --profile examples/genshin/profile.py --serial RFCR91GWXLX --timeout 600
```

The profile handles the game's entry prompt, startup preparations, minimap, unlocked waypoint and confirmation button. It supplies app-specific state recognition to the library's named-step runner, uses the selected session's capture backend, and reads rendered text through local Python OCR. To specify a particular visible waypoint, add `"position": [x,y]` or `"template": "waypoint.png"` to the corresponding touch entry. Position values are normalized to the current map. The sample requires an account with an unlocked waypoint.

For ordinary apps, write Python tests or JSON intents using the public selectors. See [Android TestKit](swiftest.md). This example adds recognition of game screens. The public API, device abstraction, matching, and CLI run independently of it.

## Historical prototype measurements

On November 6, 2026, five alternating captures per mode on the SM-G781B during the same Genshin download gave these medians. Each sample covers one requested screenshot, including transport, with the helper available.

| Capture | Response bytes | Capture time |
|---|---:|---:|
| ADB PNG | 435,542 | 0.986 s |
| On-device JPEG | 47,086 | 0.381 s |

These measurements came from the earlier Windows OCR prototype. The JPEG figure includes text-safe encoding. Initial helper upload was 2,148 bytes. These samples measure capture latency and traffic. The phone was powered by USB and downloading game content. Whole-workflow energy savings in joules remain unmeasured. Windows OCR recognized the retained error and download samples in 0.09–0.10 s per warm request on this PC.

A second comparison used five alternating samples per mode on the gameplay screen after teleportation. Median PNG size was 2,281,423 bytes and median capture time was 2.617 seconds. Median JPEG response size was 218,598 bytes and median time was 0.290 seconds. These timings exclude initial setup and OCR.

## Live teleport verification, November 7, 2026

The SM-G781B completed its game update and shader compilation. The library dismissed the active Samsung Game Booster touch overlay, reacquired the entry prompt, entered the game, and opened the map. On this UI, the character's health readout splits the green bar into two segments. The map footer shows exploration progress.

The successful map-to-arrival run selected a waypoint marked **傳送錨點** in Mondstadt's **墜星山谷**. Genshin offered **傳送**. One tap triggered loading, followed by two gameplay HUD observations. The tracked quest distance was 503 m before opening the map and 862 m after arrival.

| Successful run, starting with the map open | Measurement |
|---|---:|
| Elapsed time | 18.45 s |
| Requested frames | 10 |
| Screenshot response bytes | 1,758,660 |
| Time acquiring frames | 3.23 s |
| Time in OCR | 1.59 s |
| Game touches | 2 (select marker, teleport) |

Entry, map opening, and teleport were tested in successive runs as the game profile was calibrated. The 18.45 seconds cover the final map-to-arrival run. The marker and HUD detectors have been verified on this phone, UI language, and map view. Other map layouts and health states need additional validation. The earlier November 5 test verified recovery diagnostics for network error 4201.

Local evidence under `artifacts/genshin-teleport/` includes `20261107-002407-893199200/before.png` and the selection, arrival and result files in `20261107-002546-363460800/`.
