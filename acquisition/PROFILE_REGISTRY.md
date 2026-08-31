# Production calibration profiles

The production registry selects immutable calibration revisions for the current
camera, panel platform, game, and display geometry. It resolves a configuration once
when a camera adapter opens. Frame acquisition does not read the registry and
does not operate the phone.

## Profile model

- `rig`: HIK camera + physical rig position + panel dimensions. It owns
  camera imaging, ROI, orientation, and display normalization.
- `phone_game`: Android/panel platform + game + game display dimensions/layout. It owns the
  mini-map geometry in canonical phone-display coordinates and is usable by
  ADB-only tools.
- `phone_game_color`: an ADB-side game appearance reference. It is portable but
  never contains HIK gamma, CCM, exposure, gain, or white-balance controls.
- `rig_game`: exact `rig` + exact `phone_game`. It owns the composition needed
  for HIK mini-map/dual output.
- `rig_game_color`: a local HIK color fit derived from a portable ADB target and
  an exact local rig.

`phone` in these names means a phone/display platform, not one handset serial.
The source serial and model remain provenance. Profiles are portable between
handsets when panel and game geometry match. Panel, game, package/version, and
layout mismatches are reported as warnings for an explicit import/selection;
they do not make the data unreadable. Automatic camera-adapter selection still
requires a compatible local rig so it cannot silently open an unrelated camera.

Panel and game display dimensions are compatibility keys. Adapter behavior is
not: `full`, `minimap`, and `dual`, RGB/BGR order, normalization, ROI policy,
margin, and frame-rate policy are runtime options applied after selection.

Each publication creates (or reuses) an immutable revision under
`profiles/<kind>/.../revisions/`. SQLite in `profiles/.registry/` is the
activation authority; `profile.yaml` and `active.yaml` are commented,
human-readable records. Publishing a review candidate does not replace the
active revision. If several compatible active variants remain after applying
the observed context, resolution selects the newest revision.

## Portable calibration

Export a reviewed camera-independent revision as a directory or ZIP:

```bat
manage-profiles.bat export-portable phone-game-REVISION portable-game.zip
manage-profiles.bat export-portable phone-game-color-REVISION portable-color.zip
```

Importing is review-first. It preserves mismatch warnings and, for mini-map
geometry, composes a candidate `rig_game` revision with the active local rig:

```bat
manage-profiles.bat import-portable portable-game.zip --game-id genshin-impact
```

After reviewing the warnings and evidence, explicitly activate both imported
and locally composed revisions in one operation:

```bat
manage-profiles.bat import-portable portable-game.zip --game-id genshin-impact --activate
```

Use `--panel-size W H`, `--game-size W H`, `--camera-id`, or `--phone-id` to
describe the target system. A mismatching value warns but is accepted because
the import is explicit. The resulting `rig_game` references the active local
`rig`; camera coordinates, ROI, orientation, and rectification are never copied
from the exporting system. A portable color reference similarly requires a new
local HIK fit before it can become `rig_game_color`.

## Normal operation

Successful rig calibration publishes and activates its rig revision
automatically:

```bat
calibrate-hik-rig.bat
```

To publish canonical zigzag mini-map calibration evidence as review candidates:

```bat
manage-profiles.bat publish-minimap artifacts\game-minimap-calibration-...\calibration.json
```

After reviewing its evidence, publish and activate in one command:

```bat
manage-profiles.bat publish-minimap artifacts\game-minimap-calibration-...\calibration.json --activate
```

An existing candidate can instead be activated by immutable revision ID:

```bat
manage-profiles.bat activate phone_game-...
```

The rig reuse precheck asks the registry first. It skips recalibration only when
the active rig revision matches the connected camera/phone and the fresh image
comparison confirms that the rig has not moved.

## Camera adapter

Production use does not accept arbitrary calibration paths. Supply observed
context and let the registry select the active immutable revisions:

```python
import aria_trace.adapters.hik.compat as hikcam

camera = hikcam.HikCamera(config={
    "game_id": "genshin-impact",
    "mode": "dual",
    "camera_id": "DA9066154",       # optional when exactly one HIK is connected
    "phone_id": "RFCR91GWXLX",      # recommended when multiple variants exist
    "panel_display": {
        "natural_panel_px": [1080, 2400],
        "refresh_hz": 120.0,
    },
    "game_display": {
        "logical_frame_px": [2400, 1080],
        "game_viewport_xywh": [0, 0, 2400, 1080],
        "rotation_quarter_turns": 1,
        "ui_layout_id": "default",
    },
    "normalization": "auto",
    "color_order": "RGB",
})

with camera:
    frames = camera.get_frame()  # dual mode returns the existing dual-frame product
```

`camera.resolved_config` records exact revision IDs, paths, context, and the
adapter plan for session provenance. Resolution happens during construction;
there are zero registry reads per frame. The adapter never wakes, unlocks,
touches, launches, or powers off the phone.

`normalization="auto"` uses the adapter's best saved rectification path;
`normalization="none"` disables it for minimum latency. Explicitly forcing
`dense_remap` versus `homography` is reserved by the registry schema but is not
accepted by the current facade because it cannot yet enforce that distinction.

Automatic registry discovery checks `ARIA_PROFILE_ROOT` first. When that
environment variable is not set, it uses `profiles/` below the process's
current working directory. The supplied release helper scripts set that
directory to the extracted release root so every executable and the Python
camera adapter share one store. An explicit `profile_root` API/CLI argument
still overrides automatic discovery. `ARIA_HIK_CALIBRATION`, positional
calibration paths, and the old `calibration` config value are obsolete and
rejected. A diagnostic that genuinely needs to bypass selection must use the
deliberately conspicuous `diagnostic_calibration_override` config value or the
corresponding `--diagnostic-...-override` stream options.

The same policy applies to supporting commands:

- rig reuse checks only the active registry revision; it never scans
  `artifacts/hik-calibration-*` or chooses the newest directory;
- `wake-game-display.bat` resolves the connected HIK camera through the
  registry and no longer searches artifacts;
- zigzag dual-source capture automatically resolves the rig revision for its
  selected camera and phone, while absence/ownership failures still fall back
  to ADB-only capture;
- `demo-hik-camera.bat` treats its optional first argument only as a game ID;
  it never guesses whether the value is a file or directory;
- mutable `current.json` profile stores and recursive pointer-following are
  obsolete. SQLite activation plus exact immutable dependencies are the only
  production selection authority.

## Inspection

Resolve without opening a camera:

```bat
manage-profiles.bat resolve --camera-id DA9066154 --phone-id RFCR91GWXLX --game-id genshin-impact --mode dual
```

If display facts are omitted and more than one active display variant matches,
the command deliberately reports ambiguity. Supply the complete display context
through the Python API or use an explicit immutable revision for diagnosis.
