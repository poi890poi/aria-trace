# Production calibration profiles

The production registry selects immutable calibration revisions for the current
camera, phone, game, and display geometry. It resolves a configuration once
when a camera adapter opens. Frame acquisition does not read the registry and
does not operate the phone.

## Profile model

- `rig`: HIK camera + phone + physical rig position + panel dimensions. It owns
  camera imaging, ROI, orientation, and display normalization.
- `phone_game`: phone + game + game display dimensions/layout. It owns the
  mini-map geometry in canonical phone-display coordinates and is usable by
  ADB-only tools.
- `rig_game`: exact `rig` + exact `phone_game`. It owns the composition needed
  for HIK mini-map/dual output and optional game-matched Bayer conversion.

Panel and game display dimensions are compatibility keys. Adapter behavior is
not: `full`, `minimap`, and `dual`, RGB/BGR order, normalization, ROI policy,
margin, and frame-rate policy are runtime options applied after selection.

Each publication creates (or reuses) an immutable revision under
`profiles/<kind>/.../revisions/`. SQLite in `profiles/.registry/` is the
activation authority; `profile.yaml` and `active.yaml` are commented,
human-readable records. Publishing a review candidate does not replace the
active revision. Resolution fails on ambiguity instead of guessing.

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

An explicit calibration path remains the highest-priority diagnostic override.
For production use, omit it and supply the observed context:

```python
import acquisition.rig_calibration.hik.camera as hikcam

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

The registry root defaults to `profiles/`. Set `ARIA_PROFILE_ROOT` or pass
`profile_root` to use another store. `ARIA_HIK_CALIBRATION` or an explicit
`calibration` config value bypasses registry selection.

## Inspection

Resolve without opening a camera:

```bat
manage-profiles.bat resolve --camera-id DA9066154 --phone-id RFCR91GWXLX --game-id genshin-impact --mode dual
```

If display facts are omitted and more than one active display variant matches,
the command deliberately reports ambiguity. Supply the complete display context
through the Python API or use an explicit immutable revision for diagnosis.
