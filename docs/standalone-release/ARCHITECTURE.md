# Pipeline and architecture

The diagrams use [Mermaid](https://mermaid.js.org/) text syntax. Mermaid is
rendered by GitHub, GitLab, many Markdown viewers, and documentation systems;
the `.mmd` files can also be rendered with Mermaid CLI.

## Operator pipeline

```mermaid
flowchart LR
    Phone[Android phone] -->|ADB target display| RigExe[iris-rig-calibration.exe]
    Hik[HIK MVS camera] -->|native frames| RigExe
    RigExe -->|rig bundle + evidence| RigProfile[(rig profile)]

    Phone -->|scrcpy frames| Zigzag[iris-zigzag-acquisition.exe]
    RigProfile -->|optional normalized HIK source| Zigzag
    Hik -->|optional current-session frames| Zigzag
    Zigzag -->|ADB PNG series + optional HIK video + timestamps + YAML spaces| Session[(capture session)]

    Session --> GameExe[iris-game-calibration.exe]
    GameExe -->|screen upright| OrientationProfile[(rig-game orientation)]
    Session --> MiniExe[iris-minimap-calibration.exe]
    MiniExe -->|calibration + review evidence| MiniResult[(mini-map result)]
    Session --> ColorExe[iris-game-color-calibration.exe]
    ColorExe -->|gamma + CCM + review evidence| ColorProfile[(rig-game color profile)]

    RigProfile --> Registry[profile registry]
    MiniResult --> Registry
    ColorProfile --> Registry
    OrientationProfile --> Registry
    Registry --> Adapter[import hikcam]
    Registry -->|one-time resolved export| Embedded[generated hikcam_adapter.py]
    Adapter --> Full[rig-normalized phone]
    Adapter --> Mini[normalized mini-map]
    Adapter --> Dual[synchronized dual stream]
```

## Coordinate spaces

```mermaid
flowchart LR
    Sensor[Full HIK sensor space] -->|hardware ROI translation| ROI[HIK acquisition ROI]
    ROI -->|dense remap or homography| Rig[Rig-calibrated phone/display space]
    Rig <-->|saved rig conversion| ADB[ADB phone/display space]
    ADB -->|phone-game mini-map crop| ADBMini[ADB normalized mini-map]
    Rig -->|registered rig-game crop| HIKMini[HIK normalized mini-map]

    classDef raw fill:#fee,stroke:#933;
    classDef normalized fill:#eef,stroke:#339;
    class Sensor,ROI raw;
    class Rig,ADB,ADBMini,HIKMini normalized;
```

Coordinates and images from different boxes must not be combined directly.
Every saved capture includes its source-space description; profile transforms
are resolved before coordinates cross a boundary.

## Components and interfaces

```mermaid
classDiagram
    class HikMvsCameraAdapter {
      +open(CameraConfiguration)
      +set_roi(xywh)
      +read() FrameSample
      +close()
    }
    class RectifiedHikCamera {
      +open()
      +read() tuple
      +read_sample() FrameSample
      +release()
    }
    class ProfiledHikGameCamera {
      +mode full|minimap|dual
      +read_streams() HikGameFrameSet
      +read_sample(stream_id)
      +release()
    }
    class HikCamera {
      +get_frame() ndarray
      +get_frames() dict
      +get_iris_frame_metadata(stream_id=None) dict
      +read() tuple
      +isOpened() bool
      +release()
    }
    class ProfileRegistry {
      +resolve_adapter(context, request)
    }

    HikMvsCameraAdapter <|-- RectifiedHikCamera : acquires
    HikMvsCameraAdapter <|-- ProfiledHikGameCamera : acquires once
    RectifiedHikCamera <|-- HikCamera : full mode
    ProfiledHikGameCamera <|-- HikCamera : game modes
    ProfileRegistry --> HikCamera : resolves once at construction
```

## Runtime dependencies

```mermaid
flowchart TD
    RigExe --> OpenCV
    RigExe --> MVS[External HIK MVS runtime + driver]
    RigExe --> ADB[External ADB environment]
    ZigzagExe --> ADB
    ZigzagExe -->|continuous scrcpy mode only| Scrcpy[External scrcpy-server]
    ZigzagExe -->|continuous scrcpy mode only| FFmpeg[External FFmpeg]
    ZigzagExe -. optional .-> MVS
    MiniExe --> OpenCV
    AdapterSource[Python camera adapter source] --> OpenCV
    AdapterSource --> MVS
    AdapterSource --> Profiles[User profiles and calibration maps]
```
