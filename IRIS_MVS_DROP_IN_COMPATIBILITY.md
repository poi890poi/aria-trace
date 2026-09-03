# Replacing the MVS Python wrapper without changing an application

## Status

This document describes a feasible integration design. It is not an implemented
IRIS feature yet.

IRIS currently provides the high-level `hikcam` module. It does **not** currently
replace Hikrobot's low-level `MvCameraControl_class` module. Placing the current
`hikcam.py` beside an application that imports `MvCameraControl_class` will not
redirect that application through IRIS.

## Goal

An existing Python application may use Hikrobot's wrapper directly:

```python
from MvCameraControl_class import *

devices = MV_CC_DEVICE_INFO_LIST()
MvCamera.MV_CC_EnumDevices(MV_USB_DEVICE | MV_GIGE_DEVICE, devices)
camera = MvCamera()
```

The desired deployment keeps that application source unchanged while making its
normal MVS import resolve to an IRIS-managed camera implementation. The application
may have an IRIS runtime directory placed beside it.

This is possible through Python import precedence, provided that the application
is launched normally from source and does not override its import search path.

## Proposed deployment

```text
third-party-app/
|-- app.py
|-- MvCameraControl_class.py       # IRIS MVS compatibility shim
|-- rig_runtime/                    # IRIS Python runtime
|-- iris_tools.py
|-- profiles/                      # optional local profile registry
`-- run-app.bat
```

When Python runs `app.py`, its directory is normally the first import location.
Therefore `from MvCameraControl_class import *` loads the adjacent IRIS shim before
the MVS sample wrapper installed elsewhere.

This is a deployment change, not an application source change. It is not reliable
when the application explicitly inserts the MVS directory first in `sys.path`, uses
a custom importer, or is already frozen with the vendor module inside an executable.

## Why `hikcam.py` cannot simply be renamed

The two APIs operate at different levels.

`hikcam.HikCamera` is a Python-oriented facade. It selects IRIS profiles, opens the
camera, and returns NumPy images in documented native, rectified-phone, mini-map, or
dual-stream spaces.

`MvCameraControl_class.MvCamera` is a ctypes wrapper around the MVS C API. Its
contract includes:

- `MV_CC_*` method names and integer status codes;
- ctypes device, frame, and feature structures;
- caller-owned output buffers and pointer lifetimes;
- explicit create/open/grab/stop/close/destroy lifecycle calls;
- raw and converted pixel formats;
- image-buffer, timeout-read, and callback acquisition variants;
- direct feature-tree access for ROI, exposure, gain, trigger, and other controls.

A renamed high-level adapter would import successfully but fail as soon as the
application accessed these low-level contracts.

## Compatibility-shim design

The shim should preserve the vendor module's public surface used by the target
application while routing image acquisition through IRIS.

### 1. Load the real MVS wrapper privately

IRIS still needs Hikrobot's native driver and DLLs to control the physical camera.
The shim therefore does not eliminate the MVS installation.

The IRIS backend must load the genuine `MvCameraControl_class.py` from its resolved
absolute SDK path under a private module identity. It must not use an ordinary
`import MvCameraControl_class` after the shim is installed, because that would import
the shim recursively.

The SDK directory must remain available while loading the vendor wrapper because it
also imports modules such as `CameraParams_header`, `PixelType_header`, and
`MvErrorDefine_const`.

### 2. Re-export the vendor data contract

The compatibility module should re-export the constants, ctypes structures, callback
types, and utility symbols expected from the installed MVS version. This matters for
applications using wildcard imports and constructing structures themselves.

IRIS should not maintain hand-written copies of all these definitions. They should
come from the privately loaded vendor wrapper so structure layout remains consistent
with the installed SDK.

### 3. Provide a compatible `MvCamera`

The shim's `MvCamera` class should implement only a verified compatibility surface and
delegate unmodified vendor behavior where safe. The usual acquisition path includes:

```text
MV_CC_EnumDevices
  -> MV_CC_CreateHandle
  -> MV_CC_OpenDevice
  -> feature writes
  -> MV_CC_StartGrabbing
  -> frame retrieval
  -> MV_CC_StopGrabbing
  -> MV_CC_CloseDevice
  -> MV_CC_DestroyHandle
```

Enumeration and lifecycle operations must identify the same physical camera and
preserve expected success/error codes. Frame retrieval is the point where the shim
substitutes the selected IRIS output.

### 4. Adapt frames at the buffer boundary

For `MV_CC_GetOneFrameTimeout`, the shim can obtain an IRIS frame, validate the
caller's buffer capacity, copy the pixels, and populate the supplied frame-info
structure with dimensions, pixel type, frame number, and timestamp.

For `MV_CC_GetImageBuffer`, IRIS must retain the backing allocation until the matching
`MV_CC_FreeImageBuffer` call. Returning a temporary NumPy pointer would create a
use-after-free defect.

Callbacks require a dedicated delivery thread and stable callback buffers. Their
timing, ordering, shutdown, and exception behavior must be verified separately; they
should not be claimed compatible merely because timeout-based reads work.

The reported frame dimensions and pixel type must describe the actual IRIS output.
For example, a rectified phone image cannot be returned with native-sensor dimensions
left in `MV_FRAME_OUT_INFO_EX`.

### 5. Define feature behavior explicitly

Third-party code may configure hardware before grabbing. Each used feature must have
one of three documented behaviors:

- **Delegate:** forward a control that remains meaningful and safe.
- **Translate:** map the MVS operation into the equivalent IRIS configuration.
- **Reject:** return an appropriate MVS error when the request conflicts with the
  calibrated stream.

Silently accepting an unsupported ROI, pixel format, trigger, or exposure change is
unsafe because the caller would believe the requested camera state is active. A
calibrated profile's locked imaging and hardware ROI must remain authoritative unless
the compatibility contract explicitly permits an override.

IRIS-specific frame-space metadata has no native MVS equivalent. It may be exposed by
an additive proprietary method, but unchanged third-party applications will not call
it. The shim must still keep MVS frame metadata internally consistent.

## Compatibility levels

The implementation should state its supported level rather than claim compatibility
with the entire MVS wrapper.

### Level A: timeout-based acquisition

- Standard device enumeration and exclusive open.
- Manual feature setup used by the target application.
- `MV_CC_GetOneFrameTimeout`, including RGB or BGR variants when required.
- Normal stop, close, and destroy behavior.

This is the lowest-risk initial target.

### Level B: borrowed image buffers

- Everything in Level A.
- `MV_CC_GetImageBuffer` and `MV_CC_FreeImageBuffer` with verified allocation lifetime.

### Level C: callbacks

- Everything in Level B.
- Registered image callbacks with compatible concurrency and shutdown semantics.

Specialized functions such as recording, GenTL, serial ports, liquid lenses, raw Bayer
ISP processing, or arbitrary feature-tree access should continue to use the vendor
implementation unless a real application requires IRIS substitution there.

## What can prevent silent replacement

Import shadowing does not apply cleanly in every deployment:

- A frozen PyInstaller or similar executable may already contain the vendor module.
- The application may prepend the MVS sample directory to `sys.path` before importing.
- The application may load `MvCameraControl_class.py` by absolute filename.
- A service launcher may use a different working or script directory than expected.
- The application may import MVS header modules directly and depend on a particular SDK
  version.
- Native extensions may bypass the Python wrapper and call MVS DLLs directly.

For these cases, replacing a neighboring Python file is insufficient. The application
launcher or build configuration must control import precedence, even if application
source remains unchanged.

## Verification before deployment

Compatibility should be derived from the actual third-party call surface, not from the
full vendor wrapper. First collect every imported symbol and invoked `MV_CC_*` method.
Then test the application against both native MVS and the shim using the same camera and
settings.

The minimum evidence should include:

1. The imported `MvCameraControl_class.__file__`, proving which module was selected.
2. Device enumeration identity and ordering.
3. Requested and effective camera settings.
4. Frame dimensions, pixel type, byte count, and monotonically increasing frame IDs.
5. A native-MVS frame and corresponding IRIS frame rendered with space metadata.
6. Repeated open/grab/close cycles and recovery after an interrupted read.
7. Buffer-lifetime tests for every supported acquisition method.
8. Confirmation that an unsupported feature returns an explicit error rather than a
   false success.

The application should also be tested from its production launcher, not only from an
interactive shell, because the launcher determines Python import order and profile-root
resolution.

## Recommended implementation boundary

Do not clone Hikrobot's complete Python wrapper. The installed version contains a very
large API and changes across SDK releases. A full imitation would be fragile and would
create an unnecessary maintenance obligation.

The preferred implementation is:

- use the installed vendor wrapper as the authority for structures and unmodified
  operations;
- intercept only the methods required to substitute IRIS acquisition;
- base the supported surface on one or more real third-party applications;
- keep the high-level `hikcam` API as the recommended integration for new code;
- treat the `MvCameraControl_class` shim as a legacy, zero-source-change bridge.

Under those constraints, silently redirecting a conventional source-based MVS Python
application is practical. Claiming universal drop-in compatibility with every MVS
Python application is not.
