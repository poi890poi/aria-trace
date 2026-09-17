# UVC source for virtual HIK

Type: new source capability. The existing virtual HIK adapter accepts a UVC
camera index through `virtual-hik+uvc://`; Android identities and stored state
remain compatible. The rig pipeline still consumes its existing camera contract.

Reuse the tracked OpenCV camera adapter. Do not depend on the unrelated,
uncommitted standalone `uvccam` package. Add shell-friendly source options to
the virtual-camera control and instrument commands. No Workbench, atlas,
tracking algorithm, native HIK SDK or physical imaging-control changes.

UVC modes retain their own software ROI, zoom, imaging-state and displacement.
The first received frame establishes actual geometry. Mode fallback is reported,
and later frame-size changes fail rather than applying the wrong ROI. Timestamp
metadata identifies host reception; it is not sensor exposure time. UVC cleanup
must release the camera and lease and must never send ADB commands to its index.
Only explicit open claims a camera. Do not scan or open unrelated cameras.

Verify Android regression, UVC acquisition through mocked OpenCV, actual-size
handling, crop/displacement order and persistence, source/mode identity isolation,
single-owner locking, failed-open/read cleanup, and both command entry points.
Report hardware tests separately from these deterministic integration checks.

## Verification (2026-09-17)

- 18 Android/UVC tests pass on the Python 3.7 launcher runtime. Nine new UVC
  checks exercise the real OpenCV adapter with only VideoCapture substituted:
  delivered geometry and timestamps, displacement/ROI persistence, zoom and
  mode isolation, cross-mode ownership, failed acquisition cleanup, disconnect
  and size-change errors, URI validation, command use, and Android-only cleanup.
- 41 tests pass on Python 3.12, including those 18 and adjacent Android camera,
  HIK ROI, component-contract and adapter-export tests.
- Both Windows launcher help commands succeed and expose the UVC options.
- The broader suite on Python 3.7 has two existing compatibility errors in the
  unchanged atomic-write helper (`Path.unlink(missing_ok=True)` requires Python
  3.8+). Both affected tests pass on Python 3.12. This change does not repair that
  separate runtime limitation in the calibration/profile workflow.
- Physical UVC capture and the physical calibration sequence were not run.
  Camera delivery rate, buffering latency and backend/device compatibility
  therefore remain hardware validation items; no throughput claim is made.
