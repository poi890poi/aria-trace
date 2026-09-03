# Compatibility surfaces

The refactor preserves established imports, commands and artifact schemas while
making their owners explicit. Compatibility modules are forwarding facades; they
must contain no independent implementation or alternate behavior.

## Python and command facades

| Compatibility surface | Canonical owner |
| --- | --- |
| `acquisition.models` | `rig_runtime.domain` and `rig_runtime.domain.packets` |
| `acquisition.session`, `annotations`, `video` | `rig_runtime.adapters.filesystem` |
| `acquisition.recorder` | `rig_runtime.workflows.recording` |
| `acquisition.android_capture`, `hik_capture` | `rig_runtime.adapters`; Windows capture remains `aria_trace.adapters.windows` |
| `acquisition.minimap_*` | `rig_runtime.services.calibration.minimap` |
| `acquisition.cursor_*` | `rig_runtime.services.calibration.cursor` |
| `acquisition.scene_yaw_calibration` | `rig_runtime.services.calibration.scene_yaw` |
| `acquisition.map_*` | `aria_trace.services.mapping` |
| `acquisition.live_tracker`, `tracking_profiles` | `aria_trace.services.tracking` |
| `acquisition.route_*` | `aria_trace.services.localization.route` and route workflows |
| `acquisition.teleport_*` | `rig_runtime.domain`, plus `aria_trace` localization and teleport workflow |
| `acquisition.rig_calibration` | `rig_runtime.services.calibration.rig` |
| `acquisition.rig_calibration.hik.*` | HIK/Android adapters, rig workflows, evidence and app entry points |
| `acquisition.workbench` | `aria_trace.apps.workbench` |
| selected `poc` promoted symbols | exact aliases in `aria_trace.services` |

The root `hikcam.py` module remains a compatibility facade for existing HIK
camera consumers. The `acquisition` tree may expose explicitly named private
helpers only where an existing test/integration requires identity compatibility.

## Rules

1. A facade imports and re-exports the canonical symbol. It does not wrap,
   reinterpret defaults, catch errors or select a different algorithm.
2. New rig/acquisition code imports `rig_runtime`; integrated tracing code imports
   `aria_trace` and may consume `rig_runtime`. Neither imports `acquisition` or `poc`.
3. A behavior fix is made at the canonical owner and verified through both the
   canonical and facade entry points.
4. Architecture tests compare important facade symbols by object identity.
5. Artifact readers remain backward compatible unless a versioned migration is
   supplied. Writers emit the current schema and provenance.

## Removal criteria

A compatibility surface can be removed only when all repository callers,
packaging entry points, operator documents and external consumers have migrated;
an announced compatibility window has elapsed; and the removal has a migration
test and release note. Until then, deleting or adding behavior to a facade is an
architectural change requiring explicit review.
