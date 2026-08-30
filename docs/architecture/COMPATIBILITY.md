# Compatibility surfaces

The refactor preserves established imports, commands and artifact schemas while
making their owners explicit. Compatibility modules are forwarding facades; they
must contain no independent implementation or alternate behavior.

## Python and command facades

| Compatibility surface | Canonical owner |
| --- | --- |
| `acquisition.models` | `aria_trace.domain` and `aria_trace.domain.packets` |
| `acquisition.session`, `annotations`, `video` | `aria_trace.adapters.filesystem` |
| `acquisition.recorder` | `aria_trace.workflows.recording` |
| `acquisition.android_capture`, `windows`, `hik_capture` | `aria_trace.adapters` |
| `acquisition.minimap_*` | `aria_trace.services.calibration.minimap` |
| `acquisition.cursor_*` | `aria_trace.services.calibration.cursor` |
| `acquisition.scene_yaw_calibration` | `aria_trace.services.calibration.scene_yaw` |
| `acquisition.map_*` | `aria_trace.services.mapping` |
| `acquisition.live_tracker`, `tracking_profiles` | `aria_trace.services.tracking` |
| `acquisition.route_*` | `aria_trace.services.localization.route` and route workflows |
| `acquisition.teleport_*` | `aria_trace.domain`, localization service and teleport workflow |
| `acquisition.rig_calibration` | `aria_trace.services.calibration.rig` |
| `acquisition.rig_calibration.hik.*` | HIK/Android adapters, rig workflows, evidence and app entry points |
| `acquisition.workbench` | `aria_trace.apps.workbench` |
| selected `poc` promoted symbols | exact aliases in `aria_trace.services` |

The root `hikcam.py` module remains a compatibility facade for existing HIK
camera consumers. The `acquisition` tree may expose explicitly named private
helpers only where an existing test/integration requires identity compatibility.

## Rules

1. A facade imports and re-exports the canonical symbol. It does not wrap,
   reinterpret defaults, catch errors or select a different algorithm.
2. New production code imports `aria_trace`, never `acquisition` or `poc`.
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
