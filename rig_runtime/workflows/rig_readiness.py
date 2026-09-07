"""Validate composed game capabilities before activating a new rig configuration."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from rig_runtime.adapters.filesystem.profile_registry import ProfileContext, ProfileResolutionError
from rig_runtime.adapters.hik.game_camera import ProfiledHikGameCamera
from rig_runtime.services.calibration.rig.contracts import FrameSample


class _PlanningFrames:
    """Exercise the runtime's map/ROI path without accessing a device.

    These blank frames prove software geometry only. Headless publication also
    supplies its real camera adapter to verify programming and frame delivery.
    """
    def open(self, configuration):
        self.configuration = configuration

    def set_black_level(self, value):
        pass

    def set_manual_imaging(self, exposure, gain):
        pass

    def set_white_balance(self, red, green, blue):
        pass

    def set_bayer_conversion(self, gamma, matrix):
        pass

    def align_roi(self, roi):
        return list(roi)

    def set_roi(self, roi):
        self.roi = list(map(int, roi))
        return self.roi

    def read(self):
        return FrameSample(image=np.zeros((self.roi[3], self.roi[2], 3), np.uint8), time_ns=0)

    def close(self):
        pass


def validate_rig_configuration(rig, reconciliation, *, registry, adapter=None):
    # Color is optional. Geometry readiness must never require a fresh game
    # color fit just because the camera-to-phone transform changed.
    orientations = reconciliation["recomposed"]["rig_game_orientation"]
    reports = []
    notices = []
    for game in reconciliation["recomposed"]["rig_game"]:
        context = ProfileContext.from_dict(game["context"])
        orientation = next((item for item in orientations
                            if ProfileContext.from_dict(item["context"]).game_id == context.game_id
                            and ProfileContext.from_dict(item["context"]).game_display_signature
                            == context.game_display_signature), None)
        orientation_payload = orientation["payload"] if orientation else {}
        turns = int(orientation_payload.get(
            "camera_adapter_image_quarter_turns_clockwise_from_calibration_display", 0))
        if orientation is None:
            notices.append("Game {!r}: no optional game orientation; using calibration-display orientation".format(context.game_id))
        payload = game["payload"]
        for mode in ("full", "minimap", "dual"):
            camera = ProfiledHikGameCamera(
                registry.runtime_file(rig, "hik_camera_calibration"),
                Path(game["revision_directory"]) / "profile.json",
                mode=mode, adapter=adapter if adapter is not None else _PlanningFrames(),
                apply_game_color=False,
                output_quarter_turns_clockwise=turns,
                runtime_surface_quarter_turns_clockwise_from_natural=orientation_payload.get(
                    "game_surface_quarter_turns_clockwise_from_phone_natural"),
            )
            try:
                camera.open()
                frames = camera.read_streams()
                for stream, frame in frames.streams.items():
                    size = [int(frame.shape[1]), int(frame.shape[0])]
                    boundary = camera.get_minimap_geometry(stream)
                    cursor = camera.get_cursor_geometry(stream)
                    for label, geometry, required in (
                        ("boundary", boundary, bool(payload.get("outer_boundary"))),
                        ("cursor", cursor, bool(payload.get("cursor_geometry")) and bool(
                            (payload.get("cursor_geometry") or {}).get("rotation_center")
                            or payload.get("rotation_center"))),
                    ):
                        if not required:
                            continue
                        if not geometry.get("available_in_stream_space"):
                            raise ValueError("{} unavailable: {}".format(label, geometry.get("reason", "missing geometry")))
                        if geometry.get("image_space", {}).get("stored_size_px") != size:
                            raise ValueError("{} does not match the {} frame space".format(label, stream))
                        x, y = geometry["center_xy_px"]
                        if not (np.isfinite(x) and np.isfinite(y) and 0 <= x < size[0] and 0 <= y < size[1]):
                            raise ValueError("{} center is outside {} output".format(label, stream))
                    if (payload.get("outer_boundary") or {}).get("orientation_frame"):
                        if not boundary.get("orientation_frame"):
                            notice = "Game {!r}: optional game axes unavailable: {}".format(context.game_id, boundary.get("orientation_reason"))
                            if notice not in notices:
                                notices.append(notice)
                reports.append({"game_id": context.game_id, "mode": mode,
                                "phone_game_revision": game["dependencies"]["phone_game"]})
            except Exception as exc:
                raise ProfileResolutionError(
                    "Rig configuration is not ready for game {!r}, mode {}: {}. "
                    "The previous active configuration was preserved.".format(context.game_id, mode, exc)
                ) from exc
            finally:
                camera.release()
    return {"status": "ready", "validation": "camera_frames" if adapter is not None else "software_geometry",
            "games": reports, "notices": notices, "color_policy": "best_effort_non_gating",
            "retained_color_profiles": reconciliation.get("retained", {}).get("rig_game_color", [])}
