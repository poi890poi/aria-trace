"""Public import alias for the calibrated HIK camera adapter.

Application code can use the concise, HIK-shaped import requested by the
adapter contract::

    import hikcam

    with hikcam.HikCamera(config={"game_id": "genshin-impact"}) as camera:
        frame = camera.get_frame()

Profile selection, camera ownership, and all frame-mode behavior remain in the
verified implementation under :mod:`rig_runtime.adapters.hik.compat`.
"""

from rig_runtime.adapters.hik.compat import HikCamera, MultiHikCamera


get_cam = HikCamera.get_cam
get_cams = HikCamera.get_cams
get_all_cams = HikCamera.get_all_cams
get_all_ips = HikCamera.get_all_ips

__all__ = [
    "HikCamera",
    "MultiHikCamera",
    "get_all_cams",
    "get_all_ips",
    "get_cam",
    "get_cams",
]
