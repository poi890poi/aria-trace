"""Target-presentation and camera-frame freshness contracts."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence


HOST_MONOTONIC_CLOCK = "host_monotonic_ns"


def _size_wh(value: object) -> Optional[tuple[int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    if value[0] is None or value[1] is None:
        return None
    width, height = int(value[0]), int(value[1])
    return (width, height) if width > 0 and height > 0 else None


def matching_paint_acknowledgement(
    telemetry: Mapping[str, Any],
    presentation: object,
    canonical_target_size_wh: Sequence[int],
) -> Optional[dict[str, Any]]:
    """Return the newest acknowledgement proving the requested raster was painted.

    The canonical target raster, Android logical raster, browser surface, and panel
    are deliberately not treated as interchangeable dimensions. Presenters may
    report a rotated logical target, but must preserve the canonical target image.
    """

    expected = (int(canonical_target_size_wh[0]), int(canonical_target_size_wh[1]))
    revision = int(getattr(presentation, "revision"))
    for raw in reversed(list(telemetry.get("acknowledgements", []))):
        item = dict(raw)
        if int(item.get("revision", -1)) != revision or not bool(item.get("painted")):
            continue
        canonical = _size_wh(item.get("canonical_target_size_px"))
        if canonical is not None and canonical != expected:
            continue
        natural = _size_wh(
            [item.get("image_natural_width"), item.get("image_natural_height")]
        )
        if natural is not None and natural != expected:
            continue
        canvas = _size_wh([item.get("canvas_width"), item.get("canvas_height")])
        logical = _size_wh(item.get("logical_target_size_px"))
        expected_surface = logical or expected
        if canvas is not None and canvas != expected_surface:
            continue
        if int(item.get("server_receive_time_ns", 0)) <= 0:
            continue
        return item
    return None


def sample_host_time_ns(sample: object) -> tuple[int, str]:
    """Return a timestamp comparable with target-server host monotonic time."""

    receive_time = getattr(sample, "receive_time_ns", None)
    if receive_time is not None:
        return int(receive_time), "host_receive_monotonic"
    clock_id = str(getattr(sample, "clock_id", HOST_MONOTONIC_CLOCK))
    if clock_id != HOST_MONOTONIC_CLOCK:
        raise ValueError(
            "camera clock {!r} has no host-monotonic receive_time_ns".format(clock_id)
        )
    return int(getattr(sample, "time_ns")), "host_monotonic_adapter_time"


def presentation_freshness_boundary_ns(
    presentation: object, acknowledgement: Mapping[str, Any]
) -> int:
    return max(
        int(getattr(presentation, "issued_time_ns")),
        int(acknowledgement.get("server_receive_time_ns", 0)),
    )
