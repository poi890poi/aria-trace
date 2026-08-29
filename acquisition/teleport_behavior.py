"""Validated persistence for explicitly recorded teleport gameplay behavior."""

import json
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

from .models import TeleportBehaviorSample


def _xy(name: str, value: Optional[Sequence[float]], required: bool):
    if value is None:
        if required:
            raise ValueError("{} is required".format(name))
        return None
    if len(value) != 2:
        raise ValueError("{} must contain exactly x and y".format(name))
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(component) for component in result):
        raise ValueError("{} must contain finite coordinates".format(name))
    return result


def make_teleport_behavior_sample(
    *,
    game_profile_id: str,
    session_id: str,
    coordinate_space_id: str,
    teleport_target_global_xy: Sequence[float],
    destination_global_xy: Sequence[float],
    evidence: Mapping,
    provenance: Mapping,
    origin_global_xy: Optional[Sequence[float]] = None,
    portal_id: Optional[str] = None,
    quality: Optional[Mapping] = None,
) -> TeleportBehaviorSample:
    """Build a sample only when its coordinate system and evidence are reusable."""
    for name, value in (
        ("game_profile_id", game_profile_id),
        ("session_id", session_id),
        ("coordinate_space_id", coordinate_space_id),
    ):
        if not str(value or "").strip():
            raise ValueError("{} is required".format(name))
    if not evidence:
        raise ValueError("evidence is required")
    if not provenance:
        raise ValueError("provenance is required")
    return TeleportBehaviorSample(
        game_profile_id=str(game_profile_id),
        session_id=str(session_id),
        coordinate_space_id=str(coordinate_space_id),
        teleport_target_global_xy=_xy(
            "teleport_target_global_xy", teleport_target_global_xy, True
        ),
        destination_global_xy=_xy(
            "destination_global_xy", destination_global_xy, True
        ),
        origin_global_xy=_xy("origin_global_xy", origin_global_xy, False),
        portal_id=str(portal_id) if portal_id else None,
        evidence=dict(evidence),
        provenance=dict(provenance),
        quality=dict(quality or {}),
    )


def save_teleport_behavior_sample(
    sample: TeleportBehaviorSample, output_path: Path
) -> Path:
    """Persist one atomic, portable JSON behavior observation."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(sample.to_dict(), indent=2), encoding="utf-8")
    temporary.replace(output_path)
    return output_path
