from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class FramePacket:
    stream_id: str
    image: np.ndarray
    host_capture_time_ns: int
    host_receive_time_ns: int
    source_time_ns: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dropped_before: int = 0


@dataclass
class InputPacket:
    source_id: str
    kind: str
    host_time_ns: int
    payload: Dict[str, Any]
    source_time_ns: Optional[int] = None


@dataclass(frozen=True)
class TeleportBehaviorSample:
    """Destination-oriented evidence for one visually guarded teleport episode."""

    game_profile_id: str
    session_id: str
    coordinate_space_id: str
    teleport_target_global_xy: Tuple[float, float]
    destination_global_xy: Tuple[float, float]
    evidence: Dict[str, Any]
    provenance: Dict[str, Any]
    portal_id: Optional[str] = None
    phases: Tuple[Dict[str, Any], ...] = ()
    behavior_model: Dict[str, Any] = field(default_factory=dict)
    arrival_model: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "2.0",
            "behavior_type": "teleportation",
            "game_profile_id": self.game_profile_id,
            "session_id": self.session_id,
            "coordinate_space_id": self.coordinate_space_id,
            "teleport_target_global_xy": list(self.teleport_target_global_xy),
            "destination_global_xy": list(self.destination_global_xy),
            "portal_id": self.portal_id,
            "phases": [dict(item) for item in self.phases],
            "behavior_model": dict(self.behavior_model),
            "arrival_model": dict(self.arrival_model),
            "evidence": dict(self.evidence),
            "provenance": dict(self.provenance),
            "quality": dict(self.quality),
        }
