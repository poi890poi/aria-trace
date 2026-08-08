from dataclasses import dataclass, field
from typing import Any, Dict, Optional

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
