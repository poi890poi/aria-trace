"""Causal route alignment against a compiled replay package."""

from typing import Optional

import numpy as np

from .alignment import cosine_distances
from .descriptors import extract
from .package import ReplayPackage


class IncrementalReplayAligner:
    """Match one observation at a time without lookahead or elapsed-time coupling."""

    def __init__(
        self,
        package: ReplayPackage,
        max_advance: int = 4,
        distance_threshold: float = 0.45,
        min_margin: float = 0.0,
    ) -> None:
        if not package.references:
            raise ValueError("ReplayPackage has no reference observations")
        if max_advance < 1:
            raise ValueError("max_advance must be at least one")
        if distance_threshold <= 0.0:
            raise ValueError("distance_threshold must be positive")
        if min_margin < 0.0:
            raise ValueError("min_margin must not be negative")
        self.package = package
        self.max_advance = int(max_advance)
        self.distance_threshold = float(distance_threshold)
        self.min_margin = float(min_margin)
        self.stage_by_id = {stage["stage_id"]: stage for stage in package.stages}
        self.reset()

    def reset(self) -> None:
        self.reference_index = 0
        self.observation_index = 0

    def observe_image(self, image: np.ndarray, timestamp_ns: Optional[int] = None) -> dict:
        descriptor = extract(image, self.package.manifest["visual_descriptor"])
        return self.observe_descriptor(descriptor, timestamp_ns)

    def observe_descriptor(
        self,
        descriptor: np.ndarray,
        timestamp_ns: Optional[int] = None,
    ) -> dict:
        descriptor = np.asarray(descriptor, dtype=np.float32).reshape(1, -1)
        upper = min(len(self.package.references), self.reference_index + self.max_advance + 1)
        distances = cosine_distances(
            descriptor,
            self.package.descriptors[self.reference_index:upper],
        )[0]
        local_index = int(np.argmin(distances))
        candidate_index = self.reference_index + local_index
        distance = float(distances[local_index])
        alternatives = np.delete(distances, local_index)
        alternative_distance = float(np.min(alternatives)) if len(alternatives) else 2.0
        margin = alternative_distance - distance
        accepted = distance <= self.distance_threshold and margin >= self.min_margin
        if accepted:
            self.reference_index = candidate_index

        reference = self.package.references[self.reference_index]
        stage = self.stage_by_id[reference["stage_id"]]
        result = {
            "observation_index": self.observation_index,
            "timestamp_ns": timestamp_ns,
            "candidate_reference_index": candidate_index,
            "reference_index": self.reference_index,
            "reference_id": reference["reference_id"],
            "stage_id": stage["stage_id"],
            "stage_label": stage["label"],
            "progress": reference["progress"],
            "visual_distance": distance,
            "alternative_margin": margin,
            "accepted": bool(accepted),
            "complete": self.reference_index == len(self.package.references) - 1,
        }
        self.observation_index += 1
        return result
