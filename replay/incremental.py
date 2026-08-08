"""Stateful frame-by-frame route-progress observer."""

from typing import Optional

import numpy as np

from .alignment import cosine_distances
from .descriptors import extract
from .package import ReplayPackage


class IncrementalReplayObserver:
    """Advance through a ReplayPackage without using future observations."""

    def __init__(
        self,
        package: ReplayPackage,
        max_advance: int = 4,
        distance_threshold: float = 0.45,
        min_margin: float = 0.0,
        stay_penalty: float = 0.02,
        skip_penalty: float = 0.05,
    ) -> None:
        if max_advance < 1:
            raise ValueError("max_advance must be at least one")
        self.package = package
        self.max_advance = int(max_advance)
        self.distance_threshold = float(distance_threshold)
        self.min_margin = float(min_margin)
        self.stay_penalty = float(stay_penalty)
        self.skip_penalty = float(skip_penalty)
        self.stage_by_id = {stage["stage_id"]: stage for stage in package.stages}
        self.reset()

    def reset(self) -> None:
        self.committed_index = 0
        self.observation_count = 0
        self._scores = None

    def observe_image(self, image: np.ndarray, timestamp_ns: Optional[int] = None) -> dict:
        descriptor = extract(image, self.package.manifest["visual_descriptor"])
        return self.observe_descriptor(descriptor, timestamp_ns=timestamp_ns)

    def observe_descriptor(
        self,
        descriptor: np.ndarray,
        timestamp_ns: Optional[int] = None,
    ) -> dict:
        descriptor = np.asarray(descriptor, dtype=np.float32).reshape(1, -1)
        distances = cosine_distances(descriptor, self.package.descriptors)[0]
        if self._scores is None:
            candidate_scores = np.full(len(distances), np.inf, dtype=np.float64)
            candidate_scores[0] = float(distances[0])
        else:
            candidate_scores = self._advance_scores(distances)
        candidate_index = int(np.argmin(candidate_scores))
        distance = float(distances[candidate_index])
        alternatives = np.concatenate(
            (
                distances[: max(0, candidate_index - 2)],
                distances[min(len(distances), candidate_index + 3) :],
            )
        )
        alternative = float(np.min(alternatives)) if len(alternatives) else distance
        margin = alternative - distance
        accepted = distance <= self.distance_threshold and margin >= self.min_margin
        if accepted:
            self.committed_index = max(self.committed_index, candidate_index)
            candidate_scores[: self.committed_index] = np.inf
            finite = np.isfinite(candidate_scores)
            candidate_scores[finite] -= float(np.min(candidate_scores[finite]))
            self._scores = candidate_scores

        reference = self.package.references[self.committed_index]
        stage = self.stage_by_id[reference["stage_id"]]
        confidence = float(
            np.clip(1.0 - distance / max(self.distance_threshold, 1.0e-8), 0.0, 1.0)
            * 0.75
            + np.clip(margin / 0.15, 0.0, 1.0) * 0.25
        )
        result = {
            "observation_index": self.observation_count,
            "timestamp_ns": timestamp_ns,
            "candidate_reference_index": candidate_index,
            "reference_index": self.committed_index,
            "reference_id": reference["reference_id"],
            "stage_id": stage["stage_id"],
            "stage_label": stage["label"],
            "progress": reference["progress"],
            "visual_distance": distance,
            "alternative_margin": margin,
            "confidence": confidence,
            "accepted": bool(accepted),
            "complete": self.committed_index == len(self.package.references) - 1,
        }
        self.observation_count += 1
        return result

    def _advance_scores(self, distances: np.ndarray) -> np.ndarray:
        result = np.full(len(distances), np.inf, dtype=np.float64)
        for reference_index in range(self.committed_index, len(distances)):
            lower = max(self.committed_index, reference_index - self.max_advance)
            best = np.inf
            for prior_index in range(lower, reference_index + 1):
                if not np.isfinite(self._scores[prior_index]):
                    continue
                advance = reference_index - prior_index
                transition = (
                    self.stay_penalty
                    if advance == 0
                    else self.skip_penalty * max(0, advance - 1)
                )
                best = min(best, float(self._scores[prior_index]) + transition)
            if np.isfinite(best):
                result[reference_index] = best + float(distances[reference_index])
        return result
