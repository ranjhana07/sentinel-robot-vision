from typing import Callable, Mapping, Protocol

import numpy as np

from sentinel.tracking.tracker import Track


class IdentityResolver(Protocol):
    def resolve(self, frame: np.ndarray, track: Track) -> str | None: ...


class UnknownResolver:
    def resolve(self, frame: np.ndarray, track: Track) -> str | None:
        return None


def crop_box(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    left, top = max(int(x1), 0), max(int(y1), 0)
    right, bottom = min(int(x2), width), min(int(y2), height)
    return frame[top:bottom, left:right]


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(first, second) / denominator)


class GalleryResolver:
    def __init__(
        self,
        embed: Callable[[np.ndarray], np.ndarray],
        gallery: Mapping[str, np.ndarray],
        threshold: float = 0.6,
    ) -> None:
        self._embed = embed
        self._gallery = dict(gallery)
        self._threshold = threshold

    def resolve(self, frame: np.ndarray, track: Track) -> str | None:
        face = crop_box(frame, track.box)
        if face.size == 0:
            return None
        embedding = self._embed(face)
        best_name, best_score = None, self._threshold
        for name, reference in self._gallery.items():
            score = cosine_similarity(embedding, reference)
            if score >= best_score:
                best_name, best_score = name, score
        return best_name
