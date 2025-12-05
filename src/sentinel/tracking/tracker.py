from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from sentinel.detection.geometry import iou_matrix
from sentinel.detection.types import Box, Detection


@dataclass
class Track:
    track_id: int
    box: Box
    score: float
    hits: int = 1
    misses: int = 0
    identity: str | None = None
    identity_checked: bool = False
    velocity: tuple[float, float] = (0.0, 0.0)

    def predicted_box(self) -> Box:
        shift_x, shift_y = self.velocity
        steps = self.misses + 1
        x1, y1, x2, y2 = self.box
        return (x1 + shift_x * steps, y1 + shift_y * steps, x2 + shift_x * steps, y2 + shift_y * steps)

    @property
    def center(self) -> tuple[float, float]:
        return (self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0


@dataclass
class IouTracker:
    iou_threshold: float = 0.3
    max_age: int = 15
    min_hits: int = 2
    use_motion: bool = False
    velocity_smoothing: float = 0.6
    _tracks: list[Track] = field(default_factory=list)
    _next_id: int = 1

    def update(self, detections: Sequence[Detection]) -> list[Track]:
        matches = self._match(detections)
        matched_tracks = {track_index for track_index, _ in matches}
        matched_detections = {detection_index for _, detection_index in matches}

        for track_index, detection_index in matches:
            track = self._tracks[track_index]
            detection = detections[detection_index]
            track.velocity = self._blend_velocity(track, detection.center)
            track.box = detection.box
            track.score = detection.score
            track.hits += 1
            track.misses = 0

        for track_index, track in enumerate(self._tracks):
            if track_index not in matched_tracks:
                track.misses += 1

        for detection_index, detection in enumerate(detections):
            if detection_index not in matched_detections:
                self._tracks.append(Track(self._next_id, detection.box, detection.score))
                self._next_id += 1

        self._tracks = [track for track in self._tracks if track.misses <= self.max_age]
        return [t for t in self._tracks if t.misses == 0 and t.hits >= self.min_hits]

    def _blend_velocity(self, track: Track, new_center: tuple[float, float]) -> tuple[float, float]:
        if not self.use_motion:
            return (0.0, 0.0)
        steps = track.misses + 1
        measured = ((new_center[0] - track.center[0]) / steps, (new_center[1] - track.center[1]) / steps)
        blend = self.velocity_smoothing
        return (
            blend * track.velocity[0] + (1.0 - blend) * measured[0],
            blend * track.velocity[1] + (1.0 - blend) * measured[1],
        )

    def _match(self, detections: Sequence[Detection]) -> list[tuple[int, int]]:
        if not self._tracks or not detections:
            return []
        track_boxes = np.array(
            [t.predicted_box() if self.use_motion else t.box for t in self._tracks], dtype=np.float32
        )
        detection_boxes = np.array([d.box for d in detections], dtype=np.float32)
        overlaps = iou_matrix(track_boxes, detection_boxes)
        matches: list[tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        for flat_index in np.argsort(-overlaps, axis=None):
            track_index, detection_index = divmod(int(flat_index), overlaps.shape[1])
            if overlaps[track_index, detection_index] < self.iou_threshold:
                break
            if track_index in used_tracks or detection_index in used_detections:
                continue
            used_tracks.add(track_index)
            used_detections.add(detection_index)
            matches.append((track_index, detection_index))
        return matches
