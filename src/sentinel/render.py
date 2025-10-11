import cv2
import numpy as np

from sentinel.tracking.tracker import Track

KNOWN_COLOR = (80, 200, 80)
UNKNOWN_COLOR = (60, 60, 230)


def draw_tracks(frame: np.ndarray, tracks: list[Track], fps: float) -> np.ndarray:
    for track in tracks:
        x1, y1, x2, y2 = (int(value) for value in track.box)
        color = UNKNOWN_COLOR if track.identity is None else KNOWN_COLOR
        label = f"#{track.track_id} {track.identity or 'unknown'} {track.score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(frame, f"{fps:.1f} fps", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame
