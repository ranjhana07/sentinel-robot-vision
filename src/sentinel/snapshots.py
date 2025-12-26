from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from sentinel.tracking.tracker import Track


def padded_crop(frame: np.ndarray, box: tuple[float, float, float, float], padding: float) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    left = max(int(x1 - pad_x), 0)
    top = max(int(y1 - pad_y), 0)
    right = min(int(x2 + pad_x), width)
    bottom = min(int(y2 + pad_y), height)
    return frame[top:bottom, left:right]


class SnapshotWriter:
    def __init__(
        self,
        directory: Path | str,
        max_files: int = 200,
        padding: float = 0.25,
        quality: int = 85,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._max_files = max_files
        self._padding = padding
        self._quality = quality
        self._now = now

    def save(self, frame: np.ndarray, track: Track) -> Path | None:
        crop = padded_crop(frame, track.box, self._padding)
        if crop.size == 0:
            return None
        stamp = self._now().strftime("%Y%m%dT%H%M%S%f")
        path = self._directory / f"{stamp}_track{track.track_id}.jpg"
        if not cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, self._quality]):
            return None
        self._prune()
        return path

    def _prune(self) -> None:
        files = sorted(self._directory.glob("*.jpg"))
        for stale in files[: max(len(files) - self._max_files, 0)]:
            stale.unlink(missing_ok=True)
