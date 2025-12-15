import time
from typing import Callable

import numpy as np

from sentinel.capture.sources import FrameSource


class ReconnectingSource:
    def __init__(
        self,
        factory: Callable[[], FrameSource],
        max_failures: int = 5,
        max_attempts: int = 5,
        base_delay: float = 0.5,
        max_delay: float = 10.0,
        sleep: Callable[[float], None] = time.sleep,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> None:
        self._factory = factory
        self._max_failures = max_failures
        self._max_attempts = max_attempts
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._sleep = sleep
        self._on_event = on_event or (lambda name, fields: None)
        self._source: FrameSource | None = factory()
        self._failures = 0

    def read(self) -> np.ndarray | None:
        if self._source is None:
            return None
        while self._failures < self._max_failures:
            frame = self._source.read()
            if frame is not None:
                self._failures = 0
                return frame
            self._failures += 1
            self._on_event("camera_read_failed", {"consecutive": self._failures})
        return self._reconnect()

    def _reconnect(self) -> np.ndarray | None:
        self._source.release()
        self._source = None
        for attempt in range(1, self._max_attempts + 1):
            delay = min(self._base_delay * 2 ** (attempt - 1), self._max_delay)
            self._sleep(delay)
            self._on_event("camera_reconnect", {"attempt": attempt, "delay": delay})
            try:
                self._source = self._factory()
            except Exception as error:
                self._on_event("camera_reconnect_failed", {"attempt": attempt, "error": str(error)})
                continue
            self._failures = 0
            return self._source.read()
        return None

    def release(self) -> None:
        if self._source is not None:
            self._source.release()
            self._source = None
