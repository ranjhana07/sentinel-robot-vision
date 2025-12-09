import time
from collections import deque
from typing import Callable


class SlidingWindowLimiter:
    def __init__(
        self,
        max_events: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_events = max_events
        self._window = window_seconds
        self._clock = clock
        self._stamps: deque[float] = deque()

    def allow(self) -> bool:
        if self._max_events <= 0:
            return True
        now = self._clock()
        while self._stamps and now - self._stamps[0] >= self._window:
            self._stamps.popleft()
        if len(self._stamps) >= self._max_events:
            return False
        self._stamps.append(now)
        return True
