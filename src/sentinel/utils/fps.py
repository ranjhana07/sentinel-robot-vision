import time
from collections import deque
from typing import Callable


class FpsMeter:
    def __init__(self, window: int = 30, clock: Callable[[], float] = time.perf_counter) -> None:
        self._stamps: deque[float] = deque(maxlen=window)
        self._clock = clock

    def tick(self) -> None:
        self._stamps.append(self._clock())

    @property
    def fps(self) -> float:
        if len(self._stamps) < 2:
            return 0.0
        elapsed = self._stamps[-1] - self._stamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._stamps) - 1) / elapsed
