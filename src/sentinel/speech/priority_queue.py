import itertools
import threading
import time
from dataclasses import dataclass, field


@dataclass(order=True)
class QueuedPhrase:
    priority: int
    order: int
    text: str = field(compare=False)


class PhraseQueue:
    def __init__(self, capacity: int = 8) -> None:
        self._capacity = capacity
        self._items: list[QueuedPhrase] = []
        self._counter = itertools.count()
        self._condition = threading.Condition()

    def put(self, text: str, priority: int = 1) -> bool:
        with self._condition:
            if len(self._items) >= self._capacity:
                worst = max(self._items)
                if priority >= worst.priority:
                    return False
                self._items.remove(worst)
            self._items.append(QueuedPhrase(priority, next(self._counter), text))
            self._condition.notify()
            return True

    def get(self, timeout: float | None = None) -> str | None:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._items:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)
            best = min(self._items)
            self._items.remove(best)
            return best.text

    def __len__(self) -> int:
        with self._condition:
            return len(self._items)
