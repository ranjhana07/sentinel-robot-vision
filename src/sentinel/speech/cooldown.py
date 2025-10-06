import time
from typing import Callable, Hashable


class CooldownTable:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._last_fired: dict[Hashable, float] = {}

    def ready(self, key: Hashable, cooldown: float) -> bool:
        last = self._last_fired.get(key)
        return last is None or self._clock() - last >= cooldown

    def try_acquire(self, key: Hashable, cooldown: float) -> bool:
        if not self.ready(key, cooldown):
            return False
        self._last_fired[key] = self._clock()
        return True

    def purge(self, older_than: float) -> int:
        now = self._clock()
        stale = [key for key, fired in self._last_fired.items() if now - fired > older_than]
        for key in stale:
            del self._last_fired[key]
        return len(stale)

    def __len__(self) -> int:
        return len(self._last_fired)
