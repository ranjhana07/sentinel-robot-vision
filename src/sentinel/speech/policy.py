import time
from dataclasses import dataclass
from typing import Callable, Sequence

from sentinel.config import SpeechConfig
from sentinel.speech.cooldown import CooldownTable
from sentinel.tracking.tracker import Track


@dataclass(frozen=True)
class SpeechEvent:
    kind: str
    text: str
    track_ids: tuple[int, ...]


class SpeechPolicy:
    PURGE_AFTER_SECONDS = 600.0

    def __init__(
        self,
        config: SpeechConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._cooldowns = CooldownTable(clock)

    def evaluate(self, tracks: Sequence[Track]) -> list[SpeechEvent]:
        events: list[SpeechEvent] = []
        events.extend(self._announce_unknown(tracks))
        events.extend(self._greet_known(tracks))
        self._cooldowns.purge(self.PURGE_AFTER_SECONDS)
        return events

    def _announce_unknown(self, tracks: Sequence[Track]) -> list[SpeechEvent]:
        unknown = [track for track in tracks if track.identity is None]
        fresh = [
            track
            for track in unknown
            if self._cooldowns.try_acquire(("unknown", track.track_id), self._config.unknown_cooldown_seconds)
        ]
        if not fresh:
            return []
        if len(unknown) == 1:
            text = self._config.unknown_phrase
        else:
            text = self._config.group_phrase.format(count=len(unknown))
        return [SpeechEvent("unknown", text, tuple(track.track_id for track in fresh))]

    def _greet_known(self, tracks: Sequence[Track]) -> list[SpeechEvent]:
        events = []
        for track in tracks:
            if track.identity is None:
                continue
            if self._cooldowns.try_acquire(("greet", track.identity), self._config.cooldown_seconds):
                text = self._config.greeting.format(name=track.identity)
                events.append(SpeechEvent("greeting", text, (track.track_id,)))
        return events
