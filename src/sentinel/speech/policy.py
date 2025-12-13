import time
from dataclasses import dataclass
from datetime import datetime
from datetime import time as datetime_time
from typing import Callable, Sequence

from sentinel.config import SpeechConfig
from sentinel.speech.cooldown import CooldownTable
from sentinel.speech.ratelimit import SlidingWindowLimiter
from sentinel.tracking.tracker import Track


PRIORITY_ZONE = 0
PRIORITY_UNKNOWN = 1
PRIORITY_GREETING = 2


@dataclass(frozen=True)
class SpeechEvent:
    kind: str
    text: str
    track_ids: tuple[int, ...]
    priority: int = 1


class SpeechPolicy:
    PURGE_AFTER_SECONDS = 600.0

    def __init__(
        self,
        config: SpeechConfig,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime_time] = lambda: datetime.now().time(),
    ) -> None:
        self._config = config
        self._cooldowns = CooldownTable(clock)
        self._wall_clock = wall_clock
        self._limiter = SlidingWindowLimiter(config.max_phrases_per_minute, 60.0, clock)

    def evaluate(
        self,
        tracks: Sequence[Track],
        frame_shape: tuple[int, int] | None = None,
    ) -> list[SpeechEvent]:
        events: list[SpeechEvent] = []
        if frame_shape is not None:
            events.extend(self._check_zones(tracks, frame_shape))
        routine = self._announce_unknown(tracks) + self._greet_known(tracks)
        events.extend(event for event in routine if self._limiter.allow())
        self._cooldowns.purge(self.PURGE_AFTER_SECONDS)
        return events

    def _suppressed(self, kind: str) -> bool:
        quiet = self._config.quiet_hours
        return quiet is not None and kind in quiet.suppress and quiet.contains(self._wall_clock())

    def _announce_unknown(self, tracks: Sequence[Track]) -> list[SpeechEvent]:
        if self._suppressed("unknown"):
            return []
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
        return [SpeechEvent("unknown", text, tuple(track.track_id for track in fresh), PRIORITY_UNKNOWN)]

    def _greet_known(self, tracks: Sequence[Track]) -> list[SpeechEvent]:
        if self._suppressed("greeting"):
            return []
        events = []
        for track in tracks:
            if track.identity is None:
                continue
            if self._cooldowns.try_acquire(("greet", track.identity), self._config.cooldown_seconds):
                text = self._config.greeting.format(name=track.identity)
                events.append(SpeechEvent("greeting", text, (track.track_id,), PRIORITY_GREETING))
        return events

    def _check_zones(self, tracks: Sequence[Track], frame_shape: tuple[int, int]) -> list[SpeechEvent]:
        height, width = frame_shape
        events = []
        for rule in self._config.zones:
            cooldown = rule.cooldown_seconds or self._config.cooldown_seconds
            for track in tracks:
                if not self._rule_applies(rule.applies_to, track):
                    continue
                center_x, center_y = track.center
                if not point_in_area(center_x / width, center_y / height, rule.area):
                    continue
                if self._cooldowns.try_acquire(("zone", rule.name, track.track_id), cooldown):
                    events.append(SpeechEvent(f"zone:{rule.name}", rule.phrase, (track.track_id,), PRIORITY_ZONE))
        return events

    @staticmethod
    def _rule_applies(applies_to: str, track: Track) -> bool:
        if applies_to == "any":
            return True
        if applies_to == "known":
            return track.identity is not None
        return track.identity is None


def point_in_area(x: float, y: float, area: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = area
    return x1 <= x <= x2 and y1 <= y <= y2
