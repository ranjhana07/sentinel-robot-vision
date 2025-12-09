import pytest

from sentinel.config import SpeechConfig, ZoneRule
from sentinel.speech.cooldown import CooldownTable
from sentinel.speech.policy import SpeechPolicy, point_in_area
from sentinel.tracking.tracker import Track


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def track(track_id, x=100.0, identity=None):
    return Track(track_id, (x, 100.0, x + 40.0, 160.0), 0.9, hits=3, identity=identity)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def config():
    return SpeechConfig(cooldown_seconds=10, unknown_cooldown_seconds=30)


def test_cooldown_blocks_until_elapsed(clock):
    table = CooldownTable(clock)
    assert table.try_acquire("a", 10)
    assert not table.try_acquire("a", 10)
    clock.advance(9.9)
    assert not table.ready("a", 10)
    clock.advance(0.2)
    assert table.try_acquire("a", 10)


def test_cooldown_keys_are_independent(clock):
    table = CooldownTable(clock)
    assert table.try_acquire(("unknown", 1), 30)
    assert table.try_acquire(("unknown", 2), 30)


def test_cooldown_purge_removes_old_keys(clock):
    table = CooldownTable(clock)
    table.try_acquire("old", 1)
    clock.advance(100)
    table.try_acquire("new", 1)
    assert table.purge(50) == 1
    assert len(table) == 1


def test_unknown_face_is_announced_once_per_cooldown(config, clock):
    policy = SpeechPolicy(config, clock)
    first = policy.evaluate([track(1)])
    assert [e.text for e in first] == [config.unknown_phrase]
    assert policy.evaluate([track(1)]) == []
    clock.advance(31)
    assert len(policy.evaluate([track(1)])) == 1


def test_group_phrase_uses_total_unknown_count(config, clock):
    policy = SpeechPolicy(config, clock)
    policy.evaluate([track(1)])
    events = policy.evaluate([track(1), track(2, x=300)])
    assert events[0].text == "2 unknown people detected"
    assert events[0].track_ids == (2,)


def test_known_face_gets_greeting_with_cooldown(config, clock):
    policy = SpeechPolicy(config, clock)
    events = policy.evaluate([track(1, identity="Asha")])
    assert events[0].text == "Hello Asha"
    assert policy.evaluate([track(5, identity="Asha")]) == []
    clock.advance(11)
    assert len(policy.evaluate([track(5, identity="Asha")])) == 1


def test_known_faces_do_not_count_as_unknown(config, clock):
    policy = SpeechPolicy(config, clock)
    events = policy.evaluate([track(1, identity="Asha"), track(2, x=300)])
    kinds = {event.kind: event for event in events}
    assert kinds["unknown"].text == config.unknown_phrase


def test_zone_rule_fires_once_for_track_inside_area(clock):
    rule = ZoneRule(name="vault", area=(0.5, 0.0, 1.0, 1.0), phrase="Restricted", applies_to="unknown")
    policy = SpeechPolicy(SpeechConfig(zones=[rule], cooldown_seconds=10), clock)
    inside = Track(1, (400.0, 100.0, 440.0, 160.0), 0.9, hits=3)
    first = policy.evaluate([inside], frame_shape=(480, 640))
    assert first[0].kind == "zone:vault"
    again = policy.evaluate([inside], frame_shape=(480, 640))
    assert [e.kind for e in again] == []


def test_zone_rule_ignores_track_outside_area(clock):
    rule = ZoneRule(name="vault", area=(0.5, 0.0, 1.0, 1.0), phrase="Restricted")
    policy = SpeechPolicy(SpeechConfig(zones=[rule]), clock)
    outside = Track(1, (40.0, 100.0, 80.0, 160.0), 0.9, hits=3, identity="Asha")
    kinds = [e.kind for e in policy.evaluate([outside], frame_shape=(480, 640))]
    assert "zone:vault" not in kinds


def test_zone_rule_respects_applies_to(clock):
    rule = ZoneRule(name="vault", area=(0.0, 0.0, 1.0, 1.0), phrase="Restricted", applies_to="unknown")
    policy = SpeechPolicy(SpeechConfig(zones=[rule]), clock)
    known = Track(1, (100.0, 100.0, 140.0, 160.0), 0.9, hits=3, identity="Asha")
    kinds = [e.kind for e in policy.evaluate([known], frame_shape=(480, 640))]
    assert "zone:vault" not in kinds


def test_point_in_area_edges():
    assert point_in_area(0.5, 0.5, (0.5, 0.5, 1.0, 1.0))
    assert not point_in_area(0.49, 0.5, (0.5, 0.5, 1.0, 1.0))


def test_event_priorities_order_zone_before_unknown_before_greeting(clock):
    rule = ZoneRule(name="vault", area=(0.0, 0.0, 1.0, 1.0), phrase="Restricted", applies_to="unknown")
    policy = SpeechPolicy(SpeechConfig(zones=[rule]), clock)
    tracks = [track(1), track(2, x=300, identity="Asha")]
    events = policy.evaluate(tracks, frame_shape=(480, 640))
    priorities = {event.kind: event.priority for event in events}
    assert priorities["zone:vault"] < priorities["unknown"] < priorities["greeting"]


def test_quiet_hours_suppress_greetings_but_not_zone_alerts(clock):
    from datetime import time

    from sentinel.config import QuietHours

    rule = ZoneRule(name="vault", area=(0.0, 0.0, 1.0, 1.0), phrase="Restricted")
    config = SpeechConfig(zones=[rule], quiet_hours=QuietHours(start="22:00", end="06:00"))
    policy = SpeechPolicy(config, clock, wall_clock=lambda: time(23, 30))
    events = policy.evaluate([track(1, identity="Asha")], frame_shape=(480, 640))
    assert [event.kind for event in events] == ["zone:vault"]


def test_suppressed_phrases_do_not_burn_their_cooldown(clock):
    from datetime import time

    from sentinel.config import QuietHours

    config = SpeechConfig(quiet_hours=QuietHours(start="22:00", end="06:00"), cooldown_seconds=600)
    moments = [time(23, 0), time(7, 0)]
    policy = SpeechPolicy(config, clock, wall_clock=lambda: moments.pop(0))
    assert policy.evaluate([track(1, identity="Asha")]) == []
    assert len(policy.evaluate([track(1, identity="Asha")])) == 1
