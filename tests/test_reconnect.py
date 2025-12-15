import numpy as np
import pytest

from sentinel.capture.reconnect import ReconnectingSource

FRAME = np.zeros((2, 2, 3), dtype=np.uint8)


class ScriptedSource:
    def __init__(self, script):
        self.script = list(script)
        self.released = False

    def read(self):
        return self.script.pop(0) if self.script else None

    def release(self):
        self.released = True


class Factory:
    def __init__(self, sources):
        self.sources = list(sources)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        item = self.sources.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make(factory, events=None, **kwargs):
    sleeps = []
    source = ReconnectingSource(
        factory,
        sleep=sleeps.append,
        on_event=(lambda name, fields: events.append((name, fields))) if events is not None else None,
        **kwargs,
    )
    return source, sleeps


def test_healthy_source_passes_frames_through():
    factory = Factory([ScriptedSource([FRAME, FRAME])])
    source, sleeps = make(factory)
    assert source.read() is FRAME
    assert source.read() is FRAME
    assert factory.calls == 1
    assert sleeps == []


def test_transient_failures_do_not_trigger_reconnect():
    factory = Factory([ScriptedSource([None, None, FRAME])])
    source, _ = make(factory, max_failures=5)
    assert source.read() is FRAME
    assert factory.calls == 1


def test_reconnects_after_consecutive_failures():
    first = ScriptedSource([])
    second = ScriptedSource([FRAME])
    events = []
    source, sleeps = make(Factory([first, second]), events, max_failures=3)
    assert source.read() is FRAME
    assert first.released
    assert sleeps == [0.5]
    names = [name for name, _ in events]
    assert names.count("camera_read_failed") == 3
    assert "camera_reconnect" in names


def test_backoff_doubles_and_is_capped():
    factory = Factory([ScriptedSource([]), OSError("busy"), OSError("busy"), OSError("busy"), OSError("busy")])
    events = []
    source, sleeps = make(factory, events, max_failures=1, max_attempts=4, base_delay=1.0, max_delay=4.0)
    assert source.read() is None
    assert sleeps == [1.0, 2.0, 4.0, 4.0]
    assert [n for n, _ in events].count("camera_reconnect_failed") == 4


def test_gives_up_and_returns_none_afterwards():
    factory = Factory([ScriptedSource([]), OSError("gone")])
    source, _ = make(factory, max_failures=1, max_attempts=1)
    assert source.read() is None
    assert source.read() is None


def test_release_closes_current_source():
    inner = ScriptedSource([FRAME])
    source, _ = make(Factory([inner]))
    source.release()
    assert inner.released
