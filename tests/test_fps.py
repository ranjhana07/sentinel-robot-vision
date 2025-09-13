import pytest

from sentinel.utils.fps import FpsMeter


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_fps_is_zero_before_two_ticks():
    meter = FpsMeter(clock=FakeClock())
    assert meter.fps == 0.0
    meter.tick()
    assert meter.fps == 0.0


def test_fps_from_constant_interval():
    clock = FakeClock()
    meter = FpsMeter(clock=clock)
    for _ in range(11):
        meter.tick()
        clock.now += 0.1
    assert meter.fps == pytest.approx(10.0)


def test_window_discards_old_samples():
    clock = FakeClock()
    meter = FpsMeter(window=5, clock=clock)
    for _ in range(10):
        meter.tick()
        clock.now += 1.0
    for _ in range(5):
        meter.tick()
        clock.now += 0.1
    assert meter.fps == pytest.approx(10.0)
