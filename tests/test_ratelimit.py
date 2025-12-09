from sentinel.speech.ratelimit import SlidingWindowLimiter


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_limits_events_inside_window():
    clock = Clock()
    limiter = SlidingWindowLimiter(2, 60.0, clock)
    assert limiter.allow()
    assert limiter.allow()
    assert not limiter.allow()


def test_window_slides_forward():
    clock = Clock()
    limiter = SlidingWindowLimiter(2, 60.0, clock)
    limiter.allow()
    clock.now = 30.0
    limiter.allow()
    clock.now = 59.0
    assert not limiter.allow()
    clock.now = 60.5
    assert limiter.allow()
    assert not limiter.allow()


def test_zero_means_unlimited():
    limiter = SlidingWindowLimiter(0)
    assert all(limiter.allow() for _ in range(100))
