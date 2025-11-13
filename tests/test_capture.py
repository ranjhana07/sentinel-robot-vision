import time

import numpy as np

from sentinel.capture.threaded import ThreadedCapture


class CountingSource:
    def __init__(self, total=None, delay=0.0):
        self.total = total
        self.delay = delay
        self.produced = 0
        self.released = False

    def read(self):
        if self.total is not None and self.produced >= self.total:
            return None
        time.sleep(self.delay)
        self.produced += 1
        return np.full((4, 4, 3), self.produced % 255, dtype=np.uint8)

    def release(self):
        self.released = True


def test_stop_returns_even_if_nobody_reads():
    source = CountingSource()
    capture = ThreadedCapture(source, queue_size=2).start()
    time.sleep(0.1)
    started = time.perf_counter()
    capture.stop()
    assert time.perf_counter() - started < 1.0
    assert source.released


def test_slow_consumer_sees_recent_frames():
    source = CountingSource(delay=0.001)
    capture = ThreadedCapture(source, queue_size=2).start()
    time.sleep(0.2)
    frame = capture.read(timeout=0.5)
    capture.stop()
    assert source.produced > 10
    assert int(frame[0, 0, 0]) > 5


def test_finite_source_reports_exhausted_after_drain():
    capture = ThreadedCapture(CountingSource(total=2), queue_size=4).start()
    seen = 0
    for _ in range(50):
        if capture.read(timeout=0.1) is not None:
            seen += 1
        if capture.exhausted:
            break
    capture.stop()
    assert seen == 2
    assert capture.exhausted


def test_blocking_mode_keeps_every_frame():
    capture = ThreadedCapture(CountingSource(total=6), queue_size=1, drop_stale=False).start()
    seen = []
    for _ in range(100):
        frame = capture.read(timeout=0.1)
        if frame is not None:
            seen.append(int(frame[0, 0, 0]))
        if capture.exhausted:
            break
    capture.stop()
    assert seen == [1, 2, 3, 4, 5, 6]
