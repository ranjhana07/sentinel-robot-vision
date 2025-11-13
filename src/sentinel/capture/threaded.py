import queue
import threading

import numpy as np

from sentinel.capture.sources import FrameSource


class ThreadedCapture:
    def __init__(self, source: FrameSource, queue_size: int = 2, drop_stale: bool = True) -> None:
        self._source = source
        self._drop_stale = drop_stale
        self._frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._ended = threading.Event()
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)

    def start(self) -> "ThreadedCapture":
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            frame = self._source.read()
            if frame is None:
                self._ended.set()
                break
            if self._drop_stale:
                self._push_latest(frame)
            else:
                self._push_blocking(frame)

    def _push_latest(self, frame: np.ndarray) -> None:
        while True:
            try:
                self._frames.put_nowait(frame)
                return
            except queue.Full:
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    continue

    def _push_blocking(self, frame: np.ndarray) -> None:
        while not self._stop.is_set():
            try:
                self._frames.put(frame, timeout=0.1)
                return
            except queue.Full:
                continue

    def read(self, timeout: float = 0.5) -> np.ndarray | None:
        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def exhausted(self) -> bool:
        return self._ended.is_set() and self._frames.empty()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._source.release()
