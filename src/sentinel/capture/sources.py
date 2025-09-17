from typing import Protocol

import cv2
import numpy as np

from sentinel.config import SourceConfig


class FrameSource(Protocol):
    def read(self) -> np.ndarray | None: ...

    def release(self) -> None: ...


class OpenCVSource:
    def __init__(self, target: int | str, width: int = 640, height: int = 480) -> None:
        self._capture = cv2.VideoCapture(target)
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open video source {target!r}")
        if isinstance(target, int):
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> np.ndarray | None:
        ok, frame = self._capture.read()
        return frame if ok else None

    def release(self) -> None:
        self._capture.release()


def build_source(config: SourceConfig) -> FrameSource:
    if config.kind == "picamera":
        from sentinel.capture.picamera import PicameraSource

        return PicameraSource(config.width, config.height)
    target = config.target
    if config.kind == "camera":
        target = int(target)
    return OpenCVSource(target, config.width, config.height)
