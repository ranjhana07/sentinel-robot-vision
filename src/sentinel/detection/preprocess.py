from dataclasses import dataclass

import cv2
import numpy as np

PAD_COLOR = 114


@dataclass(frozen=True)
class LetterboxMeta:
    scale: float
    pad_left: int
    pad_top: int
    original_height: int
    original_width: int


def compute_letterbox(height: int, width: int, size: int) -> tuple[float, int, int, int, int]:
    scale = min(size / height, size / width)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    pad_left = (size - new_width) // 2
    pad_top = (size - new_height) // 2
    return scale, new_width, new_height, pad_left, pad_top


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, LetterboxMeta]:
    height, width = image.shape[:2]
    scale, new_width, new_height, pad_left, pad_top = compute_letterbox(height, width, size)
    if (new_width, new_height) != (width, height):
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), PAD_COLOR, dtype=np.uint8)
    canvas[pad_top:pad_top + new_height, pad_left:pad_left + new_width] = image
    return canvas, LetterboxMeta(scale, pad_left, pad_top, height, width)


def to_tensor(canvas: np.ndarray) -> np.ndarray:
    rgb = canvas[:, :, ::-1]
    chw = rgb.transpose(2, 0, 1)
    tensor = np.ascontiguousarray(chw, dtype=np.float32) / 255.0
    return tensor[None]


def prepare_input(image: np.ndarray, size: int) -> tuple[np.ndarray, LetterboxMeta]:
    canvas, meta = letterbox(image, size)
    return to_tensor(canvas), meta
