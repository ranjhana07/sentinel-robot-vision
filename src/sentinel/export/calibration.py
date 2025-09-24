from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from onnxruntime.quantization import CalibrationDataReader

from sentinel.detection.preprocess import prepare_input

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(folder: Path | str, limit: int | None = None) -> list[Path]:
    paths = sorted(
        path for path in Path(folder).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    return paths[:limit] if limit else paths


class ImageFolderCalibrationReader(CalibrationDataReader):
    def __init__(
        self,
        folder: Path | str,
        input_name: str,
        input_size: int = 416,
        max_images: int | None = 200,
    ) -> None:
        self._paths = list_images(folder, max_images)
        if not self._paths:
            raise ValueError(f"no calibration images found in {folder}")
        self._input_name = input_name
        self._input_size = input_size
        self._cursor: Iterator[Path] = iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)

    def get_next(self) -> dict[str, np.ndarray] | None:
        for path in self._cursor:
            image = cv2.imread(str(path))
            if image is None:
                continue
            tensor, _ = prepare_input(image, self._input_size)
            return {self._input_name: tensor}
        return None

    def rewind(self) -> None:
        self._cursor = iter(self._paths)
