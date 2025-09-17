from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort

from sentinel.config import DetectorConfig
from sentinel.detection.decode import decode_yolov8
from sentinel.detection.preprocess import prepare_input
from sentinel.detection.types import Detection


def build_session(
    model_path: Path | str,
    providers: Sequence[str] = ("CPUExecutionProvider",),
    intra_op_threads: int = 4,
) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), options, providers=list(providers))


class OnnxFaceDetector:
    def __init__(
        self,
        model_path: Path | str,
        input_size: int = 416,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        intra_op_threads: int = 4,
    ) -> None:
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.session = build_session(model_path, providers, intra_op_threads)
        self.input_name = self.session.get_inputs()[0].name

    @classmethod
    def from_config(cls, config: DetectorConfig) -> "OnnxFaceDetector":
        return cls(
            model_path=config.model_path,
            input_size=config.input_size,
            conf_threshold=config.conf_threshold,
            iou_threshold=config.iou_threshold,
            providers=config.providers,
            intra_op_threads=config.intra_op_threads,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        tensor, meta = prepare_input(frame, self.input_size)
        output = self.session.run(None, {self.input_name: tensor})[0]
        return decode_yolov8(output, meta, self.conf_threshold, self.iou_threshold)
