import numpy as np

from sentinel.detection.geometry import xywh_to_xyxy
from sentinel.detection.nms import nms
from sentinel.detection.preprocess import LetterboxMeta
from sentinel.detection.types import Detection


def scale_boxes(boxes: np.ndarray, meta: LetterboxMeta) -> np.ndarray:
    scaled = boxes.astype(np.float32).copy()
    scaled[:, [0, 2]] -= meta.pad_left
    scaled[:, [1, 3]] -= meta.pad_top
    scaled /= meta.scale
    scaled[:, [0, 2]] = scaled[:, [0, 2]].clip(0, meta.original_height)
    scaled[:, [1, 3]] = scaled[:, [1, 3]].clip(0, meta.original_width)
    return scaled


def decode_yolov8(
    output: np.ndarray,
    meta: LetterboxMeta,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int = 50,
) -> list[Detection]:
    predictions = output[0] if output.ndim == 3 else output
    predictions = predictions.T
    scores = predictions[:, 4:].max(axis=1)
    confident = scores >= conf_threshold
    predictions = predictions[confident]
    scores = scores[confident]
    if len(predictions) == 0:
        return []
    boxes = xywh_to_xyxy(predictions[:, :4])
    kept = nms(boxes, scores, iou_threshold)[:max_detections]
    boxes = scale_boxes(boxes[kept], meta)
    return [
        Detection(box=tuple(float(v) for v in box), score=float(score))
        for box, score in zip(boxes, scores[kept])
    ]
