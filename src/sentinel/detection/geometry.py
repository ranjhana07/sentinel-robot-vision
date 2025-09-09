import numpy as np


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = np.empty_like(boxes)
    half_w = boxes[:, 2] / 2.0
    half_h = boxes[:, 3] / 2.0
    converted[:, 0] = boxes[:, 0] - half_w
    converted[:, 1] = boxes[:, 1] - half_h
    converted[:, 2] = boxes[:, 0] + half_w
    converted[:, 3] = boxes[:, 1] + half_h
    return converted


def box_area(boxes: np.ndarray) -> np.ndarray:
    widths = np.clip(boxes[..., 2] - boxes[..., 0], 0.0, None)
    heights = np.clip(boxes[..., 3] - boxes[..., 1], 0.0, None)
    return widths * heights


def iou_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if len(first) == 0 or len(second) == 0:
        return np.zeros((len(first), len(second)), dtype=np.float32)
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    overlap = np.clip(bottom_right - top_left, 0.0, None)
    intersection = overlap[..., 0] * overlap[..., 1]
    union = box_area(first)[:, None] + box_area(second)[None, :] - intersection
    return intersection / np.maximum(union, 1e-9)
