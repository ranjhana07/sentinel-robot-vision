import numpy as np

from sentinel.detection.geometry import iou_matrix


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    kept: list[int] = []
    while order.size > 0:
        best = int(order[0])
        kept.append(best)
        if order.size == 1:
            break
        overlaps = iou_matrix(boxes[best:best + 1], boxes[order[1:]])[0]
        order = order[1:][overlaps <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)
