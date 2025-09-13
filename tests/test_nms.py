import numpy as np

from sentinel.detection.geometry import iou_matrix, xywh_to_xyxy
from sentinel.detection.nms import nms


def test_nms_empty_input():
    assert nms(np.zeros((0, 4)), np.zeros(0), 0.5).size == 0


def test_nms_suppresses_overlapping_lower_score():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.8, 0.9, 0.7], dtype=np.float32)
    assert nms(boxes, scores, 0.5).tolist() == [1, 2]


def test_nms_keeps_disjoint_boxes_in_score_order():
    boxes = np.array([[0, 0, 5, 5], [10, 10, 15, 15], [20, 20, 25, 25]], dtype=np.float32)
    scores = np.array([0.2, 0.9, 0.5], dtype=np.float32)
    assert nms(boxes, scores, 0.5).tolist() == [1, 2, 0]


def test_iou_matrix_values():
    first = np.array([[0, 0, 10, 10]], dtype=np.float32)
    second = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], dtype=np.float32)
    result = iou_matrix(first, second)
    assert result[0, 0] == 1.0
    assert abs(result[0, 1] - 1 / 3) < 1e-6
    assert result[0, 2] == 0.0


def test_xywh_to_xyxy():
    boxes = np.array([[10, 10, 4, 6]], dtype=np.float32)
    assert xywh_to_xyxy(boxes).tolist() == [[8.0, 7.0, 12.0, 13.0]]


def test_box_exactly_at_threshold_is_kept():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 5, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    assert nms(boxes, scores, 0.5).tolist() == [0, 1]
