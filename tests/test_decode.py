import numpy as np
import pytest

from sentinel.detection.decode import decode_yolov8
from sentinel.detection.preprocess import LetterboxMeta


def make_output(rows: list[tuple[float, float, float, float, float]]) -> np.ndarray:
    return np.array(rows, dtype=np.float32).T[None]


@pytest.fixture
def identity_meta() -> LetterboxMeta:
    return LetterboxMeta(scale=1.0, pad_left=0, pad_top=0, original_height=416, original_width=416)


def test_low_confidence_rows_are_dropped(identity_meta):
    output = make_output([(100, 100, 40, 40, 0.1), (200, 200, 40, 40, 0.2)])
    assert decode_yolov8(output, identity_meta, 0.35, 0.45) == []


def test_single_detection_is_converted_to_corners(identity_meta):
    output = make_output([(100, 100, 40, 60, 0.9)])
    detections = decode_yolov8(output, identity_meta, 0.35, 0.45)
    assert len(detections) == 1
    assert detections[0].box == pytest.approx((80.0, 70.0, 120.0, 130.0))
    assert detections[0].score == pytest.approx(0.9)


def test_duplicates_are_merged_by_nms(identity_meta):
    output = make_output(
        [(100, 100, 40, 40, 0.9), (102, 101, 40, 40, 0.8), (300, 300, 40, 40, 0.7)]
    )
    detections = decode_yolov8(output, identity_meta, 0.35, 0.45)
    assert [round(d.score, 1) for d in detections] == [0.9, 0.7]


def test_boxes_are_mapped_back_through_letterbox():
    meta = LetterboxMeta(scale=0.5, pad_left=0, pad_top=52, original_height=624, original_width=832)
    output = make_output([(208, 208, 100, 100, 0.9)])
    detection = decode_yolov8(output, meta, 0.35, 0.45)[0]
    assert detection.box == pytest.approx((316.0, 212.0, 516.0, 412.0))


def test_max_detections_is_respected(identity_meta):
    rows = [(20 + 30 * i, 50, 10, 10, 0.9 - i * 0.01) for i in range(10)]
    detections = decode_yolov8(make_output(rows), identity_meta, 0.35, 0.45, max_detections=3)
    assert len(detections) == 3
