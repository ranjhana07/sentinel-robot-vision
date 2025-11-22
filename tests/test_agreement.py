import pytest

from sentinel.detection.types import Detection
from sentinel.export.agreement import match_detections


def det(x, y, size=50, score=0.9):
    return Detection((x, y, x + size, y + size), score)


def test_identical_detections_fully_match():
    result = match_detections([det(0, 0), det(100, 100)], [det(0, 0), det(100, 100)])
    assert result.matched == 2
    assert all(iou == pytest.approx(1.0) for iou in result.ious)


def test_each_candidate_is_used_once():
    result = match_detections([det(0, 0), det(2, 2)], [det(1, 1)])
    assert result.matched == 1
    assert result.candidate_count == 1


def test_missing_and_extra_detections_are_counted():
    result = match_detections([det(0, 0), det(300, 300)], [det(1, 1), det(600, 600)])
    assert result.matched == 1
    assert result.reference_count == 2
    assert result.candidate_count == 2


def test_score_delta_is_absolute_difference():
    result = match_detections([det(0, 0, score=0.9)], [det(0, 0, score=0.7)])
    assert result.score_deltas[0] == pytest.approx(0.2)


def test_empty_inputs():
    assert match_detections([], [det(0, 0)]).matched == 0
    assert match_detections([det(0, 0)], []).matched == 0
