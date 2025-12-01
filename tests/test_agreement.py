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


class FixedDetector:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, image):
        return list(self._detections)


def test_summary_aggregates_across_images():
    from sentinel.export.agreement import MatchResult, summarise_matches

    summary = summarise_matches(
        [MatchResult(2, 2, 3, (0.9, 0.8), (0.1, 0.1)), MatchResult(0, 1, 0, (), ())]
    )
    assert summary.images == 2
    assert summary.recall == pytest.approx(2 / 3)
    assert summary.precision == pytest.approx(2 / 3)
    assert summary.mean_iou == pytest.approx(0.85)


def test_compare_detectors_runs_both_on_every_image():
    from sentinel.export.agreement import compare_detectors

    summary = compare_detectors(
        FixedDetector([det(0, 0)]), FixedDetector([det(1, 1), det(300, 300)]), [object(), object()]
    )
    assert summary.matched_faces == 2
    assert summary.candidate_faces == 4
    assert summary.recall == 1.0
    assert summary.precision == 0.5


def test_empty_summary_defaults():
    from sentinel.export.agreement import summarise_matches

    summary = summarise_matches([])
    assert summary.recall == 1.0
    assert summary.mean_iou == 0.0
