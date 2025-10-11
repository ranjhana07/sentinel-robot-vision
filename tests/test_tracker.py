from sentinel.detection.types import Detection
from sentinel.tracking.tracker import IouTracker


def face(x, y, size=50, score=0.9):
    return Detection(box=(x, y, x + size, y + size), score=score)


def test_track_needs_min_hits_before_it_is_reported():
    tracker = IouTracker(min_hits=2)
    assert tracker.update([face(10, 10)]) == []
    tracks = tracker.update([face(12, 11)])
    assert len(tracks) == 1
    assert tracks[0].track_id == 1


def test_identity_is_kept_while_face_moves():
    tracker = IouTracker(min_hits=1)
    first = tracker.update([face(10, 10)])[0].track_id
    second = tracker.update([face(20, 15)])[0].track_id
    assert first == second


def test_two_faces_get_separate_ids():
    tracker = IouTracker(min_hits=1)
    tracks = tracker.update([face(0, 0), face(300, 300)])
    assert sorted(t.track_id for t in tracks) == [1, 2]


def test_track_expires_after_max_age():
    tracker = IouTracker(min_hits=1, max_age=2)
    tracker.update([face(10, 10)])
    for _ in range(3):
        tracker.update([])
    tracks = tracker.update([face(10, 10)])
    assert tracks[0].track_id == 2


def test_track_survives_short_occlusion():
    tracker = IouTracker(min_hits=1, max_age=5)
    original = tracker.update([face(10, 10)])[0].track_id
    tracker.update([])
    tracker.update([])
    again = tracker.update([face(11, 10)])[0].track_id
    assert original == again


def test_greedy_matching_prefers_highest_overlap():
    tracker = IouTracker(min_hits=1, iou_threshold=0.1)
    tracker.update([face(0, 0), face(40, 0)])
    tracks = {t.track_id: t for t in tracker.update([face(41, 0), face(1, 0)])}
    assert tracks[1].box[0] == 1
    assert tracks[2].box[0] == 41


def test_unmatched_low_overlap_creates_new_track():
    tracker = IouTracker(min_hits=1, iou_threshold=0.5)
    tracker.update([face(0, 0)])
    tracks = tracker.update([face(45, 0)])
    assert [t.track_id for t in tracks] == [2]
