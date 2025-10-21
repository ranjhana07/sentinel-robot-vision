import cv2
import numpy as np
import pytest

from sentinel.export.frames import extract_frames


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for index in range(30):
        writer.write(np.full((48, 64, 3), index * 8, dtype=np.uint8))
    writer.release()
    return path


def test_extract_every_nth_frame(video, tmp_path):
    paths = extract_frames(video, tmp_path / "out", every=10)
    assert [p.name for p in paths] == ["frame_000000.jpg", "frame_000010.jpg", "frame_000020.jpg"]
    assert all(p.exists() for p in paths)


def test_extract_respects_limit(video, tmp_path):
    assert len(extract_frames(video, tmp_path / "out", every=1, limit=4)) == 4


def test_missing_video_raises(tmp_path):
    with pytest.raises(RuntimeError):
        extract_frames(tmp_path / "nope.avi", tmp_path / "out")
