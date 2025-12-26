from datetime import datetime, timedelta, timezone

import numpy as np

from sentinel.snapshots import SnapshotWriter, padded_crop
from sentinel.tracking.tracker import Track


def frame():
    return np.random.default_rng(0).integers(0, 255, (200, 300, 3), dtype=np.uint8)


def test_padded_crop_grows_box_and_clips_to_frame():
    crop = padded_crop(frame(), (10.0, 10.0, 50.0, 70.0), 0.25)
    assert crop.shape == (85, 60, 3)
    assert padded_crop(frame(), (0.0, 0.0, 40.0, 40.0), 0.5).shape == (60, 60, 3)


def test_snapshot_is_written_with_track_in_name(tmp_path):
    clock = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)
    writer = SnapshotWriter(tmp_path / "snaps", now=lambda: clock)
    path = writer.save(frame(), Track(7, (50.0, 50.0, 120.0, 130.0), 0.9))
    assert path is not None and path.exists()
    assert "track7" in path.name


def test_empty_crop_is_skipped(tmp_path):
    writer = SnapshotWriter(tmp_path)
    assert writer.save(frame(), Track(1, (900.0, 900.0, 950.0, 950.0), 0.9)) is None


def test_oldest_snapshots_are_pruned(tmp_path):
    clock = {"now": datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)}

    def tick():
        clock["now"] += timedelta(seconds=1)
        return clock["now"]

    writer = SnapshotWriter(tmp_path, max_files=3, now=tick)
    saved = [writer.save(frame(), Track(i, (50.0, 50.0, 120.0, 130.0), 0.9)) for i in range(6)]
    remaining = sorted(p.name for p in tmp_path.glob("*.jpg"))
    assert len(remaining) == 3
    assert remaining == sorted(p.name for p in saved[-3:])
