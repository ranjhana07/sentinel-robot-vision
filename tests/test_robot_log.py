import json
from datetime import datetime, timezone

import pytest

from sentinel.robotlog.logger import RobotLogger


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_records_are_json_lines_with_standard_fields(tmp_path):
    fixed = datetime(2025, 11, 4, 9, 12, 41, tzinfo=timezone.utc)
    logger = RobotLogger(tmp_path, now=lambda: fixed, robot_id="r2")
    logger.log("face_detected", track=3, score=0.82)
    logger.close()
    record = read_lines(logger.path)[0]
    assert record["ts"] == "2025-11-04T09:12:41.000+00:00"
    assert record["level"] == "info"
    assert record["event"] == "face_detected"
    assert record["robot"] == "r2"
    assert record["track"] == 3


def test_unserialisable_values_are_stringified(tmp_path):
    logger = RobotLogger(tmp_path)
    logger.log("startup", model=tmp_path / "model.onnx")
    logger.close()
    assert read_lines(logger.path)[0]["model"].endswith("model.onnx")


def test_unknown_level_is_rejected(tmp_path):
    logger = RobotLogger(tmp_path)
    with pytest.raises(ValueError):
        logger.log("oops", level="fatal")
    logger.close()


def test_file_rotates_and_keeps_limited_backups(tmp_path):
    logger = RobotLogger(tmp_path, max_bytes=400, backup_count=2)
    for index in range(60):
        logger.log("tick", index=index, payload="x" * 20)
    logger.close()
    names = sorted(path.name for path in tmp_path.iterdir())
    assert names == ["robot.jsonl", "robot.jsonl.1", "robot.jsonl.2"]
    for path in tmp_path.iterdir():
        assert all("event" in record for record in read_lines(path))


def test_two_loggers_do_not_share_handlers(tmp_path):
    first = RobotLogger(tmp_path / "a")
    second = RobotLogger(tmp_path / "b")
    first.log("one")
    second.log("two")
    first.close()
    second.close()
    assert len(read_lines(first.path)) == 1
    assert len(read_lines(second.path)) == 1


def test_concurrent_writers_produce_whole_lines(tmp_path):
    import threading

    logger = RobotLogger(tmp_path, max_bytes=10_000_000)

    def worker(name):
        for index in range(200):
            logger.log("tick", writer=name, index=index)

    threads = [threading.Thread(target=worker, args=(f"t{n}",)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    logger.close()
    assert len(read_lines(logger.path)) == 800
