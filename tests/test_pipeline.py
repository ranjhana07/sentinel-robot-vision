import json
import time

import numpy as np

from sentinel.config import AppConfig, SourceConfig, SpeechConfig
from sentinel.detection.types import Detection
from sentinel.pipeline import Pipeline
from sentinel.robotlog.logger import RobotLogger
from sentinel.speech.policy import SpeechPolicy
from sentinel.tracking.identity import UnknownResolver
from sentinel.tracking.tracker import IouTracker


class ScriptedSource:
    def __init__(self, frames):
        self._frames = list(frames)
        self.released = False

    def read(self):
        return self._frames.pop(0) if self._frames else None

    def release(self):
        self.released = True


class ScriptedDetector:
    def __init__(self, per_frame):
        self._per_frame = list(per_frame)

    def detect(self, frame):
        return self._per_frame.pop(0) if self._per_frame else []


class RecordingWorker:
    def __init__(self):
        self.said = []
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True
        return self

    def say(self, text, priority=1):
        self.said.append(text)
        return True

    def stop(self):
        self.stopped = True


def face(x):
    return Detection(box=(x, 60.0, x + 40.0, 120.0), score=0.9)


def build(tmp_path, per_frame, max_frames=None):
    frames = [np.zeros((200, 300, 3), dtype=np.uint8) for _ in per_frame]
    config = AppConfig(
        source=SourceConfig(kind="video", target="clip.mp4"),
        speech=SpeechConfig(unknown_cooldown_seconds=60),
    )
    worker = RecordingWorker()
    source = ScriptedSource(frames)
    pipeline = Pipeline(
        config=config,
        detector=ScriptedDetector(per_frame),
        source=source,
        tracker=IouTracker(min_hits=2),
        policy=SpeechPolicy(config.speech),
        worker=worker,
        robot_log=RobotLogger(tmp_path),
        resolver=UnknownResolver(),
    )
    return pipeline, worker, source


def read_events(tmp_path):
    lines = (tmp_path / "robot.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_unknown_face_is_announced_once_and_logged(tmp_path):
    pipeline, worker, source = build(tmp_path, [[face(50)], [face(52)], [face(54)], [face(55)]])
    assert pipeline.run() == 4
    assert worker.said == ["Unknown person detected"]
    assert worker.started and worker.stopped and source.released
    events = [event["event"] for event in read_events(tmp_path)]
    assert events[0] == "startup"
    assert events.count("face_detected") == 1
    assert events.count("speech") == 1
    assert events[-1] == "shutdown"


def test_pipeline_stops_after_max_frames(tmp_path):
    pipeline, _, _ = build(tmp_path, [[], [], [], []])
    assert pipeline.run(max_frames=2) == 2


def test_stop_request_ends_the_loop(tmp_path):
    pipeline, _, _ = build(tmp_path, [[]] * 5)
    pipeline.request_stop()
    assert pipeline.run() == 0


class StallingSource:
    def __init__(self, stall_seconds):
        self._stall_seconds = stall_seconds
        self._calls = 0

    def read(self):
        self._calls += 1
        if self._calls == 1:
            return np.zeros((200, 300, 3), dtype=np.uint8)
        if self._calls == 2:
            time.sleep(self._stall_seconds)
            return np.zeros((200, 300, 3), dtype=np.uint8)
        return None

    def release(self):
        pass


def test_stalled_source_is_reported_once_and_recovers(tmp_path):
    config = AppConfig(source=SourceConfig(kind="video", target="x", stall_seconds=0.1))
    pipeline = Pipeline(
        config=config,
        detector=ScriptedDetector([]),
        source=StallingSource(0.5),
        tracker=IouTracker(),
        policy=SpeechPolicy(config.speech),
        worker=None,
        robot_log=RobotLogger(tmp_path),
        resolver=UnknownResolver(),
        read_timeout=0.05,
    )
    assert pipeline.run() == 2
    events = [event["event"] for event in read_events(tmp_path)]
    assert events.count("frame_stall") == 1
    assert events.count("frame_resume") == 1


def test_unknown_face_snapshot_is_saved_and_logged(tmp_path):
    from sentinel.snapshots import SnapshotWriter

    pipeline, _, _ = build(tmp_path / "logs", [[face(50)], [face(52)], [face(54)]])
    pipeline.snapshots = SnapshotWriter(tmp_path / "snaps")
    pipeline.run()
    events = read_events(tmp_path / "logs")
    snapshot_events = [event for event in events if event["event"] == "snapshot"]
    assert len(snapshot_events) == 1
    assert len(list((tmp_path / "snaps").glob("*.jpg"))) == 1
