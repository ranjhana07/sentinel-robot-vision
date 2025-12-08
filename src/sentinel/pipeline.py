import logging
import threading
import time
from dataclasses import dataclass, field

import cv2

from sentinel.capture.sources import FrameSource, build_source
from sentinel.capture.threaded import ThreadedCapture
from sentinel.config import AppConfig
from sentinel.detection.onnx_engine import OnnxFaceDetector
from sentinel.render import draw_tracks
from sentinel.robotlog.logger import RobotLogger
from sentinel.speech.factory import build_backend
from sentinel.speech.policy import SpeechPolicy
from sentinel.speech.worker import SpeechWorker
from sentinel.tracking.identity import IdentityResolver, UnknownResolver
from sentinel.tracking.tracker import IouTracker
from sentinel.utils.fps import FpsMeter

logger = logging.getLogger(__name__)

FPS_LOG_INTERVAL_SECONDS = 10.0


@dataclass
class Pipeline:
    config: AppConfig
    detector: OnnxFaceDetector
    source: FrameSource
    tracker: IouTracker
    policy: SpeechPolicy
    worker: SpeechWorker | None
    robot_log: RobotLogger
    resolver: IdentityResolver
    display: bool = False
    _stop: threading.Event = field(default_factory=threading.Event)

    def request_stop(self) -> None:
        self._stop.set()

    def run(self, max_frames: int | None = None) -> int:
        capture = ThreadedCapture(
            self.source,
            self.config.source.queue_size,
            drop_stale=self.config.source.kind != "video",
        ).start()
        if self.worker is not None:
            self.worker.start()
        meter = FpsMeter()
        frames = 0
        last_fps_log = time.monotonic()
        self.robot_log.log("startup", model=str(self.config.detector.model_path))
        try:
            while not self._stop.is_set() and (max_frames is None or frames < max_frames):
                frame = capture.read(timeout=0.5)
                if frame is None:
                    if capture.exhausted:
                        break
                    continue
                tracks = self.tracker.update(self.detector.detect(frame))
                for track in tracks:
                    if not track.identity_checked:
                        track.identity = self.resolver.resolve(frame, track)
                        track.identity_checked = True
                        self.robot_log.log(
                            "face_detected",
                            track=track.track_id,
                            identity=track.identity or "unknown",
                            score=round(track.score, 3),
                            box=[round(v, 1) for v in track.box],
                        )
                for event in self.policy.evaluate(tracks, frame.shape[:2]):
                    queued = self.worker.say(event.text, event.priority) if self.worker else False
                    self.robot_log.log(
                        "speech", kind=event.kind, text=event.text, tracks=list(event.track_ids), queued=queued
                    )
                meter.tick()
                frames += 1
                if time.monotonic() - last_fps_log >= FPS_LOG_INTERVAL_SECONDS:
                    self.robot_log.log("fps", fps=round(meter.fps, 2))
                    last_fps_log = time.monotonic()
                if self.display:
                    cv2.imshow("sentinel", draw_tracks(frame, tracks, meter.fps))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            capture.stop()
            if self.worker is not None:
                self.worker.stop()
            if self.display:
                cv2.destroyAllWindows()
            self.robot_log.log("shutdown", frames=frames)
            self.robot_log.close()
        return frames


def build_pipeline(config: AppConfig, display: bool = False) -> Pipeline:
    worker = None
    if config.speech.enabled:
        worker = SpeechWorker(build_backend(config.speech), config.speech.queue_size)
    return Pipeline(
        config=config,
        detector=OnnxFaceDetector.from_config(config.detector),
        source=build_source(config.source),
        tracker=IouTracker(
            config.tracker.iou_threshold,
            config.tracker.max_age,
            config.tracker.min_hits,
            config.tracker.use_motion,
            config.tracker.velocity_smoothing,
        ),
        policy=SpeechPolicy(config.speech),
        worker=worker,
        robot_log=RobotLogger.from_config(config.logging),
        resolver=UnknownResolver(),
        display=display,
    )
