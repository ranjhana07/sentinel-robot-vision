import threading
import time

from sentinel.speech.worker import SpeechWorker


class RecordingBackend:
    def __init__(self, delay=0.0, fail_on=None):
        self.spoken = []
        self.delay = delay
        self.fail_on = fail_on
        self.closed = False
        self.release = threading.Event()

    def speak(self, text):
        if self.fail_on == text:
            raise RuntimeError("audio device busy")
        if self.delay:
            self.release.wait(self.delay)
        self.spoken.append(text)

    def close(self):
        self.closed = True


def test_messages_are_spoken_in_order():
    backend = RecordingBackend()
    worker = SpeechWorker(backend).start()
    for phrase in ("one", "two", "three"):
        assert worker.say(phrase)
    worker.stop()
    assert backend.spoken == ["one", "two", "three"]
    assert backend.closed


def test_say_does_not_block_when_queue_is_full():
    backend = RecordingBackend(delay=5.0)
    worker = SpeechWorker(backend, max_queue=1).start()
    worker.say("first")
    time.sleep(0.05)
    assert worker.say("second") is True
    started = time.perf_counter()
    assert worker.say("third") is False
    assert time.perf_counter() - started < 0.1
    backend.release.set()
    worker.stop()


def test_backend_failure_does_not_kill_worker():
    backend = RecordingBackend(fail_on="bad")
    worker = SpeechWorker(backend).start()
    worker.say("bad")
    worker.say("good")
    worker.stop()
    assert backend.spoken == ["good"]


def test_urgent_phrase_jumps_the_queue():
    backend = RecordingBackend(delay=5.0)
    worker = SpeechWorker(backend, max_queue=4).start()
    worker.say("first")
    time.sleep(0.05)
    worker.say("greeting", priority=2)
    worker.say("alert", priority=0)
    backend.release.set()
    worker.stop()
    assert backend.spoken == ["first", "alert", "greeting"]
