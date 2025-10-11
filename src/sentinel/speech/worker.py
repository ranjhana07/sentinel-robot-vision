import logging
import queue
import threading

from sentinel.speech.tts import TtsBackend

logger = logging.getLogger(__name__)


class SpeechWorker:
    def __init__(self, backend: TtsBackend, max_queue: int = 8) -> None:
        self._backend = backend
        self._messages: queue.Queue[str | None] = queue.Queue(maxsize=max_queue)
        self._thread = threading.Thread(target=self._run, name="speech", daemon=True)

    def start(self) -> "SpeechWorker":
        self._thread.start()
        return self

    def say(self, text: str) -> bool:
        try:
            self._messages.put_nowait(text)
        except queue.Full:
            return False
        return True

    def _run(self) -> None:
        while True:
            text = self._messages.get()
            if text is None:
                return
            try:
                self._backend.speak(text)
            except Exception:
                logger.exception("speech backend failed")

    def stop(self, timeout: float = 5.0) -> None:
        self._messages.put(None)
        self._thread.join(timeout)
        self._backend.close()
